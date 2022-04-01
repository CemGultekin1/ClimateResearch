#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import os
import argparse
from datetime import date
import json
import copy
import scipy
import climate_data
import matplotlib.pyplot as plt
from itertools import product, combinations_with_replacement


# In[2]:


def ganlossfun(output,target,mask,eps=0):
    '''pout=torch.sum(output*mask,dim=(1,2,3))/torch.sum(mask,dim=(1,2,3))
    return torch.mean(torch.log((pout)*target+(1-target)*(1-pout)+eps))'''
    #pout=torch.sum(output*mask,dim=(1,2,3))/torch.sum(mask,dim=(1,2,3))
    return torch.sum((target-output)**2*mask)/torch.sum(mask)
def lossfun(output,target,mask,neglog=False,logloss=False,heteroscedastic=False,outsize=2):
    loss=0
    if neglog:
        if heteroscedastic:
            outsize+=1
        output,probs= torch.split(output, [outsize,output.shape[1]-outsize], dim=1)
        loss+=negloglikelihood(probs,mask)
    if heteroscedastic:
        loss+=heteroscedasticGaussianLoss(output, target,mask,outsize=outsize)
    else:
        output,_= torch.split(output, [outsize,output.shape[1]-outsize], dim=1)
        if logloss:
            loss+=logLoss(output, target,mask)
        else:
            loss+=l2Loss(output, target,mask)
    return loss
def negloglikelihood(probs,mask,eps=1e-2):
    numclass=probs.shape[1]
    N=torch.sum(mask)
    #-probs*torch.log10(probs + eps)-(1-probs)*torch.log10(1-probs+eps)
    probs=torch.mean(torch.log10(probs+eps)+torch.log10(1-probs+eps),dim=1,keepdim=True)
    probs=probs[mask>0]
    return torch.sum(probs)/N

def heteroscedasticGaussianLoss(output, target,mask,eps=1e-5,outsize=2):
    mean, precision = torch.split(output, [outsize,output.shape[1]-outsize], dim=1)
    precision=precision + eps
    nprecision=precision.shape[1]
    N=torch.sum(mask)
    if nprecision>1:
        err2=(target - mean)**2
        loss=torch.mean(                - 1 / 2 *  torch.log(precision)                 +  1 / 2 * err2 * precision                 + 1 / 2 * np.log(1e3),dim=1,keepdim=True)
        loss=loss[mask>0]
        loss=torch.sum(loss)
    else:
        err2=torch.sum((target - mean)**2,dim=1,keepdim=True)
        err2=err2[mask>0]
        precision=precision[mask>0]
        loss=torch.sum(                - 1 / 2 *  torch.log(precision)                 +  1 / 2 * err2 * precision                 + 1 / 2 * np.log(1e3))
    return loss/N

def logLoss(output, target,mask,eps=1e-5):
    l2 = torch.log( (target - output)**2 + eps) - np.log(eps)
    l2 = torch.sum(l2,dim=1,keepdim=True)
    l2=l2[mask>0]
    l2 = torch.mean(l2)
    return l2

def l2Loss(output, target,mask):
    l2 =  1 / 2 * (target - output)**2
    l2 = torch.sum(l2,dim=1,keepdim=True)
    l2=l2[mask>0]
    l2=torch.mean(l2)
    return l2


# In[3]:


def approximate_widths(def_width,def_filters,filters):
    nlyr=len(def_filters)
    num_param=np.zeros(nlyr)
    for i in range(nlyr):
        num_param[i]=def_width[i]*def_width[i+1]*(def_filters[i]**2)/(filters[i]**2)
    widths=np.zeros(nlyr+1)
    widths[0]=def_width[0]
    for i in range(1,nlyr+1):
        widths[i]=int(num_param[i-1]/widths[i-1])
    widths[-1]=def_width[-1]
    widths=[int(w) for w in widths]
    return widths[1:]
def lcnn_architecture(width_scale,filter_size,mode=0):
    widths=[128,64,32,32,32,32,32,3]
    widths=[np.ceil(width_scale*w) for w in widths]
    filters21=[5,5,3,3,3,3,3,3]
    if filter_size<21:
        filter_size=(filter_size//2)*2+1
        cursize=21
        filters=np.array(filters21)
        while cursize>filter_size:
            filters=filter_shrink_method(filters,mode)
            cursize=np.sum(filters)-len(filters)+1
    else:
        filters=filters21
    #widths=approximate_widths(widths,filters21,filters)
    net=LCNN()
    nparam0=net.nparam
    net=LCNN(filter_size=filters)
    nparam1=net.nparam
    rat=np.sqrt(nparam0/nparam1)
    widths=[int(np.ceil(rat*w)) for w in widths]
    widths[-1]=3
    net=LCNN(filter_size=filters,width=widths)
    return widths,filters,net.nparam
def filter_shrink_method(filters,mode):
    if mode==0:
        # Default
        i=np.where(filters==np.amax(filters))[0][-1]
        filters[i]-=2
    elif mode==1:
        # top-to-bottom equal shrink
        i=np.where(filters==np.amax(filters))[0][-1]
        filters[i]-=1
    elif mode==2:
        # top-to-bottom aggressive shrink
        i=np.where(filters!=1)[0][-1]
        filters[i]-=1
    elif mode==3:
        # bottom-to-top aggressive shrink
        i=np.where(filters!=1)[0][0]
        filters[i]-=1
    else:
        np.random.seed(mode)
        order=np.argsort(np.random.rand(len(filters)))
        I=np.where(filters==np.amax(filters))[0]
        I=np.array([i for i in order if i in I])
        i=I[0]
        filters[i]-=1
    return filters
def unet_receptive_field_compute(filter_size,pools,deep_filters):
    receptive_field=1
    for i in range(len(pools)):
        ww=np.sum(deep_filters[-1-i])-len(deep_filters[-1-i])
        receptive_field=(receptive_field+ww)*pools[-i-1]
    receptive_field+=np.sum(filter_size[:3])-3
    return receptive_field
def unet_architecture(sigma):
    sigma1=4
    receptive_field1=102
    receptive_field=(int(receptive_field1/sigma*sigma1)//2+1)*2
    filter_size=[5,5,3,3,3,3,3,3]
    deep_filters=[[3,3,3,1,1,1],[3,3,3,1,1,1],[3,3,3,1,1,1]]
    widths=[64,128,256,512]
    if sigma==sigma1:
        return widths,filter_size,deep_filters
    pools=[2,2,2]
    org_filter_size__=copy.deepcopy(filter_size)
    org_deep_filters__=copy.deepcopy(deep_filters)
    rec=unet_receptive_field_compute(filter_size,pools,deep_filters)
    nomoreleft=False
    while rec>receptive_field:
        filter_size__=copy.deepcopy(filter_size)
        deep_filters__=copy.deepcopy(deep_filters)
        lvl=len(deep_filters)-1
        while lvl>=0:
            dlvl=np.array(deep_filters[lvl])
            if not np.all(dlvl==1):
                break
            lvl-=1
        if lvl>=0:
            I=np.where(dlvl>1)[0][-1]
            deep_filters[lvl][I]-=1
        else:
            ff=np.array(filter_size[:3])
            if not np.all(ff==1):
                I=np.where(ff>1)[0][-1]
                filter_size[I]-=2
            else:
                nomoreleft=True
                break
        rec=unet_receptive_field_compute(filter_size,pools,deep_filters)
    if nomoreleft:
        nomoreleft=False
        while rec>receptive_field:
            filter_size__=copy.deepcopy(filter_size)
            deep_filters__=copy.deepcopy(deep_filters)
            ff=np.array(filter_size[3:])
            if not np.all(ff==np.amax(ff)):
                I=np.where(ff==np.amax(ff))[0][-1]
                filter_size[I+3]-=2
            elif not np.all(ff==1):
                I=np.where(ff>1)[0][-1]
                filter_size[I+3]-=2
            else:
                nomoreleft=True
                break
            rec=unet_receptive_field_compute(filter_size,pools,deep_filters)
        if nomoreleft:
            print('WTF!')
        else:
            filter_size=copy.deepcopy(filter_size__)
            deep_filters=copy.deepcopy(deep_filters__)
    else:
        filter_size=copy.deepcopy(filter_size__)
        deep_filters=copy.deepcopy(deep_filters__)
    rec=unet_receptive_field_compute(filter_size,pools,deep_filters)

    net=UNET()
    nparam0=net.nparam
    net=UNET(filter_size=filter_size,deep_filters=deep_filters)
    nparam1=net.nparam
    rat=np.sqrt(nparam0/nparam1)
    
    widths=[int(np.ceil(w*rat)) for w in widths]
    net=UNET(widths=widths,filter_size=filter_size,deep_filters=deep_filters)
    nparam2=net.nparam
    return widths,filter_size,deep_filters

def qcnn_receptive_field_compute(filter_size):
    return np.sum(filter_size)-len(filter_size)+1
def qcnn_architecture(sigma):
    sigma1=4
    
    receptive_field1=21
    receptive_field=int(receptive_field1/sigma*sigma1)//2*2+1
    
    filter_size=[5,5,3,3,3,3,3,3]
    qwidth=64
    widths=[128,64,32,32,32,32,32,1]
    if sigma==sigma1:
        return widths,filter_size,qwidth,[11,11]
    
    qfilt1=(receptive_field+1)//2
    if qfilt1%2==0:
        qfilt2=qfilt1-1
        qfilt1+=1
    else:
        qfilt2=qfilt1
    qfilt=[qfilt1,qfilt2]
    
    org_filter_size__=copy.deepcopy(filter_size)
    rec=qcnn_receptive_field_compute(filter_size)
    nomoreleft=False
    
    while rec>receptive_field:
        filter_size__=copy.deepcopy(filter_size)
        ff=np.array(filter_size)
        if not np.all(ff==np.amax(ff)):
            I=np.where(ff==np.amax(ff))[0][-1]
            filter_size[I]-=2
        elif not np.all(ff==1):
            I=np.where(ff>1)[0][-1]
            filter_size[I]-=2
        else:
            nomoreleft=True
            break
        rec=qcnn_receptive_field_compute(filter_size)
    if nomoreleft:
        print('WTF!')
    net=QCNN()
    nparam0=net.nparam
    net=QCNN(filter_size=filter_size,qfilt=qfilt)
    nparam1=net.nparam
    rat=np.sqrt(nparam0/nparam1)
    
   
    
    qwidth=int(np.ceil(qwidth*rat**2))
    widths=[int(np.ceil(w*rat)) for w in widths]
    
    net=QCNN(width=widths,qwidth=qwidth,filter_size=filter_size,qfilt=qfilt)
    nparam2=net.nparam
    return widths,filter_size,qwidth,qfilt


# In[1]:


'''
MODEL COMPARE: (2x2 = 4)  [0-3]
    LCNN/QCNN
    u+v+T -> Su+Sv+ST
    Surf/Deep
    (testing 1pct CO2)

BASE TESTS: do each with
    u+v+T -> Su+Sv+ST 
    LCNN
    Surf/Deep
    (testing 1pct CO2)
filtersize (21) 15 9 7 5 3 1          (2x6)   = 10 [1000-1011]
coarse-grain (4) 8 12 16              (2x3)   = 6  [2000-2005]
geophys (4dom) + glbl + glbl lat + glbl long   (2x3)x2 = 12 [3000-3011]
    (also with UNET)
'''
def golden_model_bank(args,descriptive=False,configure=False,verbose=True,only_description=False):
    model_id=int(args.model_id)
    model_bank_id=args.model_bank_id
    
    if not only_description:
        data_info=climate_data.get_dict(model_bank_id,model_id)
    
    folder_root='/scratch/zanna/data/cm2.6/'
    data_root=['coarse-surf-data-sigma-',                    'coarse-3D-data-sigma-',                    'coarse-1pct-CO2-surf-data-sigma-',                    'coarse-1pct-CO2-3D-data-sigma-']
    model_names=['LCNN','QCNN','UNET','GAN','REG']
    sigma_vals=[4,8,12,16]
    filter_sizes=[21,15,9,7,5,3,1]
    
    STEP=1000
    test_type=model_id//STEP
    test_num=model_id%STEP
    # default parameter choices
    surf_deep=0
    lat_features=False
    long_features=False
    direct_coords=False
    residue_training=False
    temperature=[True,True]
    co2test_flag= args.co2==1
    physical_dom_id=0
    depthvals=[5.03355 , 55.853249,  110.096153, 181.312454,  330.007751,1497.56189 , 3508.633057]
    sigma_id=0
    filt_mode=0
    arch_id=0
    filter_size=21
    outwidth=3
    inwidth=3
    depthind=2
    resnet=False
    # index parameter init
    tt=test_num
    if test_type==1:
        # FILTERSIZE
        # 15 9 7 5 3 1
        surf_deep=tt%2
        tt=tt//2
        filter_size_id=tt%(len(filter_sizes)-1)+1
        filter_size=filter_sizes[filter_size_id]
    elif test_type==2:
        # COARSE-GRAIN
        physical_dom_id=3
        args.batch=2
        surf_deep=tt%2
        tt=tt//2
        sigma_id=tt%(len(sigma_vals))
        sigma=sigma_vals[sigma_id]
        args.batch=int(2*(sigma/4)**2)
        filter_size=np.int(np.ceil(21/sigma*4))//2*2+1
    elif test_type==4:
        # FULL TYPE TRAINING
        
        # DATASET (2)
        # SURF/DEEP 

        # FILTERSIZE (7)
        # 21 15 9 7 5 3 1
        
        # SIGMAVALS (4)
        # 4 8 12 16 
        
        # GEOPHYS (3)
        # NONE - GLOBAL - (GLOBAL+COORDS)
        
        # RESIDUE TARGET(2)
        # YES - NO
        
        
        '''
        EXPERIMENT 1
            Filtersize + Sigmaval + GEOPHY
                15/9/5/1 + 4  + GLBCOORDS
                15/9/5/1 + 8  + GLBCOORDS
                15/9/5/1 + 16 + GLBCOORDS
            VALUES 
               4114,4115,4116,4117,4120,4121,4124,4125,
               4128,4129,4130,4131,4134,4135,4138,4139, 
               4156,4157,4158,4159,4162,4163,4166,4167
       EXPERIMENT 2
            Filtersize + Sigmaval + GEOPHY + NO RESIDUE
                15/9/5/1 + 4  + GLBCOORDS
                15/9/5/1 + 8  + GLBCOORDS
                15/9/5/1 + 16 + GLBCOORDS
            VALUES 
               4282,4283,4284,4285,4288,4289,4292,4293,
               4296,4297,4298,4299,4302,4303,4306,4307,
               4324,4325,4326,4327,4330,4331,4334,4335
        '''
        
        surf_deep=tt%2
        tt=tt//2
        
        filter_size_id=tt%len(filter_sizes)
        tt=tt//len(filter_sizes)
        
        sigma_id=tt%len(sigma_vals)
        tt=tt//len(sigma_vals)
        
        sigma=sigma_vals[sigma_id]
        filter_size=filter_sizes[filter_size_id]
        
        args.batch=256+64
        
        geophys=tt%3
        if geophys>0:
            physical_dom_id=3
            args.batch=int(2*(sigma/4)**2)
        if geophys==2:
            lat_features =True
            direct_coords=True
        
        tt=tt//3
        residue_training=(tt%2)==0
    elif test_type==5:
        # FULL TYPE TRAINING
        
        # DATASET (2)
        # SURF/DEEP 

        # ARCHITECTURE (3)
        # LCNN/QCNN/UNET
        
        # SIGMAVALS (4)
        # 4 8 12 16 
        
        # GEOPHYS (3)
        # NONE - GLOBAL - (GLOBAL+COORDS)
        
        # RESIDUE TARGET(2)
        # YES - NO
        '''
        EXPERIMENT 1
            Dataset (2) + LCNN/QCNN (2) + Sigmavals (4) + 4 Domains + Res
                5000,5001,5002,5003,5006,5007,5008,5009,5012,5013,5014,5015,5018,5019,5020,5021
        EXPERIMENT 2
            Dataset (2) + LCNN/QCNN/UNET (3) + Sigmavals (4) + GLBL + Res 
            Dataset (2) + LCNN/UNET (2) + Sigmavals (4) + COORDS + Res 
                5024,5025,5026,5027,5028,5029,5030,5031,5032,5033,5034,5035,5036,5037,5038,5039,\
                5040,5041,5042,5043,5044,5045,5046,5047,5048,5049,5052,5053,5054,5055,5058,5059,\
                5060,5061,5064,5065,5066,5067,5070,5071
        EXPERIMENT 1.5
            Dataset (2) + LCNN/QCNN (2) + Sigmavals (4) + 4 Domains + No Res
                5072,5073,5074,5075,5078,5079,5080,5081,5084,5085,5086,5087,5090,5091,5092,5093
        EXPERIMENT 2.5
            Dataset (2) + LCNN/QCNN/UNET (3) + Sigmavals (4) + GLBL + No Res 
                5096,5097,5098,5099,5100,5101,5102,5103,5104,5105,5106,5107,5108,5109,5110,5111,5112,5113,5114,5115,5116,5117,5118,5119
        '''
        
        C=[2,3,4,3,2]
        title='full-training'
        names=[['dataset', 'surf','depth 110m'],                       ['architecture','LCNN','QCNN','UNET'],                       ['sigma']+[str(sig) for sig in sigma_vals],                       ['training-doms','4regions','global','global+coords'],                       ['residue','yes','no']]
        surf_deep=tt%2
        tt=tt//2
        
        arch_id=tt%3
        tt=tt//3
        
        sigma_id=tt%len(sigma_vals)
        tt=tt//len(sigma_vals)
        
        sigma=sigma_vals[sigma_id]
        filter_size=int(21*4/sigma//2)*2+1
        args.batch=128#4#256
        
        geophys=tt%3
        if geophys>0:
            physical_dom_id=3
            args.batch=int(2*(sigma/4)**2)
        if geophys==2:
            lat_features =True
            #direct_coords=True
        tt=tt//3
        residue_training=(tt%2)==0
    elif test_type==6:
        '''
        Regression model with various settings
        EXPERIMENT
            Dataset (2) + Regression (1) + Sigmavals (4) + Training(3) + Res/No (2)
            6000-6048
        '''
        
        C=[2,4,3,2]
        title='linear regression'
        names=[['dataset', 'surf','depth 110m'],                       ['sigma']+[str(sig) for sig in sigma_vals],                       ['training-doms','4regions','global','global+coords'],                       ['residue','yes','no']]
        
        surf_deep=tt%2
        tt=tt//2
        
        arch_id=4
        
        sigma_id=tt%len(sigma_vals)
        tt=tt//len(sigma_vals)
        
        sigma=sigma_vals[sigma_id]
        args.batch=4
        
        geophys=tt%3
        if geophys>0:
            args.batch=1
            physical_dom_id=3
        if geophys==2:
            lat_features =True
            direct_coords=True
        tt=tt//3
        residue_training=(tt%2)==0
    elif test_type==7:
        '''
        Testing various shrinkage types
        EXPERIMENT
            Sigmavals (4) + Shrinkage (6)
        '''
        
        C=[4,6]
        title='shrinkage procedures'
        names=[['sigma']+[str(sig) for sig in sigma_vals],                   ['shrinkage type']+[str(sig) for sig in range(6)]]
        
        sigma_id=tt%4
        tt=tt//4
        sigma=sigma_vals[sigma_id]
        filter_size=int(21*4/sigma//2)*2+1
        args.batch=4
        filt_mode=tt+1
    elif test_type==8:
        # FULL TYPE TRAINING
        
        # DATASET (2)
        # SURF/DEEP 

        # FILTERSIZE (9)
        # 21 15 9 7 5 4 3 2 1
        
        # SIGMAVALS (4)
        # 4 8 12 16 
        
        # RESIDUE TARGET(2)
        # YES - NO
        
        filter_sizes=[21,15,9,7,5,4,3,2,1]
        
        C=[2,9,4,2]
        title='filter size training'
        names=[['dataset', 'surf','depth 110m'],                       ['filter sozes']+[str(sig) for sig in filter_sizes],                       ['sigma']+[str(sig) for sig in sigma_vals],                       ['residue','yes','no']]
        filt_mode=1
        
        surf_deep=tt%2
        tt=tt//2
        
        filter_size_id=tt%len(filter_sizes)
        tt=tt//len(filter_sizes)
        
        sigma_id=tt%len(sigma_vals)
        tt=tt//len(sigma_vals)
        
        residue_training=(tt%2)==0
        
        sigma=sigma_vals[sigma_id]
        filter_size=filter_sizes[filter_size_id]
        
        geophys=1
        physical_dom_id=3
        args.batch=int(2*(sigma/4)**2)
    elif test_type==9:
        C=[2,2,2,2,len(sigma_vals),3]
        title='root improvement'
        names=[['temp','no','yes'],                    ['global','no','yes'],                          ['res','no','yes'],                              ['geophys','no','yes'],                                ['sigma']+[str(sig) for sig in sigma_vals],                                      ['widths']+[str(sig) for sig in [0,1,2]]]
        
        
        resnet=True
        surf_deep=0
        temperature[0]=1-(tt%2 == 0)
        temperature[1]=1-(tt%2 == 0)
        tt=tt//2
        
        if not temperature[0]:
            inwidth=2
        if not temperature[1]:
            outwidth=2 
        
        if tt%2==0:
            physical_dom_id=0
        else:
            physical_dom_id=3
        tt=tt//2
        
        
        residue_training=(tt%2)!=0
        tt=tt//2
        
        lat_features=(tt%2)!=0
        tt=tt//2
        
        sigma_id=tt%len(sigma_vals)
        tt=tt//len(sigma_vals)
        print(sigma_id,sigma_vals)
        sigma=sigma_vals[sigma_id]
        
        width_id=tt
        if sigma==4:
            # spread = 10
            filters=[3]*10+[1]*6
        elif sigma==8:
            # spread = 5
            filters=[3]*5+[1]*11
        elif sigma==12:
            # spread = 4
            filters=[3]*4+[1]*12
        elif sigma==16:
            # spread = 3
            filters=[3]*3+[1]*13
        widths=[[64,32,1],[128,64,1],[256,128,1]]
        widths=widths[width_id]
        
        if physical_dom_id==3:
            args.batch=int(2*(sigma/4)**2)
            if width_id==2:
                args.batch=int(args.batch/2)
        else:
            args.batch=165
            
        filter_size=int(21*4/sigma/2)*2+1
    elif test_type==3:
        C=[2,2,2,2,len(sigma_vals)]
        title='root improvement'
        names=[['temp','no','yes'],                    ['global','no','yes'],                          ['res','no','yes'],                              ['geophys','no','yes'],                                  ['sigma']+[str(sig) for sig in sigma_vals]]
        
        
        surf_deep=0
        temperature[0]=1-(tt%2 == 0)
        temperature[1]=1-(tt%2 == 0)
        tt=tt//2
        if tt%2==0:
            physical_dom_id=0
        else:
            physical_dom_id=3
        tt=tt//2
        
        residue_training=(tt%2)!=0
        tt=tt//2
        
        lat_features=(tt%2)!=0
        #direct_coords=(tt%2)!=0
        tt=tt//2
        
        sigma_id=tt
        sigma=sigma_vals[sigma_id]
        tt=tt//len(sigma_vals)
        
        if physical_dom_id==3:
            args.batch=int(2*(sigma/4)**2)
        else:
            args.batch=165
            
        filter_size=int(21*4/sigma/2)*2+1
        if not temperature[0]:
            inwidth=2
        if not temperature[1]:
            outwidth=2
    elif test_type==0:
        C=[2,2,2,7,len(sigma_vals)]
        depthvals=[5.03355,55.853249,110.096153,181.312454,330.007751, 1497.56189 , 3508.633057]
        title='depth test'
        names=[['temp','yes','no'],                    ['res','no','yes'],                        ['geophys','no','yes'],                            ['training-depth']+[str(i) for i in range(7)],                              ['sigma']+[str(i) for i in sigma_vals]]
        
        
        surf_deep=1
        temperature[0]=1-(tt%2)
        temperature[1]=temperature[0]
        tt=tt//2
        
        if not temperature[0]:
            inwidth=2
        if not temperature[1]:
            outwidth=2 
        
        
        residue_training=(tt%2)!=0
        tt=tt//2
        
        lat_features=(tt%2)!=0
        #direct_coords=(tt%2)!=0
        tt=tt//2
        
        physical_dom_id=3
        
        
        
        depthind=tt%7
        tt=tt//7
        
        
        sigma_id=tt
        sigma=sigma_vals[tt]
        
        if physical_dom_id==3:
            args.batch=int(2*(sigma/4)**2)
        else:
            args.batch=128
        filter_size=int(21*4/sigma/2)*2+1
            
    if only_description:
        title+=' '+str(STEP*test_type)
        if verbose:
            print(title)
            for i in range(len(names)):
                print('\t'+names[i][0])
                outputstr='\t\t'
                for j in range(1,len(names[i])):
                    outputstr+=names[i][j]+' - '
                print(outputstr)
        return C,names
    if co2test_flag:
        surf_deep+=2
    sigma=sigma_vals[sigma_id]
    args.data_address=folder_root+data_root[surf_deep]+str(sigma)
    
    
    args.data_address+='.zarr'
    
    if arch_id==0: #LCNN
        width_scale=1
        if not resnet:
            widths,filters,nparam=lcnn_architecture(width_scale,filter_size,mode=filt_mode)
        else:
            _,filters,nparam=lcnn_architecture(width_scale,filter_size,mode=filt_mode)
        net=LCNN(initwidth=inwidth,outwidth=outwidth,                 filter_size=filters,                 width=widths,                 nprecision=outwidth,                 latsig=lat_features,                 latsign=lat_features,                 longitude=long_features,                 freq_coord=lat_features and not direct_coords,                 direct_coord=direct_coords,                 skipcons=resnet)
    elif arch_id==1: #QCNN
        widths,filter_size__,qwidth,qfilt=qcnn_architecture(sigma)
        net=QCNN(width=widths,qwidth=qwidth,filter_size=filter_size__,qfilt=qfilt,                 initwidth=inwidth,outwidth=outwidth,                 nprecision=outwidth,                 latsig=lat_features,                 latsign=lat_features,                 longitude=long_features,                 freq_coord=lat_features)
    elif arch_id==2: #UNET
        widths,filter_size__,deep_filters=unet_architecture(sigma)
        net=UNET(widths=widths,deep_filters=deep_filters,filter_size=filter_size__,                 initwidth=inwidth,outwidth=outwidth,                 nprecision=outwidth,                 latsig=lat_features,                 latsign=lat_features,                 longitude=long_features,                 freq_coord=lat_features)
    elif arch_id==3: #GAN
        net=GAN(initwidth=inwidth,outwidth=outwidth,                latsig=lat_features,                 latsign=lat_features,                 longitude=long_features,                 freq_coord=lat_features)
    elif arch_id==4: #Regression
        net=RegressionModel(initwidth=inwidth,                 latsig=lat_features,                 latsign=lat_features,                 direct_coord=direct_coords)
    if arch_id!=3:
        loss=lambda output, target, mask:             lossfun(output, target, mask,heteroscedastic=True,outsize=outwidth)
    else:
        loss=lambda output, target, mask:             ganlossfun(output, target, mask)
    description=model_names[arch_id]
    if arch_id==0:
        stt=str(filter_size)
        stt=stt+'x'+stt
        description+=' + '+stt
    if surf_deep%2==0:
        description+=' + '+'surface'
    elif surf_deep%2==1:
        depthval=depthvals[depthind]
        depthval=str(int(np.round(depthval)))
        description+=' + '+'deep ('+str(depthval)+'m)'
        
    if surf_deep//2==1:
        description+=' +1%CO2'
    if residue_training:
        description+=' + '+'res'
    if physical_dom_id==0:
        description+=' + '+'4 domains'
    elif physical_dom_id==3:
        description+=' + '+'glbl'
        if lat_features:
            description+=' + '+'lat'
        if long_features:
            description+=' + '+'long'
    
    description+=' + '+'coarse('+str(sigma)+')'
    if verbose:
        print(description+' + '+'batch= '+str(args.batch), flush=True)
    partition=climate_data.physical_domains(physical_dom_id)
    ds_zarr=climate_data.load_ds_zarr(args)
    model_bank_id='G'
    
    if configure:
        data_info['direct_coord']=direct_coords
        data_info['freq_coord']=lat_features
        data_info['lat_feat']=lat_features
        data_info['long_feat']=long_features
        data_info['inputs']="usurf vsurf surface_temp".split()
        if not temperature[0]:
            data_info['inputs']=data_info['inputs'][:2]
            
        if residue_training:
            data_info['outputs']="Su_r Sv_r ST_r".split()
        else:
            data_info['outputs']="Su Sv ST".split()
        if not temperature[1]:
            data_info['outputs']=data_info['outputs'][:2]
        maskloc='/scratch/cg3306/climate/masks/'
        if surf_deep==0:
            maskloc+='surf'
        elif surf_deep==1:
            maskloc+='deep'
        maskloc+='-sigma'+str(sigma)
        maskloc+='-filter'+str(filter_size)
        if physical_dom_id==0:
            maskloc+='-dom4'
        if physical_dom_id==3:
            maskloc+='-glbl'
        if resnet:
            maskloc+='-padded'
        maskloc+='.npy'
        data_info['maskloc']=maskloc
    if not descriptive and configure:
        climate_data.update_model_info(data_info,model_bank_id,model_id)
    data_init=lambda partit : climate_data.Dataset2(ds_zarr,partit,model_id,model_bank_id,                                                    net,subtime=args.subtime,parallel=args.nworkers>1,                                                    depthind=depthind)
    
    if not descriptive:
        return net,loss,data_init,partition
    else:
        return description


# In[5]:


'''import numpy as np
C=[2,3,4,3,2]
x=[[0,1],[0,1,2],[0,1,2,3],[1,2],[1]]
lxi=[len(xx) for xx in x]
n=np.prod(lxi)
y=[]
for i in range(n):
    ii=i
    tt=0
    TT=1
    for j in range(len(x)):
        tt+=x[j][ii%lxi[j]]*TT
        ii=ii//lxi[j]
        TT*=C[j]
    y.append(tt)
y=[tt+5000 for tt in y] 
str(y).replace(' ','')'''


# In[6]:


def model_bank(args,descriptive=False,configure=False,verbose=True):
    try:
        model_bank_id=int(args.model_bank_id)
    except:
        if args.model_bank_id=='G':
            return golden_model_bank(args,descriptive=descriptive,configure=configure,verbose=verbose)
    model_id=int(args.model_id)
    hetsc=1
    #if not descriptive:
    data_info=climate_data.get_dict(model_bank_id,model_id)
    if configure:
        data_info['direct_coord']=False
        data_info['freq_coord']=False
    if model_bank_id>12 and model_bank_id<=16:
        folder_root='/scratch/zanna/data/cm2.6/'
        data_root=['coarse-surf-data-sigma-',                        'coarse-3D-data-sigma-',                        'coarse-1pct-CO2-surf-data-sigma-',                        'coarse-1pct-CO2-3D-data-sigma-4-',                           ]
    elif model_bank_id==11 or model_bank_id==12:
        args.data_address='/scratch/zanna/data/cm2.6/coarse-3D-data-sigma-4-1.zarr/'
        
        if model_id<10:
            physical_dom_id=0
        else:
            physical_dom_id=3
        outwidth=3
        initwidth=2
        if model_bank_id==12:
            outwidth=2
            initwidth=3
        if configure:
            data_info['inputs']="usurf vsurf".split()
            data_info['outputs']="Su Sv".split()
            
            if model_bank_id==12:
                data_info['inputs']="usurf vsurf surface_temp".split()
                data_info['outputs']="ST".split()

            data_info['st_ocean']=[model_id%5]
            data_info['maskloc']='/scratch/cg3306/climate/masks/'+                            'coarse-3D-data-sigma-4-physdoms-'+str(physical_dom_id)+                            '-depth-'+str(model_id%5)+'.np'
        if model_id//5==0:
            if not descriptive:
                net=LCNN(initwidth=initwidth,outwidth=outwidth)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            description='LCNN ocean depth = '+str(model_id%5)
        elif model_id//5==1:
            if not descriptive:
                net=QCNN(initwidth=initwidth,outwidth=outwidth)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            description='QCNN ocean depth = '+str(model_id%5)
        elif model_id//5==2:
            if not descriptive:
                net=LCNN(initwidth=initwidth,outwidth=outwidth,latsig=True,latsign=True,longitude=True,freq_coord=True)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            description='LCNN ocean glbl-coords depth = '+str(model_id%5)
        elif model_id//5==3:
            if not descriptive:
                net=LCNN(initwidth=initwidth,outwidth=outwidth)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            description='LCNN ocean glbl depth = '+str(model_id%5)
        elif model_id>=20:
            filtid=model_id-20
            physical_dom_id=0
            def_width=[2,256,128,64,64,64,32,32,3]
            def_filters=[5,5,3,3,3,3,3,3] # 21
            if configure:
                data_info['st_ocean']=[2]
            if filtid==0:
                filters=[7,7,5,3,3,3,3,3] # 27
            elif filtid==1:
                filters=[3,3,3,3,3,3,3,1] # 15
            elif filtid==2:
                filters=[3,3,3,3,1,1,1,1] # 9
            elif filtid==3:
                filters=[3,3,3,1,1,1,1,1] # 7
            elif filtid==4:
                filters=[3,3,1,1,1,1,1,1] # 5
            elif filtid==5:
                filters=[3,1,1,1,1,1,1,1] # 3
            widths=approximate_widths(def_width,def_filters,filters)
            if not descriptive:
                net=LCNN(initwidth=initwidth,outwidth=outwidth,filter_size=filters,width=widths)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            description='LCNN ocean depth = 2, input-box size = '+str(np.sum(filters)-len(filters)+1)
    elif model_bank_id==10:
        physical_dom_id=0
        
        args.data_address='/scratch/zanna/data/cm2.6/coarse-surf-data-sigma-4.zarr/'
        if model_id==0:
            hetsc=5
            if configure:
                data_info['inputs']="usurf vsurf".split()
                data_info['outputs']="Su Sv".split()
            if not descriptive:
                net=LCNN(initwidth=2,outwidth=3)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True,outsize=2)
            description='LCNN surface vel'
        elif model_id==1:
            if configure:
                data_info['inputs']="usurf vsurf surface_temp".split()
                data_info['outputs']="ST".split()
            if not descriptive:
                net=LCNN(initwidth=3,outwidth=2)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True,outsize=1)
            description='LCNN surface temp'
        elif model_id==2:
            if configure:
                data_info['inputs']="usurf vsurf surface_temp".split()
                data_info['outputs']="ST".split()
            if not descriptive:
                net=QCNN(initwidth=3,outwidth=2)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True,outsize=1)
            description='QCNN surface temp'
        elif model_id==3:
            physical_dom_id=3
            if configure:
                data_info['inputs']="usurf vsurf surface_temp".split()
                data_info['outputs']="ST".split()
            if not descriptive:
                net=LCNN(initwidth=3,outwidth=2,latsig=True,latsign=True,longitude=True,freq_coord=True)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True,outsize=1)
            description='LCNN surface temp glbl-coord'
        elif model_id==4:
            physical_dom_id=3
            if configure:
                data_info['inputs']="usurf vsurf surface_temp".split()
                data_info['outputs']="ST".split()
            if not descriptive:
                net=LCNN(initwidth=3,outwidth=2)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True,outsize=1)
            description='LCNN surface temp glbl'
        elif model_id>=5:
            
            def_width=[2,256,128,64,64,64,32,32,3]
            def_filters=[5,5,3,3,3,3,3,3] # 21
            filtid=model_id-5
            if filtid==0:
                filters=[7,7,5,3,3,3,3,3] # 27
            elif filtid==1:
                filters=[3,3,3,3,3,3,3,1] # 15
            elif filtid==2:
                filters=[3,3,3,3,1,1,1,1] # 9
            elif filtid==3:
                filters=[3,3,3,1,1,1,1,1] # 7
            elif filtid==4:
                filters=[3,3,1,1,1,1,1,1] # 5
            elif filtid==5:
                filters=[3,1,1,1,1,1,1,1] # 3
      
            widths=approximate_widths(def_width,def_filters,filters)
            if configure:
                data_info['inputs']="usurf vsurf surface_temp".split()
                data_info['outputs']="ST".split()
                
           
            filtersize=np.sum(filters)-len(filters)+1
            if not descriptive:
                net=LCNN(initwidth=3,outwidth=2,filter_size=filters,width=widths)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True,outsize=1)
            description='LCNN surface temp - filter size= '+str(filtersize)
        if configure:
            if model_id<5:
                data_info['maskloc']='/scratch/cg3306/climate/masks/coarse-surf-data-sigma-4-mask-'+str(physical_dom_id)+'.np'
            elif model_id>=5:
                data_info['maskloc']='/scratch/cg3306/climate/masks/'+                    'coarse-surf-data-sigma-4-mask-filtersize-'+str(filtersize)+'.np'
    elif model_bank_id==9:
        physical_dom_id=0
        
        args.data_address='/scratch/zanna/data/cm2.6/coarse-surf-data-sigma-4.zarr/'
        
        def_width=[2,256,128,64,64,64,32,32,3]
        def_filters=[5,5,3,3,3,3,3,3] # 21
        mm=5
        filtid=model_id
        if filtid==0:
            filters=[7,7,5,3,3,3,3,3] # 27
        elif  filtid==1:
            filters=[5,5,3,3,3,3,3,3] # 21
        elif filtid==2:
            filters=[3,3,3,3,3,3,3,1] # 15
        elif filtid==3:
            filters=[3,3,3,3,1,1,1,1] # 9
        elif filtid==4:
            filters=[3,3,3,1,1,1,1,1] # 7
        elif filtid==5:
            filters=[3,3,1,1,1,1,1,1] # 5
        elif filtid==6:
            filters=[3,1,1,1,1,1,1,1] # 3
        
        widths=approximate_widths(def_width,def_filters,filters)
        
        filtersize=np.sum(filters)-len(filters)+1
        if configure:
            data_info['inputs']="usurf vsurf".split()
            data_info['outputs']="Su Sv".split()
            data_info['maskloc']='/scratch/cg3306/climate/masks/'+                            'coarse-surf-data-sigma-4-filtersize-'+str(filtersize)+'.np'
        if not descriptive:
            net=LCNN(initwidth=2,outwidth=3,filter_size=filters,width=widths)
            loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True,outsize=2)
        description='LCNN surface flow - filter size= '+str(np.sum(filters)-len(filters)+1)
    elif model_bank_id==8:
        physical_dom_id=3
        if model_bank_id==8:
            geofeat=False
        else:
            geofeat=True
        def_width=[2,256,128,64,64,64,32,32,3]
        def_filters=[5,5,3,3,3,3,3,3]
        if model_id==0:
            sigma=2
            filters=[11,11,5,5,5,5,5,5]
        elif model_id==1:
            sigma=4
            filters=[5,5,3,3,3,3,3,3]
        elif model_id==2:
            sigma=6
            filters=[3,3,3,3,3,3,3,3]
        elif model_id==3:
            sigma=8
            filters=[3,3,3,3,3,1,1,1]
        elif model_id==4:
            sigma=12
            filters=[3,3,3,1,1,1,1,1]
        elif model_id==5:
            sigma=16
            filters=[3,3,3,1,1,1,1,1]
        #rescale=[1/10,1/10]
        scales=np.load('/scratch/cg3306/climate/climate_research/scales.npy')
        ind=np.where(scales[:,0]==sigma)[0][0]
        rescale=[scales[ind,1],scales[ind,2]]
        if not descriptive:
            widths=approximate_widths(def_width,def_filters,filters)
            net=LCNN(latsig=geofeat,latsign=geofeat, longitude=geofeat,                         filter_size=filters,                         width=widths,                            rescale=rescale)
            loss=lambda output, target, mask: lossfun(output, target, mask, heteroscedastic=True)
        description='LCNN sigma='+str(sigma)
        args.data_address='/scratch/cg3306/climate/data-read/data/sigma-'+str(sigma)+'-data.zarr'
        
    elif model_bank_id==7:
        if model_id==0:
            physical_dom_id=3
        elif model_id==1:
            physical_dom_id=2
        if not descriptive:
            net=CQCNN()
            loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True,neglog=True)
        description='CQCNN'
    elif model_bank_id==6:
        physical_dom_id=3
        if not descriptive:
            net=Improved_QCNN()
            loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
        fsize=21
        description='Improved QCNN'
    elif model_bank_id==5:
        physical_dom_id=3
        if model_id==0:
            if not descriptive:
                net=MatReg(order=2,width=64)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='Local regression order 2 width 64 global training'
        elif model_id==1:
            if not descriptive:
                net=NonlinearReg(order=2,width=64)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='Local nonlinear order 2 global training'
        elif model_id==2:
            if not descriptive:
                net=NonlinearReg(order=3,width=64)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='Local nonlinear order 3 global training'    
        if model_id==3:
            if not descriptive:
                net=MatReg(order=3,width=64)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='Local regression order 3 width 64 global training'
    elif model_bank_id==4:
        physical_dom_id=3
        loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
        if model_id==0:
            net=UNET(latsig=True,latsign=True,longitude=True,freq_coord=True)
            description='UNET-coords'
        elif model_id==1:
            net=UNET()
            loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            description='UNET'
    elif model_bank_id==3:
        physical_dom_id=3
        if model_id==0:
            if not descriptive:
                net=LCNN()
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='LCNN global training'
        elif model_id==1:
            if not descriptive:
                net=LCNN(latsig=True)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            
            description='LCNN global lat val training'
        elif model_id==2:
            if not descriptive:
                net=LCNN(latsig=True,latsign=True)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='LCNN latitude val+sign'
        elif model_id==3:
            if not descriptive:
                net=LCNN(latsig=True,latsign=True,longitude=True)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='LCNN latitude val+sign, longitude'
        elif model_id==4:
            if not descriptive:
                net=LCNN(direct_coord=True)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='LCNN direct coord'
    elif model_bank_id==2:
        physical_dom_id=3
        if model_id==0:
            physical_dom_id=0
            if not descriptive:
                net=LCNN(latsig=True,physical_domain_id=0)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='LCNN('+str(fsize)+'x'+str(fsize)+') wt=0.05 heteroscedastic small domain'
        elif model_id==1:
            if not descriptive:
                net=LCNN(latsig=True,direct_coord=False,longitude=True)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='LCNN('+str(fsize)+'x'+str(fsize)+') freq encoded geo global training'
        elif model_id==2:
            if not descriptive:
                net=LCNN(latsig=True,direct_coord=True,longitude=True)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='LCNN('+str(fsize)+'x'+str(fsize)+') direct geo global training'
        elif model_id==3:
            if not descriptive:
                net=LCNN(latsig=True,direct_coord=True,longitude=True,physical_force_features=True)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='LCNN('+str(fsize)+'x'+str(fsize)+') freq encoded additional physical features global training'
        elif model_id==4:
            if not descriptive:
                net=LCNN(latsig=True,direct_coord=True,longitude=True,timeshuffle=False)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='LCNN('+str(fsize)+'x'+str(fsize)+') freq encoded heteroscedastic no time shuffle'
        elif model_id==5:
            physical_dom_id=4
            if not descriptive:
                net=LCNN(latsig=True,direct_coord=True,longitude=True,timeshuffle=False)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='LCNN('+str(fsize)+'x'+str(fsize)+') freq encoded heteroscedastic equator'
    elif model_bank_id==1:
        physical_dom_id=0
        #widths=[128,64,32,32,32,32,32,32,3],\
                    #filter_sizes=[5,5,3,3,3,3,3,3,3]
        
        widths=[
                [128,64,32, 32, 32, 32, 32, 3],\
                [128,64,32, 32, 32, 32, 32, 3]
        ]
        
        # 11 5 3
        filter_sizes=[
            [5,5,3,1,1,1,1,1],\
            [5,1,1,1,1,1,1,1]
        ]
        coarsen_levels=[
            2,\
            4
        ]
        filter_sizes2=[
            [7,5],\
            [3,3]
        ]
        if model_id<2:
            coarsen_level=coarsen_levels[model_id]
            if not descriptive:
                net=LCNN(coarsen=coarsen_level)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=21
            description='LCNN('+str(fsize)+'x'+str(fsize)+') coarsen('+str(coarsen_level)+') heteroscedastic'
        else:
            width=widths[model_id-2]
            filter_size=filter_sizes[model_id-2]
            if not descriptive:
                net=LCNN(width=width,filter_size=filter_size)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            fsize=np.sum(filter_size)-len(filter_size)+1
            description='LCNN('+str(fsize)+'x'+str(fsize)+') coarsen(0) heteroscedastic'
        
    elif model_bank_id==0:
        physical_dom_id=0
        if model_id==0:
            if not descriptive:
                net=LCNN()
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            
            description='LCNN(21x21) coarsen(0) heteroscedastic'
        elif model_id==1:
            if not descriptive:
                net=LCNN()
                loss=lambda output, target, mask: lossfun(output, target, mask)
            else:
                description='LCNN(21x21) coarsen(0) l2'
        elif model_id==2:
            if not descriptive:
                net=LCNN()
                loss=lambda output, target, mask: lossfun(output, target, mask,logloss=True)
            description='LCNN(21x21) coarsen(0) log'
        elif model_id==3:
            if not descriptive:
                net=QCNN(qwidth=128)
                loss=lambda output, target, mask: lossfun(output, target, mask,heteroscedastic=True)
            description='QCNN(21) heteroskedastic'
        elif model_id==4:
            if not descriptive:
                net=QCNN(qwidth=128)
                loss=lambda output, target, mask: lossfun(output, target, mask,logloss=True)
            description='QCNN(21) log'
    print(description, flush=True)
    partition=climate_data.physical_domains(physical_dom_id)
    #if model_bank_id<8:
    ds_zarr=climate_data.load_ds_zarr(args)
    readfile2=False
    if model_bank_id==8:
        readfile2=True
    if model_bank_id < 9:
        data_init=lambda partit : climate_data.Dataset1(ds_zarr,partit,net,subtime=args.subtime,readfile2=readfile2)
    else:
        if not descriptive and configure:
            climate_data.update_model_info(data_info,model_bank_id,model_id)
        #print(args.subtime)
        data_init=lambda partit : climate_data.Dataset2(ds_zarr,partit,model_id,model_bank_id,net,subtime=args.subtime,heteroscrescale=hetsc)
    if not descriptive:
        return net,loss,data_init,partition
    else:
        return description


# In[7]:


class ClimateNet(nn.Module):
    def __init__(self,spread=0,coarsen=0,rescale=[1/10,1/1e7],latsig=False,                 timeshuffle=True,direct_coord=True,longitude=False,latsign=False,gan=False):
        super(ClimateNet, self).__init__()
        if torch.cuda.is_available():  
            device = "cuda:0" 
        else:  
            device = "cpu"  
        self.generative=False
        self.timeshuffle=timeshuffle
        self.device = torch.device(device) 
        self.spread=spread
        self.latsig=latsig
        self.direct_coord=direct_coord
        self.longitude=longitude
        self.latsign=latsign
        self.coarsen=coarsen
        self.coarse_grain_filters=[]
        self.coarse_grain_filters.append([])
        self.nn_layers = nn.ModuleList()
        self.init_coarsen=coarsen
        self.rescale=rescale
        self.gan=gan
        self.nprecision=0
        for m in range(1,9):
            gauss1=torch.zeros(2*m+1,2*m+1,dtype=torch.float32,requires_grad=False)
            for i in range(m):
                for j in range(m):
                    gauss1[i,j]=np.exp( -(j**2+i**2)/((2*m)**2)/2)
            gauss1=gauss1/gauss1.sum()
            self.coarse_grain_filters.append(torch.reshape(gauss1,[1,1,2*m+1,2*m+1]).to(device))
    def coarse_grain(self,x,m):
        if m==0:
            return x
        b=x.shape[0]
        c=x.shape[1]
        h=x.shape[2]
        w=x.shape[3]
        return F.conv2d(x.view(b*c,1,h,w),self.coarse_grain_filters[m]).view(b,c,h-2*m,w-2*m)
    def set_coarsening(self,c):
        c_=self.coarsen
        self.coarsen=c
        self.spread=self.spread-c_+c
    def initial_coarsening(self,):
        self.spread=self.spread+self.init_coarsen-self.coarsen
        self.coarsen=self.init_coarsen
        
def physical_forces(x):
    dudy=x[:,:2,2:,1:-1]-x[:,:2,:-2,1:-1]
    dudx=x[:,:2,1:-1,2:]-x[:,:2,1:-1,:-2]
    x=x[:,:,1:-1,1:-1]
    u_=x[:,0:1]
    v_=x[:,1:2]
    x=torch.cat([x,dudy,dudx,u_*dudy,v_*dudy,u_*dudx,v_*dudx],dim=1)
    return x

class LCNN(ClimateNet):
    def __init__(self,spread=0,heteroscedastic=True,coarsen=0,                    width=[128,64,32,32,32,32,32,3],                    filter_size=[5,5,3,3,3,3,3,3],                    latsig=False,                    latsign=False,                    direct_coord=False,                    freq_coord=False,                    timeshuffle=False,                    physical_force_features=False,                    longitude=False,                    rescale=[1/10,1/1e7],                    initwidth=2,                    outwidth=2,                    nprecision=1,                    skipcons=False):
        super(LCNN, self).__init__(spread=spread,coarsen=coarsen,latsig=latsig,timeshuffle=timeshuffle)
        device=self.device
        self.nn_layers = nn.ModuleList()
        spread=0
        self.nparam=0
        self.rescale=torch.tensor(rescale,dtype=torch.float32,requires_grad=False)
        self.freq_coord=freq_coord
        self.heteroscedastic=heteroscedastic
        width[-1]=outwidth+nprecision
        self.width=width
        self.outwidth=outwidth
        self.initwidth=initwidth
        self.filter_size=filter_size
        self.num_layers=len(filter_size)
        self.direct_coord=direct_coord
        self.longitude=longitude
        self.latsign=latsign
        self.latsig=latsig
        self.nprecision=nprecision
        self.skipcons=skipcons
        self.physical_force_features=physical_force_features
        
        self.bnflag=True#self.latsig or self.latsign or self.direct_coord
        if self.direct_coord:
            if self.latsig:
                initwidth+=1
            if self.latsign:
                initwidth+=1
            if self.longitude:
                initwidth+=1
        elif self.freq_coord:
            if self.latsig:
                initwidth+=1
            if self.latsign:
                initwidth+=1
            if self.longitude:
                initwidth+=2
        if physical_force_features:
            initwidth+=12
            i=len(filter_size)
            while i>0:
                i-=1
                if filter_size[i]>1:
                    break
            filter_size[i]-=2
            spread+=1
                
        self.padding=[ff//2*self.skipcons for ff in filter_size]
        if not self.skipcons:
            self.nn_layers.append(nn.Conv2d(initwidth, width[0], filter_size[0],padding=self.padding[0]).to(device) )

            self.nparam+=initwidth*width[0]*filter_size[0]**2
            spread+=(filter_size[0]-1)/2
            for i in range(1,self.num_layers):
                if self.bnflag:
                    self.nn_layers.append(nn.BatchNorm2d(width[i-1]).to(device) )
                    self.nparam+=width[i-1]
                self.nn_layers.append(nn.Conv2d(width[i-1], width[i], filter_size[i],padding=self.padding[i]).to(device) )
                self.nparam+=width[i-1]*width[i]*filter_size[i]**2
                spread+=(filter_size[i]-1)/2
        else:
            w0=width[0]
            w1=width[1]
            w2=width[-1]
            self.nn_layers.append(nn.Conv2d(initwidth, w0, 1,padding=0).to(device) )
            self.nn_layers.append(nn.BatchNorm2d(w0).to(device) )
            self.nparam+=initwidth*w0+w0
            for i in range(self.num_layers):
                self.nn_layers.append(nn.Conv2d(w0, w1, 1,padding=0).to(device) )
                self.nn_layers.append(nn.BatchNorm2d(w1).to(device) )
                self.nn_layers.append(nn.Conv2d(w1, w1, filter_size[i],padding=self.padding[i]).to(device) )
                self.nn_layers.append(nn.BatchNorm2d(w1).to(device) )
                self.nn_layers.append(nn.Conv2d(w1, w0, 1,padding=0).to(device) )
                self.nparam+=w0*w1*2+w1**2*filter_size[i]**2+w0+w1*2
                spread+=(filter_size[i]-1)/2
            self.nn_layers.append(nn.Conv2d(w0, w2, 1,padding=0).to(device) )
            self.nparam+=w2*w0                
        self.nn_layers.append(nn.Softplus().to(device))
        spread+=coarsen
        self.spread=np.int64(spread)
        self.receptive_field=self.spread*2+1
    def forward(self, x):
        #x=x/self.rescale[0]
        #if self.physical_force_features:
        #    x=physical_forces(x)
        #x=self.coarse_grain(x,self.coarsen)
        if not self.skipcons:
            cn=0
            for i in range(self.num_layers-1):            
                x = self.nn_layers[cn](x)
                cn+=1
                if self.bnflag:
                    x = F.relu(self.nn_layers[cn](x))
                    cn+=1
                else:
                    x = F.relu(x)
            x=self.nn_layers[cn](x)
            cn+=1
        else:
            cn=0
            x = self.nn_layers[cn](x)
            cn+=1
            x = self.nn_layers[cn](x)
            cn+=1
            for i in range(self.num_layers):
                init=x*1                
                x = self.nn_layers[cn](x)
                cn+=1
                x = F.relu(self.nn_layers[cn](x))
                cn+=1
                x = self.nn_layers[cn](x)
                cn+=1
                x = F.relu(self.nn_layers[cn](x))
                cn+=1
                x = self.nn_layers[cn](x)
                cn+=1
                x+=init
            x = self.nn_layers[cn](x)
            cn+=1
        mean,precision=torch.split(x,[x.shape[1]-self.nprecision,self.nprecision],dim=1)
        precision=self.nn_layers[cn](precision)
        x=torch.cat([mean,precision],dim=1)
        return x


# In[8]:


class RegressionModel(ClimateNet):
    def __init__(self,spread=1, degree=3,initwidth=3,outwidth=3,latsig=False,                    latsign=False,                    direct_coord=False,                    freq_coord=False):
        super(RegressionModel, self).__init__()
        device=self.device
        self.nn_layers = nn.ModuleList()
        self.spread=spread
        self.nparam=0
        self.freq_coord=freq_coord
        self.heteroscedastic=False
        self.direct_coord=direct_coord
        self.longitude=False
        self.latsign=latsign
        self.latsig=latsig
        self.degree=degree
        receptive_field=2*spread+1
        self.receptive_field=receptive_field
        if self.direct_coord:
            if self.latsig:
                initwidth+=1
            if self.latsign:
                initwidth+=1
            if self.longitude:
                initwidth+=1
        elif self.freq_coord:
            if self.latsig:
                initwidth+=1
            if self.latsign:
                initwidth+=1
            if self.longitude:
                initwidth+=2
        self.initwidth=initwidth
        self.outwidth=outwidth
        featurenum=initwidth*receptive_field**2
        self.featurenum=featurenum
        T=nn.Conv2d(initwidth, featurenum+1, 2*spread+1)
        W=T.weight.data
        W=W*0
        W.requires_grad=False
        names=[]
        for i in range(initwidth):
            for j in range(featurenum):
                j_=j
                j1=j_%receptive_field
                j_=j_//receptive_field
                j2=j_%receptive_field
                j_=j_//receptive_field
                j3=j_%initwidth
                W_=torch.zeros(receptive_field,receptive_field)
                W_[j1,j2]=1
                W[j,i]=W_.view(1,1,receptive_field,receptive_field)
                names.append([j3,[j1-spread,j2-spread]])
        self.basic_names=names.copy()
        T.weight.data=W
        T.bias.data=T.bias.data*0
        for j in range(featurenum,featurenum+1):
            T.bias.data[j]=1.
        T.bias.data.requires_grad=False
        self.nn_layers.append(T.to(device) )
        N=outwidth+1
        self.res = list(combinations_with_replacement(range(N), degree))
        self.res = [torch.tensor(I) for I in self.res]
        self.names=[]
        self.outputdimen=len(self.res)
    def compute_names(self,):
        outwidth=self.outwidth
        self.names=[]
        names=self.basic_names.copy()
        for i in range(len(self.res)):
            I=self.res[i]
            D=torch.zeros(featurenum)
            for j in range(featurenum):
                D[j]=torch.sum(I==j)
            stt=[]
            for j in range(featurenum):
                if D[j]>0:
                    K=names[j].copy()
                    K.append(int(D[j].item()))
                    stt.append(K)
            self.names.append(stt)
    def forward(self,x,w):
        bnum=x.shape[0]
        ysp=x.shape[2]
        xsp=x.shape[3]
        spread=self.spread
        featurenum=self.featurenum
        initwidth=self.initwidth
        receptive_field=self.receptive_field
        y=torch.zeros(bnum,w.shape[1],ysp-2*spread,xsp-2*spread)
        for j in range(featurenum):
            j_=j
            j1=j_%receptive_field
            j_=j_//receptive_field
            j2=j_%receptive_field
            j_=j_//receptive_field
            j3=j_%initwidth

            y0=spread-(j1-spread)
            y1=ysp-spread-(j1-spread)

            x0=spread-(j2-spread)
            x1=xsp-spread-(j2-spread)
            for i in range(w.shape[1]):
                y[:,i,:,:]+=w[j3,i]*x[:,j3,y0:y1,x0:x1]
        return y
    def cross_products(self, x,y,mask=[]):
        bnum=x.shape[0]
        ysp=x.shape[2]
        xsp=x.shape[3]
        spread=self.spread
        outwidth=self.outwidth
        initwidth=self.initwidth
        receptive_field=self.receptive_field
        x_=torch.zeros(bnum,outwidth+1,ysp-2*spread,xsp-2*spread)
        for j in range(outwidth):
            j_=j
            j1=j_%receptive_field
            j_=j_//receptive_field
            j2=j_%receptive_field
            j_=j_//receptive_field
            j3=j_%initwidth

            y0=spread-(j1-spread)
            y1=ysp-spread-(j1-spread)

            x0=spread-(j2-spread)
            x1=xsp-spread-(j2-spread)

            x_[:,j,:,:]=x[:,j3,y0:y1,x0:x1]
        x_[:,outwidth]=x_[:,outwidth]+1.
        #x=self.nn_layers[0](x)
        x=x_
        bnum=x.shape[0]
        nchan=x.shape[1]
        outnchan=y.shape[1]
        ysp=x.shape[2]
        xsp=x.shape[3]
        sp=ysp*xsp
        if len(mask)>0:
            x=x*mask
            y=y*mask
        x=torch.reshape(x,(bnum,nchan,sp))
        y=torch.reshape(y,(bnum,outnchan,sp))
        x=x.permute((1,0,2))
        y=y.permute((1,0,2))
        x=torch.reshape(x,(nchan,bnum*sp))
        y=torch.reshape(y,(outnchan,bnum*sp))
        degree=self.degree
        nfeat=len(self.res)
        x=x.to(torch.device("cpu"))
        y=y.to(torch.device("cpu"))
        X=torch.zeros(nfeat,bnum*sp).to(torch.device("cpu"))
        for i in range(nfeat):
            X[i]=torch.prod(x[self.res[i]],dim=0)
        X2=X@X.T
        XY=X@y.T
        Y2=torch.sum(torch.square(y),dim=1)
        return X2,XY,Y2


# In[9]:


class GAN(ClimateNet):
    def __init__(self,spread=0,                    width_generator=[3,128,64,32,32,32,32,32,3],                    filter_size_generator=[3,3,3,3,3,3,3,3],                    width_discriminator=[3,128,64,32,32,32,32,32,1],                    filter_size_discriminator=[9,9,3,3,1,1,1,1],                    latsig=False,                    latsign=False,                    direct_coord=False,                    freq_coord=False,                    longitude=False,                    initwidth=3,                    outwidth=3,                    random_field=1):
        super(GAN, self).__init__(gan=True)
        device=self.device
        self.freq_coord=freq_coord
        
        self.outwidth=outwidth
        self.initwidth=initwidth
        self.random_field=random_field
        self.width_generator=width_generator
        self.width_discriminator=width_discriminator
        self.filter_size_generator=filter_size_generator
        self.filter_size_discriminator=filter_size_discriminator
        self.direct_coord=direct_coord
        self.longitude=longitude
        self.latsign=latsign
        self.latsig=latsig
        self.generator_layers=[]
        self.discriminator_layers=[]
        
        if self.direct_coord:
            if self.latsig:
                initwidth+=1
            if self.latsign:
                initwidth+=1
            if self.longitude:
                initwidth+=1
        elif self.freq_coord:
            if self.latsig:
                initwidth+=1
            if self.latsign:
                initwidth+=1
            if self.longitude:
                initwidth+=2
 
        # Discriminator build
        discriminator=ClimateNet()
        width=copy.deepcopy(width_discriminator)
        filter_size=copy.deepcopy(filter_size_discriminator)
        spread=0
        self.nparam=0
        width[0]=initwidth
        for i in range(len(filter_size)):
            if i==2:
                discriminator.nn_layers.append(nn.BatchNorm2d(outwidth).to(device) )
                self.nparam+=outwidth
                width[i]+=outwidth
            discriminator.nn_layers.append(nn.Conv2d(width[i], width[i+1], filter_size[i]).to(device) )
            self.nparam+=width[i]*width[i+1]*filter_size[i]**2
            spread+=(filter_size[i]-1)/2
            
            discriminator.nn_layers.append(nn.BatchNorm2d(width[i+1]).to(device) )
            self.nparam+=width[i+1]
            if i<len(filter_size)-1:
                discriminator.nn_layers.append(nn.ReLU(inplace=True).to(device)) 
        self.receptive_field=np.int64(spread*2+1)
        self.discriminator=discriminator
        
        # Generator build
        generator=ClimateNet()
        width=copy.deepcopy(width_generator)
        filter_size=copy.deepcopy(filter_size_generator)
        spread=0
        width[0]=initwidth+random_field
        width[-1]=outwidth
        for i in range(len(filter_size)):
            generator.nn_layers.append(nn.Conv2d(width[i], width[i+1], filter_size[i]).to(device) )
            self.nparam+=width[i]*width[i+1]*filter_size[i]**2
            spread+=(filter_size[i]-1)/2
            if i<len(filter_size)-1:
                generator.nn_layers.append(nn.BatchNorm2d(width[i+1]).to(device) )
                self.nparam+=width[i+1]
                generator.nn_layers.append(nn.ReLU(inplace=True).to(device)) 
            else:
                generator.nn_layers.append(nn.BatchNorm2d(width[i+1]).to(device) )
                self.nparam+=width[i+1]
        self.spread=np.maximum(np.int64(spread),self.spread)
        self.generator=generator
    def discriminator_forward(self,x,y):#,yhat):
        for i in range(6):
            x = self.discriminator.nn_layers[i](x)
        i=6
        #y=torch.cat([y,yhat],dim=1)
        y =self.discriminator.nn_layers[i](y)#-yhat)
        x=torch.cat([x,y],dim=1)
        for i in range(7,len(self.discriminator.nn_layers)):
            x = self.discriminator.nn_layers[i](x)
        return 1/(1+torch.exp(x))
    def generator_forward(self,x,z):
        x=torch.cat([x,z],dim=1)
        for i in range(len(self.generator.nn_layers)):
            x = self.generator.nn_layers[i](x)
        return x#torch.tanh(x)*50


# In[10]:


class QCNN(ClimateNet):
    def __init__(self,qwidth=64,qfilt=[11,11],spread=0,heteroscedastic=True,coarsen=0,                    width=[128,64,32,32,32,32,32,1],                    filter_size=[5,5,3,3,3,3,3,3],                    latsig=False,                    latsign=False,                    freq_coord=False,                    direct_coord=False,                    timeshuffle=False,                    physical_force_features=False,                    longitude=False,                    rescale=[1/10,1/1e7],                    initwidth=2,                    outwidth=2,                    nprecision=1):
        super(QCNN, self).__init__(spread=spread,coarsen=coarsen,latsig=latsig,timeshuffle=timeshuffle)
        device=self.device
        self.nn_layers = nn.ModuleList()
        spread=0
        
        self.rescale=torch.tensor(rescale,dtype=torch.float32,requires_grad=False)
        self.freq_coord=freq_coord
        self.heteroscedastic=heteroscedastic
        self.initwidth=initwidth
        self.outwidth=outwidth
        self.width=width
        self.filter_size=filter_size
        self.num_layers=len(filter_size)
        self.direct_coord=direct_coord
        self.longitude=longitude
        self.latsign=latsign
        self.latsig=latsig
        self.nprecision=nprecision
        self.physical_force_features=physical_force_features
        self.nparam=0
        self.bnflag=True
        
        self.nn_layers.append(nn.Conv2d(initwidth, qwidth, qfilt[0]).to(device) )
        self.nparam+=initwidth*qwidth*qfilt[0]**2
        self.nn_layers.append(nn.BatchNorm2d(qwidth).to(device) )
        self.nparam+=qwidth
        self.nn_layers.append(nn.BatchNorm2d(qwidth).to(device) )
        self.nparam+=qwidth
        self.nn_layers.append(nn.Conv2d(qwidth, outwidth, qfilt[1]).to(device) )
        self.nparam+=outwidth*qwidth*qfilt[1]**2
        self.nn_layers.append(nn.Conv2d(initwidth, width[0], filter_size[0]).to(device) )
        self.nparam+=initwidth*width[0]*filter_size[0]**2
        spread+=(filter_size[0]-1)/2
        width[-1]=nprecision
        for i in range(1,self.num_layers):
            self.nn_layers.append(nn.BatchNorm2d(width[i-1]).to(device) )
            self.nparam+=width[i-1]
            self.nn_layers.append(nn.Conv2d(width[i-1], width[i], filter_size[i]).to(device) )
            self.nparam+=width[i-1]*width[i]*filter_size[i]**2
            spread+=(filter_size[i]-1)/2
        
        self.nn_layers.append(nn.Softplus().to(device))
        spread+=coarsen
        self.spread=np.int64(spread)
        self.receptive_field=self.spread*2+1
    def forward(self, x):
        u=x*1
        cn=0
        u = self.nn_layers[cn](u)
        cn+=1
        u = torch.square(self.nn_layers[cn](u))
        cn+=1
        u = self.nn_layers[cn](u)
        cn+=1
        u = self.nn_layers[cn](u)
        cn+=1
        
        for i in range(self.num_layers-1):
            x = self.nn_layers[cn](x)
            cn+=1
            x = F.relu(self.nn_layers[cn](x))
            cn+=1
            
        x=self.nn_layers[cn](x)
        cn+=1
        precision=self.nn_layers[cn](x)
        x=torch.cat([u,precision],dim=1)
        return x


# In[11]:


class PCNN(ClimateNet):
    def __init__(self,qwidth=64,spread=0,heteroscedastic=True,coarsen=0,                    latsig=False,                    latsign=False,                    freq_coord=False,                    direct_coord=False,                    timeshuffle=False,                    physical_force_features=False,                    longitude=False,                    rescale=[1/10,1/1e7],                    initwidth=2,                    outwidth=3,                    degree=4):
        super(PCNN, self).__init__(spread=spread,latsig=latsig,timeshuffle=timeshuffle)
        device=self.device
        self.nn_layers = nn.ModuleList()
        spread=0
        
        self.rescale=torch.tensor(rescale,dtype=torch.float32,requires_grad=False)
        self.freq_coord=freq_coord
        self.heteroscedastic=heteroscedastic
        self.initwidth=initwidth
        self.outwidth=outwidth
        
        
        self.direct_coord=direct_coord
        self.longitude=longitude
        self.latsign=latsign
        self.latsig=latsig

        self.physical_force_features=physical_force_features
        
        self.degree=degree
        width=[64,64,32]
        # 7 5 
        # 
        filter_size=[7,7,5,5]
        for j in range(2):
            for i in range(degree-1):
                self.nn_layers.append(nn.Conv2d(initwidth, width[0], filter_size[0],bias=False).to(device) )
                self.nn_layers.append(nn.Conv2d(width[0], width[1], filter_size[1],bias=False).to(device) )
                self.nn_layers.append(nn.BatchNorm2d(width[1]).to(device) )
                self.nn_layers.append(nn.BatchNorm2d(width[1]).to(device) )
                self.nn_layers.append(nn.Conv2d(width[1], width[2], filter_size[2],bias=False).to(device) )
                if j==0:
                    self.nn_layers.append(nn.Conv2d(width[2], outwidth-1, filter_size[3],bias=i==0).to(device) )
                else:
                    self.nn_layers.append(nn.Conv2d(width[2], 1, filter_size[3],bias=i==0).to(device) )
        self.nn_layers.append(nn.BatchNorm2d(1).to(device) )
            
        spread=0
        for i in range(len(filter_size)):
            spread+=(filter_size[i]-1)/2
        
     
        
        self.nn_layers.append(nn.Softplus().to(device))
        self.spread=np.int64(spread)
        self.receptive_field=self.spread*2+1
    def forward(self, x,decomposed=False):
        degree=self.degree
        cn=0
        ymean=[[] for i in range(degree-1)]
        for i in range(degree-1):
            u=x*1
            for j in range(2):
                u = self.nn_layers[cn](u)
                cn+=1
            u = self.nn_layers[cn](u)#-self.nn_layers[cn].bias.reshape(1,-1,1,1)
            cn+=1
            u = u**(i+1)
            
            u = self.nn_layers[cn](u)#-self.nn_layers[cn].bias.reshape(1,-1,1,1)
            cn+=1
            for j in range(2):
                u = self.nn_layers[cn](u)
                cn+=1
            ymean[i]=u
            
        yprec=[[] for i in range(degree-1)]
        for i in range(degree-1):
            u=x*1
            for j in range(2):
                u = self.nn_layers[cn](u)
                cn+=1
            u = self.nn_layers[cn](u)#-self.nn_layers[cn].bias.reshape(1,-1,1,1)
            cn+=1
            
            u = u**(i+1)
            
            u = self.nn_layers[cn](u)#-self.nn_layers[cn].bias.reshape(1,-1,1,1)
            cn+=1
            
            for j in range(2):
                u = self.nn_layers[cn](u)
                cn+=1
            yprec[i]=u
        if decomposed:
            return ymean,yprec
        precision_=yprec[0]/degree
        mean_=ymean[0]/degree
        for i in range(1,degree-1):
            precision_+=yprec[i]/degree
            mean_+=ymean[i]/degree
        for j in range(2):
            precision_=self.nn_layers[cn](precision_)
            cn+=1
        x=torch.cat([mean_,precision_],dim=1)
        return x


# In[12]:


class QMat(ClimateNet):
    def __init__(self,filter_size=11,yequispaced=False):
        super(QMat, self).__init__()
        device = self.device
        self.yequispaced=yequispaced
        self.spread=np.int64((filter_size-1)/2)
        self.filter_size=filter_size
        m=filter_size
        m2=m**2
        self.nn_layers = nn.ModuleList()
        SHT=nn.Conv2d(2, 2*m2, m,bias=False).to(device)
        SHT.weight.data=SHT.weight.data*0
        for i in range(m2):
            i0,i1=np.unravel_index(i,[m,m])
            SHT.weight.data[i,0,i0,i1]=1
            SHT.weight.data[i+m2,1,i0,i1]=1
        SHT.weight.requires_grad=False
        self.shift_conv=SHT
        self.nn_layers.append( nn.Linear(2*m2,2*m2,bias=False).to(device))
        self.nn_layers.append( nn.Linear(2*m2,2*m2,bias=False).to(device))
    def forward(self, x):
        b=x.shape[0]
        d1=x.shape[2]
        d2=x.shape[3]
        m2=2*self.filter_size**2
        X=self.shift_conv(x).view(b,m2,-1)
        X=X.permute(0,2,1)
        
        M0X=self.nn_layers[0](X)
        Y0=torch.mul(M0X,X).sum(2)/m2
        
        M1X=self.nn_layers[1](X)
        Y1=torch.mul(M1X,X).sum(2)/m2
        Y=torch.stack([Y0,Y1],dim=1).view(b,2,d1-self.spread*2,d2-self.spread*2)
        return Y


# In[13]:


class DQCNN(ClimateNet):
    def __init__(self,spread=0,heteroscedastic=True,coarsen=0,                    width=[64,32,16,16,16,16,16,2],                    filter_size=[5,5,3,3,3,3,3,3],                    latsig=False,                    latsign=False,                    direct_coord=False,                    timeshuffle=False,                    physical_force_features=False,                    longitude=False,                    rescale=[1/10,1/1e7]):
        super(DQCNN, self).__init__(spread=spread,coarsen=coarsen,latsig=latsig,timeshuffle=timeshuffle)
        device=self.device
        self.nn_layers = nn.ModuleList()
        spread=0
        
        self.rescale=torch.tensor(rescale,dtype=torch.float32,requires_grad=False)
        
        self.heteroscedastic=heteroscedastic
        self.width=width
        self.filter_size=filter_size
        self.num_layers=len(filter_size)
        self.direct_coord=direct_coord
        self.longitude=longitude
        self.latsign=latsign
        self.latsig=latsig
        
        initwidth=2
        self.physical_force_features=physical_force_features
        
        self.bnflag=True#self.latsig or self.latsign or self.direct_coord

        self.nn_layers.append(nn.Conv2d(initwidth, width[0], filter_size[0]).to(device) )
        spread+=(filter_size[0]-1)/2
        for i in range(1,self.num_layers):
            self.nn_layers.append(nn.BatchNorm2d(width[i-1]).to(device) )
            self.nn_layers.append(nn.Conv2d(width[i-1], width[i], filter_size[i]).to(device) )
            self.nn_layers.append(nn.Conv2d(width[i], 3, 1).to(device) )
            spread+=(filter_size[i]-1)/2
        self.nn_layers.append(nn.BatchNorm2d(1).to(device) )
        self.nn_layers.append(nn.Softplus().to(device))
        self.spread=np.int64(spread)
        self.receptive_field=self.spread*2+1
    def forward(self, x):
        cn=0
        y=torch.zeros(x.shape[0],3,x.shape[2]-2*self.spread,x.shape[3]-2*self.spread).to(self.device)
        nspread=self.spread
        x = self.nn_layers[cn](x)
        cn+=1
        for i in range(self.num_layers-1):
            x = torch.square(self.nn_layers[cn](x))
            cn+=1
            
            x = self.nn_layers[cn](x)
            cn+=1
            
            x_ = self.nn_layers[cn](x)
            cn+=1
            
            nspread=(x_.shape[2]-y.shape[2])//2
            if nspread>0:
                y += x_[:,:,nspread:-nspread,nspread:-nspread]
            else:
                y += x_
        
        mean,precision=torch.split(y,[2,1],dim=1)
        precision=self.nn_layers[cn](precision)
        cn+=1
        precision=self.nn_layers[cn](precision)
        x=torch.cat([mean,precision],dim=1)
        return x


# In[12]:


def ismember_(a,b):
    for a_ in b:
        if a==a_:
            return True
    return False


# In[16]:


class NonlinearReg(ClimateNet):
    def __init__(self, order=3,width=64):
        super(NonlinearReg, self).__init__()
        device=self.device
        self.order=order
        inwidth=3**self.order*2 +3
        width0=width +3
        width1=width
        outwidth=3
        self.depth=7
        self.indepth=[1,3,5]
        self.spread=self.order
        self.nn_layers=nn.ModuleList()
        widths=[]*self.depth
        for i in range(self.depth):
            if ismember_(i,self.indepth):
                widths.append([width0,width1])
            else:
                widths.append([width0,width0])
                
        widths[-1][1]=outwidth
        widths[0][0]=inwidth
        self.nn_layers.append(nn.Conv2d(widths[0][0], widths[0][1],1).to(device))
        for i in range(self.depth-2):
            self.nn_layers.append(nn.BatchNorm2d(widths[i][1]).to(device))
            self.nn_layers.append(nn.Conv2d(widths[i+1][0], widths[i+1][1],1).to(device))
          
        self.nn_layers.append(nn.BatchNorm2d(widths[self.depth-2][1]).to(device))
        self.nn_layers.append(nn.Conv2d(widths[self.depth-1][0], widths[self.depth-1][1],1).to(device))
        
        self.preclyr=nn.Softplus().to(device)
    def forward(self,x):
        (u,geo)=torch.split(x,[2,x.shape[1]-2],dim=1)
        u=self.take_derivatives(u)
        geo=geo[:,:,self.order:-self.order,self.order:-self.order]
        u=torch.cat([u,geo],dim=1)
        i=0
        for t in range(self.depth-1):
            if self.nn_layers[i].weight.data.shape[1]>u.shape[1]:
                u=torch.cat([u,geo],dim=1)
            u=self.nn_layers[i](u)
            i+=1
            u=torch.selu(self.nn_layers[i](u))
            i+=1
        if self.nn_layers[i].weight.data.shape[1]>u.shape[1]:
            u=torch.cat([u,geo],dim=1)
        u=self.nn_layers[i](u)
        (mean,prec)=torch.split(u,[2,1],dim=1)
        prec=self.preclyr(prec)
        y=torch.cat([mean,prec],dim=1)
        return y
    def take_derivatives(self,u):
        for i in range(self.order):
            dudy=u[:,:,2:,1:-1]-u[:,:,:-2,1:-1]
            dudx=u[:,:,1:-1,2:]-u[:,:,1:-1,:-2]
            u=u[:,:,1:-1,1:-1]
            u=torch.cat([u,dudy,dudx],dim=1)
        return u


# In[17]:


class MatReg(ClimateNet):
    def __init__(self,order=2,width=64):
        super(MatReg, self).__init__()
        device=self.device
        inwidth=3
        self.order=order
        self.direct_coord=True
        outwidth=((3**self.order*2)*2+2)*3
        self.width=width
        self.depth=5
        self.spread=self.order
        self.nn_layers=nn.ModuleList()
        self.nn_layers.append(nn.Conv2d(inwidth, width,1).to(device))
        for i in range(self.depth-2):
            self.nn_layers.append(nn.BatchNorm2d(width).to(device))
            self.nn_layers.append(nn.Conv2d(width, width,1).to(device))
            
        self.nn_layers.append(nn.BatchNorm2d(width).to(device))
        self.nn_layers.append(nn.Conv2d(width, outwidth,1).to(device))
        for i in range(3):
            self.nn_layers.append(nn.BatchNorm2d(outwidth//3).to(device))
            self.nn_layers.append(nn.Conv2d(outwidth//3, 1,1).to(device))
        self.preclyr=nn.Softplus().to(device)
    def forward(self,x):
        (u,geo)=torch.split(x,[2,x.shape[1]-2],dim=1)
        i=0
        for _ in range(self.depth-1):
            geo=self.nn_layers[i](geo)
            i+=1
            geo=torch.selu(self.nn_layers[i](geo))
            i+=1
        geo=self.nn_layers[i](geo)
        i+=1
        geo=geo[:,:,self.order:-self.order,self.order:-self.order]
        U=self.take_derivatives(u)
        u=U[:,:2]
        U=torch.cat([u,U*u[:,0:1],U*u[:,1:2]],dim=1)
        GEO=torch.split(geo,geo.shape[1]//3,dim=1)
        Y=[U*geo for geo in GEO]
        for j in range(3):
            Y[j]=self.nn_layers[i](Y[j])
            i+=1
            Y[j]=self.nn_layers[i](Y[j])
            i+=1
        Y[2]=self.preclyr(Y[2])
        y=torch.cat(Y,dim=1)
        return y
    def take_derivatives(self,u):
        for i in range(self.order):
            dudy=u[:,:,2:,1:-1]-u[:,:,:-2,1:-1]
            dudx=u[:,:,1:-1,2:]-u[:,:,1:-1,:-2]
            u=u[:,:,1:-1,1:-1]
            u=torch.cat([u,dudy,dudx],dim=1)
        return u


# In[18]:


class UNET(ClimateNet):
    def __init__(self,spread=0,heteroscedastic=True,                    #width=[128,64,32,32,32,32,32,3],\
                     widths=[64,128,256,512],\
                    pools=[2,2,2],\
                    filter_size=[5,5,3,3,3,3,3,3],\
                    deep_filters=[[3,3,3,1,1],[3,3,3,1,1],[3,3,3,1,1]],\
                    latsig=False,\
                    latsign=False,\
                    direct_coord=False,\
                    freq_coord=False,\
                    timeshuffle=False,\
                    physical_force_features=False,\
                    longitude=False,\
                    rescale=[1/10,1/1e7],\
                    initwidth=2,\
                    outwidth=2,\
                    nprecision=1,\
                    verbose=False):
        super(UNET, self).__init__()
        device=self.device
        bnflag=True
        self.bnflag=bnflag
        self.direct_coord=direct_coord
        self.freq_coord=freq_coord
        self.longitude=longitude
        self.latsig=latsig
        self.latsign=latsign
        self.nn_layers=nn.ModuleList()
        self.verbose=verbose
        self.nprecision=nprecision
        self.nparam=0
        self.bnflag=True#self.latsig or self.latsign or self.direct_coord
        if self.direct_coord:
            initwidth+=3
        elif self.freq_coord:
            if self.latsig:
                initwidth+=1
            if self.latsign:
                initwidth+=1
            if self.longitude:
                initwidth+=2
        self.initwidth=initwidth
        self.outwidth=outwidth
        self.spread=int((np.sum(filter_size)-len(filter_size))/2)
        
        
        
        widthin=initwidth
        widthout=outwidth+nprecision
        
        #widths=[64,128,256,512]#[2,4,8,16]#
        nlevel=len(widths)
        self.nlevel=nlevel
        self.locs=[]
        self.pools=pools
        
        self.receptive_field=1
        for i in range(len(pools)):
            ww=np.sum(deep_filters[-1-i])-len(deep_filters[-1-i])
            self.receptive_field=(self.receptive_field+ww)*pools[-i-1]
        self.receptive_field+=np.sum(filter_size[:3])-3
        #self.add_conv_layers([5,5,3],[widths[0]]*4,widthin=widthin) 
        self.add_conv_layers(filter_size[:3],[widths[0]]*4,widthin=widthin) 
        self.add_conv_layers(1,[widths[0]]*3)
        for i in range(nlevel-1):
            pool=[pools[i],pools[i]]
            self.add_down_sampling(pool)
            self.add_conv_layers(deep_filters[i],[widths[i+1]]*(len(deep_filters[i])+1),widthin=widths[i])
            #self.add_conv_layers(1,[widths[i+1]]*3)
            
        
        for i in range(nlevel-2):
            pool=[pools[-1-i],pools[-1-i]]
            self.add_up_sampling(1,[widths[-1-i],widths[-1-i]//2],pool)
            self.add_conv_layers(deep_filters[-1-i],[widths[-2-i]]*(len(deep_filters[-1-i])+1),widthout=widths[-2-i],widthin=widths[-1-i])
            
        
        i=nlevel-2
        pool=[pools[-1-i],pools[-1-i]]
        self.add_up_sampling(1,[widths[-1-i],widths[-1-i]//2],pool)
        self.add_conv_layers(filter_size[3:],[widths[-2-i]]*6,widthout=widthout,widthin=widths[-1-i])
        self.to_device()
        self.precisionlyr=nn.Softplus().to(device)
    def add_conv_layers(self,conv,width,widthin=0,widthout=0,final_nonlinearity=False):
        loc0=len(self.nn_layers)
        if widthin!=0:
            width[0]=widthin
        if widthout!=0:
            width[-1]=widthout
        if type(conv)==int:
            conv=[conv]*(len(width)-1)
        for i in range(len(width)-2):
            self.nn_layers.append(nn.Conv2d(width[i], width[i+1],conv[i]))
            self.nparam+=width[i]*width[i+1]*conv[i]**2
            self.nn_layers.append(nn.BatchNorm2d(width[i+1]))
            self.nparam+=width[i+1]
            self.nn_layers.append(nn.ReLU(inplace=True))
        self.nn_layers.append(nn.Conv2d(width[-2], width[-1],conv[-1]))
        self.nparam+=width[-2]*width[-1]*conv[-1]**2
        if final_nonlinearity:
            self.nn_layers.append(nn.BatchNorm2d(width[-1]))
            self.nparam+=width[-1]
            self.nn_layers.append(nn.ReLU(inplace=True))
        loc1=len(self.nn_layers)
        self.locs.append([loc0,loc1])
    def add_down_sampling(self,pool):
        loc0=len(self.nn_layers)
        self.nn_layers.append(nn.MaxPool2d(pool, stride=pool))
        loc1=len(self.nn_layers)
        self.locs.append([loc0,loc1])
    def add_up_sampling(self,conv,width,pool):
        loc0=len(self.nn_layers)
        self.nn_layers.append(nn.ConvTranspose2d(width[0], width[1],conv,stride=pool))
        self.nparam+=width[0]*width[1]*conv**2
        loc1=len(self.nn_layers)
        self.locs.append([loc0,loc1])
    def apply_layers(self,x,K):
        for i in range(self.locs[K][0],self.locs[K][1]):
            x=self.nn_layers[i](x)
        return x
    def to_device(self,):
        for i in range(len(self.nn_layers)):
            self.nn_layers[i]= self.nn_layers[i].to(self.device)       

    def trim_merge(self,f,x):
        if type(x)==torch.Tensor:
            ny=(f.shape[2]-x.shape[2])
            nx=(f.shape[3]-x.shape[3])
        else:
            ny=(f.shape[2]-x[0])
            nx=(f.shape[3]-x[1])
        ny0,ny1=ny//2,ny//2
        nx0,nx1=nx//2,nx//2
        if ny0+ny1<ny:
            ny1+=1
        if nx0+nx1<nx:
            nx1+=1
        f=f[:,:,ny0:-ny1,nx0:-nx1]
        
        if type(x)==torch.Tensor:
            return torch.cat([f,x],dim=1)
        else:
            return f
    def zeropad(self,f,x):
        if type(x)==torch.Tensor:
            diffY=(f.shape[2]-x.shape[2])
            diffX=(f.shape[3]-x.shape[3])
        else:
            diffY=(f.shape[2]-x.shape[2])
            diffX=(f.shape[3]-x.shape[3])
        
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        if type(x)==torch.Tensor:
            return torch.cat([f,x],dim=1)
        else:
            return f
    def forward(self,u):
        locs=self.locs
        
        features=[]
        nlevel=self.nlevel
        t=0
        x=u*1
        x=self.apply_layers(x,t) # convolutions
        if self.verbose:
            print('conv: '+str(0)+' '+str(t)+  '  '+ str(x.shape))
        t+=1
        
        f=x*1
        f=self.apply_layers(f,t) # ptswise
        if self.verbose:
            print('ptswise: '+str(0)+' '+str(t)+  '  '+ str(f.shape))
        t+=1
        features.append(f)
        
        for i in range(nlevel-1):
            x=self.apply_layers(x,t) # downsampling
            if self.verbose:
                print('down: '+str(i+1)+' '+str(t)+  '  '+ str(x.shape))
            t+=1
            
            x=self.apply_layers(x,t) # convolutions
            if self.verbose:
                print('conv: '+str(i+1)+' '+str(t)+  '  '+ str(x.shape))
            t+=1
            
            f=x*1
            '''
            f=self.apply_layers(f,t) # ptswise
            if self.verbose:
                print('ptswise: '+str(i+1)+' '+str(t)+  '  '+ str(f.shape))
            t+=1'''
            features.append(f)

        f=features[-1]
        for jj in range(1,nlevel):
            j=nlevel-jj-1
            x=self.apply_layers(f,t) # upsample
            if self.verbose:
                print('upsample: ('+str(j)+ ', '+str(0)+') '+str(t)+ '  '+ str(x.shape))
            t+=1
            f=features[j]
            f=self.zeropad(f,x)
            f=self.apply_layers(f,t) # convolutions
            if self.verbose:
                print('conv: '+str(i+1)+' '+str(t)+  '  '+ str(f.shape))
            t+=1
        (mean,prec)=torch.split(f,[self.outwidth,self.nprecision],dim=1)
        prec=self.precisionlyr(prec)
        y=torch.cat([mean,prec],dim=1)
        return y


# In[19]:


class CQCNN(ClimateNet):
    def __init__(self,width=32,classnum=8,direct_coord=False):
        super(CQCNN, self).__init__()
        latsig=True
        latsign=True
        longitude=True
        heteroscedastic=True
        self.heteroscedastic=heteroscedastic
        self.spread=10
        self.direct_coord=direct_coord
        self.longitude=longitude
        self.latsig=latsig
        self.latsign=latsign
        
        self.nn_layers = nn.ModuleList()
        self.other_learnables = []
        self.locs=[]
        self.layers={}
        geoin=4
        flowin=2
        width_in=geoin+flowin
        self.classnum=classnum
        self.width=width
        #width=[128,64,32,32,32,32,32,3],\
                    #filter_size=
        self.create_conv_layers([5,5,3,3,3,3,3,3],[width_in,64,32,32,32,32,32,32,classnum],                                final_nonlinearity=False,                                label='class-conv')
        self.create_batch_norm_layer(classnum,label='presoftmax')
        self.create_softmax_layer(label='softmax')
        for i in range(classnum):
            self.create_conv_layers([7,5],[flowin,width,width],                                    mid_nonlinearity=False,final_nonlinearity=False,label='fun-conv0-'+str(i))
            self.create_poly_nonlinearity(width,2,label='poly-nnlnr-'+str(i))
            #self.create_batch_norm_layer(width,label='post-poly-nnlnr-'+str(i))
            self.create_conv_layers([7,5],[width,width,2],                                    mid_nonlinearity=False,final_nonlinearity=False,label='fun-conv1-'+str(i))
        for i in range(classnum):
            self.create_conv_layers([7,5],[flowin,width,width],                                    mid_nonlinearity=False,final_nonlinearity=False,label='h-fun-conv0-'+str(i))
            self.create_poly_nonlinearity(width,4,label='h-poly-nnlnr-'+str(i))
            #self.create_batch_norm_layer(width,label='h-post-poly-nnlnr-'+str(i))
            self.create_conv_layers([7,5],[width,width,1],                                    mid_nonlinearity=False,final_nonlinearity=False,label='h-fun-conv1-'+str(i))
        self.create_batch_norm_layer(1,label='h-pre-softplus-nnlnr')
        self.to_device()
        self.precisionlyr=nn.Softplus().to(self.device)
        
    def to_device(self,):
        for i in range(len(self.nn_layers)):
            self.nn_layers[i]= self.nn_layers[i].to(self.device)   
        for i in range(len(self.other_learnables)):
            self.other_learnables[i]= self.other_learnables[i].to(self.device)   
    def forward(self,x):
        uv,_=torch.split(x,[2,x.shape[1]-2],dim=1)
        t=0
        x=self.apply_layers(x,label='class-conv') #convolution
        x=self.apply_layers(x,label='presoftmax') #batchnorm
        cl0=self.apply_layers(x,label='softmax') #softmax
        cl=torch.split(cl0,1,dim=1)
        mean=torch.zeros(cl[0].shape[0],2,cl[0].shape[2],cl[0].shape[3]).to(self.device)
        pre_precision=torch.zeros(cl[0].shape[0],1,cl[0].shape[2],cl[0].shape[3]).to(self.device)
        for i in range(self.classnum):
            y=self.apply_layers(uv,label='fun-conv0-'+str(i)) 
            #y=self.apply_layers(y,label='pre-poly-nnlnr-'+str(i))
            y=self.apply_poly_nonlinearity(y,label='poly-nnlnr-'+str(i))
            mean+=cl[i]*self.apply_layers(y,label='fun-conv1-'+str(i))
        for i in range(self.classnum):
            y=self.apply_layers(uv,label='h-fun-conv0-'+str(i)) 
            #y=self.apply_layers(y,label='h-pre-poly-nnlnr-'+str(i))
            y=self.apply_poly_nonlinearity(y,label='h-poly-nnlnr-'+str(i))
            pre_precision+=cl[i]*self.apply_layers(y,label='h-fun-conv1-'+str(i))
        pre_precision=self.apply_layers(pre_precision,label='h-pre-softplus-nnlnr')
        prec=self.precisionlyr(pre_precision)
        #print(mean.shape,prec.shape,cl0.shape)
        return torch.cat([mean,prec,cl0],dim=1)
    def quadratic_forward(self,x,class_index=0):
        uv,_=torch.split(x,[2,x.shape[1]-2],dim=1)
        i=class_index
        y=self.apply_layers(uv,label='fun-conv0-'+str(i)) 
        #y=self.apply_layers(y,label='pre-poly-nnlnr-'+str(i))
        y=self.apply_poly_nonlinearity(y,label='poly-nnlnr-'+str(i))
        y=self.apply_layers(y,label='fun-conv1-'+str(i))
        return y
    def register_layers(self,loc0,loc1,label):
        self.layers[label]=[loc0,loc1]
    def create_poly_nonlinearity(self,width,degree,label='poly-nnlnr'):
        loc0=len(self.nn_layers)
        self.nn_layers.append(nn.Conv2d(1,1,1,bias=False))
        for i in range(degree):
            self.nn_layers.append(nn.BatchNorm2d(width))
            self.nn_layers.append(nn.Conv2d(1,1,1,bias=False))
        loc1=len(self.nn_layers)
        self.register_layers(loc0,loc1,label)
    def create_batch_norm_layer(self,width,label='batchnorm'):
        loc0=len(self.nn_layers)
        self.nn_layers.append(nn.BatchNorm2d(width))
        loc1=len(self.nn_layers)
        self.register_layers(loc0,loc1,label)
    def create_softmax_layer(self,label='softmax'):
        loc0=len(self.nn_layers)
        self.nn_layers.append(nn.Softmax(dim=1)) 
        loc1=len(self.nn_layers)
        self.register_layers(loc0,loc1,label)
    def apply_poly_nonlinearity(self,x,label):
        loc0=self.layers[label][0]
        loc1=self.layers[label][1]
        y=self.nn_layers[loc0].weight.data+x*0
        j=1
        for i in range(loc0+1,loc1,2):
            b=self.nn_layers[i](x)
            y+=self.nn_layers[i+1].weight.data*(b**j)/np.math.factorial(i-loc0)
            j+=1
        return y
    def apply_layers(self,x,label):
        for i in range(self.layers[label][0],self.layers[label][1]):
            x=self.nn_layers[i](x)
        return x
    def create_conv_layers(self,conv,width,widthin=0,widthout=0,mid_nonlinearity=True,final_nonlinearity=False,label='conv'):
        loc0=len(self.nn_layers)
        if widthin!=0:
            width[0]=widthin
        if widthout!=0:
            width[-1]=widthout
        if type(conv)==int:
            conv=[conv]*(len(width)-1)
        for i in range(len(width)-2):
            self.nn_layers.append(nn.Conv2d(width[i], width[i+1],conv[i]))
            if mid_nonlinearity:
                self.nn_layers.append(nn.BatchNorm2d(width[i+1]))
                self.nn_layers.append(nn.ReLU())
        self.nn_layers.append(nn.Conv2d(width[-2], width[-1],conv[-1]))
        if final_nonlinearity:
            self.nn_layers.append(nn.BatchNorm2d(width[-1]))
            self.nn_layers.append(nn.ReLU())
        loc1=len(self.nn_layers)
        self.register_layers(loc0,loc1,label)


# In[20]:


class Improved_QCNN(ClimateNet):
    def __init__(self,width=64,filter_size=[11,11],                    direct_coord=False):
        super(Improved_QCNN, self).__init__()
        latsig=True
        latsign=True
        timeshuffle=False
        longitude=True
        heteroscedastic=True
        self.heteroscedastic=heteroscedastic
        self.spread=10
        self.direct_coord=direct_coord
        self.longitude=longitude
        self.latsig=latsig
        self.latsign=latsign
        
        self.nn_layers = nn.ModuleList()
        self.locs=[]
        
        geoin=4
        flowin=2
        
        widths=[geoin,64,32,16]
        
        
        self.add_conv_layers([11,7,5],[geoin,16,16,16],final_nonlinearity=True)
        self.add_transconv_layers([5,7,11],[16,16,16,width],final_nonlinearity=False)
        self.add_conv_layers([1],[flowin,width],final_nonlinearity=False)
        self.add_batch_norm_layer(width)
        self.add_conv_layers([11],[width,width])
        self.add_batch_norm_layer(width)
        self.add_conv_layers([11],[width,16],final_nonlinearity=False)
        self.add_conv_layers([1,1,1,1],[16,16,16,16,3],final_nonlinearity=False)
        self.to_device()
        self.precisionlyr=nn.Softplus().to(self.device)
        
    def to_device(self,):
        for i in range(len(self.nn_layers)):
            self.nn_layers[i]= self.nn_layers[i].to(self.device)   
    def forward(self,x):
        uv,geo=torch.split(x,[2,x.shape[1]-2],dim=1)
        t=0
        geo=self.apply_layers(geo,t) #convolution
        #print('geo: '+str(geo.shape))
        t+=1
        geo=self.apply_layers(geo,t) #deconvolution
        #print('geo: '+str(geo.shape))
        t+=1
        uv=self.apply_layers(uv,t) #upwidth
        #print('uv: '+str(uv.shape))
        t+=1
        uv=uv*(1+geo)
        uv=self.apply_layers(uv,t) #batch normalization
        t+=1
        uv=self.apply_layers(uv,t) #convolution
        t+=1
        uv=torch.square(uv) #squaring
        uv=self.apply_layers(uv,t) #batch normalization
        t+=1
        uv=self.apply_layers(uv,t) #convolution
        t+=1
        uv=self.apply_layers(uv,t) #pointwise nonlinearity
        t+=1
        mean,prec=torch.split(uv,[2,1],dim=1)
        prec=self.precisionlyr(prec)
        return torch.cat([mean,prec],dim=1)
    def add_batch_norm_layer(self,width):
        loc0=len(self.nn_layers)
        self.nn_layers.append(nn.BatchNorm2d(width)) #filter multip
        loc1=len(self.nn_layers)
        self.locs.append([loc0,loc1])
        
    def add_conv_layers(self,conv,width,widthin=0,widthout=0,final_nonlinearity=False):
        loc0=len(self.nn_layers)
        if widthin!=0:
            width[0]=widthin
        if widthout!=0:
            width[-1]=widthout
        if type(conv)==int:
            conv=[conv]*(len(width)-1)
        for i in range(len(width)-2):
            self.nn_layers.append(nn.Conv2d(width[i], width[i+1],conv[i]))
            self.nn_layers.append(nn.BatchNorm2d(width[i+1]))
            self.nn_layers.append(nn.ReLU())
        self.nn_layers.append(nn.Conv2d(width[-2], width[-1],conv[-1]))
        if final_nonlinearity:
            self.nn_layers.append(nn.BatchNorm2d(width[-1]))
            self.nn_layers.append(nn.ReLU())
        loc1=len(self.nn_layers)
        self.locs.append([loc0,loc1])
    def add_transconv_layers(self,conv,width,pool=[1,1],widthin=0,widthout=0,nnlnr=True,final_nonlinearity=False):
        loc0=len(self.nn_layers)
        if widthin!=0:
            width[0]=widthin
        if widthout!=0:
            width[-1]=widthout
        if type(conv)==int:
            conv=[conv]*(len(width)-1)
        for i in range(len(width)-2):
            self.nn_layers.append(nn.ConvTranspose2d(width[i], width[i+1],conv[i]))
            self.nn_layers.append(nn.BatchNorm2d(width[i+1]))
            if nnlnr:
                self.nn_layers.append(nn.ReLU())
        self.nn_layers.append(nn.ConvTranspose2d(width[-2], width[-1],conv[-1]))
        if final_nonlinearity:
            self.nn_layers.append(nn.BatchNorm2d(width[-1]))
            if nnlnr:
                self.nn_layers.append(nn.ReLU())
        loc1=len(self.nn_layers)
        self.locs.append([loc0,loc1])
        
    def apply_layers(self,x,K):
        for i in range(self.locs[K][0],self.locs[K][1]):
            x=self.nn_layers[i](x)
        return x
    


# In[21]:


class Autoencoder(ClimateNet):
    def __init__(self,):
        super(Autoencoder, self).__init__()
        latsig=False
        latsign=False
        timeshuffle=False
        longitude=False
        heteroscedastic=False
        direct_coord=False
        self.heteroscedastic=heteroscedastic
        self.spread=0
        self.generative=True
        self.rescale=[]
        self.direct_coord=direct_coord
        self.longitude=longitude
        self.latsig=latsig
        self.latsign=latsign
        
        self.nn_layers = nn.ModuleList()
        self.locs=[]
        
        #self.add_conv_layers([9,7,5,2],[2,32,32,64,256],final_nonlinearity=True)
        self.add_conv_layers([5,5,3,3,3,3,3],[2,32,64,64,64,64,128,128],final_nonlinearity=True)
        #self.add_transconv_layers([2,5,7,9],[256,64,32,32,2],final_nonlinearity=False)
        self.add_transconv_layers([3,3,3,3,3,5,5],[128,128,64,64,64,64,32,2],final_nonlinearity=False)
        self.to_device()
        
    def to_device(self,):
        for i in range(len(self.nn_layers)):
            self.nn_layers[i]= self.nn_layers[i].to(self.device)   
    def forward(self,x):
        uv,geo=torch.split(x,[2,x.shape[1]-2],dim=1)
        for t in range(2):
            uv=self.apply_layers(uv,t) 
        return uv
    def encode(self,x):
        uv,geo=torch.split(x,[2,x.shape[1]-2],dim=1)
        return self.apply_layers(uv,0) 
    def decode(self,x):
        uv,geo=torch.split(x,[2,x.shape[1]-2],dim=1)
        return self.apply_layers(uv,1) 
    def add_batch_norm_layer(self,width):
        loc0=len(self.nn_layers)
        self.nn_layers.append(nn.BatchNorm2d(width)) #filter multip
        loc1=len(self.nn_layers)
        self.locs.append([loc0,loc1])
        
    def add_conv_layers(self,conv,width,stride=[],widthin=0,widthout=0,final_nonlinearity=False):
        loc0=len(self.nn_layers)
        if widthin!=0:
            width[0]=widthin
        if widthout!=0:
            width[-1]=widthout
        if type(conv)==int:
            conv=[conv]*(len(width)-1)
        if len(stride)==0:
            stride=[1]*(len(width)-1)
        for i in range(len(width)-2):
            self.nn_layers.append(nn.Conv2d(width[i], width[i+1],conv[i],stride=stride[i]))
            self.nn_layers.append(nn.BatchNorm2d(width[i+1]))
            self.nn_layers.append(nn.ReLU())
        self.nn_layers.append(nn.Conv2d(width[-2], width[-1],conv[-1],stride=stride[-1]))
        if final_nonlinearity:
            self.nn_layers.append(nn.BatchNorm2d(width[-1]))
            self.nn_layers.append(nn.ReLU())
        loc1=len(self.nn_layers)
        self.locs.append([loc0,loc1])
    def add_transconv_layers(self,conv,width,stride=[],pool=[1,1],widthin=0,widthout=0,nnlnr=True,final_nonlinearity=False):
        loc0=len(self.nn_layers)
        if widthin!=0:
            width[0]=widthin
        if widthout!=0:
            width[-1]=widthout
        if type(conv)==int:
            conv=[conv]*(len(width)-1)
        if len(stride)==0:
            stride=[1]*(len(width)-1)
        for i in range(len(width)-2):
            self.nn_layers.append(nn.ConvTranspose2d(width[i], width[i+1],conv[i],stride=stride[i]))
            self.nn_layers.append(nn.BatchNorm2d(width[i+1]))
            if nnlnr:
                self.nn_layers.append(nn.ReLU())
        self.nn_layers.append(nn.ConvTranspose2d(width[-2], width[-1],conv[-1],stride=stride[-1]))
        if final_nonlinearity:
            self.nn_layers.append(nn.BatchNorm2d(width[-1]))
            if nnlnr:
                self.nn_layers.append(nn.ReLU())
        loc1=len(self.nn_layers)
        self.locs.append([loc0,loc1])
        
    def apply_layers(self,x,K):
        for i in range(self.locs[K][0],self.locs[K][1]):
            x=self.nn_layers[i](x)
        return x
    


# In[22]:


'''net=UNET(verbose=True,ypad=150,xpad=200)'''


# In[ ]:





# In[23]:


'''(645-y.shape[2]) // 2,  (900-y.shape[3])//2'''


# In[24]:


'''y=net.forward(x)'''


# In[193]:


'''x=torch.randn(2,6,net.ypad*2+645,net.xpad*2+900)'''


# In[137]:





# In[ ]:




