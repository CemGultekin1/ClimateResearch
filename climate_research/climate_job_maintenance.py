#!/usr/bin/env python
# coding: utf-8

# In[2]:


import xarray as xr
import torch
from torch.nn import functional as F
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import date
import json
import copy
from scipy.interpolate import RectBivariateSpline
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product

import climate_train
import climate_data
import climate_models


# In[3]:


def jobnums(C=[2,3,4,3,2],x=[[0,1],[0,1,2],[0,1,2,3],[1,2],[0]],offset=5000):
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
    y=[tt+offset for tt in y] 
    return y
def factor_jobnum(C,x):
    x=[xx%1000 for xx in x]
    n=len(x)
    y=[]
    for i in range(n):
        num=x[i]
        y_=[]
        for j in range(len(C)):
            y_.append(num%C[j])
            num=num//C[j]
        y.append(y_)
    return y
def configure_models(modelnums):
    for i in modelnums:
        string_input="--b 1 -e 10 --nworkers 10             --subtime 0.005 --disp 1 --relog 1 --rerun 1             --lr 0.01 --model_id " +str(i)+ " --model_bank_id G"
        args=climate_train.options(string_input=string_input.split())
        _=climate_models.model_bank(args,configure=True,verbose=False)

        net,criterion,(data_init,partition),logs,(PATH0,PATH1,LOG,root)=climate_train.load_from_save(args)
        print(net.nparam,net.receptive_field)
        (training_set,training_generator),                    (val_set,val_generator),                        (test_set,test_generator),                            (glbl_set,glbl_gen)=climate_train.load_data(data_init,partition,args)
        #landmasks=climate_data.get_land_masks(val_generator)
        training_set.save_scales()
        training_set.save_masks()
        landmasks=training_set.get_masks()
        X,dom_num,Y=training_set[0]
        print('\t\t'+str(landmasks[dom_num].shape)+' '+str(Y.shape))
        print(str(i),flush=True)
    climate_data.safe()
def check_model(modelnums):
    final_lrs=[]
    num_iter=[]
    eval_vals=[]
    grprb_vals=[]
    
    for i in range(len(modelnums)):
        if modelnums[i]//1000!=6:
            LOG='/scratch/cg3306/climate/runs/G-'+str(modelnums[i])+'/log.json'

            try:
                with open(LOG, 'r') as outfile:
                    logs=json.load(outfile)
                if np.isnan(logs['test-loss'][-1]):
                    flr=np.nan
                    niter=np.nan
                else:
                    flr=logs['lr'][-1]
                    niter=len(logs['lr'])
            except:
                flr=1
                niter=0
            final_lrs.append(flr)
            num_iter.append(niter)
            DIR='/scratch/cg3306/climate/runs/G-'+str(modelnums[i])+'/'
            if os.path.exists(DIR):
                evalflag=('MSE.npy' in os.listdir(DIR) or 'MSE-depth.npy' in os.listdir(DIR)) and                             ('MSE-co2.npy' in os.listdir(DIR) or 'MSE-co2-depth.npy' in os.listdir(DIR))
            else:
                evalflag=0
            eval_vals.append(evalflag)
            gradprobeflag=0
            if os.path.exists(DIR):
                if 'grad-probe-data-0.npy' in os.listdir(DIR):
                    gradprobeflag=1
            grprb_vals.append(gradprobeflag)
        else:
            final_lrs.append(0)
            grprb_vals.append(0)
            DIR='/scratch/cg3306/climate/runs/G-'+str(modelnums[i])+'/'
            num=0
            if 'X2-train.npy' in os.listdir(DIR):
                num+=1
            if 'X2-val.npy' in os.listdir(DIR):
                num+=1
            if 'X2-test.npy' in os.listdir(DIR):
                num+=1
            num_iter.append(num)
            evalflag=('MSE.npy' in os.listdir(DIR) or 'MSE-depth.npy' in os.listdir(DIR)) and                             ('MSE-co2.npy' in os.listdir(DIR) or 'MSE-co2-depth.npy' in os.listdir(DIR))
            eval_vals.append(evalflag)
    x=[final_lrs,num_iter,eval_vals,grprb_vals]
    x=[np.array(xx) for xx in x ]
    x=np.array(x)
    return x
def select_models(C,x,conditional,offset):
    modelnums=jobnums(C=C,x=x,offset=offset)
    modelnums=np.array(modelnums)
    x=check_model(modelnums)
    y=np.array([conditional(x[:,i]) for i in range(x.shape[1])])
    I=np.where(y)[0]
    return modelnums[I]
def report_progress(C,x,offset=5000):
    report_short=lambda str_,I:  print(str_+':\n'+str(I.tolist()).replace(' ',''))
    lower_lim=1e-6
    if offset==6000:
        Instrt=select_models(C,x,lambda x: x[1]==0,offset)
        Inonfinished=select_models(C,x,lambda x: x[1]<3,offset)
        Ifinished=select_models(C,x,lambda x: x[1]==3,offset)
        Istrt=select_models(C,x,lambda x: x[1]>0,offset)
        
        Inonevalco2=select_models(C,x,lambda x: x[2]//2==0,offset)
        Iallevals=select_models(C,x,lambda x: x[2]==3,offset)
        Inonevalnonco2=select_models(C,x,lambda x: x[2]%2==0,offset)
        report_short('Hasnt started',Instrt)
        report_short('Files exist',Istrt)
        report_short('No test file',Inonfinished)
        report_short('Test file',Ifinished)
        report_short('No co2 eval',Inonevalco2)
        report_short('No nonco2 eval',Inonevalnonco2)
        report_short('All evals done',Iallevals)
        return
    Instp=select_models(C,x,lambda x: x[1]>0 and x[0]>1e-7,offset)
    Ilate_eval=select_models(C,x,lambda x: x[0]>lower_lim and x[0]<1e-3 and not x[2],offset)

    Instrt=select_models(C,x,lambda x: x[1]==0,offset)
    Ieval0=select_models(C,x,lambda x: (x[0]<=lower_lim) and not x[2],offset)
    Ieval1=select_models(C,x,lambda x: (x[0]<=lower_lim) and x[2],offset)
    Ieval0_expand=np.concatenate([Ieval0,Ieval0+500],axis=0)
    Ilate_eval=np.concatenate([Ilate_eval,Ilate_eval+500],axis=0)
    Instp_expand=np.concatenate([Instp,Instp+500],axis=0)
    Inan=select_models(C,x,lambda x: np.isnan(x[1]),offset)
    
    
    Igrad0=select_models(C,x,lambda x: (x[0]<=lower_lim) and not x[3],offset)
    Igrad1=select_models(C,x,lambda x: (x[0]<=lower_lim) and  x[3],offset)
    
    report_short('Hasnt started',Instrt)
    report_short('Hasnt finished',Instp)
    report_short('Nan',Inan)
    report_short('Needs eval',Ieval0_expand)
    report_short('Done eval',Ieval1)
    report_short('Late eval job',Ilate_eval)
    report_short('Needs grad-probe',Igrad0)
    report_short('Done rad-probe',Igrad1)


# In[ ]:




