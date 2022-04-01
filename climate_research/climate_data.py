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
import time
import matplotlib.pyplot as plt
from scipy import ndimage

def ismember_(a,b):
    for a_ in b:
        if a==a_:
            return True
    return False

def physical_domains(domain_sel_id,validation=True):
    partition={}
    partition['train']={}
    print('domain id: '+str(domain_sel_id))
    if domain_sel_id==0:
        partition['train']['xmin']=[-50,-180,-110,-48]
        partition['train']['xmax']=[-20,-162,-92,-30]
        partition['train']['ymin']=[35,-40,-20,0]
        partition['train']['ymax']=[50,-25,-5,15]
    elif domain_sel_id==1:
        partition['train']['xmin']=[-47]#,-150,-117]
        partition['train']['xmax']=[-38.7]#,-130,-97]
        partition['train']['ymin']=[43]#,-50,-20]
        partition['train']['ymax']=[48.7]#,-35,-5]  
    elif domain_sel_id==2:
        partition['train']['xmin']=[-150]
        partition['train']['xmax']=[-130]
        partition['train']['ymin']=[-50]
        partition['train']['ymax']=[-35]
    elif domain_sel_id==3:
        partition['train']={
        'xmin': [-1e3],'xmax':[1e3],\
        'ymin': [-1e3],'ymax':[1e3],\
        'tmin': 0,'tmax':1}
        '''u=np.load('/scratch/cg3306/climate/physical_land_11.npy')
        x=np.load('/scratch/cg3306/climate/physical_x.npy')
        y=np.load('/scratch/cg3306/climate/physical_y.npy')
        nx,ny=len(x),len(y)
        spread=10
        domlenx=15
        domleny=5
        yy=torch.linspace(0,ny,ny//domleny).to(torch.int).numpy().tolist()
        xx=torch.linspace(0,nx,nx//domlenx).to(torch.int).numpy().tolist()
        partition['train']['xmin']=[]
        partition['train']['xmax']=[]
        partition['train']['ymin']=[]
        partition['train']['ymax']=[]
        
        m=1
        landmass=np.zeros((len(yy)-1+2*m,len(xx)-1+2*m))
        
        for j in range(len(xx)-1):
            for i in range(len(yy)-1):
                j0=np.maximum(xx[j]-spread,0)
                j1=np.minimum(xx[j+1]+spread,nx-1)
                i0=np.maximum(yy[i]-spread,0)
                i1=np.minimum(yy[i+1]+spread,ny-1)
                landmass[i+m,j+m]=np.sum(u[i0:i1+1,j0:j1+1])>0
        
        for j in range(len(xx)-1):
            for i in range(len(yy)-1):
                j0=np.maximum(xx[j]-spread,0)
                j1=np.minimum(xx[j+1]+spread,nx-1)
                i0=np.maximum(yy[i]-spread,0)
                i1=np.minimum(yy[i+1]+spread,ny-1)
                #landmassflag=np.sum(u[i0:i1+1,j0:j1+1])
                #totsize=(i1-i0)*(j1-j0)
                landmassflag=np.sum(landmass[i:i+2*m,j:j+2*m])
                if landmassflag==0:#landmass/totsize<0.1:
                    partition['train']['xmin'].append(x[j0])
                    partition['train']['xmax'].append(x[j1])
                    partition['train']['ymin'].append(y[i0])
                    partition['train']['ymax'].append(y[i1])'''
    elif domain_sel_id==4:
        u=np.load('/scratch/cg3306/climate/physical_land_11.npy')
        x=np.load('/scratch/cg3306/climate/physical_x.npy')
        y=np.load('/scratch/cg3306/climate/physical_y.npy')
        nx,ny=len(x),len(y)
        spread=10
        domlen=20
        yy=torch.linspace(0,ny,ny//domlen).to(torch.int).numpy().tolist()
        xx=torch.linspace(0,nx,nx//domlen).to(torch.int).numpy().tolist()
        partition['train']['xmin']=[]
        partition['train']['xmax']=[]
        partition['train']['ymin']=[]
        partition['train']['ymax']=[]
        
        m=1
        landmass=np.zeros((len(yy)-1+2*m,len(xx)-1+2*m))
        
        for j in range(len(xx)-1):
            for i in range(len(yy)-1):
                j0=np.maximum(xx[j]-spread,0)
                j1=np.minimum(xx[j+1]+spread,nx-1)
                i0=np.maximum(yy[i]-spread,0)
                i1=np.minimum(yy[i+1]+spread,ny-1)
                landmass[i+m,j+m]=np.sum(u[i0:i1+1,j0:j1+1])>0
        
        for j in range(len(xx)-1):
            for i in range(len(yy)-1):
                j0=np.maximum(xx[j]-spread,0)
                j1=np.minimum(xx[j+1]+spread,nx-1)
                i0=np.maximum(yy[i]-spread,0)
                i1=np.minimum(yy[i+1]+spread,ny-1)
                #landmassflag=np.sum(u[i0:i1+1,j0:j1+1])
                #totsize=(i1-i0)*(j1-j0)
                landmassflag=np.sum(landmass[i:i+2*m,j:j+2*m])
                if landmassflag==0 and y[i0]>-25 and y[i1]<25:
                    partition['train']['xmin'].append(x[j0])
                    partition['train']['xmax'].append(x[j1])
                    partition['train']['ymin'].append(y[i0])
                    partition['train']['ymax'].append(y[i1])
        
                
    partition['train']['tmin']=0.0
    partition['train']['tmax']=0.7
    if validation:
        partition['validation']=copy.deepcopy(partition['train'])
        partition['validation']['tmin']=0.75
        partition['validation']['tmax']=0.8
    else:
        partition['train']['tmax']=0.8
    partition['test']=copy.deepcopy(partition['train'])
    partition['test']['tmin']=0.85
    partition['test']['tmax']=1
    
    partition['earth']={
        'xmin': [-1e3],'xmax':[1e3],\
        'ymin': [-1e3],'ymax':[1e3],\
        'tmin': 0.85,'tmax':1}
    return partition

def load_ds_zarr(args):
    ds_zarr=xr.open_zarr(args.data_address)
    tot_time=ds_zarr.time.shape[0]
    
    
    if args.testrun>0:
        sub_tot_time=args.testrun
        rng = np.random.default_rng(0)
        indices=np.sort(rng.choice(tot_time,size=sub_tot_time,replace=False))
        ds_zarr=ds_zarr.sel(time=ds_zarr.time[indices].data)
    '''usc=0
    uss=0
    M=20
    give_sc=lambda AA: np.mean(np.abs(AA[AA==AA]))
    for j in range(M):
        dsi=ds_zarr.isel(time=np.arange(j,j+1))
        u,v,S_x,S_y=dsi.usurf.values[0],dsi.vsurf.values[0],dsi.S_x.values[0],dsi.S_y.values[0]
        u=np.stack([u,v],axis=0)
        S=np.stack([S_x,S_y],axis=0)
        usc+=give_sc(u)
        uss+=give_sc(S)
    usc=usc/M
    uss=uss/M
    
    ds_zarr['usurf']=ds_zarr['usurf']/usc
    ds_zarr['vsurf']=ds_zarr['vsurf']/usc
    ds_zarr['S_x']=ds_zarr['S_x']/uss
    ds_zarr['S_y']=ds_zarr['S_y']/uss
    '''
    return ds_zarr

def get_land_masks(val_gen):
    if isinstance(val_gen.dataset,GlobalDataset):
        masks=val_gen.dataset.get_landmask()
        return 1-masks
    val_gen.dataset.no_more_mask_flag=False
    ii=0
    for local_batch,local_masks,local_labels in val_gen:
        if ii==0:
            bsize=local_batch.shape[0]
            masks=torch.zeros((val_gen.dataset.num_domains//bsize + 1)*bsize,local_masks.shape[1],local_masks.shape[2],local_masks.shape[3])
        masks[ii:ii+local_masks.shape[0]]=local_masks
        ii+=local_masks.shape[0]
        if ii>=val_gen.dataset.num_domains:
            break
    masks=masks[0:val_gen.dataset.num_domains]
    val_gen.dataset.no_more_mask_flag=True
    return masks


def zigzag_freq(n,m,f0,df,d=1,reps=100):
    x=np.linspace(0,m,n)
    for _ in range(m):
        x=np.abs(1-np.abs(1-x))
    x=x**d*df+f0
    x=np.cumsum(x)
    x=x/x[-1]*2*np.pi*reps
    return x

def sigmoid_freq(n,f0,df,d=20,reps=100):
    x=np.linspace(-1,1,n)
    x=1/(1+np.exp(-x*d))
    x=x*df+f0
    x=np.cumsum(x)
    x=x/x[-1]*2*np.pi*reps
    return x
def geographic_features(y,x):
    lat1=zigzag_freq(len(y),2,(30*645)//len(y),40,d=1,reps=55)
    lat2=sigmoid_freq(len(y), (30*645)//len(y),30,d=15,reps=55)
    lng1=zigzag_freq(len(x),2,(30*645)//len(y),50,d=1,reps=70)
    lng2=zigzag_freq(len(x),4,(30*645)//len(y),50,d=1,reps=70)
    return lat1, lat2, lng1, lng2



def hat_freq(n,span):
    p0=1/2
    p1=4
    m=2
    x=np.linspace(0,m,n)
    for _ in range(m):
        x=np.abs(1-np.abs(1-x))
    Pmin=span*p0
    Pmax=span*p1

    Fmin=1/Pmax
    Fmax=1/Pmin
    dF=(Fmax-Fmin)
    x=x*dF+Fmin
    x=np.cumsum(x)
    return x*2*np.pi

def sigmoid_freq(n,span):
    p0=1/2
    p1=4
    m=2
    d=20
    
    x=np.linspace(-1,1,n)
    x=1/(1+np.exp(-x*d))
    
    Pmin=span*p0
    Pmax=span*p1

    Fmin=1/Pmax
    Fmax=1/Pmin
    dF=(Fmax-Fmin)
    
    x=x*dF+Fmin
    x=np.cumsum(x)
    return x*2*np.pi
def geographic_features2(n,span):
    lat1=hat_freq(n,span)
    lat2=sigmoid_freq(n,span)
    lng1=lat1
    lng2=lat2
    return lat1, lat2, lng1, lng2


# In[2]:


class Dataset1(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ds_data,domains,net,cropped=False,subtime=1,datatype=0,roundearth=False,readfile2=False):
        #print('hereeeee')
        self.spread=net.spread
        self.latsig=net.latsig
        self.direct_coord=net.direct_coord
        timeshuffle=net.timeshuffle
        self.longitude=net.longitude
        self.latsign=net.latsign
        self.scale_u=net.rescale[0]
        self.scale_S=net.rescale[1]
        self.readfile2=readfile2
        self.ds_data = ds_data
        self.domains=domains
        self.cropped=cropped
        self.datatype=datatype
        
        self.num_domains=len(self.domains['xmin'])
        tot_time=self.ds_data.isel(xu_ocean=[0],yu_ocean=[0]).time.shape[0]
        self.time_st=np.int64(np.floor(tot_time*self.domains['tmin']))
        self.time_tr=np.int64(np.ceil(tot_time*self.domains['tmax']))
        self.tot_time=self.time_tr-self.time_st
        self.num_time=np.int64(np.ceil(self.tot_time*subtime))
   
        self.filename='/scratch/cg3306/climate/physical_land_'+str(len(self.ds_data.xu_ocean.values))+'.npy'

        dimens=[]
        '''
        self.time_per_domain=[]
        for nd in range(self.num_domains):
            self.time_per_domain.append(\
                        np.sort(np.random.choice(self.tot_time,self.num_time))+self.time_st)
        
        if not timeshuffle:
            for i in range(1,nd):
                self.time_per_domain[i]=self.time_per_domain[0]'''
            
        for nd in range(self.num_domains):
            datsel=self.ds_data.sel(time=self.ds_data.time[0].data,                                    xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]),                                    yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))
            dimens.append([datsel.yu_ocean.shape[0],datsel.xu_ocean.shape[0]])
        dimens=torch.tensor(dimens)
        self.dimens=torch.amax(dimens,dim=0)
        coords=[]
        icoords=[]
        geo=[]
        
        y=self.ds_data.yu_ocean.values
        x=self.ds_data.xu_ocean.values
        
        if not self.direct_coord:
            lat1, lat2, lng1, lng2 = geographic_features(y,x)
        else:
            lat=y/y[-1]
            lng=x/x[-1]
        for nd in range(self.num_domains):
            datsel=self.ds_data.sel(time=self.ds_data.time[0].data,                                    xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]),                                    yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))
            
            xx=datsel.xu_ocean.values
            yy=datsel.yu_ocean.values
            
            i0=np.argmin(np.abs(yy[0]-y))
            i1=np.argmin(np.abs(yy[-1]-y))+1
            
            j0=np.argmin(np.abs(xx[0]-x))
            j1=np.argmin(np.abs(xx[-1]-x))+1
            
            if not self.direct_coord:
                locgeo = lat1[i0:i1], lat2[i0:i1], lng1[j0:j1], lng2[j0:j1]
            else:
                locgeo = lat[i0:i1], lng[j0:j1]
            locgeo = [torch.tensor(ll,dtype=torch.float32) for ll in locgeo]
            geo.append(locgeo)
            
            coords.append([yy,xx])
            icoords.append([i0,i1,j0,j1])
        
        self.coords=coords
        self.geo=geo
        self.icoords=icoords
        self.no_more_mask_flag=True
        
        self.pooler=nn.MaxPool2d(2*self.spread+1,stride=1)
        
        y,x=self.coords[0]
        dy=y[1:]-y[:-1]
        dx=x[1:]-x[:-1]

        mdy=np.mean(dy)
        mdx=np.mean(dx)

        self.box_km=[mdy*111,mdx*85]
        
    def __len__(self):
        'Denotes the total number of samples'
        return self.num_domains*self.num_time
    
    def __getitem__(self, index):
        #return torch.randn(1,10,10),torch.randn(1,10,10),torch.randn(1,10,10)
        'Generates one sample of data'
        spread=self.spread
        usc=self.scale_u
        ssc=self.scale_S
        
        nd=index%self.num_domains
        nt=np.int64(np.floor(index/self.num_domains))+self.time_st
        #nt=self.time_per_domain[nd][nt]
        datsel=self.ds_data.isel(time=[nt]).                                sel(xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]),                                yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))

        usurf=datsel.usurf.values
        vsurf=datsel.vsurf.values

        usurf=torch.tensor(usurf,dtype=torch.float32)
        vsurf=torch.tensor(vsurf,dtype=torch.float32)

        #if resc:
        usurf=usurf/usc
        vsurf=vsurf/usc
        
        
        if not (self.latsig or self.latsign or self.direct_coord):
            X=torch.cat([usurf,vsurf],dim=0)
        else:
            locgeo=self.geo[nd]
            xx=torch.ones(usurf.shape)
            if not self.direct_coord:
                wt=0
                latcod=torch.cos(locgeo[0]+nt*wt*2*np.pi)
                lat1=latcod.view(1,-1,1)*xx
                
                latcod=torch.cos(locgeo[1]+nt*wt*2*np.pi)
                lat2=latcod.view(1,-1,1)*xx
                
                if self.longitude:
                    latcod=torch.cos(locgeo[2]+nt*wt*2*np.pi)
                    lng1=xx*latcod.view(1,1,-1)
                    latcod=torch.cos(locgeo[3]+nt*wt*2*np.pi)
                    lng2=xx*latcod.view(1,1,-1)
                    X=torch.cat([usurf,vsurf,lat1,lat2,lng1,lng2],dim=0)
                else:
                    if self.latsign:
                        X=torch.cat([usurf,vsurf,lat1,lat2],dim=0)
                    else:
                        X=torch.cat([usurf,vsurf,lat1],dim=0)
            else:
                '''latcod=locgeo[0]
                lat=latcod.view(1,-1,1)*xx
                latcod=locgeo[1]
                lng=latcod.view(1,1,-1)*xx'''
                x=datsel.xu_ocean.values
                y=datsel.yu_ocean.values
                dy=np.zeros((len(y),))
                dy[1:]=y[1:]-y[:-1]
                dy[:-1]=dy[:-1]+dy[1:]
                dy[1:-1]=dy[1:-1]/2
                dy=(dy-np.mean(dy))/np.std(dy)
                cosx=np.cos(x/180*np.pi)
                siny=np.sin(y/80*np.pi)

                cosx=torch.reshape(torch.tensor(cosx,dtype=torch.float32),[1,1,-1])
                siny=torch.reshape(torch.tensor(siny,dtype=torch.float32),[1,-1,1])
                dy=torch.reshape(torch.tensor(dy,dtype=torch.float32),[1,-1,1])
                fcosx=cosx+siny*0
                fsiny=cosx*0+siny
                fdy=cosx*0+dy
                if self.longitude:
                    X=torch.cat([usurf,vsurf,fcosx,fsiny,fdy],dim=0)
                else:
                    X=torch.cat([usurf,vsurf,fsiny,fdy],dim=0)
        
        sx=datsel.S_x[0,spread:-spread,spread:-spread].values
        sy=datsel.S_y[0,spread:-spread,spread:-spread].values
        sx=torch.tensor(sx,dtype=torch.float32)
        sy=torch.tensor(sy,dtype=torch.float32)


        sx=sx/self.scale_S
        sy=sy/self.scale_S

        Y=torch.stack([sx,sy],dim=0)
        
            
        if not self.no_more_mask_flag:
            if self.readfile2:
                u=np.load(self.filename)
            else:
                u=np.load('/scratch/cg3306/climate/physical_land_11.npy')
            i0,i1,j0,j1=self.icoords[nd]
            M=torch.tensor(u[i0:i1,j0:j1],dtype=torch.float32)
            #M=torch.zeros(X.shape[0],X.shape[1],X.shape[2])
            #M[X!=X]=1
            #M=torch.stack([M],dim=1)
            M=torch.stack([M],dim=0)
            M=torch.stack([M],dim=0)
            M=self.pooler(M)
            M=1-M[0]
            
        
        Y[Y!=Y]=0
        X[X!=X]=0
        
        '''X[0:2]=X[0:2]*10.
        Y=Y*1.e7'''
        
        if not self.cropped:
            p3d = (0, self.dimens[1]-2*spread-Y.shape[2], 0, self.dimens[0]-2*spread-Y.shape[1]) 
            Y = F.pad(Y, p3d, "constant", 0)
            
            if not self.no_more_mask_flag:
                p3d = (0, self.dimens[1]-2*spread-M.shape[2], 0, self.dimens[0]-2*spread-M.shape[1]) 
                M = F.pad(M, p3d, "constant", 0)
            
            p3d = (0, self.dimens[1]-X.shape[2], 0, self.dimens[0]-X.shape[1]) 
            X = F.pad(X, p3d, "constant", 0)
        if not self.no_more_mask_flag:
            return X,M,Y
        else:
            return X,torch.tensor(nd),Y


# In[3]:


def reset():
    data_info={}
    lookupfile='/scratch/cg3306/climate/climate_research/model_data_info.json'
    with open(lookupfile,'w') as infile:
        json.dump(data_info,infile)
        
def safe():
    lookupfile='/scratch/cg3306/climate/climate_research/model_data_info.json'
    with open(lookupfile) as infile:
        data_info=json.load(infile)
    
    lookupfile='/scratch/cg3306/climate/climate_research/model_data_info_2.json'
    with open(lookupfile,'w') as infile:
        json.dump(data_info,infile)
def recover():
    lookupfile='/scratch/cg3306/climate/climate_research/model_data_info_2.json'
    random_wait()
    with open(lookupfile) as infile:
        data_info=json.load(infile)
    
    lookupfile='/scratch/cg3306/climate/climate_research/model_data_info.json'
    random_wait()
    with open(lookupfile,'w') as infile:
        json.dump(data_info,infile)
        
def get_dict(model_bank_id,model_id,parallel=True):
    expand_dict(model_bank_id,model_id)
    lookupfile='/scratch/cg3306/climate/climate_research/model_data_info.json'
    random_wait(parallel=parallel)
    with open(lookupfile) as infile:
        data_info_=json.load(infile)
    return data_info_[str(model_bank_id)][str(model_id)]
def expand_dict(model_bank_id,model_id,parallel=True):
    lookupfile='/scratch/cg3306/climate/climate_research/model_data_info.json'
    random_wait(parallel=parallel)
    with open(lookupfile) as infile:
        data_info_=json.load(infile)
    iskey1=False
    iskey2=False
    if ismember_(str(model_bank_id),list(data_info_.keys()),):
        iskey1=True
        if ismember_(str(model_id),list(data_info_[str(model_bank_id)].keys())):
            iskey2=True
            
    #print('expansion report: '+str(iskey1)+'  '+str(iskey2))
    if not iskey1:
        data_info_[str(model_bank_id)]={}
    if not iskey2:
        data_info_[str(model_bank_id)][str(model_id)]={}
    random_wait(parallel=parallel)
    with open(lookupfile,'w') as infile:
        json.dump(data_info_,infile)
def update_model_info(data_info,model_bank_id,model_id,parallel=True):
    expand_dict(model_bank_id,model_id)
    lookupfile='/scratch/cg3306/climate/climate_research/model_data_info.json'
    random_wait(parallel=parallel)
    with open(lookupfile) as infile:
        data_info_=json.load(infile)
    data_info_[str(model_bank_id)][str(model_id)]=data_info.copy()
    random_wait(parallel=parallel)
    with open(lookupfile,'w') as infile:
        json.dump(data_info_,infile)


# In[4]:


def random_wait(secs=2,parallel=True):
    tt=np.random.rand()*5
    time.sleep(tt)


# In[2]:


class Dataset2(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self,ds_data,domains,model_id,model_bank_id,net,                 subtime=1,heteroscrescale=1,parallel=True,depthind=2):
        lookupfile='/scratch/cg3306/climate/climate_research/model_data_info.json'
        self.lookupfile=lookupfile
        random_wait(parallel=parallel)
        with open(lookupfile) as infile:
            data_info_=json.load(infile)
        data_info_=self.expand_(data_info_,model_bank_id,model_id).copy()
        data_info=data_info_[str(model_bank_id)][str(model_id)].copy()
        self.depthind=depthind
        depthvals=[]
        if 'st_ocean' in list(ds_data.coords.keys()):
#             depth values are
#             [   5.03355 ,   55.853249,  110.096153,  181.312454,  330.007751,
#                       1497.56189 , 3508.633057]
            depthvals=ds_data.coords['st_ocean'].values
        if hasattr(net, 'skipcons'):
            self.padded=net.skipcons
        else:
            self.padded=False
        self.depthvals=depthvals
        self.spread=net.spread
        data_info['spread']=int(net.spread)
        
        self.freq_coord=net.freq_coord
        data_info['freq_coord']=self.freq_coord
        
        self.direct_coord=net.direct_coord
        data_info['direct_coord']=self.direct_coord
        
        self.heteroscrescale=heteroscrescale
        
        self.longitude=net.longitude
        self.latsign=net.latsign
        self.latsig=net.latsig
        
        self.ds_data = ds_data.sel(yu_ocean=slice(-85, 85))
        self.domains=domains
        
            
        if ismember_('inscales',list(data_info.keys())):
            self.inscales=data_info['inscales']
            self.outscales=data_info['outscales']
        else:
            self.inscales=[]
            self.outscales=[]

        self.inputs=data_info['inputs']
        self.outputs=data_info['outputs']
        self.parallel=parallel
        self.num_domains=len(self.domains['xmin'])
        tot_time=self.ds_data.isel(xu_ocean=[0],yu_ocean=[0]).time.shape[0]
        self.time_st=np.int64(np.floor(tot_time*self.domains['tmin']))
        self.time_tr=np.int64(np.ceil(tot_time*self.domains['tmax']))
        self.tot_time=self.time_tr-self.time_st
        self.num_time=np.int64(np.ceil(self.tot_time*subtime))
        
        self.model_bank_id=model_bank_id
        self.model_id=model_id
        self.filename=data_info['maskloc']
        
        dimens=[]
        for nd in range(self.num_domains):
            datsel=self.ds_data.sel(time=self.ds_data.time[0].data,                                    xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]),                                    yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))
            dimens.append([datsel.yu_ocean.shape[0],datsel.xu_ocean.shape[0]])
        dimens=torch.tensor(dimens)
        self.dimens=torch.amax(dimens,dim=0)
        self.glbl_data=self.ds_data.yu_ocean.shape[0]*self.ds_data.xu_ocean.shape[0]==self.dimens[0]*self.dimens[1]
        if self.glbl_data:
            self.dimens[1]+=self.spread*2
        coords=[]
        icoords=[]
        geo=[]
        
        y=self.ds_data.yu_ocean.values
        x=self.ds_data.xu_ocean.values
        
        if not self.direct_coord:
            #lat1, lat2, lng1, lng2 = geographic_features(y,x)
            lat1, lat2, lng1, lng2 = geographic_features2(len(y),net.spread*2+1)
        else:
            lat=y/y[-1]
            lng=x/x[-1]
        for nd in range(self.num_domains):
            datsel=self.ds_data.sel(time=self.ds_data.time[0].data,                                    xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]),                                    yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))
            
            xx=datsel.xu_ocean.values
            yy=datsel.yu_ocean.values
            
            i0=np.argmin(np.abs(yy[0]-y))
            i1=np.argmin(np.abs(yy[-1]-y))+1
            
            j0=np.argmin(np.abs(xx[0]-x))
            j1=np.argmin(np.abs(xx[-1]-x))+1
            
            if not self.direct_coord:
                locgeo = lat1[i0:i1], lat2[i0:i1], lng1[j0:j1], lng2[j0:j1]
            else:
                locgeo = lat[i0:i1], lng[j0:j1]
            locgeo = [torch.tensor(ll,dtype=torch.float32) for ll in locgeo]
            geo.append(locgeo)
            
            coords.append([yy,xx])
            icoords.append([i0,i1,j0,j1])
        
        self.coords=coords
        self.geo=geo
        self.icoords=icoords
        self.no_more_mask_flag=True
        
        if not net.gan:
            self.mask_spread=net.spread
        else:
            self.mask_spread=(net.receptive_field-1)//2
        
        self.pooler=nn.MaxPool2d(2*self.mask_spread+1,stride=1)
        
        y,x=self.coords[0]
        dy=y[1:]-y[:-1]
        dx=x[1:]-x[:-1]

        mdy=np.mean(dy)
        mdx=np.mean(dx)

        self.box_km=[mdy*111,mdx*85]
        
        #self.save_info_file(data_info)
        self.data_info=data_info.copy()
    def expand_(self,data_info_,model_bank_id,model_id):
        sufficient_len=False
        if ismember_(str(model_bank_id),list(data_info_.keys())):
            if ismember_(str(model_id),list(data_info_[str(model_bank_id)].keys())):
                sufficient_len=True
        if not sufficient_len:
            #print('expand_ resets')
            data_info_[str(model_bank_id)][str(model_bank)]=data_info_[str(model_bank_id)]['0'].copy()
        return data_info_
    def __len__(self):
        'Denotes the total number of samples'
        return self.num_domains*self.num_time
    def get_info_file(self):
        with open(self.lookupfile) as infile:
            data_info=json.load(infile)
        return data_info[str(self.model_bank_id)][str(self.model_id)].copy()
    def save_info_file(self,data_info):
        random_wait(parallel=self.parallel)
        with open(self.lookupfile) as infile:
            data_info_=json.load(infile)
        data_info_[str(self.model_bank_id)][str(self.model_id)]=data_info.copy()
        with open(self.lookupfile,'w') as infile:
            json.dump(data_info_,infile)
    def save_masks(self,tag=''):
        Ms=[[] for nd in range(len(self.icoords))]
        for nd in range(len(self.icoords)):
            if self.glbl_data:
                X,Y=self.input_output(nd,scale=False,crop=True,periodic_lon_expand=True)
            else:
                X,Y=self.input_output(nd,scale=False,crop=True)
            #X,_=self.input_output(nd)
            u=X[0]
            u[u==0]=np.nan
            u[u==u]=0
            u[u!=u]=1
            M=u.clone().detach().type(torch.float)
            
            M=torch.stack([M],dim=0)
            
            M=torch.stack([M],dim=0)
            
            M=self.pooler(M)
            M=1-M[0]
            if self.padded:
                shp=list(M.numpy().shape)
                shp[1]+=2*self.spread
                shp[2]+=2*self.spread
                Z=torch.zeros(shp)
                Z[:,self.spread:-self.spread,self.spread:-self.spread]=M
                Ms[nd]=self.pad_with_zero(Z,0)
            else:
                Ms[nd]=self.pad_with_zero(M,self.mask_spread)
        
        Ms=torch.stack(Ms).detach().numpy()
        data_info=self.get_info_file()
        filename=data_info['maskloc'+tag]
        with open(filename, 'wb') as f:
            np.save(f, Ms)
    def get_masks(self,glbl=False,padded=False):
        data_info=self.get_info_file()
        model_bank_id=self.model_bank_id
        model_id=self.model_id
        filename=data_info['maskloc']
        if glbl:
            filename=filename.replace('dom4','glbl')
        if padded:
            if not 'padded' in filename:
                filename=filename.replace('.npy','-padded.npy')
        M=np.load(filename)
        M=torch.tensor(M,dtype=torch.float32)
        M.requires_grad=False
        return M
    def get_mask_address(self,):
        data_info=self.get_info_file()
        model_bank_id=self.model_bank_id
        model_id=self.model_id
        filename=data_info['maskloc']
        return filename
    def compute_scales(self):
        data_info=self.get_info_file()
        Ms=[]
        innum=len(self.inputs)
        outnum=len(self.outputs)
        inscales=torch.zeros(innum)
        outscales=torch.zeros(outnum)
        M=50
        for nd in range(M):
            if self.glbl_data:
                X,Y=self.input_output(nd,scale=False,crop=True,periodic_lon_expand=True)
            else:
                X,Y=self.input_output(nd,scale=False,crop=True)
            u=X[0]
            y=Y[0]
            u[u==0]=np.nan
            y[y==0]=np.nan
            inscales+=torch.mean(torch.abs(X[:innum,u==u]),dim=(1))/M
            outscales+=torch.mean(torch.abs(Y[:outnum,y==y]),dim=(1))/M
        inscales,outscales=inscales.numpy(),outscales.numpy()
        return inscales,outscales
    def save_scales(self):
        data_info=self.get_info_file()
        Ms=[]
        innum=len(self.inputs)
        outnum=len(self.outputs)
        inscales=torch.zeros(innum)
        outscales=torch.zeros(outnum)
        M=50
        for nd in range(M):
            if self.glbl_data:
                X,Y=self.input_output(nd,scale=False,crop=True,periodic_lon_expand=True)
            else:
                X,Y=self.input_output(nd,scale=False,crop=True)
            #X,Y=self.input_output(nd,periodic_lon_expand=True)
            u=X[0]
            y=Y[0]
            u[u==0]=np.nan
            y[y==0]=np.nan
            inscales+=torch.mean(torch.abs(X[:innum,u==u]),dim=(1))/M
            outscales+=torch.mean(torch.abs(Y[:outnum,y==y]),dim=(1))/M
        inscales,outscales=inscales.numpy(),outscales.numpy()
        
        model_bank_id=self.model_bank_id
        model_id=self.model_id
        data_info['inscales']=inscales.tolist()
        data_info['outscales']=outscales.tolist()
        self.inscales=inscales
        self.outscales=outscales
        self.save_info_file(data_info)
    def domain_index(self,index):
        return index%self.num_domains
    def input_output(self,index,scale=True,crop=True,periodic_lon_expand=False):
        'Generates one sample of data'
        spread=self.spread
        nd=index%self.num_domains
        nt=np.int64(np.floor(index/self.num_domains))+self.time_st
        #nt=self.time_per_domain[nd][nt]
        datsel=self.ds_data
        if 'st_ocean' in list(self.ds_data.coords.keys()):
            datsel=datsel.isel(st_ocean=self.depthind)
        datsel=datsel.isel(time=[nt]).                                sel(xu_ocean=slice(self.domains['xmin'][nd], self.domains['xmax'][nd]),                                yu_ocean=slice(self.domains['ymin'][nd], self.domains['ymax'][nd]))
        if 'st_ocean' in self.data_info:
            datsel=datsel.isel(st_ocean=self.data_info['st_ocean']).sum(dim='st_ocean')
        
        X=[datsel[self.inputs[i]].values for i in range(len(self.inputs))]
        if self.spread>0 and crop and not self.padded:
            Y=[datsel[self.outputs[i]].values                       [:,self.spread:-self.spread,:]                       for i in range(len(self.outputs))]
            if not periodic_lon_expand:
                Y=[Y[i][:,:,self.spread:-self.spread]                       for i in range(len(self.outputs))]
        else:
            Y=[datsel[self.outputs[i]].values                          for i in range(len(self.outputs))]
        
        if scale:
            X=[X[i]/self.inscales[i] for i in range(len(self.inputs))]
            Y=[Y[i]/self.outscales[i]/self.heteroscrescale for i in range(len(self.outputs))]
        
        usurf=datsel.usurf.values
        vsurf=datsel.vsurf.values
        X=[torch.tensor(x,dtype=torch.float32) for x in X]
        Y=[torch.tensor(y,dtype=torch.float32) for y in Y]
        
        X=torch.cat(X,dim=0)
        Y=torch.cat(Y,dim=0)
        if self.freq_coord or self.direct_coord:
            locgeo=self.geo[nd]
            xx=torch.ones(X[0].shape)
            if not self.direct_coord:
                wt=0
                latcod=torch.cos(locgeo[0]+nt*wt*2*np.pi)
                lat1=latcod.view(1,-1,1)*xx
                
                latcod=torch.cos(locgeo[1]+nt*wt*2*np.pi)
                lat2=latcod.view(1,-1,1)*xx
                '''latcod=torch.cos(locgeo[2]+nt*wt*2*np.pi)
                lng1=xx*latcod.view(1,1,-1)
                latcod=torch.cos(locgeo[3]+nt*wt*2*np.pi)
                lng2=xx*latcod.view(1,1,-1)'''
                if self.latsign:
                    X=torch.cat([X,lat1],dim=0)
                if self.latsig:
                    X=torch.cat([X,lat2],dim=0)
                if self.longitude:
                    X=torch.cat([X,lng1,lng2],dim=0)
               
            else:
                x=datsel.xu_ocean.values
                y=datsel.yu_ocean.values
                dy=np.zeros((len(y),))
                dy[1:]=y[1:]-y[:-1]
                dy[:-1]=dy[:-1]+dy[1:]
                dy[1:-1]=dy[1:-1]/2
                dy=(dy-np.mean(dy))/np.std(dy)
                cosx=np.cos(x/180*np.pi)
                siny=np.sin(y/80*np.pi)

                cosx=torch.reshape(torch.tensor(cosx,dtype=torch.float32),[1,1,-1])
                siny=torch.reshape(torch.tensor(siny,dtype=torch.float32),[1,-1,1])
                dy=torch.reshape(torch.tensor(dy,dtype=torch.float32),[1,-1,1])
                fcosx=cosx+siny*0
                fsiny=cosx*0+siny
                fdy=cosx*0+dy
                if self.latsign:
                    X=torch.cat([X,fsiny],dim=0)
                if self.latsig:
                    X=torch.cat([X,fdy],dim=0)
                if self.longitude:
                    X=torch.cat([X,fcosx],dim=0)
        if self.spread>0 and periodic_lon_expand:
            X=torch.cat([X[:,:,-self.spread:],X,X[:,:,:self.spread]], axis=2)
            if self.padded:
                Y=torch.cat([Y[:,:,-self.spread:],Y,Y[:,:,:self.spread]], axis=2)
        return X,Y
    def __getitem__(self, index):      
        if self.glbl_data:
            X,Y=self.input_output(index,crop=True,periodic_lon_expand=True)
        else:
            X,Y=self.input_output(index,crop=True)
        Y[Y!=Y]=0
        X[X!=X]=0
        if not self.padded:
            X,Y=self.pad_with_zero(X,0),self.pad_with_zero(Y,self.spread)
        else:
            X,Y=self.pad_with_zero(X,0),self.pad_with_zero(Y,0)
        return X,torch.tensor(self.domain_index(index)),Y
    def pad_with_zero(self,Y,spread,padding_val=0,centered=False):
        #p3d = (0, self.dimens[1]-2*spread-Y.shape[2], 0, self.dimens[0]-2*spread-Y.shape[1]) 
        if not centered:
            p3d = (0, self.dimens[1]-2*spread-Y.shape[2], 0, self.dimens[0]-2*spread-Y.shape[1]) 
        else:
            d1=self.dimens[1]-2*spread-Y.shape[2]
            d2=self.dimens[0]-2*spread-Y.shape[1]
            p3d = (d1//2,d1-d1//2 , d2//2, d2-d2//2) 
        Y = F.pad(Y, p3d, "constant", padding_val)
        return Y


# In[4]:


def subeddy_forcing(ds_zarr,sigma,t,informative=False):
    x=ds_zarr.xu_ocean.values
    y=ds_zarr.yu_ocean.values
    x_=x[1:-1]
    y_=y[1:-1]
    x_=x_[::sigma]
    y_=y_[::sigma]
    if informative:
        return x_,y_
    dx=x[2:]-x[:-2]
    dy=y[2:]-y[:-2]
    dy=np.reshape(dy,[-1,1])
    dx=np.reshape(dx,[1,-1])
    dxy=dx*dy
    #x=np.reshape(x,[1,-1])
    #y=np.reshape(y,[-1,1])

    u=ds_zarr.isel(time=np.arange(t,t+1)).usurf.values[0]
    v=ds_zarr.isel(time=np.arange(t,t+1)).vsurf.values[0]

    a=np.zeros((4*sigma+1,4*sigma+1))
    for i in range(-2*sigma,2*sigma+1):
        for j in range(-2*sigma,2*sigma+1):
            a[i+sigma,j+sigma]=np.exp( - (i**2+j**2)/(sigma/2)**2/2)
    a=a/np.sum(a)

    A=np.stack([a],axis=0)

    adxy=ndimage.convolve(dxy, a, mode='mirror')

    adxy=np.stack([adxy],axis=0)
    dudy=(u[2:,1:-1]-u[:-2,1:-1])/dy
    dudx=(u[1:-1,2:]-u[1:-1,2:])/dx
    dvdy=(v[2:,1:-1]-v[:-2,1:-1])/dy
    dvdx=(v[1:-1,2:]-v[1:-1,2:])/dx
    u,v=u[1:-1,1:-1],v[1:-1,1:-1]
    U=np.stack((u,v,dudx,dudy,dvdx,dvdy),axis=0)

    S=np.stack([U[0]*U[2]+U[1]*U[3],U[0]*U[4]+U[1]*U[5]],axis=0)
    U=ndimage.convolve(U, A, mode='mirror')/adxy
    S=ndimage.convolve(S, A, mode='mirror')/adxy
    S=S[:,::sigma,::sigma]
    U=U[:,::sigma,::sigma]
    S=S-np.stack([U[0]*U[2]+U[1]*U[3],U[0]*U[4]+U[1]*U[5]],axis=0)
    U=U[:2]
    return U,S

'''
def geographic_features2(y,x):
    lat1=zigzag_freq(len(y),2,30,40,d=1,reps=np.int(len(y)/650*55))
    lat2=sigmoid_freq(len(y),30,30,d=15,reps=np.int(len(y)/650*55))
    lng1=zigzag_freq(len(x),2,30,50,d=1,reps=np.int(len(x)/900*70))
    lng2=zigzag_freq(len(x),4,30,50,d=1,reps=np.int(len(x)/900*70))
    return lat1, lat2, lng1, lng2
'''



class GlobalDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, ds_data,domains,ypad=75,xpad=300,subtime=1):
        self.ds_data = ds_data
        self.domains=domains
        self.xpad=xpad
        self.ypad=ypad
        tot_time=self.ds_data.isel(xu_ocean=[0],yu_ocean=[0]).time.shape[0]
        self.time_st=np.int64(np.floor(tot_time*self.domains['tmin']))
        self.time_tr=np.int64(np.ceil(tot_time*self.domains['tmax']))
        self.tot_time=self.time_tr-self.time_st
        self.num_time=np.int64(np.ceil(self.tot_time*subtime))
        self.time_downsample=np.sort(np.random.choice(                                        self.tot_time,self.num_time,replace=False))+self.time_st
    def __len__(self):
        'Denotes the total number of samples'
        return self.num_time
    def get_landmask(self):
        land_mass=torch.tensor(np.load('/scratch/cg3306/climate/physical_land_11.npy'),dtype=torch.float32)
        land_mass=torch.stack([land_mass],dim=0)
        land_mass=torch.stack([land_mass],dim=0)
        return land_mass
    def __getitem__(self, nt):
        'Generates one sample of data'
        nt=self.time_downsample[nt]
        datsel=self.ds_data.isel(time=[nt])
        usurf=datsel.usurf.values
        vsurf=datsel.vsurf.values
        
        x=datsel.xu_ocean.values
        y=datsel.yu_ocean.values
        dy=np.zeros((len(y),))
        dy[1:]=y[1:]-y[:-1]
        dy[:-1]=dy[:-1]+dy[1:]
        dy[1:-1]=dy[1:-1]/2
        dy=(dy-np.mean(dy))/np.std(dy)
        cosx=np.cos(x/180*np.pi)
        siny=np.sin(y/80*np.pi)
        
        
        usurf=torch.tensor(usurf,dtype=torch.float32)
        vsurf=torch.tensor(vsurf,dtype=torch.float32)
        
        cosx=torch.reshape(torch.tensor(cosx,dtype=torch.float32),[1,1,-1])
        siny=torch.reshape(torch.tensor(siny,dtype=torch.float32),[1,-1,1])
        dy=torch.reshape(torch.tensor(dy,dtype=torch.float32),[1,-1,1])
        fcosx=cosx+siny*0
        fsiny=cosx*0+siny
        fdy=cosx*0+dy
        
        
        land_mass=torch.tensor(np.load('/scratch/cg3306/climate/physical_land_11.npy'),dtype=torch.float32)
        land_mass=torch.stack([land_mass],dim=0)
        X=torch.cat([usurf,vsurf,land_mass,fcosx,fsiny,fdy],dim=0)
        X[X!=X]=0
        X[:1]=X[:1]*(1-X[2:3])
        
        mx=self.xpad
        my=self.ypad
        
        
        sx=datsel.S_x.values
        sy=datsel.S_y.values
        sx=torch.tensor(sx,dtype=torch.float32)
        sy=torch.tensor(sy,dtype=torch.float32)
        

        Y=torch.cat([sx,sy],dim=0)
        
        Y[Y!=Y]=0
        Y=Y*(1-X[2:3])
        
        X=torch.cat([X[:,:,-mx:],X,X[:,:,:mx]],dim=2)
        X=torch.cat([torch.flip(X[:,:my],[1]),X,torch.flip(X[:,-my:],[1])],dim=1)
        
        return X,torch.tensor(0),Y


# In[ ]:




