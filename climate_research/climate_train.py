#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
from datetime import date

'''

os.system("jupyter nbconvert --to script 'climate_data.ipynb'")
os.system("jupyter nbconvert --to script 'climate_models.ipynb'")
'''

import climate_data
import climate_models

def options(string_input=[]):
    parser=argparse.ArgumentParser()
    parser.add_argument("-b","--batch",type=int,default=4)
    parser.add_argument("-e","--epoch",type=int,default=4)
    parser.add_argument("-r","--rootdir",type=str,default='/scratch/cg3306/climate/runs/')
    parser.add_argument("--nworkers",type=int,default=1)
    parser.add_argument("-o","--outdir",type=str,default="")
    parser.add_argument("--testrun",type=int,default=0)
    parser.add_argument("--action",type=str,default="train")
    parser.add_argument("--model_id",type=str,default="0")
    parser.add_argument("--data_address",type=str,default=                        '/scratch/ag7531/mlruns/19/bae994ef5b694fc49981a0ade317bf07/artifacts/forcing/')
    parser.add_argument("--relog",type=int,default=0)
    parser.add_argument("--rerun",type=int,default=0)
    parser.add_argument("--lr",type=float,default=0.01)
    parser.add_argument("--model_bank_id",type=str,default="0")
    parser.add_argument("--physical_dom_id",type=int,default=0)
    parser.add_argument("--subtime",type=float,default=1)
    parser.add_argument("--disp",type=int,default=500)
    parser.add_argument("--co2",type=int,default=0)
    parser.add_argument("--depth",type=int,default=0)
    if len(string_input)==0:
        return parser.parse_args()
    return parser.parse_args(string_input)
def default_collate(batch):
    if len(batch[0])==3:
        data = torch.stack([item[0] for item in batch])
        mask = torch.stack([item[1] for item in batch])
        target = torch.stack([item[2] for item in batch])
        return data,mask,target
    else:
        data = torch.stack([item[0] for item in batch])
        target = torch.stack([item[1] for item in batch])
        return data,target
def load_from_save(args):
    if len(args.outdir)>0:
        root = args.rootdir +str(args.outdir) +str(args.model_id)
    else:
        root = args.rootdir +str(args.model_bank_id) + "-"+str(args.model_id)
    PATH0 = root + "/last-model"
    PATH1 = root + "/best-model"
    LOG = root + "/log.json"
    net,loss,data_init,partition=climate_models.model_bank(args)
    
    isdir = os.path.isdir(args.rootdir)
    if not isdir:
        os.mkdir(args.rootdir)
    isdir = os.path.isdir(root) 
    if not isdir:
        os.mkdir(root)
    if not args.rerun:
        try:
            net.load_state_dict(torch.load(PATH1,map_location=get_device()))
            net.train()
            print("Loaded the existing model")
        except IOError:
            print("No existing model found - new beginnings")
    try: 
        with open(LOG) as f:
            logs = json.load(f)
    except IOError:
        if not net.gan:
            logs = {"epoch":[],"train-loss":[],"test-loss":[],"val-loss":[],"lr":[],"batchsize":[]}
        else:
            logs = {"epoch":[],"train-generator-loss":[],"train-discriminator-loss":[],                                "test-generator-loss":[],"test-discriminator-loss":[],                                "val-generator-loss":[],"val-discriminator-loss":[],                                 "lr-generator":[],"lr-discriminator":[],                            "batchsize":[]}
    if args.relog:
        if not net.gan:
            logs = {"epoch":[],"train-loss":[],"test-loss":[],"val-loss":[],"lr":[],"batchsize":[]}
        else:
            logs = {"epoch":[],"train-generator-loss":[],"train-discriminator-loss":[],                                "test-generator-loss":[],"test-discriminator-loss":[],                                "val-generator-loss":[],"val-discriminator-loss":[],                                 "lr-generator":[],"lr-discriminator":[],                            "batchsize":[]}
    return net,loss,(data_init,partition), logs,(PATH0,PATH1,LOG,root)
def load_data( data_init,partition,args):
    '''if args.testrun>100 or args.testrun==0:
        max_epochs=args.epoch
    if args.testrun>0
        max_epochs=4'''
    '''args.physical_dom_id=physical_dom_id
    partition=climate_data.physical_domains(args.physical_dom_id)'''
    timeout=180
    params={'batch_size':args.batch,'shuffle':True, 'num_workers':args.nworkers,'timeout':timeout}
        
    training_set = data_init(partition['train']) #climate_data.Dataset(ds_zarr,partition['train'],net,subtime=args.subtime)
    training_generator = torch.utils.data.DataLoader(training_set, **params,collate_fn=default_collate)
    
    
    params={'batch_size':args.batch,'shuffle':False, 'num_workers':args.nworkers,'timeout':timeout}
    
    val_set = data_init(partition['validation'])#climate_data.Dataset(ds_zarr,partition['validation'],net,subtime=args.subtime)
    val_generator = torch.utils.data.DataLoader(val_set,                                        **params,collate_fn=default_collate)

    test_set = data_init(partition['test'])#climate_data.Dataset(ds_zarr,partition['test'],net,subtime=args.subtime)
    test_generator = torch.utils.data.DataLoader(test_set,**params,collate_fn=default_collate)
    
    earth_set = data_init(partition['earth'])#climate_data.Dataset(ds_zarr,partition['earth'],net,subtime=args.subtime)
    earth_generator = torch.utils.data.DataLoader(earth_set,                                          **params,collate_fn=default_collate)
    return (training_set,training_generator),            (val_set,val_generator),                (test_set,test_generator),                    (earth_set,earth_generator)
def get_device():
    use_cuda=torch.cuda.is_available()
    device=torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark=True
    return device
def prep(args):
    print('prepping',flush=True)
    net,criterion,(data_init,partition),logs,(PATH0,PATH1,LOG,root)=load_from_save(args)
    
    (training_set,training_generator),_,_,_=load_data(data_init,partition,args)
    
    training_set.save_scales()
    training_set.save_masks()

def linear_feature_run(generator,indim,outdim,net,save_iter,landmasks,root,tag):
    if root[-1]=='/':
        root=root[:-1]
    tt=0
    X2=torch.zeros(indim,indim)
    XY=torch.zeros(indim,outdim)
    Y2=torch.zeros(outdim)
    device=get_device()
    for X,M, Y in generator:
        with torch.set_grad_enabled(False):
            X=X.to(device)
            X2_,XY_,Y2_=net.cross_products(X,Y,mask=landmasks[M])
        X2+=X2_
        XY+=XY_
        Y2+=Y2_
        tt+=1
        if tt%save_iter==0:
            np.save(root+'/X2-'+tag+'.npy',X2.numpy())
            np.save(root+'/XY-'+tag+'.npy',XY.numpy())
            np.save(root+'/Y2-'+tag+'.npy',Y2.numpy())
            print('\t\t'+str(tt),flush=True)
    np.save(root+'/X2-'+tag+'.npy',X2.numpy())
    np.save(root+'/XY-'+tag+'.npy',XY.numpy())
    np.save(root+'/Y2-'+tag+'.npy',Y2.numpy())
def train(args):
    net,criterion,(data_init,partition),logs,(PATH0,PATH1,LOG,root)=load_from_save(args)
    (training_set,training_generator),(val_set,val_generator),(test_set,test_generator),(glbl_set,glbl_gen)=load_data(data_init,partition,args)
    if isinstance(training_set,climate_data.Dataset2):
        landmasks=training_set.get_masks()
    else:
        landmasks=climate_data.get_land_masks(val_generator)
    device=get_device()
    landmasks=landmasks.to(device)
    landmasks.requires_grad=False
    if isinstance(net, climate_models.RegressionModel):
        outdim=len(training_set.outputs)
        indim=net.outputdimen
        save_iter=10

        generators=[training_generator,val_generator,test_generator]
        tags=['train','val','test']
        for i in range(len(tags)):
            generator=generators[i]
            tag=tags[i]
            print(tag)
            linear_feature_run(generator,indim,outdim,net,save_iter,landmasks,root,tag)
        return
    
    
    
    max_epochs = args.epoch
    kgan=2
    if not net.gan:
        if len(logs['lr'])==0:
            optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
        else:
            optimizer = optim.SGD(net.parameters(), lr=logs['lr'][-1], momentum=0.9)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=5)
    else:
        if len(logs['lr-generator'])==0:
            discriminator_optimizer = optim.SGD(net.discriminator.parameters(), lr=args.lr, momentum=0.9)
            generator_optimizer = optim.SGD(net.generator.parameters(), lr=args.lr, momentum=0.9)
        else:
            discriminator_optimizer = optim.SGD(net.discriminator.parameters(), lr=logs['lr-discriminator'][-1], momentum=0.9)
            generator_optimizer = optim.SGD(net.generator.parameters(), lr=logs['lr-generator'][-1], momentum=0.9)
        scheduler_discriminator=torch.optim.lr_scheduler.                        ReduceLROnPlateau(discriminator_optimizer,factor=0.5,patience=5)
        scheduler_generator=torch.optim.lr_scheduler.                        ReduceLROnPlateau(generator_optimizer,factor=0.5,patience=5)

    best_counter=0
    print("epochs started")
    for epoch in range(max_epochs):
        if not net.gan:
            logs['train-loss'].append([])
        else:
            logs['train-generator-loss'].append([])
            logs['train-discriminator-loss'].append([])
        tt=0
        for local_batch,dom_num, local_labels in training_generator:
            local_batch, local_labels = local_batch.to(device),local_labels.to(device)
            #print('train: '+str(local_batch.shape))
            if not net.gan:
                outputs = net.forward(local_batch)
                loss = criterion(outputs, local_labels,landmasks[dom_num])
                logs['train-loss'][-1].append(loss.item())
                scheduler.optimizer.zero_grad()
                loss.backward()
                scheduler.optimizer.step()
            else:
                shp=torch.tensor(local_batch.shape)
                shp[1]=1
                random_field=torch.randn(shp.tolist()).to(device)
                yhat = net.generator_forward(local_batch,random_field)
                phat = net.discriminator_forward(local_batch,yhat)#,local_labels)
                p = net.discriminator_forward(local_batch,local_labels)#,local_labels)
                if tt%kgan!=kgan-1:
                    loss = (criterion(p, 1,landmasks[dom_num])+criterion(phat, 0,landmasks[dom_num]))/2
                    logs['train-discriminator-loss'][-1].append(loss.item())
                    scheduler_discriminator.optimizer.zero_grad()
                    scheduler_generator.optimizer.zero_grad()  
                    loss.backward()
                    scheduler_discriminator.optimizer.step()
                else:
                    loss = criterion(phat, 1,landmasks[dom_num])
                    logs['train-generator-loss'][-1].append(loss.item())
                    scheduler_discriminator.optimizer.zero_grad()
                    scheduler_generator.optimizer.zero_grad()  
                    loss.backward()
                    scheduler_generator.optimizer.step()
            tt+=1
            if tt%args.disp==0:
                if not net.gan:
                    print('\t\t\t train-loss: ',str(np.mean(np.array(logs['train-loss'][-1]))),                          '\t ±',                          str(np.std(np.array(logs['train-loss'][-1]))),flush=True)
                else:
                    print('\t\t\t train-generator-loss: ',str(np.mean(np.array(logs['train-generator-loss'][-1]))),                          '\t ±',                          str(np.std(np.array(logs['train-generator-loss'][-1]))),flush=True)
                    print('\t\t\t train-discriminator-loss: ',str(np.mean(np.array(logs['train-discriminator-loss'][-1]))),                          '\t ±',                          str(np.std(np.array(logs['train-discriminator-loss'][-1]))),flush=True)
        with torch.set_grad_enabled(False):
            if not net.gan:
                val_loss=0.
            else:
                val_loss_d=0.
                val_loss_g=0.
            num_val=0
            for local_batch,dom_num, local_labels in val_generator:
                local_batch,local_labels = local_batch.to(device),local_labels.to(device)
                if not net.gan:
                    outputs = net.forward(local_batch)
                    loss = criterion(outputs, local_labels,landmasks[dom_num])
                    val_loss+=loss.item()
                else:
                    shp=torch.tensor(local_batch.shape)
                    shp[1]=1
                    random_field=torch.randn(shp.tolist()).to(device)
                    yhat = net.generator_forward(local_batch,random_field)
                    phat = net.discriminator_forward(local_batch,yhat)#,local_labels)
                    p = net.discriminator_forward(local_batch,local_labels)#,local_labels)
                    loss_g=criterion(phat, 1,landmasks[dom_num])
                    loss_d=criterion(p, 1,landmasks[dom_num])+criterion(phat, 0,landmasks[dom_num])
                    val_loss_d+=loss_d.item()
                    val_loss_g+=loss_g.item()
                num_val+=1
            if not net.gan:
                logs['val-loss'].append(val_loss/num_val)
                logs['lr'].append(scheduler.optimizer.param_groups[0]['lr'])
                scheduler.step(logs['val-loss'][-1])
            else:
                logs['val-discriminator-loss'].append(val_loss_d/num_val)
                logs['val-generator-loss'].append(val_loss_d/num_val)
                
                logs['lr-generator'].append(scheduler_generator.optimizer.param_groups[0]['lr'])
                scheduler_generator.step(logs['val-generator-loss'][-1])
                #logs['val-generator-loss'][-1]=logs['val-generator-loss'][-1]
                
                logs['lr-discriminator'].append(scheduler_discriminator.optimizer.param_groups[0]['lr'])
                scheduler_generator.step(logs['val-discriminator-loss'][-1])
                #logs['val-discriminator-loss'][-1]=logs['val-discriminator-loss'][-1]


        with torch.set_grad_enabled(False):
            if not net.gan:
                val_loss=0.
            else:
                val_loss_d=0.
                val_loss_g=0.
            num_val=0
            for local_batch,dom_num, local_labels in val_generator:
                local_batch,local_labels = local_batch.to(device),local_labels.to(device)
                if not net.gan:
                    outputs = net.forward(local_batch)
                    loss = criterion(outputs, local_labels,landmasks[dom_num])
                    val_loss+=loss.item()
                else:
                    shp=torch.tensor(local_batch.shape)
                    shp[1]=1
                    random_field=torch.randn(shp.tolist()).to(device)
                    yhat = net.generator_forward(local_batch,random_field)
                    phat = net.discriminator_forward(local_batch,yhat)#,local_labels)
                    p = net.discriminator_forward(local_batch,local_labels)#,local_labels)
                    loss_g=criterion(phat, 1,landmasks[dom_num])
                    loss_d=criterion(p, 1,landmasks[dom_num])+criterion(phat, 0,landmasks[dom_num])
                    val_loss_d+=loss_d.item()
                    val_loss_g+=loss_g.item()
                num_val+=1
            if not net.gan:
                logs['test-loss'].append(val_loss/num_val)
                logs['batchsize'].append(args.batch)
            else:
                logs['test-discriminator-loss'].append(val_loss_d/num_val)
                logs['test-generator-loss'].append(val_loss_g/num_val)

        if len(logs['epoch'])>0:
            logs['epoch'].append(logs['epoch'][-1]+1)
        else:
            logs['epoch'].append(1)
        if not net.gan:
            print('#epoch ',str(logs['epoch'][-1]),' ',                      ' test-loss: ',str(logs['test-loss'][-1]),                      ' val-loss: ',str(logs['val-loss'][-1]),                      ' train-loss: ',str(np.mean(np.array(logs['train-loss'][-1]))),                      ' lr: ',str(logs['lr'][-1]),flush=True)
        else:
            print('#epoch :',str(logs['epoch'][-1]),' ',                      ' (generator) test-loss: ',str(logs['test-generator-loss'][-1]),                      ' val-loss: ',str(logs['val-generator-loss'][-1]),                      ' train-loss: ',str(np.mean(np.array(logs['train-generator-loss'][-1]))),                      ' lr: ',str(logs['lr-generator'][-1]),flush=True)
            print('\t (discriminator):',                      ' test-loss: ',str(logs['test-discriminator-loss'][-1]),                      ' val-loss: ',str(logs['val-discriminator-loss'][-1]),                      ' train-loss: ',str(np.mean(np.array(logs['train-discriminator-loss'][-1]))),                      ' lr: ',str(logs['lr-discriminator'][-1]),flush=True)


        torch.save(net.state_dict(), PATH0)
        with open(LOG, 'w') as outfile:
            json.dump(logs, outfile)
        if not net.gan:
            if np.min(logs['test-loss']) == logs['test-loss'][-1]:
                torch.save(net.state_dict(), PATH1)
                best_counter=0
            else:
                best_counter+=1
            if logs['lr'][-1]<1e-7:
                break
        else:
            torch.save(net.state_dict(), PATH1)
            '''if np.min(logs['test-generator-loss']) == logs['test-generator-loss'][-1]:
                torch.save(net.state_dict(), PATH1)
                best_counter=0
            else:
                best_counter+=1'''
            if logs['lr-generator'][-1]<1e-7 and logs['lr-discriminator'][-1]<1e-7:
                break
def grad_probe_features2(uv,uv1,g,yc,xc,listout=False,projection=[],geoflag=True):
    if listout:        
        '''[y_coords[mid],x_coords[mid],           uv5,nuv,duvdt,duvdy,duvdx,duvdyy,duvdxy,duvdxx,              g5,r,cl2]'''
        names=[]
        t=0
        names.append(['coords',0,t+2])
        t+=2
        names.append(['uv5',t,t+2*5**2])
        t+=2*5**2
        names.append(['nuv',t,t+2])
        t+=2
        names.append(['duvdt',t,t+2])
        t+=2
        names.append(['duvdy',t,t+2])
        t+=2
        names.append(['duvdx',t,t+2])
        t+=2
        names.append(['duvdyy',t,t+2])
        t+=2
        names.append(['duvdxy',t,t+2])
        t+=2
        names.append(['duvdxx',t,t+2])
        t+=2
        names.append(['g5',t,t+2*5**2])
        t+=2*5**2
        names.append(['r',t,t+2*11])
        t+=2*11
        if geoflag:
            names.append(['cl2',t,t+6])
            t+=6
        else:
            names.append(['cl2',t,t+4])
            t+=4
        if len(projection)>0:
            names.append(['l-g5',t,t+2*5**2])
            t+=2*5**2
            names.append(['l-r',t,t+2*11])
            t+=2*11
            if geoflag:
                names.append(['l-cl2',t,t+6])
                t+=6
            else:
                names.append(['l-cl2',t,t+4])
                t+=4
        return names
    
    width=len(xc)
    mid=(width-1)//2
    
    uv=torch.reshape(uv,[-1,width,width])
    uv1=torch.reshape(uv1,[-1,width,width])
    g=torch.reshape(g,[-1,width,width])
    nchan=uv.shape[0]
    

    
    uv,_=uv.split([2,uv.shape[0]-2],dim=0)
    uv1,_=uv1.split([2,uv1.shape[0]-2],dim=0)
    
    
    
    area=(yc[-1]-yc[0])*(xc[-1]-xc[0])
    
    dyc= yc[1:]-yc[:-1]
    dxc= xc[1:]-xc[:-1]
    
    
    ddyc= (dyc[1:] + dyc[:-1])/2
    ddxc= (dxc[1:] + dxc[:-1])/2
    
    dyc=dyc.reshape([1,-1,1])
    dxc=dxc.reshape([1,1,-1])
    ddyc=ddyc.reshape([1,-1,1])
    ddxc=ddxc.reshape([1,1,-1])
    
    duvdy=(uv[:,:-1]-uv[:,1:])/dyc
    duvdx=(uv[:,:,:-1]-uv[:,:,1:])/dxc
    
    duvdyy=(duvdy[:,:-1]-duvdy[:,1:])/ddyc
    duvdxy=(duvdy[:,:,:-1]-duvdy[:,:,1:])/dxc
    duvdxx=(duvdx[:,:,:-1]-duvdx[:,:,1:])/ddxc
    
    duvdy=torch.sum(duvdy**2,dim=[1,2])*area
    duvdx=torch.sum(duvdx**2,dim=[1,2])*area
    
    duvdyy=torch.sum(duvdyy**2,dim=[1,2])*area
    duvdxy=torch.sum(duvdxy**2,dim=[1,2])*area
    duvdxx=torch.sum(duvdxx**2,dim=[1,2])*area
    
    duvdt=torch.sum((uv1-uv)**2,dim=[1,2])*area
    

    nuv=(uv**2).sum(axis=(1,2))*area
    
    
   
    
    
    uv5=uv[:2,mid-2:mid+3,mid-2:mid+3]
    
    
    ng=(g**2).sum(axis=(1,2),keepdim=True)*area
    g5=g[:2,mid-2:mid+3,mid-2:mid+3]
    r=torch.zeros(2,mid+1)
    r[:,0]=g[:2,mid,mid]**2
    
    for i in range(1,mid+1):
        r[:,i]+=torch.sum(g[:2,mid-i:mid+i,mid+i]**2,dim=1)
        r[:,i]+=torch.sum(g[:2,mid+i,mid-i:mid+i]**2,dim=1)
        r[:,i]+=torch.sum(g[:2,mid-i:mid+i,mid-i]**2,dim=1)
        r[:,i]+=torch.sum(g[:2,mid-i,mid-i:mid+i]**2,dim=1)
        r[:,i]=r[:,i]/( (2*i+1)*4-4)
    cl2=ng/torch.sqrt(torch.sum(ng**2))
    
    F=[yc[mid:mid+1],xc[mid:mid+1],           uv5,nuv,duvdt,duvdy,duvdx,duvdyy,duvdxy,duvdxx,              g5,r,cl2]
    
    
    if len(projection)>0:
        g_=(projection@(projection.T@(g[:2].reshape([-1])))).reshape([-1,width,width])
        g=torch.cat([g_,g[2:]],axis=0)
        ng=(g**2).sum(axis=(1,2),keepdim=True)*area
        g5_=g[:2,mid-2:mid+3,mid-2:mid+3]
        r_=torch.zeros(2,mid+1)
        r_[:,0]=g[:2,mid,mid]**2
        for i in range(1,mid+1):
            r_[:,i]+=torch.sum(g[:2,mid-i:mid+i,mid+i]**2,dim=1)
            r_[:,i]+=torch.sum(g[:2,mid+i,mid-i:mid+i]**2,dim=1)
            r_[:,i]+=torch.sum(g[:2,mid-i:mid+i,mid-i]**2,dim=1)
            r_[:,i]+=torch.sum(g[:2,mid-i,mid-i:mid+i]**2,dim=1)
            r_[:,i]=r[:,i]/( (2*i+1)*4-4)
        cl2_=ng/torch.sqrt(torch.sum(ng**2))
        F=F+[g5_,r_,cl2_]
    F=[f.reshape([-1]) for f in F]
    return torch.cat(F,dim=0)
def grad_probe_features3(g,yc,xc,inchan,spread,listout=False,geoflag=False):
    if listout:        
        names=[]
        t=0
        names.append(['coords',0,t+2])
        t+=2
        names.append(['r',t,t+inchan*(spread+1)])
        t+=inchan*(spread+1)
        if geoflag:
            names.append(['cl2',t,t+inchan+2])
            t+=6
        return names
    
    width=len(xc)
    mid=spread
    
    g=torch.reshape(g,[inchan,width,width])
    #nchan=uv.shape[0]
    
    
    r=torch.zeros(inchan,mid+1)
    r[:,0]=g[:inchan,mid,mid]**2
    
    for i in range(1,mid+1):
        r[:,i]+=torch.sum(g[:inchan,mid-i:mid+i,mid+i]**2,dim=1)
        r[:,i]+=torch.sum(g[:inchan,mid+i,mid-i:mid+i]**2,dim=1)
        r[:,i]+=torch.sum(g[:inchan,mid-i:mid+i,mid-i]**2,dim=1)
        r[:,i]+=torch.sum(g[:inchan,mid-i,mid-i:mid+i]**2,dim=1)
    for i in range(mid+1):
        r[:,i]=torch.sum(r[:,i:],dim=1)
    if geoflag:   
        ng=(g**2).sum(axis=(1,2),keepdim=True)
        cl2=ng/torch.sqrt(torch.sum(ng**2))
        F=[yc[mid:mid+1],xc[mid:mid+1],               r,cl2]
    else:
        F=[yc[mid:mid+1],xc[mid:mid+1],               r]
    F=[f.reshape([-1]) for f in F]
    return torch.cat(F,dim=0)
def grad_probe(args):
    net,criterion,(data_init,partition),logs,(PATH0,PATH1,LOG,root)=load_from_save(args)
    (training_set,training_generator),(val_set,val_generator),(test_set,test_generator),(dataset,glbl_gen)=load_data(data_init,partition,args)
    if isinstance(training_set,climate_data.Dataset2):
        landmasks=training_set.get_masks(glbl=True)
    else:
        landmasks=climate_data.get_land_masks(val_generator)
    device=get_device()
    MASK=landmasks.to(device)
    yc,xc=dataset.coords[0]
    xc=torch.tensor(xc)
    yc=torch.tensor(yc)

    if isinstance(dataset,climate_data.Dataset2):
        MASK=MASK[0,0]
        MASK[MASK==0]=np.nan
        MASK[MASK==MASK]=1
        MASK[MASK!=MASK]=0
    

    device=get_device()
    net.eval()
    for i in range(len(net.nn_layers)-1):
        try:
            net.nn_layers[i].weight.requires_grad=False
        except:
            QQ=1
    geoflag=net.freq_coord
    inchan=net.nn_layers[0].weight.data.shape[1]#net.initwidth
    outchan=net.outwidth
    names=grad_probe_features3([],[],[],inchan,net.spread,listout=True)#,geoflag=geoflag)
    numprobe=names[-1][-1]


    maxsamplecount=1000
    samplecount=np.minimum(len(dataset),maxsamplecount)
    dt=np.maximum(len(dataset)//samplecount,1)
    numbatch=3
    width=net.receptive_field

    chss=np.arange(maxsamplecount)*dt
    np.random.shuffle(chss)
    
    tot=0
    ii=0
    dd=0
    TOTMAX=np.inf
    F=[]
    spread=net.spread
    MASKK=MASK[:-width,:-width]
    stsz=[np.maximum(1,MASKK.shape[i]//70) for i in range(2)]
    MASKK=MASKK[::stsz[0],::stsz[1]]
    KK,LL=np.where(MASKK>0)
    snum=0
    dd=0
    
    samplecount1=20
    GRADS=torch.zeros(samplecount1,len(KK),outchan,inchan,width,width)
    GS=torch.zeros(samplecount,len(KK),numprobe*outchan)

    for i in range(samplecount):
        UV,_,S = dataset[chss[i]]
        for j in range(len(KK)):
            K,L=KK[j]*stsz[0],LL[j]*stsz[1]
            for chan in range(outchan):
                uv=torch.stack([UV[:,K:K+width,L:L+width]],dim=0).to(device)
                uv.requires_grad=True
                output=net.forward(uv)
                x0=output.shape[2]
                x1=output.shape[3]
                m0=(x0-1)//2
                m1=(x1-1)//2
                ou=output[0,chan,m0,m1]
                ou.backward(retain_graph=True)
                g=uv.grad
                uv.grad=None
                uv=uv.to(torch.device("cpu")).detach()
                g=g.to(torch.device("cpu")).detach()
                sample=grad_probe_features3(g,yc[K:K+width],xc[L:L+width],inchan,net.spread)#,geoflag=geoflag)
                GS[i, j,chan*len(sample):(chan+1)*len(sample)]=sample
                if i<samplecount1:
                    GRADS[i,j,chan]=g.reshape([inchan,width,width])
            dd+=1
        if i%args.disp==0:
            print(i,samplecount,flush=True)
            np.save(root+'/grad-probe-data.npy', GS[:i+1])
            if i<samplecount1:
                np.save(root+'/grad-samples.npy', GRADS[:i+1])
        if i==samplecount1:
            np.save(root+'/grad-samples.npy', GRADS)
                


    np.save(root+'/grad-probe-data.npy', GS)


def grad_analysis(args):
    net,criterion,(data_init,partition),logs,(PATH0,PATH1,LOG,root)=load_from_save(args)
    _,_,_,(dataset,datagen)=load_data(data_init,partition,args)
    
    MASK=climate_data.get_land_masks(datagen)[0,0]
    device=get_device()
    net.eval()
    for i in range(len(net.nn_layers)-1):
        try:
            net.nn_layers[i].weight.requires_grad=False
        except:
            QQ=1
    width=net.receptive_field
    spread=net.spread
    dx=spread
    dy=spread
    
    W=np.reshape(np.arange(spread+1),[-1,1])
    sx=dataset.dimens[1]-width+1
    sy=dataset.dimens[0]-width+1
    
    xx=np.arange(0,sx,dx)
    yy=np.arange(0,sy,dy)
    nx=len(xx)
    ny=len(yy)
    
    
    UV,_,_ = dataset[0]
    nchan=UV.shape[0]
    G=np.zeros((nchan*3,ny*width, nx*width))
    width=net.receptive_field
    maxsamplenum=4*width**2+2
    CN=np.zeros((4*width**2+2,4*width**2+2))
    MS=np.zeros((4*width**2+2,maxsamplenum))
    samplecount=0
    tot=0
    ii=0
    dd=0
    TOTMAX=np.inf
    for i in range(len(dataset)):
        UV,_,_ = dataset[i]
        #for local_batch,nan_mask, _ in datagen:
        #uv, nan_mask = local_batch.to(device),nan_mask.to(device)
        for k in range(ny):
            for l in range(nx):
                dd+=1
                K,L=yy[k],xx[l]
                if MASK[K,L]>0:  
                    uv=torch.stack([UV[:,K:K+width,L:L+width]],dim=0).to(device)
                    uv.requires_grad=True
                    output=net.forward(uv)
                    x0=output.shape[2]
                    x1=output.shape[3]
                    m0=(x0-1)//2
                    m1=(x1-1)//2
                    ou=output[0,0,m0,m1]
                    ou.backward(retain_graph=True)
                    g=uv.grad
                    uv.grad=None
                    uv=uv.to(torch.device("cpu")).detach().numpy()
                    g=g.to(torch.device("cpu")).detach().numpy()
                    g=np.reshape(g,[nchan,width,width])
                    for j in range(nchan):
                        G[j,k*width:(k+1)*width,l*width:(l+1)*width]+=g[j]**2
                        fg=np.abs(np.fft.fftshift(np.fft.fft2(g[j])))
                        G[j+nchan,k*width:(k+1)*width,l*width:(l+1)*width]+=fg**2
                        G[j+2*nchan,k*width:(k+1)*width,l*width:(l+1)*width]+=np.abs(g[j])/np.sum(np.abs(g[j]))
                    
                    ii+=1
                    uv_=np.reshape(uv[0,:2],[-1,1])
                    g_=np.reshape(g[:2],[-1,1])
                    ou=np.reshape(ou.to(torch.device("cpu")).detach().numpy(),[-1,1])
                    vec_=np.concatenate((uv_,ou,g_,np.ones((1,1))),axis=0)
                    CN=CN+vec_@vec_.T
                    tot+=1
                    if samplecount<maxsamplenum:
                        if np.random.randint(2, size=1)[0]==0:
                            MS[:,samplecount:samplecount+1]=vec_
                            samplecount+=1
                    if tot>TOTMAX:
                        break
                else:
                    G[:,k*width:(k+1)*width,l*width:(l+1)*width]=np.nan
                if dd%1000==0:
                    print('\t\t '+str(dd/nx/ny),flush=True)
            if tot>TOTMAX:
                break
                
        print(tot,flush=True)
        with open(root+'/global-grad.npy', 'wb') as f:
            np.save(f, G/tot)
        with open(root+'/global-grad-covariance.npy', 'wb') as f:
            np.save(f, CN/tot)
        with open(root+'/global-grad-samples.npy', 'wb') as f:
            np.save(f, MS)
        if tot>TOTMAX:
            break
def data_fourier_analysis(args):
    # WILL NOT WORK BECAUSE OF CHANGES
    net,criterion,logs,PATH0,PATH1,LOG,root=load_from_save(args)

    width=41
    dx=(width-1)//2
    dy=(width-1)//2

    net.spread=dx
    _,_,_,(dataset,datagen)=load_data(net,args)
    MASK=climate_data.get_land_masks(datagen)[0,0]

    
    
    
    sx=dataset.dimens[1]-2*dx-2*dx
    sy=dataset.dimens[0]-2*dx-2*dx
    
    xx=np.arange(0,sx,dx)
    yy=np.arange(0,sy,dy)
    nx=len(xx)
    ny=len(yy)
    
    G=np.zeros((8,ny*width, nx*width))
    
    
    tot=0
    
    for i in range(len(dataset)):
        UV,_,SXY = dataset[i]
        #for local_batch,nan_mask, _ in datagen:
        #uv, nan_mask = local_batch.to(device),nan_mask.to(device)
        for k in range(ny):
            for l in range(nx):
                K,L=yy[k],xx[l]
                if MASK[K,L]>0:    
                    uv=UV[:,K:K+width,L:L+width].numpy()
                    sxy=SXY[:,K:K+width,L:L+width].numpy()
                    ff=[np.abs(np.fft.fftshift(np.fft.fft2(uv[oo]))) for oo in range(2)]
                    ff=ff+[np.abs(np.fft.fftshift(np.fft.fft2(sxy[oo]))) for oo in range(2)]
                    for j in range(len(ff)):
                        G[j,k*width:(k+1)*width,l*width:(l+1)*width]+=ff[j]**2
                else:
                    G[:,k*width:(k+1)*width,l*width:(l+1)*width]=np.nan
        tot+=1
        print(tot,flush=True)
        NG=torch.tensor(G)
        NG[:4]=NG[:4]/tot
        for k in range(ny):
            for l in range(nx):
                NN=NG[:4,k*width:(k+1)*width,l*width:(l+1)*width]
                MNN=torch.sum(NN,dim=[1,2],keepdim=True)
                NN=NN/MNN
                NG[4:,k*width:(k+1)*width,l*width:(l+1)*width]=NN
        with open('/scratch/cg3306/climate/global-fourier-analysis.npy', 'wb') as f:
            np.save(f, NG.numpy()) 

            
def data_covariance(args):
    net,criterion,(data_init,partition),logs,(PATH0,PATH1,LOG,root)=load_from_save(args)
    _,_,_,(dataset,datagen)=load_data(data_init,partition,args)
    
    MASK=climate_data.get_land_masks(datagen)[0,0]
    device=get_device()
    net.eval()
    width=net.receptive_field
    spread=net.spread
    dx=spread
    dy=spread
    
    W=np.reshape(np.arange(spread+1),[-1,1])
    sx=dataset.dimens[1]-width+1
    sy=dataset.dimens[0]-width+1
    
    xx=np.arange(0,sx,dx)
    yy=np.arange(0,sy,dy)
    nx=len(xx)
    ny=len(yy)
    
    
    UV,_,_ = dataset[0]
    nchan=UV.shape[0]
    G=np.zeros((nchan*2,ny*width, nx*width))
    width=net.receptive_field
    COV=np.zeros((2*width**2+1,2*width**2+1))
    samplecount=0
    tot=0
    TOTMAX=np.inf
    for i in range(len(dataset)):
        UV,_,_ = dataset[i]
        #for local_batch,nan_mask, _ in datagen:
        #uv, nan_mask = local_batch.to(device),nan_mask.to(device)
        for k in range(ny):
            for l in range(nx):
                K,L=yy[k],xx[l]
                if MASK[K,L]>0:  
                    uv=torch.reshape(UV[:2,K:K+width,L:L+width],(1,-1))
                    uv=torch.cat([uv,torch.ones(1,1)],dim=1)
                    COV=COV+(uv.T@uv).numpy()
                    tot+=1
                    if tot>TOTMAX:
                        break
            if tot>TOTMAX:
                break
        if tot>TOTMAX:
            break
        if i%10==0:
            print('\t\t '+str(tot/nx/ny/len(dataset)),flush=True)
        with open('/scratch/cg3306/climate/data-covariance.npy', 'wb') as f:
            np.save(f, COV/tot) 
            
def projection_analysis(args):
    model_id=int(args.model_id)%4
    sigma_id=int(args.model_id)//4
    sigma_vals=[4,8,12,16]
    sigma=sigma_vals[sigma_id]
    data_root='/scratch/zanna/data/cm2.6/coarse-'
    raw_add=['3D-data-sigma-'+str(sigma)+'.zarr','surf-data-sigma-'+str(sigma)+'.zarr']
    raw_co2=['1pct-CO2-'+ss for ss in raw_add]
    raw_add=raw_add+raw_co2
    raw_data_address=raw_add[model_id]
    ds_data=xr.open_zarr(data_root+raw_data_address)
    MSELOC='/scratch/cg3306/climate/projection_analysis/'+raw_data_address.replace('.zarr','')+'MSE.npy'
    SC2LOC='/scratch/cg3306/climate/projection_analysis/'+raw_data_address.replace('.zarr','')+'SC2.npy'
    noutsig=3
    names='Su Sv ST'.split()
    MSE=torch.zeros(noutsig,ds_data.Su.shape[-2], ds_data.Su.shape[-1])
    SC2=torch.zeros(noutsig,ds_data.Su.shape[-2], ds_data.Su.shape[-1])
    print(MSELOC)
    T=len(ds_data.time.values)
    for i in range(T):
        uv=ds_data.isel(time=i)
        for j in range(len(names)):
            Sxy=uv[names[j]].values
            output=Sxy-uv[names[j]+'_r'].values
            SC2[j]+=Sxy**2
            MSE[j]+=(Sxy-output)**2
        if i%50==0:
            with open(MSELOC, 'wb') as f:
                np.save(f, MSE/(i+1))
            with open(SC2LOC, 'wb') as f:
                np.save(f, SC2/(i+1))
            print('\t #'+str(i),flush=True)
    with open(MSELOC, 'wb') as f:
        np.save(f, MSE/(i+1))
    with open(SC2LOC, 'wb') as f:
        np.save(f, SC2/(i+1))
def global_averages():
    DIR='/scratch/zanna/data/cm2.6/'
    roots=['3D-data',           'surf-data']
    roots=roots+['1pct-CO2-'+r for r in roots]
    roots=['coarse-'+r for r in roots]

    saveloc='/scratch/cg3306/climate/portrays/'
    sigmavals=[4,8,12,16]
    varnames=['Su','Sv','ST']
    save_buff=500
    maxtime=-1
    varnames=varnames+[var+'_r' for var in varnames]
    for file in roots:
        for i in range(len(sigmavals)):
            fname=file+'-sigma-'+str(sigmavals[i])
            print(fname)
            ds_zarr=xr.open_zarr(DIR+fname+'.zarr')
            ny=len(ds_zarr.yu_ocean)
            nx=len(ds_zarr.xu_ocean)
            X=np.zeros((len(varnames),ny,nx))
            if maxtime>0:
                T=np.minimum(len(ds_zarr.time),maxtime)
            else:
                T=len(ds_zarr.time)
            for t in range(T):
                ds=ds_zarr.isel(time=t)
                for j in range(len(varnames)):
                    X[j]+=ds[varnames[j]].values**2
                if t%save_buff==0 or t==T-1:
                    print('\t\t'+str(t),flush=True)
                    np.save(saveloc+fname,X/(t+1))
def linear_model_fit(root):
    filenames=['X2','XY','Y2']
    tags=['train','val','test']
    D=[]
    for i in range(len(tags)):
        D.append([])
        for j in range(len(filenames)):
            D[-1].append(np.load(root+'/'+filenames[j]+'-'+tags[i]+'.npy'))
    lmbds=10**np.linspace(-8,1,num=100)
    errs=np.zeros((len(lmbds),3))
    xmean=np.mean(np.abs(D[0][0]))
    for i in range(len(lmbds)):
        lmbd=lmbds[i]
        w_=np.linalg.solve(D[0][0]+lmbd*xmean*np.eye(D[0][0].shape[0]),D[0][1])
        for j in range(w_.shape[1]):
            w=w_[:,j]
            errs[i,j]=w.T@(D[1][0]@w)-2*D[1][1][:,j]@w+D[1][2][j]
    I=np.argmin(errs,axis=0)
    w=np.zeros((len(w),3))
    for i in range(3):
        lmbd=lmbds[I[i]]
        w_=np.linalg.solve(D[0][0]+lmbd*xmean*np.eye(D[0][0].shape[0]),D[0][1])
        w[:,i]=w_[:,i]
    return w 
def binned_r2_analysis(args,save=True):
    net,criterion,(data_init,partition),logs,(PATH0,PATH1,LOG,root)=load_from_save(args)
    (training_set,training_generator),(val_set,val_generator),(test_set,test_generator),(dataset,glbl_gen)=load_data(data_init,partition,args)
    residue_flag=dataset.outputs[0]=='Su_r'
    if residue_flag:
        numoutput=len(dataset.outputs)
        dataset.outputs+=['Su','Sv','ST']
        dataset.outscales=dataset.outscales*2
    if isinstance(net, climate_models.RegressionModel):
        w=linear_model_fit(root)
    device=get_device()
    net.eval()

    MSELOC=root+'/binned-mse.npy'
    MSERESLOC=root+'/binned-mse-res.npy'
    SC2LOC=root+'/binned-sc2.npy'
    FREQLOC=root+'/binned-freq.npy'
    EDGESLOC=root+'/binned-edges.npy'
    EQEDGESLOC=root+'/binned-equid-edges.npy'
    
    LOCS=[MSELOC,MSERESLOC,SC2LOC,FREQLOC,EDGESLOC,EQEDGESLOC]
    if args.co2:
        LOCS=[LL.replace('.npy','-co2.npy') for LL in LOCS]
            
    print(root)
    
    spread=net.spread
    noutsig=net.outwidth



    if args.depth:
        LOCS=[LL.replace('.npy','-depth.npy') for LL in LOCS]
        numdepths=np.maximum(len(training_set.depthvals),1)
    else:
        numdepths=1
        
    MSELOC,MSERESLOC,SC2LOC,FREQLOC,EDGESLOC,EQEDGESLOC=LOCS
    print(' doing binned R2 analysis')

    nedges=101

    num_init_samples=30
    pooler=nn.MaxPool2d(2*dataset.mask_spread+1,stride=1)

    arr=np.arange(len(dataset))
    np.random.shuffle(arr)

    uv,Sxy=dataset.input_output(arr[0],periodic_lon_expand=True)

    def gen_mask(uv,pooler):
        if len(uv.shape)==3:
            MASK=uv[:1]*1
        elif len(uv.shape)==2:
            MASK=uv*1
        MASK[MASK==MASK]=0
        MASK[MASK!=MASK]=1
        MASK=1-pooler(MASK)
        return MASK[0]

    MASK=gen_mask(uv,pooler)
    sp_extend=np.int64(torch.sum(MASK).item())
    SU=torch.zeros(numdepths,noutsig,sp_extend*num_init_samples)

    for di in range(num_init_samples*numdepths):
        i=di%num_init_samples
        depthind=di//num_init_samples
        if args.depth:
            dataset.depthind=depthind
            uv,Sxy=dataset.input_output(arr[i],scale=False,periodic_lon_expand=True)
            if i==0:
                training_set.depthind=depthind
                insc,outsc=training_set.compute_scales()
                insc,outsc=np.reshape(insc,[-1,1,1]),np.reshape(outsc,[-1,1,1])
                if residue_flag:
                    outsc=np.concatenate([outsc,outsc],axis=0)
                MASK=gen_mask(uv,pooler)
        else:
            uv,Sxy=dataset.input_output(arr[i],scale=False,periodic_lon_expand=True)
        if residue_flag:
            _,Sxy=torch.split(Sxy,[3,3],dim=0)
        Sxy=Sxy[:,MASK==1]
        SU[depthind,:,i*sp_extend:(i+1)*sp_extend]=Sxy

    ASU=torch.abs(SU)
    SSU,_=torch.sort(ASU,dim=2)
    I=np.arange(0,nedges)*((sp_extend*num_init_samples)//nedges)
    #I=np.concatenate([I,[(sp_extend*num_init_samples)-1]])
    I=torch.tensor(I)
    edges1=SSU[:,:,I]
    medges1=(edges1[:,:,1:]+edges1[:,:,:-1])/2

    lastel=edges1[:,:,-1]
    edges=torch.zeros(numdepths,noutsig,nedges)
    for i in range(edges1.shape[0]):
        for j in range(edges1.shape[1]):
            edges[i,j,:]=torch.linspace(0,lastel[i,j],nedges)

    def update_subroutine(SIG,DSIG,MSE,S2,FREQ,edges,edges1,depthind,only_mse_update=False):
        ASIG=torch.abs(SIG)
        i=depthind
        for j in range(edges.shape[1]):
            for k in range(edges.shape[2]-1):
                e1=edges1[i,j,k]
                e2=edges1[i,j,k+1]
                I=(ASIG[i,j]>=e1)*(ASIG[i,j]<=e2)
                MSE[i,j,k]+=torch.sum(DSIG[i,j,I]**2)
                if not only_mse_update:
                    S2[i,j,k]+=torch.sum(SIG[i,j,I]**2)
                    e1=edges[i,j,k]
                    e2=edges[i,j,k+1]
                    I=(ASIG[i,j]>=e1)*(ASIG[i,j]<=e2)
                    FREQ[i,j,k]+=torch.sum(I)
        return MSE,S2,FREQ

    MSE=torch.zeros(numdepths,noutsig,nedges-1)
    S2=MSE*1
    FREQ=torch.zeros(numdepths,noutsig,nedges-1)
    
    if save:
        with open(EDGESLOC, 'wb') as f:
            np.save(f,edges.numpy())
        with open(EQEDGESLOC, 'wb') as f:
            np.save(f,edges1.numpy())
    

    if residue_flag:
        MSE_RES=MSE*1

    for di in range(len(dataset)*numdepths):
        i=di%len(dataset)
        depthind=di//len(dataset)
        if args.depth:
            dataset.depthind=depthind
            uv,Sxy=dataset.input_output(arr[i],scale=False,periodic_lon_expand=True)
            if i==0:
                training_set.depthind=depthind
                insc,outsc=training_set.compute_scales()
                insc,outsc=np.reshape(insc,[-1,1,1]),np.reshape(outsc,[-1,1,1])
                if residue_flag:
                    outsc=np.concatenate([outsc,outsc],axis=0)
                MASK=gen_mask(uv,pooler)
            uv=uv/insc
            Sxy=Sxy/outsc
            Sxy[Sxy!=Sxy]=0
            uv[uv!=uv]=0
        else:
            outsc=np.array(dataset.outscales).reshape([-1,1,1])
            uv,Sxy=dataset.input_output(arr[i],scale=True,periodic_lon_expand=True)
            Sxy[Sxy!=Sxy]=0
            uv[uv!=uv]=0
        if residue_flag:
            Sxy1,Sxy=torch.split(Sxy,[numoutput,numoutput],dim=0)
        uv=torch.stack([uv]).to(device)
        with torch.set_grad_enabled(False):
            if isinstance(net, climate_models.RegressionModel):
                output=net.forward(uv,w)
            else:
                output=net.forward(uv)
        output=output[0].to(torch.device("cpu"))
        output,prec,_=torch.split(output,[noutsig,net.nprecision,output.shape[0]-noutsig-net.nprecision],dim=0)
        output=output[:,MASK==1]
        Sxy=Sxy[:,MASK==1]
        if not residue_flag:
            MSE,S2,FREQ=update_subroutine(Sxy*outsc,(Sxy-output)*outsc,MSE,S2,FREQ,edges,edges1,depthind)
        else:
            Sxy1=Sxy1[:,MASK==1]
            MSE,S2,FREQ=update_subroutine(Sxy*outsc,(Sxy1-output)*outsc,MSE,S2,FREQ,edges,edges1,depthind)
            MSE_RES,_,_=update_subroutine(Sxy*outsc,Sxy1*outsc,MSE_RES,S2,FREQ,edges,edges1,depthind,only_mse_update=True)
        if di%10==0:
                MSE_=MSE.numpy()
                MSE_[:depthind]=MSE_[:depthind]/len(dataset)
                MSE_[depthind]=MSE_[depthind]/(i+1)
                SC2_=S2.numpy()
                SC2_[:depthind]=SC2_[:depthind]/len(dataset)
                SC2_[depthind]=SC2_[depthind]/(i+1)
                if save:
                    with open(MSELOC, 'wb') as f:
                        np.save(f, MSE_)
                    with open(SC2LOC, 'wb') as f:
                        np.save(f, SC2_)
                    with open(FREQLOC, 'wb') as f:
                        np.save(f, FREQ.numpy())
                if residue_flag:
                    MSE_=MSE_RES.numpy()
                    MSE_[:depthind]=MSE_[:depthind]/len(dataset)
                    MSE_[depthind]=MSE_[depthind]/(i+1)
                    if save:
                        with open(MSERESLOC, 'wb') as f:
                            np.save(f, MSE_)
                if not args.depth:
                    print('\t #'+str(i),flush=True)
                else:
                    print('\t depth# '+str(depthind)+', time# '+str(i),flush=True)
                
    MSE_=MSE.numpy()
    MSE_[:depthind]=MSE_[:depthind]/len(dataset)
    MSE_[depthind]=MSE_[depthind]/(i+1)
    SC2_[:depthind]=SC2_[:depthind]/len(dataset)
    SC2_[depthind]=SC2_[depthind]/(i+1)
    if save:
        with open(MSELOC, 'wb') as f:
            np.save(f, MSE_)
        with open(SC2LOC, 'wb') as f:
            np.save(f, SC2_)
        with open(FREQLOC, 'wb') as f:
            np.save(f, FREQ.numpy())
        if residue_flag:
            MSE_=MSE_RES.numpy()
            MSE_[:depthind]=MSE_[:depthind]/len(dataset)
            MSE_[depthind]=MSE_[depthind]/(i+1)
            if save:
                with open(MSERESLOC, 'wb') as f:
                    np.save(f, MSE_)
    print('analysis is done',flush=True)
    


# In[5]:


def analysis(args):
    net,criterion,(data_init,partition),logs,(PATH0,PATH1,LOG,root)=load_from_save(args)
    (training_set,training_generator),(val_set,val_generator),(test_set,test_generator),(dataset,glbl_gen)=load_data(data_init,partition,args)
    residue_flag=dataset.outputs[0]=='Su_r'
    if residue_flag:
        numoutputs=len(dataset.outputs)
        dataset.outputs+=[ss.replace('_r','') for ss in dataset.outputs]
        dataset.outscales=dataset.outscales*2
    if isinstance(net, climate_models.RegressionModel):
        w=linear_model_fit(root)
    device=get_device()
    net.eval()
    if args.co2:
        MSELOC=root+'/MSE-co2.npy'
        SC2LOC=root+'/SC2-co2.npy'
    else:
        MSELOC=root+'/MSE.npy'
        SC2LOC=root+'/SC2.npy'

    spread=net.spread
    noutsig=net.outwidth



    if args.depth:
        MSELOC=MSELOC.replace('.npy','-depth.npy')
        SC2LOC=SC2LOC.replace('.npy','-depth.npy')
        numdepths=len(training_set.depthvals)
    else:
        numdepths=1
    recfield=0
    if not dataset.padded:
        recfield=2*spread
    MSE=torch.zeros(numdepths,noutsig,dataset.dimens[0]-recfield, dataset.dimens[1]-recfield)
    #LIKE=torch.zeros(noutsig,dataset.dimens[0]-spread*2, dataset.dimens[1]-spread*2)
    SC2=torch.zeros(numdepths,noutsig,dataset.dimens[0]-recfield, dataset.dimens[1]-recfield)
    print(MSELOC)
    arr=np.arange(len(dataset))
    np.random.shuffle(arr)

    for di in range(len(dataset)*numdepths):
        i=di%len(dataset)
        depthind=di//len(dataset)
        if args.depth:
            dataset.depthind=depthind
            uv,Sxy=dataset.input_output(arr[i],scale=False,periodic_lon_expand=True)
            if i==0:
                training_set.depthind=depthind
                insc,outsc=training_set.compute_scales()
                insc,outsc=np.reshape(insc,[-1,1,1]),np.reshape(outsc,[-1,1,1])
                if residue_flag:
                    outsc=np.concatenate([outsc,outsc],axis=0)
            uv[:insc.shape[0]]=uv[:insc.shape[0]]/insc
            Sxy[:outsc.shape[0]]=Sxy[:outsc.shape[0]]/outsc
            uv[uv!=uv]=0
            Sxy[Sxy!=Sxy]=0
            uv,Sxy=dataset.pad_with_zero(uv,0),dataset.pad_with_zero(Sxy,dataset.spread)
        else:
            uv,_,Sxy=dataset[arr[i]]
        if residue_flag:
            Sxy1,Sxy=torch.split(Sxy,[numoutputs,numoutputs],dim=0)
        uv=torch.stack([uv]).to(device)
        #net.set_coarsening(0)
        with torch.set_grad_enabled(False):
            if isinstance(net, climate_models.RegressionModel):
                output=net.forward(uv,w)
            else:
                output=net.forward(uv)
        output=output[0].to(torch.device("cpu"))
        output,prec,_=torch.split(output,[noutsig,net.nprecision,output.shape[0]-noutsig-net.nprecision],dim=0)


        SC2[depthind]=SC2[depthind] + Sxy**2
        if residue_flag:
            MSE[depthind]=MSE[depthind] + (Sxy1-output)**2
        else:
            MSE[depthind]=MSE[depthind] + (Sxy-output)**2
        if di%10==0:
            MSE_=MSE.numpy()/(i+1)
            SC2_=SC2.numpy()/(i+1)
            
            with open(MSELOC, 'wb') as f:
                np.save(f, MSE_)
            with open(SC2LOC, 'wb') as f:
                np.save(f, SC2_)
            if not args.depth:
                print('\t #'+str(i),flush=True)
            else:
                print('\t depth# '+str(depthind)+', time# '+str(i),flush=True)
                
    MSE_=MSE.numpy()/len(dataset)
    SC2_=SC2.numpy()/len(dataset)

    with open(MSELOC, 'wb') as f:
        np.save(f, MSE_)
    with open(SC2LOC, 'wb') as f:
        np.save(f, SC2_)
    print('analysis is done',flush=True)


# In[6]:


def shift_geo_analysis(args):    
    net,criterion,(data_init,partition),logs,(PATH0,PATH1,LOG,root)=load_from_save(args)
    _,_,_,(dataset,datagen)=load_data(data_init,partition,args)
    
    MASK=climate_data.get_land_masks(datagen)[0,0]
    ymax=MASK.shape[0]
    xmax=MASK.shape[1]
    xx=torch.linspace(20,xmax-20,10,dtype=torch.long)
    yy=torch.linspace(20,ymax-20,10,dtype=torch.long)
    MM=MASK[yy,:]
    MM=MM[:,xx]
    ycord,xcord=np.where(MM>0)
    ycord=yy[ycord]
    xcord=xx[xcord]
    PTS=[[ycord[i].item(),xcord[i].item()] for i in range(len(xcord))]
    PTS=np.array(PTS)
    device=get_device()

    net.eval()

    R2LOC=root+'/R2-shift.npy'
    PTSLOC=root+'/PTS-shift.npy'
    spread=net.spread
    mm=(2*spread+1)
    MSE=torch.zeros(len(PTS),(dataset.dimens[0]-spread*2)//mm+1, (dataset.dimens[1]-spread*2)//mm+1)
    SC2=torch.zeros(len(PTS),(dataset.dimens[0]-spread*2)//mm+1, (dataset.dimens[1]-spread*2)//mm+1)
    coarsening=net.init_coarsen>0
    arr=np.arange(len(dataset))
    np.random.shuffle(arr)

    dd=0
    PTSMAP=MSE*0
    for ppi in range(len(PTS)):
        pp=PTS[ppi]
        PTSMAP[ppi,pp[0]//mm,pp[1]//mm]=1
        for i in range(len(arr)):
            uv,_,Sxy=dataset[arr[i]]
            geo=uv[2:]
            uv=uv[:2]
            UV=uv[:,pp[0]:pp[0]+mm,pp[1]:pp[1]+mm]

            #uv_=uv[:,pp[0]%mm:,pp[1]%mm:]
            uv=torch.roll(uv,shifts=(-(pp[0]%mm),-(pp[1]%mm)),dims=(1,2))
            geo=torch.roll(geo,shifts=(-(pp[0]%mm),-(pp[1]%mm)),dims=(1,2))
            for t in range(uv.shape[1]//mm):
                for tt in range(uv.shape[2]//mm):
                    uv[:,t*mm:(t+1)*mm,tt*mm:(tt+1)*mm]=UV
            uv=torch.cat([uv,geo],dim=0)
            Sxy=Sxy[:,pp[0],pp[1]].view(-1,1,1)
            uv=torch.stack([uv]).to(device)
            with torch.set_grad_enabled(False):
                outschedulerput=net.forward(uv)
            output=output[0].to(torch.device("cpu"))
            output,_=torch.split(output,[2,output.shape[0]-2],dim=0)
            output=output[:,::mm,::mm]
            SC2[ppi,:output.shape[1],:output.shape[2]]+= torch.sum((Sxy-output*0)**2,dim=0)
            MSE[ppi,:output.shape[1],:output.shape[2]]+= torch.sum((Sxy-output)**2,dim=0)
            dd+=1
            if dd%300==0:
                print(ppi,i)
                R2=1-MSE/SC2
                with open(R2LOC, 'wb') as f:
                    np.save(f, R2.numpy())
                with open(PTSLOC, 'wb') as f:
                    np.save(f, PTSMAP.numpy())
    R2=1-MSE/SC2
    with open(R2LOC, 'wb') as f:
        np.save(f, R2.numpy())
    with open(PTSLOC, 'wb') as f:
        np.save(f, PTSMAP.numpy())

def save_scales():
    sigmas=[2,4,6,8,12,16]
    scales=np.zeros((len(sigmas),3))
    M=50
    
    for i in range(len(sigmas)):
        sigma=sigmas[i]
        ds_data=xr.open_zarr('/scratch/cg3306/climate/data-read/data/sigma-'+str(sigma)+'-data.zarr')
        scales[i,0]=sigma
        usc=0
        uss=0
        Ts=np.random.random_integers(0,len(ds_data.time)-1,M)
        give_sc=lambda AA: np.mean(np.abs(AA[AA==AA]))
        for j in Ts:
            dsi=ds_data.isel(time=np.arange(j,j+1))
            u,v,S_x,S_y=dsi.usurf.values[0],dsi.vsurf.values[0],dsi.S_x.values[0],dsi.S_y.values[0]
            u=np.stack([u,v],axis=0)
            S=np.stack([S_x,S_y],axis=0)
            usc=np.maximum(usc,give_sc(u))
            uss=np.maximum(uss,give_sc(S))
        #usc=usc/M
        #uss=uss/M
        scales[i,1]=usc
        scales[i,2]=uss
        print(scales[i])
    np.save('/scratch/cg3306/climate/climate_research/scales.npy',scales)
def ismember_(a,b):
    for a_ in b:
        if a==a_:
            return True
    return False

def quadratic_fit(args):
    args.batch=1
    net,criterion,(data_init,partition),logs,(PATH0,PATH1,LOG,root)=load_from_save(args)
    
    (training_set,training_generator),(val_set,val_generator),(test_set,test_generator),(glbl_set,glbl_gen)=load_data(data_init,partition,args)
    landmasks=climate_data.get_land_masks(val_generator)
    M=np.load('/scratch/cg3306/climate/runs/'+str(args.model_bank_id)+'-'+str(args.model_id)+'/quadratic_M.npy')
    M=torch.tensor(M).type(torch.float)
    Mfit=M*0
    landmasks=landmasks.to(device)
    landmasks.requires_grad=False
    max_epochs = args.epoch
    if len(logs['lr'])==0:
        optimizer = optim.SGD(Mfit, lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.SGD(Mfit, lr=logs['lr'][-1], momentum=0.9)
        
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=5)
    
    best_counter=0
    print("epochs started")
    for epoch in range(max_epochs):
        logs['train-loss'].append([])
        #for local_batch,nan_mask, local_labels in training_generator:
        tt=0
        for local_batch,dom_num, _ in training_generator:
            local_batch, local_labels = local_batch.to(device),local_labels.to(device)
            outputs = net.forward(local_batch)
            if not net.generative:
                loss = criterion(outputs, local_labels,landmasks[dom_num])
            else:
                loss = torch.mean((local_batch-outputs)**2)
            logs['train-loss'][-1].append(loss.item())
            scheduler.optimizer.zero_grad()
            loss.backward()
            scheduler.optimizer.step()
            tt+=1
            if tt%args.disp==0:
                print('\t\t\t train-loss: ',str(np.mean(np.array(logs['train-loss'][-1]))),                      '\t ±',                      str(np.std(np.array(logs['train-loss'][-1]))),flush=True)
        with torch.set_grad_enabled(False):
            val_loss=0.
            num_val=0
            for local_batch,dom_num, ldeocal_labels in val_generator:
                local_batch,local_labels = local_batch.to(device),local_labels.to(device)
                #print('val: '+str(local_batch.shape))
                outputs = net.forward(local_batch)
                if not net.generative:
                    loss = criterion(outputs, local_labels,landmasks[dom_num])
                else:
                    loss = torch.mean((local_batch-outputs)**2)
                val_loss+=loss.item()
                num_val+=1
            logs['val-loss'].append(val_loss/num_val)
        logs['lr'].append(scheduler.optimizer.param_groups[0]['lr'])
        scheduler.step(logs['val-loss'][-1])
        with torch.set_grad_enabled(False):
            val_loss=0.
            num_val=0
            for local_batch,dom_num, local_labels in test_generator:
                local_batch,local_labels = local_batch.to(device), local_labels.to(device)
                outputs = net.forward(local_batch)
                if not net.generative:
                    loss = criterion(outputs, local_labels,landmasks[dom_num])
                else:
                    loss = torch.mean((local_batch-outputs)**2)
                val_loss+=loss.item()
                num_val+=1
            logs['test-loss'].append(val_loss/num_val)
            logs['batchsize'].append(args.batch)
        
        if len(logs['epoch'])>0:
            logs['epoch'].append(logs['epoch'][-1]+1)
        else:
            logs['epoch'].append(1)
    
        print('#epoch ',str(logs['epoch'][-1]),' ',                  ' test-loss: ',str(logs['test-loss'][-1]),                  ' val-loss: ',str(logs['val-loss'][-1]),                  ' train-loss: ',str(np.mean(np.array(logs['train-loss'][-1]))),                  ' lr: ',str(logs['lr'][-1]),flush=True)
        
        
        torch.save(net.state_dict(), PATH0)
        with open(LOG, 'w') as outfile:
            json.dump(logs, outfile)
        if np.min(logs['test-loss']) == logs['test-loss'][-1]:
            torch.save(net.state_dict(), PATH1)
            best_counter=0
        else:
            best_counter+=1
        if logs['lr'][-1]<1e-7:
            break


# In[7]:


'''args=options(string_input=    "--b 2 -e 4 --nworkers 10 --subtime 0.1 --lr 0.01 --model_id 0 --model_bank_id 4".split())'''


# In[8]:


def error_analysis(args):
        net,criterion,logs,PATH0,PATH1,LOG,root=load_from_save(args)
        net.set_coarsening(0)
        (_,training_generator),(_,_),(_,test_generator),_=load_data(net,args)
        device=get_device()
        net.eval()
        logs['total-err']=np.zeros(2)
        tot=0
 
        for local_batch,nan_mask, local_labels in training_generator:
            local_batch, nan_mask, local_labels = local_batch.to(device),nan_mask.to(device), local_labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = net.forward(local_batch)*nan_mask
            outputs,_=torch.split(outputs,[2,outputs.shape[1]-2],dim=1)
            err=outputs-local_labels
            logs['total-err'][0]+=torch.sum(err**2).item()
            err=err*nan_mask+(1-nan_mask)*1e9
            err=err.to(torch.device("cpu"))
            err=err.numpy()
            tot+=np.sum( (np.abs(err)<1e9).astype(int))
        
        logs['total-err'][0]=logs['total-err'][0]/tot
        tot=0
        
        
        for local_batch,nan_mask, local_labels in test_generator:
            local_batch, nan_mask, local_labels = local_batch.to(device),nan_mask.to(device), local_labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = net.forward(local_batch)*nan_mask
            outputs,_=torch.split(outputs,[2,outputs.shape[1]-2],dim=1)
            err=outputs-local_labels
            logs['total-err'][1]+=torch.sum(err**2).item()
            err=err*nan_mask+(1-nan_mask)*1e9
            err=err.to(torch.device("cpu"))
            err=err.numpy()
            tot+=np.sum( (np.abs(err)<1e9).astype(int))
            
        
        logs['total-err'][1]=logs['total-err'][1]/tot
        logs['total-err']=logs['total-err'].tolist()
        with open(LOG, 'w') as outfile:
            json.dump(logs, outfile)
            
        coarsening=net.init_coarsen>0
        if not coarsening:
            return 
        net.initial_coarsening()
        (_,training_generator),(_,_),(_,test_generator),_,ds_zarr=load_data(net,args)
        
        logs['total-err-coarse']=np.zeros(2)
        tot=0    
        for local_batch,nan_mask, local_labels in training_generator:
            local_batch, nan_mask, local_labels = local_batch.to(device),nan_mask.to(device), local_labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = net.forward(local_batch)*nan_mask
            outputs,_=torch.split(outputs,[2,outputs.shape[1]-2],dim=1)
            err=outputs-local_labels
            logs['total-err-coarse'][0]+=torch.sum(err**2).item()
            err=err*nan_mask+(1-nan_mask)*1e9
            err=err.to(torch.device("cpu"))
            err=err.numpy()
            tot+=np.sum( (np.abs(err)<1e9).astype(int))
        
        
        logs['total-err-coarse'][0]=logs['total-err-coarse'][0]/tot
        #hist=np.zeros(len(edges)-1)
        tot=0
        
        for local_batch,nan_mask, local_labels in test_generator:
            local_batch, nan_mask, local_labels = local_batch.to(device),nan_mask.to(device), local_labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = net.forward(local_batch)*nan_mask
            outputs,_=torch.split(outputs,[2,outputs.shape[1]-2],dim=1)
            err=outputs-local_labels
            logs['total-err-coarse'][1]+=torch.sum(err**2).item()
            err=err*nan_mask+(1-nan_mask)*1e9
            err=err.to(torch.device("cpu"))
            err=err.numpy()
            tot+=np.sum( (np.abs(err)<1e9).astype(int))
            
        
        logs['total-err-coarse'][1]=logs['total-err-coarse'][1]/tot
        logs['total-err-coarse']=logs['total-err-coarse'].tolist()
        with open(LOG, 'w') as outfile:
            json.dump(logs, outfile)
        


# In[9]:


def quadratic_model_matrix(args):
    print('loading net')
    net,criterion,(data_init,partition),logs,(PATH0,PATH1,LOG,root)=load_from_save(args)
    net.eval()
    m=net.spread*2+1
    if isinstance(net,climate_models.CQCNN):
        ncl=net.classnum
        M=torch.zeros(ncl,2,2*m*m,2*m*m)
        b=torch.zeros(ncl,2,2*m*m)
        for j in range(ncl):
            uv=torch.zeros(1,2,m,m).to(get_device())
            print('filling b-'+str(j))
            b[j,0,:]=torch.reshape(give_dev_wrt_input(net,m,dim=0,class_index=j,index=-1),[2*m*m])
            b[j,1,:]=torch.reshape(give_dev_wrt_input(net,m,dim=1,class_index=j,index=-1),[2*m*m])
            print('filling c')
            #c[j]=give_dev_wrt_input(net,m,class_index=j,constant=True)#torch.reshape(net(uv).detach(),[-1])
            uv.data.zero_()
            print('filling M-'+str(j))
            for i in range(M.shape[2]):
                M[j,0,i,:]=(torch.reshape(give_dev_wrt_input(net,m,dim=0,class_index=j,index=i),[2*m*m])-b[j,0])/2
                M[j,1,i,:]=(torch.reshape(give_dev_wrt_input(net,m,dim=1,class_index=j,index=i),[2*m*m])-b[j,1])/2
        print('\t saving')
        with open(root+'/quadratic_M.npy', 'wb') as f:
            np.save(f, M.numpy())
        with open(root+'/quadratic_b.npy', 'wb') as f:
            np.save(f, b.numpy())
        '''with open(root+'/quadratic_c.npy', 'wb') as f:
            np.save(f, c.numpy())  '''
    else:
        ncl=-1
        M=torch.zeros(2,2*m*m,2*m*m)
        b=torch.zeros(2,2*m*m)
        uv=torch.zeros(1,2,m,m).to(get_device())
        print('filling b')
        b[0,:]=torch.reshape(give_dev_wrt_input(net,m,dim=0,index=-1),[2*m*m])
        b[1,:]=torch.reshape(give_dev_wrt_input(net,m,dim=1,index=-1),[2*m*m])
        print('filling c')
        uv.data.zero_()
        print('filling M')
        for i in range(M.shape[2]):
            M[0,i,:]=(torch.reshape(give_dev_wrt_input(net,m,dim=0,index=i),[2*m*m])-b[0])/2
            M[1,i,:]=(torch.reshape(give_dev_wrt_input(net,m,dim=1,index=i),[2*m*m])-b[1])/2
        print('\t saving')
        with open(root+'/quadratic_M.npy', 'wb') as f:
            np.save(f, M.numpy())
        with open(root+'/quadratic_b.npy', 'wb') as f:
            np.save(f, b.numpy())

    
    #c=torch.zeros(ncl)
    

def give_dev_wrt_input(net,m,dim=0,index=0,class_index=-1,constant=False):
    i=index
    uv=torch.zeros(1,2,m,m)
    if constant:
        if class_index<0:
            y=net.forward(uv)
        else:
            y=net.quadratic_forward(uv,class_index=class_index)
        return y
    if i>=0:
        I=np.unravel_index(i, [2,m,m])
        uv[0,I[0],I[1],I[2]]=1
    #uv=uv.to(get_device())
    uv.requires_grad_(True)
    uv.grad=None
    if class_index<0:
        y=net.forward(uv)
    else:
        y=net.quadratic_forward(uv,class_index=class_index)
    y=torch.reshape(y,[-1])
    
    y[dim].backward()
    g=uv.grad
    return g


# In[10]:


def main():
    today = date.today()
    print("Today's date:", today,flush=True)
    args=options()
    print(args)
    if args.action=="train":
        train(args)
    if args.action=="analysis":
        analysis(args)
    if args.action=="binned-r2":
        binned_r2_analysis(args)
    if args.action=="quadratic":
        quadratic_model_matrix(args)
    if args.action=="error-analysis":
        error_analysis(args)
    if args.action=="grad-analysis":
        grad_analysis(args)
    if args.action=="data-cov-analysis":
        data_covariance(args)
    if args.action=="fourier-analysis":
        data_fourier_analysis(args)
    if args.action=="shift-geo-analysis":
        shift_geo_analysis(args)
    if args.action=="grad-probe":
        grad_probe(args)
    if args.action=="prep":
        prep(args)
    if args.action=="projection-analysis":
        projection_analysis(args)
    if args.action=="global-averages":
        global_averages()
if __name__=='__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




