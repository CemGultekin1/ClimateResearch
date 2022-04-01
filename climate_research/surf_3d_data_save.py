#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import intake
from intake import open_catalog
from matplotlib import colors, cm, pyplot as plt
def advections(u_v_field: xr.Dataset, grid_data: xr.Dataset,typenum=0):
    dxu = grid_data['dxu']
    dyu = grid_data['dyu']
    gradient_x = u_v_field.diff(dim='xu_ocean') / dxu
    gradient_y = u_v_field.diff(dim='yu_ocean') / dyu
    # Interpolate back the gradients
    interp_coords = dict(xu_ocean=u_v_field.coords['xu_ocean'],
                         yu_ocean=u_v_field.coords['yu_ocean'])
    gradient_x = gradient_x.interp(interp_coords)
    gradient_y = gradient_y.interp(interp_coords)
    u, v = u_v_field['usurf'], u_v_field['vsurf']
    
    adv = u * gradient_x + v * gradient_y
    names=list(u_v_field.keys())
    
    adv_u = u * gradient_x['usurf'] + v * gradient_y['usurf']
    adv_v = u * gradient_x['vsurf'] + v * gradient_y['vsurf']
    adv_T = u * gradient_x['surface_temp'] + v * gradient_y['surface_temp']
    result = xr.Dataset({'Su': adv_u,'Sv': adv_v,'ST': adv_T })
    
    return result
def spatial_filter(data: np.ndarray, sigma: float):
    
    ndim=len(data.shape)
    #print(ndim)
    if ndim==3:
        result = np.zeros_like(data)
        for t in range(data.shape[0]):
            data_t = data[t, ...]
            result_t = gaussian_filter(data_t, sigma, mode='constant')
            result[t, ...] = result_t
    elif ndim==4:
        preshape=data.shape
        data=data.reshape([preshape[0]*preshape[1],preshape[2],preshape[3]])
        result = np.zeros(data.shape)
        for t in range(data.shape[0]):
            data_t = data[t, ...]
            result_t = gaussian_filter(data_t, sigma, mode='constant')
            result[t, ...] = result_t
        result=result.reshape([preshape[0],preshape[1],result.shape[1],result.shape[2]])
    return result

def spatial_filter_dataset(dataset: xr.Dataset, grid_info: xr.Dataset,
                           sigma: float):
    area_u = grid_info['dxu'] * grid_info['dyu'] / 1e8
    dataset = dataset * area_u
    # Normalisation term, so that if the quantity we filter is constant
    # over the domain, the filtered quantity is constant with the same value
    norm = xr.apply_ufunc(lambda x: gaussian_filter(x, sigma, mode='constant'),
                          area_u, dask='parallelized', output_dtypes=[float, ])
    filtered = xr.apply_ufunc(lambda x: spatial_filter(x, sigma), dataset,
                              dask='parallelized', output_dtypes=[float, ])
    return filtered / norm

def eddy_forcing(u_v_dataset : xr.Dataset, grid_data: xr.Dataset,
                 scale: int, method: str = 'mean',
                 nan_or_zero: str = 'zero', scale_mode: str = 'factor',
                 debug_mode=False,typenum=0) -> xr.Dataset:

    # Replace nan values with zeros.
    if nan_or_zero == 'zero':
        u_v_dataset = u_v_dataset.fillna(0.0)
    if scale_mode == 'factor':
        #print('Using factor mode')
        scale_x = scale
        scale_y = scale

    scale_filter = (scale_x / 2, scale_y / 2)
    # High res advection terms
    adv = advections(u_v_dataset, grid_data,typenum=typenum)
    # Filtered advections
    filtered_adv = spatial_filter_dataset(adv, grid_data, scale_filter)
    # Filtered u,v field and temperature
    u_v_filtered = spatial_filter_dataset(u_v_dataset, grid_data, scale_filter)
    # Advection term from filtered velocity field
    adv_filtered = advections(u_v_filtered, grid_data,typenum=typenum)
    # Forcing
    forcing = adv_filtered - filtered_adv
    #if typenum==0:
    #forcing = forcing.rename({'adv': 'S'})
    # Merge filtered u,v, temperature and forcing terms
    forcing = forcing.merge(u_v_filtered)
    # Coarsen
    #print('scale factor: ', scale)
    forcing_coarse = forcing.coarsen({'xu_ocean': int(scale_x),
                                      'yu_ocean': int(scale_y)},
                                     boundary='trim')
    if method == 'mean':
        forcing_coarse = forcing_coarse.mean()
    else:
        raise ValueError('Passed coarse-graining method not implemented.')
    if nan_or_zero == 'zero':
        # Replace zeros with nans for consistency
        forcing_coarse = forcing_coarse.where(forcing_coarse['usurf'] != 0)
    if not debug_mode:
        return forcing_coarse
    u_v_dataset = u_v_dataset.merge(adv)
    filtered_adv = filtered_adv.rename({'adv_x': 'f_adv_x',
                                        'adv_y': 'f_adv_y'})
    adv_filtered = adv_filtered.rename({'adv_x': 'adv_f_x',
                                        'adv_y': 'adv_f_y'})
    u_v_filtered = u_v_filtered.rename({'usurf': 'f_usurf',
                                        'vsurf': 'f_vsurf'})
    u_v_dataset = xr.merge((u_v_dataset, u_v_filtered, adv, filtered_adv,
                            adv_filtered, forcing[['S_x', 'S_y']]))
    return u_v_dataset, forcing_coarse

def coarsen_save(sigma,typenum,testflag=False,projection=False,rewrite=False):
    if typenum==0:
        raw_data_address='/scratch/zanna/data/cm2.6/surf-data.zarr'
        u_v_dataset=xr.open_zarr(raw_data_address).chunk(chunks={"time":1})
        if not testflag:
            filename='/scratch/zanna/data/cm2.6/coarse-surf-data-sigma-'+str(sigma)+'.zarr'
        else:
            filename='/scratch/zanna/data/cm2.6/coarse-surf-data-sigma-'+str(sigma)+'-test.zarr'
        if projection:
            QQx,QQy,_,_=compute_projections(sigma,stratind=0)
    elif typenum==1:
        raw_data_address='/scratch/zanna/data/cm2.6/3D-data.zarr'
        #stratinds=np.array([0,10,20,30])
        stratinds=np.array([0,5,10,15,20,30,40])
        #stratinds=10#np.array([10])
        u_v_dataset=xr.open_zarr(raw_data_address)                        .chunk(chunks={"time":1})                        .isel(st_ocean=stratinds)                        .drop("salt nv st_edges_ocean xt_ocean yt_ocean".split())                        .rename({"u":"usurf","v":"vsurf","temp":"surface_temp"})
        stratvals=u_v_dataset.st_ocean.values#np.concatenate([[0],u_v_dataset.st_ocean.values],axis=(0))
        if not testflag:
            filename='/scratch/zanna/data/cm2.6/coarse-3D-data-sigma-'+str(sigma)+'.zarr'
        else:
            filename='/scratch/zanna/data/cm2.6/coarse-3D-data-sigma-'+str(sigma)+'-test.zarr'
        if projection:
            QQx,QQy,_,_=compute_projections(sigma,stratind=1)
    elif typenum==2:
        raw_data_address='/scratch/zanna/data/cm2.6/1pct-CO2-surf-data.zarr'
        u_v_dataset=xr.open_zarr(raw_data_address)                        .chunk(chunks={"time":1})
        if not testflag:
            filename='/scratch/zanna/data/cm2.6/coarse-1pct-CO2-surf-data-sigma-'+str(sigma)+'.zarr'
        else:
            filename='/scratch/zanna/data/cm2.6/coarse-1pct-CO2-surf-data-sigma-'+str(sigma)+'-test.zarr'
        if projection:
            QQx,QQy,_,_=compute_projections(sigma,raw_data_address=raw_data_address)
    elif typenum==3:
        raw_data_address='/scratch/zanna/data/cm2.6/1pct-CO2-3D-data.zarr'
        stratinds=np.array([0,5,10,15,20,30,40])
        u_v_dataset=xr.open_zarr(raw_data_address)                        .chunk(chunks={"time":1})                        .isel(st_ocean=stratinds)                        .drop("nv st_edges_ocean xt_ocean yt_ocean".split())                        .rename({"u":"usurf","v":"vsurf","temp":"surface_temp"})
        stratvals=u_v_dataset.st_ocean.values
        if not testflag:
            filename='/scratch/zanna/data/cm2.6/coarse-1pct-CO2-3D-data-sigma-'+str(sigma)+'.zarr'
        else:
            filename='/scratch/zanna/data/cm2.6/coarse-1pct-CO2-3D-data-sigma-'+str(sigma)+'-test.zarr'
        if projection:
            QQx,QQy,_,_=compute_projections(sigma,stratind=1,raw_data_address=raw_data_address)
    if testflag:
        print('running test')
        u_v_dataset=u_v_dataset.isel(time=np.arange(4))
    nb=sigma*4

    x=u_v_dataset.xu_ocean.values
    y=u_v_dataset.yu_ocean.values
    dx=x[1]-x[0]
    xbeg=x[0]
    xter=x[-1]
    x=np.concatenate([dx*np.arange(-nb,0)+xbeg,x,dx*np.arange(1,nb+1)+xter],axis=0)
    dx=x[1:]-x[:-1]
    dy=y[1:]-y[:-1]

    dx=np.reshape(dx,(1,-1))
    dx=[dx for i in range(len(y)-1)]
    dx=np.concatenate(dx,axis=0)
    #dx=np.stack([dx],axis=0)

    dy=np.reshape(dy,(-1,1))
    dy=[dy for i in range(len(x)-1)]
    dy=np.concatenate(dy,axis=1)
    grid_data=xr.Dataset(data_vars=dict(dxu=(["yu_ocean","xu_ocean"],dx),                                        dyu=(["yu_ocean","xu_ocean"],dy)),                            coords=dict(xu_ocean=x[:-1],yu_ocean=y[:-1]))
    print('saving '+str(sigma)+'...',flush=True)
    print(filename)
    tmax=len(u_v_dataset.time)
    try:
        ds=xr.open_zarr(filename)
        tmin=ds.time.values[-1]
    except:
        tmin=0
    if testflag:
        tmin=0
    if rewrite:
        tmin=0
    print("tmin: "+str(tmin)+",  tmax: "+str(tmax))
    time_chunk=sigma
    print("time_chunksize: "+str(time_chunk))

    no_depth=typenum%2==0
    for j in range(tmin//time_chunk,tmax//time_chunk):
        ter=min(tmax,(j+1)*time_chunk)
        print('\t\t'+str(j*time_chunk)+' : '+ str(tmax),flush=True)
        times_=np.arange(j*time_chunk,ter)
        for times__ in times_:
            times=np.arange(times__,times__+1)
            u_v_datasetj=u_v_dataset.isel(time=times)
            if no_depth:
                u=np.stack([u_v_datasetj.usurf.values,                            u_v_datasetj.vsurf.values,                              u_v_datasetj.surface_temp.values],axis=0)
            else:
                u=np.concatenate([u_v_datasetj.usurf.values,                            u_v_datasetj.vsurf.values,                              u_v_datasetj.surface_temp.values],axis=0)
            if projection:
                uL=u*1
                uH=u*1
                for t in range(u.shape[0]*u.shape[1]):
                    t1=t%u.shape[0]
                    t2=t//u.shape[0]
                    ut=u[t1,t2]*1
                    mask=ut*1
                    mask[mask==mask]=1
                    ut[ut!=ut]=0
                    u0=(QQx@(QQx.T@ut)@QQy)@QQy.T
                    uL[t1,t2]=u0*mask
                    uH[t1,t2]=(ut-u0)*mask
                uL=np.concatenate([uL[:,:,:,-nb:],uL,uL[:,:,:,:nb]],axis=3)
                uH=np.concatenate([uH[:,:,:,-nb:],uH,uH[:,:,:,:nb]],axis=3)
            u=np.concatenate([u[:,:,:,-nb:],u,u[:,:,:,:nb]],axis=3)
            if no_depth:
                uv=xr.Dataset(data_vars=dict(usurf=(["time","yu_ocean","xu_ocean"],u[0]),                                                vsurf=(["time","yu_ocean","xu_ocean"],u[1]),                                                 surface_temp=(["time","yu_ocean","xu_ocean"],u[2])),                                    coords=dict(xu_ocean=x,yu_ocean=y,time=times))  
                if projection:
                    uvL=xr.Dataset(data_vars=dict(usurf=(["time","yu_ocean","xu_ocean"],uL[0]),                                            vsurf=(["time","yu_ocean","xu_ocean"],uL[1]),                                             surface_temp=(["time","yu_ocean","xu_ocean"],uL[2])),                                coords=dict(xu_ocean=x,yu_ocean=y,time=times))
                    uvH=xr.Dataset(data_vars=dict(usurf=(["time","yu_ocean","xu_ocean"],uH[0]),                                            vsurf=(["time","yu_ocean","xu_ocean"],uH[1]),                                             surface_temp=(["time","yu_ocean","xu_ocean"],uH[2])),                                coords=dict(xu_ocean=x,yu_ocean=y,time=times))
            else:
                uv=xr.Dataset(data_vars=dict(usurf=(["time","st_ocean","yu_ocean","xu_ocean"],u[0:1]),                                    vsurf=(["time","st_ocean","yu_ocean","xu_ocean"],u[1:2]),                                     surface_temp=(["time","st_ocean","yu_ocean","xu_ocean"],u[2:3])),                        coords=dict(st_ocean=stratvals,xu_ocean=x,yu_ocean=y,time=times))  
                if projection:
                    uvL=xr.Dataset(data_vars=dict(usurf=(["time","st_ocean","yu_ocean","xu_ocean"],uL[0:1]),                                            vsurf=(["time","st_ocean","yu_ocean","xu_ocean"],uL[1:2]),                                             surface_temp=(["time","st_ocean","yu_ocean","xu_ocean"],uL[2:3])),                                coords=dict(st_ocean=stratvals,xu_ocean=x,yu_ocean=y,time=times))
                    uvH=xr.Dataset(data_vars=dict(usurf=(["time","st_ocean","yu_ocean","xu_ocean"],uH[0:1]),                                            vsurf=(["time","st_ocean","yu_ocean","xu_ocean"],uH[1:2]),                                             surface_temp=(["time","st_ocean","yu_ocean","xu_ocean"],uH[2:3])),                                coords=dict(st_ocean=stratvals,xu_ocean=x,yu_ocean=y,time=times))
            forcing1=eddy_forcing(uv, grid_data,sigma,nan_or_zero='nan',typenum=typenum).sel(yu_ocean=slice(-85, 85))                .sel(xu_ocean=slice(xbeg,xter))
            if projection:
                forcingL=eddy_forcing(uvL, grid_data,sigma,nan_or_zero='nan',typenum=typenum).sel(yu_ocean=slice(-85, 85))                    .sel(xu_ocean=slice(xbeg,xter))
                forcingH=eddy_forcing(uvH, grid_data,sigma,nan_or_zero='nan',typenum=typenum).sel(yu_ocean=slice(-85, 85))                    .sel(xu_ocean=slice(xbeg,xter))
                nms=['Su','Sv','ST']
                for nm in nms:
                    forcingL[nm].values=forcing1[nm].values-forcingL[nm].values-forcingH[nm].values
                    forcingH[nm].values=forcingL[nm].values+forcingH[nm].values
                forcingL=forcingL.drop("usurf vsurf surface_temp".split()).rename({'Su': 'Su_LH',
                                                'Sv': 'Sv_LH','ST':'ST_LH'})
                forcingH=forcingH.drop("usurf vsurf surface_temp".split()).rename({'Su': 'Su_r',
                                                'Sv': 'Sv_r','ST':'ST_r'})
                forcing1=forcing1.merge(forcingL)
                forcing1=forcing1.merge(forcingH)
            if times__==times_[0]:
                forcing=forcing1.copy(deep=True)
            else:
                forcing=forcing.merge(forcing1)
            print('\t\t\t\t'+str((times__+1)/len(times_)),flush=True)
        forcing=forcing.chunk(chunks={"time":len(forcing.time),"xu_ocean":len(forcing.xu_ocean),"yu_ocean":len(forcing.yu_ocean)})
        if j==0:
            forcing.to_zarr(filename,mode='w')
        else:
            forcing.to_zarr(filename,append_dim="time",mode='a')
    print('\t\t '+str(sigma)+' is done',flush=True)
def compute_projections(sigma,stratind=-1,raw_data_address='/scratch/zanna/data/cm2.6/3D-data.zarr'):
    u_v_dataset=xr.open_zarr(raw_data_address)
    u_v_dataset=u_v_dataset.isel(time=np.arange(1)).drop("time".split())
    if stratind>=0:
        u_v_dataset=u_v_dataset.isel(st_ocean=stratind)                    .drop("nv st_edges_ocean xt_ocean yt_ocean st_ocean".split())
        u_v_dataset=u_v_dataset['u'.split()]
    else:
        u_v_dataset=u_v_dataset.rename({'usurf':'u','vsurf':'v','surface_temp':'temp'})
    nb=sigma*4
    scale_x,scale_y=sigma,sigma
    scale_filter = (scale_x / 2, scale_y / 2)

    x=u_v_dataset.xu_ocean.values
    y=u_v_dataset.yu_ocean.values
    dx=x[1]-x[0]
    xbeg=x[0]
    xter=x[-1]
    x=np.concatenate([dx*np.arange(-nb,0)+xbeg,x,dx*np.arange(1,nb)+xter],axis=0)
    y=u_v_dataset.yu_ocean.values
    dx=x[1:]-x[:-1]
    dy=y[1:]-y[:-1]
    dy=np.concatenate([dy,dy[-1:]])
    dx=np.concatenate([dx,dx[-1:]])

    dx=np.reshape(dx,(1,-1))
    dx=[dx for i in range(len(y))]
    dx=np.concatenate(dx,axis=0)

    dy=np.reshape(dy,(-1,1))
    dy=[dy for i in range(len(x))]
    dy=np.concatenate(dy,axis=1)

    grid_data=xr.Dataset(data_vars=dict(dxu=(["yu_ocean","xu_ocean"],dx),                                        dyu=(["yu_ocean","xu_ocean"],dy)),                            coords=dict(xu_ocean=x,yu_ocean=y))


    x=u_v_dataset.xu_ocean.values.reshape([-1])
    y=u_v_dataset.yu_ocean.values.reshape([-1])
    dx=x[1]-x[0]
    xbeg=x[0]
    xter=x[-1]
    x=np.concatenate([dx*np.arange(-nb,0)+xbeg,x,dx*np.arange(1,nb)+xter],axis=0)
    dx=x[1:]-x[:-1]
    dy=y[1:]-y[:-1]
    dy=np.concatenate([dy,dy[-1:]])
    dx=np.concatenate([dx,dx[-1:]])

    
    ny=u_v_dataset.u.shape[-2]
    area_u=dy
    norm=gaussian_filter(area_u, scale_filter[0], mode='constant').reshape([-1])
    for i in range(ny):
        uy=np.zeros(ny)
        uy[i]=1
        filtered=gaussian_filter(uy, sigma/2, mode='constant').reshape([-1])/norm
        temp=xr.Dataset(data_vars=dict(u=(["x"],filtered)),                            coords=dict(x=np.arange(filtered.shape[0]))).coarsen({'x': sigma},
             boundary='trim').mean()
        cuy=temp.u.values
        if i==0:
            Qy=np.zeros((ny,len(cuy)))  
        Qy[i]=cuy


    nx=u_v_dataset.u.shape[-1]
    nb=sigma*4
    for i  in range(nx):
        ux=np.zeros(nx)
        ux[i]=1
        ux=np.concatenate([ux[-nb:],ux,ux[:nb]],axis=0)
        filtered=gaussian_filter(ux, scale_filter[1], mode='constant').reshape([-1])
        temp=xr.Dataset(data_vars=dict(u=(["x"],filtered)),                            coords=dict(x=np.arange(filtered.shape[0]))).coarsen({'x': sigma},
             boundary='trim').mean()
        cux=temp.u.values
        if i==0:
            Qx=np.zeros((nx,len(cux)))  
        Qx[i]=cux

    [QQx,_]=np.linalg.qr(Qx,mode='reduced')
    [QQy,_]=np.linalg.qr(Qy,mode='reduced')
    return QQy,QQx,Qx,Qy
def compute_quadratic_form(yc,dy,dx,sigma):
    sigma2=sigma//2
    s=sigma*4+1
    m=sigma*5+1
    n=sigma*6+1
    yh=yc*sigma-4*sigma2

    u=np.zeros(s)
    u[sigma*2]=1
    g=gaussian_filter(u,sigma2,mode='constant')

    gy=np.zeros(m)
    gx=np.zeros(m)
    
    gy_1=np.zeros(m)
    gx_1=np.zeros(m)
    
    dgy=np.zeros(n)
    dgx=np.zeros(n)

    yh_1=yh-sigma
    for i in range(sigma):
        gx[i:i+s]+=g/np.sum(g)/sigma
        gy[i:i+s]+=g*dy[yh+i:yh+i+s]/np.sum(g*dy[yh+i:yh+i+s])/sigma
        gx_1[i:i+s]+=g/np.sum(g)/sigma
        gy_1[i:i+s]+=g*dy[yh_1+i:yh_1+i+s]/np.sum(g*dy[yh_1+i:yh_1+i+s])/sigma
    
    dgy[:m]=-gy_1
    dgy[-m:]+=gy
    dgy=dgy/np.sum(dy[yh_1:yh])
    
    dgx[:m]=-gx_1
    dgx[-m:]+=gx
    dgx=dgx/dx
    
    n1=n+1
    Kx=np.zeros((m**2,n1**2))
    Ky=np.zeros((m**2,n1**2))
    for i in range(m**2):
        iy=i//m
        ix=i%m
        gi=gy[iy]*gx[ix]
        for j in range(n1**2):
            jy=j//n1
            jx=j%n1
            jx_1=jx-sigma
            jy_1=jy-sigma
            if jx_1>=0 and jx_1<m and jy<m :
                gjx_1=gy[jy]*gx[jx_1]/dx
            else:
                gjx_1=0
            if jy<m and jx<m:
                gj=gy[jy]*gx[jx]/dx
            else:
                gj=0
            K1=gi*(gjx_1-gj)

            if jy_1>=0 and jy_1<m and jx<m:
                gjy_1=gy[jy_1]*gx[jx]/dy[yh+jy_1]
            else:
                gjy_1=0
            if jy<m and jx<m:
                gj=gy[jy]*gx[jx]/dy[yh+jy]
            else:
                gj=0
            K2=gi*(gjy_1-gj)


            if iy==jy and ix==jx-1:
                K3=gi/dx
            elif iy==jy and ix==jx:
                K3=-gi/dx
            else:
                K3=0

            if ix==jx and iy==jy-1:
                K4=gi/dy[iy+yh]
            elif ix==jx and iy==jy:
                K4=-gi/dy[iy+yh]
            else:
                K4=0

            Kx[i,j]=K1-K3
            Ky[i,j]=K2-K4
    return Kx,Ky
def download_raw(typenum):
    if typenum==0:
        cat = open_catalog("https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/ocean/GFDL_CM2.6.yaml")
        ds=cat["GFDL_CM2_6_control_ocean_surface"].to_dask()
        ds=ds.drop(                    'biomass_p chl dic htotal irr_mix kw o2 po4 sea_level sea_level_sq surface_salt nv st_ocean_sub01 xt_ocean yt_ocean'.split()                       )
        filename='/scratch/zanna/data/cm2.6/surf-data.zarr'
        print('saving 3d data',flush=True)
    else:
        cat = open_catalog("https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/ocean/GFDL_CM2.6.yaml")
        ds  = cat["GFDL_CM2_6_control_ocean_3D"].to_dask()
        filename='/scratch/zanna/data/cm2.6/3D-data.zarr'
    save_fun(ds,filename)
    
    if typenum==0:
        print('\t saved 3d data successfully',flush=True)
    else:
        print('\t saved surf data successfully',flush=True)
def options(string_input=[]):
    parser=argparse.ArgumentParser()
    parser.add_argument("-s","--sigma",type=int,default=0)
    parser.add_argument("-t","--type",type=int,default=0)
    parser.add_argument("--test",type=int,default=0)
    parser.add_argument("-p","--projection",type=int,default=0)
    parser.add_argument("-b","--batch",type=int,default=4)
    parser.add_argument("--clean3d",type=int,default=0)
    parser.add_argument("--rewrite",type=int,default=0)
    if len(string_input)==0:
        return parser.parse_args()
    else:
        return parser.parse_args(string_input)


# In[2]:


'''
sigma=args.sigma
typenum=args.type
testflag=bool(args.test)
projection=bool(args.projection)'''


'''args=options("-s 4 -t 1 --test 1 -p 1".split())
coarsen_save(args.sigma,\
             args.type,\
             testflag=bool(args.test),\
             projection=bool(args.projection))'''


# In[3]:


def visualize_residues(dimens=[30,40,-50,-35],lognorm=False,d1=10,d2=20):
    Z0=xr.open_zarr('/scratch/zanna/data/cm2.6/coarse-surf-data-sigma-4.zarr').isel(time=0)
    nrows=3
    ncols=3
    if len(dimens)==0:
        ZZ=Z0#.sel(xu_ocean=slice(30,40),yu_ocean=slice(-45,-30))
    else:
        ZZ=Z0.sel(xu_ocean=slice(dimens[0],dimens[1]),yu_ocean=slice(dimens[2],dimens[3]))
    xx=ZZ.xu_ocean.values
    yy=ZZ.yu_ocean.values[::-1]
    fig,axs=plt.subplots(nrows,ncols,figsize=(d2,d1))
    vrnames=['Su','Sv','ST']
    for ii in range(nrows):
        vrname=vrnames[ii]
        forcing=['','_r']
        titles=['','_0','_r']
        titles=[vrname+tt for tt in titles]
        forcing=[vrname+ff for ff in forcing]
        Sr=ZZ[forcing[1]].values
        S=ZZ[forcing[0]].values
        S=[S,S-Sr,Sr]
        for j in range(len(S)):
            jj=j
            ax=axs[ii,jj]
            SIJ=S[j][::-1]
            smax=np.amax(np.abs(SIJ[SIJ==SIJ]))
            if lognorm:
                SIJ=np.log10(np.abs(SIJ))
                smax=np.log10(smax)
                neg=ax.imshow(SIJ,vmin=-4,vmax=smax,cmap='PuOr',extent=[xx[0],xx[-1],yy[-1],yy[0]])
            else:
                neg=ax.imshow(SIJ,vmin=-smax,vmax=smax,cmap='PuOr',extent=[xx[0],xx[-1],yy[-1],yy[0]])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(neg,cax=cax)
            if lognorm:
                ax.set_title('log10 |'+titles[j]+'|')
            else:
                ax.set_title(titles[j]+' (1e-14 m2s-4)')
def adjust_reflections(Qx,sigma):
    nb=4*sigma
    Qx_=Qx[nb:-nb,:]*1
    Qx1=Qx[:nb,:]
    Qx1=Qx1[::-1,:]
    Qx2=Qx[-nb:,:]
    Qx2=Qx2[::-1,:]
    Qx_[:nb,:]+=Qx1
    Qx_[-nb:,:]+=Qx2
    return Qx_            


# In[4]:


'''visualize_residues(lognorm=True,dimens=[],d1=10,d2=20)'''


# In[5]:


'''sigmas=[4,8,12,16]
QXS=[]
QYS=[]
for i in range(len(sigmas)):
    sigma=sigmas[i]
    QQx,QQy,Qx,Qy=compute_projections(sigma,stratind=0)
    QXS.append(Qx)
    QYS.append(Qy)
fig,axs=plt.subplots(1,4,figsize=(20,5))
for i in range(len(sigmas)):
    sigma=sigmas[i]
    ax=axs[i]
    Qx,Qy=QXS[i],QYS[i]
    _,sx,_=np.linalg.svd(Qx)
    ax.semilogy(sx[:-8*sigma]/sx[0],label='across longitude')
    _,sy,_=np.linalg.svd(Qy)
    ax.semilogy(sy/sy[0],label='across latitude')
    ax.set_title('Singular Values of Coarse-graining sigma='+str(sigma))
    ax.legend()
    ax.grid()
    ax.set_ylabel('singular values')
    ax.set_xlabel('index')'''


# In[6]:


def plot_inverse_span(sigma):
    QQx,QQy,Qx,Qy=compute_projections(sigma,stratind=0)
    Qx=adjust_reflections(Qx,sigma)
    u,s,v=np.linalg.svd(Qx)
    iQx=u[:,:len(s)-4*sigma]@np.diag(1/s[:-4*sigma])@v[:-4*sigma,:]

    u,s,v=np.linalg.svd(Qy)
    iQy=u[:,:len(s)]@np.diag(1/s)@v
    SQx=iQx*1
    for i in range(iQx.shape[1]):
        qx=iQx[:,i]
        ii=np.argmax(np.abs(qx))
        qx=np.concatenate([qx[ii:],qx[:ii]],axis=0)
        SQx[:,i]=qx
    mx=iQx.shape[0]//2
    SQx=np.abs(np.concatenate([SQx[mx:],SQx[:mx]],axis=0))


    SQy=iQy*1
    for i in range(iQy.shape[1]):
        qx=iQy[:,i]
        ii=np.argmax(qx)
        SQy[:,i]=np.concatenate([qx[ii:],qx[:ii]],axis=0)
    my=iQy.shape[0]//2
    SQy=np.abs(np.concatenate([SQy[my:],SQy[:my]],axis=0))
    MX=iQx.shape[1]
    MX4=MX//4
    mx2=mx//2
    S=12*sigma
    span=np.arange(-S,S)
    nspan=span/sigma
    fig,axs=plt.subplots(1,2,figsize=(15,5))
    mSQx=np.mean(SQx,axis=1)
    sSQx=np.std(SQx,axis=1)
    axs[0].semilogy(nspan,mSQx[mx+span],label='average')
    axs[0].fill_between(nspan,mSQx[mx+span]-sSQx[mx+span],mSQx[mx+span]+sSQx[mx+span],alpha=0.2,color='blue',label='+-1 std')
    axs[0].set_title('Inverse Coarse-Graining('+str(sigma)+') Span in Longitude')
    axs[0].set_ylim([5e-4,np.amax(SQx)*1.2])
    axs[0].set_xlabel('longitude grid points from center ('+str(sigma*7.5)+'km each)')
    axs[0].axvline(x=-3,linestyle='--',color='r')
    axs[0].axvline(x=3,linestyle='--',color='r')
    axs[0].legend()

    mSQy=np.mean(SQy,axis=1)
    sSQy=np.std(SQy,axis=1)
    axs[1].semilogy(nspan,mSQy[my+span],label='average')
    axs[1].fill_between(nspan,mSQy[my+span]-sSQy[my+span],mSQy[my+span]+sSQy[my+span],alpha=0.2,color='blue',label='+-1 std')
    axs[1].set_ylim([1e-4,np.amax(SQy)*1.1])
    axs[1].set_title('Inverse Coarse-Graining('+str(sigma)+') Span in Latitude')
    axs[1].set_xlabel('latitude grid points from center ('+str(sigma*7.5)+'km each at equator)')
    axs[1].axvline(x=-3,linestyle='--',color='r')
    axs[1].axvline(x=3,linestyle='--',color='r')
    axs[1].legend()


# In[7]:


def plotting_projections():
    sigmas=[4,8,12,16]
    QXS=[]
    QYS=[]
    for i in range(len(sigmas)):
        sigma=sigmas[i]
        QQx,QQy,Qx,Qy=compute_projections(sigma,stratind=0)
        QXS.append(Qx)
        QYS.append(Qy)
    fig,axs=plt.subplots(1,4,figsize=(20,5))
    for i in range(len(sigmas)):
        sigma=sigmas[i]
        ax=axs[i]
        Qx,Qy=QXS[i],QYS[i]
        _,sx,_=np.linalg.svd(Qx)
        ax.semilogy(sx[:-8*sigma]/sx[0],label='across longitude')
        _,sy,_=np.linalg.svd(Qy)
        ax.semilogy(sy/sy[0],label='across latitude')
        ax.set_title('Singular Values of Coarse-graining sigma='+str(sigma))
        ax.legend()
        ax.grid()
        ax.set_ylabel('singular values')
        ax.set_xlabel('index')

    SQQx=QQx*1
    for i in range(QQx.shape[1]):
        qx=QQx[:,i]
        ii=np.argmax(qx)
        qx=np.concatenate([qx[ii:],qx[:ii]],axis=0)
        SQQx[:,i]=qx
    mx=QQx.shape[0]//2
    SQQx=np.concatenate([SQQx[mx:],SQQx[:mx]],axis=0)


    SQQy=QQy*1
    for i in range(QQy.shape[1]):
        qx=QQy[:,i]
        ii=np.argmax(qx)
        SQQy[:,i]=np.concatenate([qx[ii:],qx[:ii]],axis=0)
    my=QQy.shape[0]//2
    SQQy=np.concatenate([SQQy[my:],SQQy[:my]],axis=0)


    SSQQx=np.sort(SQQx**2,axis=1)
    SSQQx=SSQQx[:,::-1]
    MX=QQx.shape[1]
    MX4=MX//4
    mx2=mx//2
    S=40
    span=np.arange(-S,S)

    plt.semilogy(np.abs(SSQQx[:,0]))


# In[8]:


def clean_3d_data():
    root='/scratch/zanna/data/cm2.6/'
    filename=root+'3D-data.zarr'
    filename1=root+'3D-data-1.zarr'
    ds_data=xr.open_zarr(filename)            .isel(st_ocean=np.array([0,10,15,20])).drop('salt')
    save_fun(ds_data,filename1)


# In[9]:


def save_fun(uv_data,filename):
    dd=4
    for j in range(len(uv_data.time.values)//dd):
        if j==0:
            uv_data.isel(time=range(j*dd,(j+1)*dd)).to_zarr(filename,mode='w')
        else:
            uv_data.isel(time=range(j*dd,(j+1)*dd)).to_zarr(filename,append_dim="time",mode='a')
        print('\t\t'+str(j*dd),flush=True)


# In[10]:


'''yc=200
Kx,Ky=compute_quadratic_form(yc,dy,dx,sigma)

raw_data_address='/scratch/zanna/data/cm2.6/3D-data.zarr'
stratinds=np.array([0])
u_v_dataset=xr.open_zarr(raw_data_address)\
                .chunk(chunks={"time":1})\
                .isel(st_ocean=stratinds)\
                .drop("salt nv st_edges_ocean xt_ocean yt_ocean".split())
u_v_dataset=u_v_dataset.isel(time=np.arange(1),st_ocean=np.arange(1))
u_v_dataset=u_v_dataset.drop("time st_ocean".split())

yi=yc*sigma
u_v_datasetj=u_v_dataset
Su=np.zeros(u_v_datasetj.u.shape[2]-n1)
uu=np.stack([u_v_datasetj.u.values[0,0],\
                u_v_datasetj.v.values[0,0],\
                  u_v_datasetj.temp.values[0,0] ],axis=0)
for xi in range(u_v_datasetj.u.shape[2]-n1):
    t=uu[2,yi:yi+n1,xi:xi+n1].reshape([-1,1])
    v=uu[1,yi:yi+m,xi:xi+m].reshape([1,-1])
    u=uu[0,yi:yi+m,xi:xi+m].reshape([1,-1])
    Su[xi]=u@Kx@t+v@Ky@t

x=u_v_dataset.xu_ocean.values
y=u_v_dataset.yu_ocean.values
dx=x[1:]-x[:-1]
dy=y[1:]-y[:-1]

dx=np.reshape(dx,(1,-1))
dx=[dx for i in range(len(y)-1)]
dx=np.concatenate(dx,axis=0)
#dx=np.stack([dx],axis=0)

dy=np.reshape(dy,(-1,1))
dy=[dy for i in range(len(x)-1)]
dy=np.concatenate(dy,axis=1)
#dy=np.stack([dy],axis=0)
grid_data=xr.Dataset(data_vars=dict(dxu=(["yu_ocean","xu_ocean"],dx),\
                                    dyu=(["yu_ocean","xu_ocean"],dy)),\
                        coords=dict(xu_ocean=x[:-1],yu_ocean=y[:-1]))



u_v_datasetj=u_v_dataset

u=np.stack([u_v_datasetj.u.values,\
            u_v_datasetj.v.values,\
              u_v_datasetj.temp.values ],axis=0)

uv=xr.Dataset(data_vars=dict(usurf=(["time","yu_ocean","xu_ocean"],u[0,0]),\
                                vsurf=(["time","yu_ocean","xu_ocean"],u[1,0]),\
                                 surface_temp=(["time","yu_ocean","xu_ocean"],u[2,0])),\
                    coords=dict(xu_ocean=x,yu_ocean=y,time=np.arange(1)))
forcing=eddy_forcing(uv, grid_data,sigma,nan_or_zero='nan',typenum=0)'''


# In[14]:


def main():
    args=options()
    print(args)
    if args.clean3d==1:
        clean_3d_data()
    if args.sigma==0:
        #download_raw(args.type)
        print(' uncomment the part and resubmit the job, this action requires download')
    elif args.sigma>0:
        coarsen_save(args.sigma,                     args.type,                     testflag=bool(args.test),                     projection=bool(args.projection),rewrite=bool(args.rewrite))
if __name__=='__main__':
    main()

