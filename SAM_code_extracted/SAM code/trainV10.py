import os
from matplotlib import pyplot as plt
from tool import Generator, data_utils,select_device,findPotential,fault2dis,showPotential,checkPotential,collate_function,RandomScaleRotateBatch,Generator_new,resize_array_with_zoom3d,resize_array_with_zoom,downsample_h5,count_parameters

import torch
#import tensorflow as tf
import platform
import random
#random.seed(0)
import sys
from h5py import File
import numpy as np
import os 
from scipy.ndimage import zoom
from scipy.signal import resample
from scipy import fft
from matplotlib import cm
from config import args,write_config


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
# PATCH (v6): derive the h5 from --dataset so any dataset dir works without
# editing this file (e.g. --dataset earth_dataset_global/key_train.lst).
_dsd = os.path.dirname(args.dataset)
if _dsd and os.path.exists(os.path.join(_dsd, 'faults.h5')):
    trainFile = os.path.join(_dsd, 'faults.h5')
else:
    trainFile = 'earth_dataset_v3/faults.h5'
dataFile = 'earth_dataset/ca_offshore_faults.h5'
#trainFile = '/H5/all_eqA_296_tomark_other.h5'

if args.job =='outputAllKey':
    #找到outputH5文件夹里所有jpg后缀的文件，将文件名写入key_all.lst并计数
    from glob import glob
    count = 0
    with open('key_all.lst','w') as f:
        for file in glob('outputH5/*.jpg'):
            file = os.path.basename(file)
            f.write(file[:-4]+'\n')
            count +=1
    print(count)
    exit()

if args.job =='outputAllKeyOther':
    #找到outputH5_new文件夹里所有jpg后缀的文件，将文件名写入key_all_new.lst并计数
    from glob import glob
    count = 0
    with open('key_all_new.lst','w') as f:
        for file in glob('outputH5_new/*.jpg'):
            file = os.path.basename(file)
            f.write(file[:-4]+'\n')
            count +=1
    print(count)
    exit()

if args.job =='outputAllKeyOtherSelect':
    #选在旧图片里且在旧list里的图片+新图片，写在新文件key_select_new.lst
    from glob import glob
    keys_all = [os.path.basename(file)[:-4] for file in glob('outputH5_new/*.jpg')]
    keys_all_old = [os.path.basename(file)[:-4] for file in glob('outputH5/*.jpg')]
    keys_select_old=  []
    with open('key.lst','r') as f:
        for line in f:
            keys_select_old.append(line.strip())
    with open('key_select_new.lst','w') as f:
        for key in keys_all:
            if key in keys_all_old:
                if key not in keys_select_old:
                    continue
            f.write(key+'\n')
                
    
        

keylist_train = args.dataset
keylist_train_added = args.datasetAdd
keyL_h5_added = []
with open(keylist_train,'r') as f:
    keyL_h5 = f.read().split('\n')[:-1]

if keylist_train_added != '':
    with open(keylist_train_added,'r') as f:
        tmp = f.read().split('\n')[:-1]
    for key in tmp:
        if key not in keyL_h5:
            keyL_h5_added.append(key)
print('keyL_h5_added:',len(keyL_h5_added))

random.shuffle(keyL_h5)




dataEnd = np
BatchSize=args.batchsize
inter=1

mn = args.mn.split(',')
n = int(mn[0])
m = int(mn[-1])

strideL = np.zeros([100,2],'int')+1



dataMode = args.dataMode
from numba import njit
from tool import point2length


if __name__ == '__main__':
    
    #from UnetModelTensor import UModel as UnetModel,IoU,backend,keras
    from torchUnet import UNet,norm2,denorm2,huber_loss,torch,WarmupDecayScheduler,IoU,conv_sam,Dice,FocalTversky
    
    if args.job == 'extract_value':
        from shapefile import Reader
        
        faultFile = args.faultFile
        
        
        faultFile_value = faultFile.replace('.shp','_value.h5')
        k0 = args.value
        file = args.valueFile
        with File(file,'r') as f:
            data = f[k0][:]
            la = f['la'][:]
            lo = f['lo'][:]
        plt.close()
        plt.figure(figsize=(10,10))
        plt.imshow(data[-1],cmap='bwr')
        plt.colorbar()
        plt.gca().set_aspect('equal')
        plt.savefig(faultFile_value.replace('.h5',f'_{k0}.jpg'),dpi=300)
        #valueL = []
        with File(faultFile_value,'a') as res:
            with Reader(faultFile) as f:
                for i ,shape in enumerate(f.shapes()):
                    #shape =f.shapes()[1000]
                    print(i,'in',len(f),i/len(f))
                    fault_name  = str(i)
                    points = shape.points
                    points = np.array(points)
                    if fault_name not in res:
                        fault_group = res.create_group(fault_name)
                        fault_group.create_dataset('points',data=points)
                    else:
                        fault_group = res[fault_name]
                    if k0 in fault_group:
                        del fault_group[k0]
                    
                    #import numpy as np

                    # 假设 la 和 lo 是升序排列的
                    I = np.searchsorted(la, points[:,1])
                    J = np.searchsorted(lo, points[:,0])
                    
                    

                    # 限制 I, J 在合法范围内
                    I = np.clip(I, 0, len(la) - 3)
                    J = np.clip(J, 0, len(lo) - 3)

                    #value = data[:, I, J]
                    # 先获取所有可能的索引组合
                    #value = np.take(np.take(data, I, axis=1), J, axis=2)
                    # 生成 I, J 的唯一组合
                    #unique_pairs, unique_idx = np.unique(np.stack([I, J], axis=1), axis=0, return_inverse=True)
                    #unique_I, unique_J = unique_pairs[:, 0], unique_pairs[:, 1]

                    # 只查询唯一的 I, J 组合，减少重复访问
                    #unique_values = data[:, unique_I, unique_J]

                    # 还原回原来的索引顺序
                    #value = unique_values[:, unique_idx]
                    #I  = np.abs(la[np.newaxis]-points[:,1][:,np.newaxis]).argmin(axis=1)
                    #J  = np.abs(lo[np.newaxis]-points[:,0][:,np.newaxis]).argmin(axis=1)
                    #print(I,J,points[:,0],points[:,1],la[::100],data.shape)
                    #exit()
                    value = data[:,I,J]/4+data[:,I+1,J+1]/4+data[:,I+1,J]/4+data[:,I,J+1]/4
                    #value1 = data[:,J,I]
                    #print(value.max(),value1.max())#,I,J,points[:,0])
                    print(la.shape,lo.shape,data.shape)
                    fault_group.create_dataset(k0,data=value.astype('float16'))
                    
                    #exit()
                    #valueL.append(value.mean(axis=1))
            #valueL = np.stack(valueL,axis=0)
            #np.save(faultFile_value,valueL)
        exit()
    if args.job == 'show_value':
        from shapefile import Reader
        file = args.valueFile
        faultFile = args.faultFile
        k0 = args.value
        
        faultFile_value = faultFile.replace('.shp','_value.h5')
        
        plotDir = faultFile.replace('.shp','/')
        if not os.path.exists(plotDir):
            os.makedirs(plotDir)
        
        with File(faultFile_value,'r') as f:
            for k in list(f.keys())[::100]:
                group = f[k]
                print(group)
                value = group[k0][:]/255
                points = group['points'][:]
                plt.close()
                plt.figure(figsize=(10,10))
                plt.plot(points[:,0],points[:,1],'k')
                plt.scatter(points[:,0],points[:,1],c=value,cmap='bwr')
                plt.colorbar()
                plt.gca().set_aspect('equal')
                plt.savefig(f'{plotDir}{k}.jpg',dpi=300)
                
        exit()
    if args.job == 'show_tsne':
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        compressionFile  = 'data/20240311_f/tectonics_Compression.h5'
        extensionFile  = 'data/20240311_f/tectonics_Extention.h5'
        featureL_compression = []
        minN = 30
        plotDir = args.resDir+'tsne/'
        if not os.path.exists(plotDir):
            os.makedirs(plotDir)
        with File(compressionFile,'r') as f:
            for k in list(f.keys())[::1]:
                #print(f[k].keys())
                if f[k]['feature'].shape[1]<minN:
                    continue
                featureL_compression.append(f[k]['feature'][:].mean(axis=1))
        featureL_compression = np.stack(featureL_compression,axis=0)
        featureL_extension = []
        with File(extensionFile,'r') as f:
            for k in list(f.keys())[::1]:
                if f[k]['feature'].shape[1]<minN:
                    continue
                featureL_extension.append(f[k]['feature'][:].mean(axis=1))
        featureL_extension = np.stack(featureL_extension,axis=0)
        featureL = np.concatenate([featureL_compression,featureL_extension],axis=0)
        labelL = np.concatenate([np.zeros([featureL_compression.shape[0]]),np.ones([featureL_extension.shape[0]])])
        featureL_norm = (featureL)/(featureL.std(axis=0,keepdims=True)+1e-4)
        featureL_norm = featureL_norm /((featureL_norm**2).sum(axis=1,keepdims=True)+1e-4)**0.5
        tsne = TSNE(n_components=2,random_state=0)
        #tsne = PCA(n_components=2)
        x = tsne.fit_transform(featureL_norm)
        
        plt.close()
        plt.plot(x[:len(featureL_compression),0],x[:len(featureL_compression),1],'.r',ms=3,alpha=0.5,label='c',mec='None')
        plt.plot(x[len(featureL_compression):,0],x[len(featureL_compression):,1],'.b',ms=3,alpha=0.05,label='e',mec='None')
        
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.savefig(f'{plotDir}/tsne.jpg',dpi=600)
        
        plt.close()
        plt.plot(x[:len(featureL_compression),0],x[:len(featureL_compression),1],'.r',ms=3,alpha=0.5,label='c',mec='None')
        plt.plot(x[len(featureL_compression):,0],x[len(featureL_compression):,1],'.b',ms=3,alpha=0.0,label='e',mec='None')
        #plt.plot(vnslo,vnsla,'k',lw=0.5)
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.savefig(f'{plotDir}/tsne_c.jpg',dpi=600)
        
        plt.close()
        plt.plot(x[:len(featureL_compression),0],x[:len(featureL_compression),1],'.r',ms=3,alpha=0.0,label='c',mec='None')
        plt.plot(x[len(featureL_compression):,0],x[len(featureL_compression):,1],'.b',ms=3,alpha=0.05,label='e',mec='None')
        #plt.plot(vnslo,vnsla,'k',lw=0.5)
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.savefig(f'{plotDir}/tsne_e.jpg',dpi=600)
        exit()
    if args.job == 'show_feature':
        k0 = args.value
        file = args.valueFile
        with File(file,'r') as f:
            data = f[k0][:,::2,::2]
            la = f['la'][::2]
            lo = f['lo'][::2]
        plotDir = args.resDir+'features/'
        if not os.path.exists(plotDir):
            os.makedirs(plotDir)
        for i in range(len(data)):
            print(i)
            plt.close()
            plt.figure(figsize=(10,10))
            #plt.imshow(data[i],cmap='jet')
            plt.pcolormesh(lo,la,data[i],cmap='jet',rasterized=True)
            plt.colorbar()
            plt.gca().set_aspect('equal')
            plt.savefig(f'{plotDir}/{i}.jpg',dpi=300)
    if args.job == 'cal_feature_corr':
        k0 = args.value
        file = args.valueFile
        with File(file,'r') as f:
            data = f[k0][:,::4,::4]
            la = f['la'][::4]
            lo = f['lo'][::4]
        corr = np.zeros([data.shape[0],data.shape[0]])
        
        data  =torch.from_numpy(data).float()
        data_norm = (data-data.mean(axis=(1,2),keepdims=True))/(data.std(axis=(1,2),keepdims=True)+1e-4)
        data_norm = data_norm.to('cuda')
        
        for i in range(data.shape[0]):
            for j in range(i,data.shape[0]):
                data0 = data_norm[i]#.to('cuda')
                data1 = data_norm[j]#.to('cuda')
                Corr = (data0*data1).mean().cpu().numpy()
                corr[i,j] = Corr
                corr[j,i] = Corr
                print(i,j,corr[i,j])
        plt.close()
        plt.figure(figsize=(10,10))
        plt.imshow(corr,cmap='bwr',vmin=-1,vmax=1)
        plt.colorbar()
        plt.gca().set_aspect('equal')
        plt.savefig(f'{args.resDir}corr.jpg',dpi=300)
        np.savetxt(f'{args.resDir}corr.txt',corr)
    if args.job == 'an_feature_corr':
        corr = np.loadtxt(f'{args.resDir}corr.txt')
        
        PCA = np.linalg.eig(corr)
        threshod = PCA[0].max()*0.005
        plt.close()
        plt.figure(figsize=(10,10))
        plt.plot(PCA[0])
        plt.plot([0,256],[threshod,threshod],'k')
        plt.savefig(f'{args.resDir}PCA.jpg',dpi=300)
        print((PCA[0]>threshod).sum())
        exit()
    if args.job == 'show_feature_corr':
        corr = np.loadtxt(f'{args.resDir}corr.txt')
        PCA = np.linalg.eig(corr)
        k0 = args.value
        file = args.valueFile
        plotDir = args.resDir+'features/'
        if not os.path.exists(plotDir):
            os.makedirs(plotDir)
        with File(file,'r') as f:
            data = f[k0][:,::4,::4]
            la = f['la'][::4]
            lo = f['lo'][::4]
        threshod = PCA[0].max()*0.005
        data  =torch.from_numpy(data).float()
        data_norm = (data-data.mean(axis=(1,2),keepdims=True))/(data.std(axis=(1,2),keepdims=True)+1e-4)
        data_norm = data_norm.to('cuda')
        for i in range((PCA[0]>threshod).sum()):
            v = PCA[1][i]
            v = torch.from_numpy(v).float().to('cuda')
            feature = torch.einsum('kij,k->ij',data_norm,v).cpu().numpy()
            feature = feature/feature.std()
            plt.close()
            plt.figure(figsize=(10,5))
            #plt.imshow(data[i],cmap='jet')
            plt.pcolormesh(lo,la,feature,cmap='RdYlBu',rasterized=True,vmin=-3,vmax=3)
            plt.colorbar()
            plt.gca().set_aspect('equal')
            plt.savefig(f'{plotDir}/feature_{i}.jpg',dpi=300)
    if args.job in ['show_kmean','show_kmean_o','show_kmean_river']:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        from scipy import io as sio
        if args.job ==  'show_kmean':
            compressionFile  = 'data/0zero9/tectonics_Compression_value.h5'
            extensionFile  = 'data/0zero9/tectonics_Extention_value.h5'
            #little_riverFile = 'data/river/Hynek-valley_value.h5'
            plotDir = args.resDir+'tsne/'
        elif args.job == 'show_kmean_river':
            compressionFile  = 'data/20240311_f/tectonics_Compression_value.h5'
            extensionFile  = 'data/20240311_f/tectonics_Extention_value.h5'
            little_riverFile = 'data/river/Hynek-valley_value.h5'
            large_riverFile = 'data/fluvial_tanaka/fluvial_tanaka2014_value.h5'
            plotDir = args.resDir+'tsne_river/'
        else:
            compressionFile  = 'data/tectonics_Compression_value.h5'
            extensionFile  = 'data/tectonics_Extention_value.h5'
            plotDir = args.resDir+'tsne_o/'
        featureL_compression = []
        lolaL_compression = []
        pointsL_compression = []
        distanceL_compression = []
        minN = 10
        minDis = 0.2
        
        
        
        if not os.path.exists(plotDir):
            os.makedirs(plotDir)
            
            
            
        from shapefile import Reader
        vnsla = []
        vnslo = []
        with Reader('data/VNs/valley.shp') as f:
            for i,shape in enumerate(f.shapes()):
                points = np.array(shape.points)
                vnsla.append(points[:,1])
                vnslo.append(points[:,0])
                vnsla.append(np.ones(1)*np.nan)
                vnslo.append(np.ones(1)*np.nan)
        vnsla = np.concatenate(vnsla)
        vnslo = np.concatenate(vnslo)
        
        plt.close()
        plt.plot(vnslo,vnsla,'k',lw=0.5)
        plt.gca().set_aspect('equal')
        plt.savefig(f'{plotDir}/vns.jpg',dpi=600)
        
        with File(compressionFile,'r') as f:
            for k in list(f.keys())[::1]:
                #print(f[k].keys())
                if f[k]['feature'].shape[1]<minN:
                    continue
                distance = point2length(f[k]['points'][:])
                if distance.min()<minDis:
                    continue
                featureL_compression.append(f[k]['feature'][:].mean(axis=1))
                lolaL_compression.append(f[k]['points'][:].mean(axis=0))
                pointsL_compression.append(f[k]['points'][:])
                distanceL_compression.append(distance)
        featureL_compression = np.stack(featureL_compression,axis=0)
        lolaL_compression = np.stack(lolaL_compression,axis=0)
        distanceL_compression = np.stack(distanceL_compression,axis=0)
        
        featureL_extension = []
        lolaL_extension = []
        pointsL_extension = []
        distanceL_extension = []
        
        with File(extensionFile,'r') as f:
            for k in list(f.keys())[::1]:
                if f[k]['feature'].shape[1]<minN:
                    continue
                distance = point2length(f[k]['points'][:])
                if distance.min()<minDis:
                    continue
                featureL_extension.append(f[k]['feature'][:].mean(axis=1))
                lolaL_extension.append(f[k]['points'][:].mean(axis=0))
                pointsL_extension.append(f[k]['points'][:])
                distanceL_extension.append(distance)
        featureL_extension = np.stack(featureL_extension,axis=0)
        lolaL_extension = np.stack(lolaL_extension,axis=0)
        if args.job == 'show_kmean_river':
            featureL_little_river = []
            lolaL_little_river = []
            pointsL_little_river = []
            distanceL_little_river = []
            with File(little_riverFile,'r') as f:
                for k in list(f.keys())[::1]:
                    #print(f[k].keys())
                    if f[k]['feature'].shape[1]<minN:
                        continue
                    distance = point2length(f[k]['points'][:])
                    if distance.min()<minDis:
                        continue
                    featureL_little_river.append(f[k]['feature'][:].mean(axis=1))
                    lolaL_little_river.append(f[k]['points'][:].mean(axis=0))
                    pointsL_little_river.append(f[k]['points'][:])
                    distanceL_little_river.append(distance)
            featureL_little_river = np.stack(featureL_little_river,axis=0)
            lolaL_little_river = np.stack(lolaL_little_river,axis=0)
            distanceL_little_river = np.stack(distanceL_little_river,axis=0)
            
            
            featureL_large_river = []
            lolaL_large_river = []
            pointsL_large_river = []
            distanceL_large_river = []
            with File(large_riverFile,'r') as f:
                for k in list(f.keys())[::1]:
                    #print(f[k].keys())
                    if f[k]['feature'].shape[1]<minN:
                        continue
                    distance = point2length(f[k]['points'][:])
                    if distance.min()<minDis:
                        continue
                    featureL_large_river.append(f[k]['feature'][:].mean(axis=1))
                    lolaL_large_river.append(f[k]['points'][:].mean(axis=0))
                    pointsL_large_river.append(f[k]['points'][:])
                    distanceL_large_river.append(distance)  
                    
            featureL_large_river = np.stack(featureL_large_river,axis=0)
            lolaL_large_river = np.stack(lolaL_large_river,axis=0)
            distanceL_large_river = np.stack(distanceL_large_river,axis=0)
            
            
            
        
        if args.job == 'show_kmean_river':
            featureL = np.concatenate([featureL_compression,featureL_extension,featureL_little_river,featureL_large_river],axis=0)
            labelL = np.concatenate([np.zeros([featureL_compression.shape[0]]),np.ones([featureL_extension.shape[0]]),np.ones([featureL_little_river.shape[0]])*2,np.ones([featureL_large_river.shape[0]])*3])
            lolaL = np.concatenate([lolaL_compression,lolaL_extension,lolaL_little_river,lolaL_large_river],axis=0)
            distanceL_extension= np.stack(distanceL_extension,axis=0)
            distanceL = np.concatenate([distanceL_compression,distanceL_extension,distanceL_little_river,distanceL_large_river],axis=0)
            pointsL = pointsL_compression+pointsL_extension+pointsL_little_river+pointsL_large_river
            featureL_norm = (featureL)/(featureL.std(axis=0,keepdims=True)+1e-4)
            featureL = featureL_norm/((featureL_norm**2).sum(axis=1,keepdims=True)+1e-4)**0.5
            
        else:
            featureL = np.concatenate([featureL_compression,featureL_extension],axis=0)
            labelL = np.concatenate([np.zeros([featureL_compression.shape[0]]),np.ones([featureL_extension.shape[0]])])
            lolaL = np.concatenate([lolaL_compression,lolaL_extension],axis=0)
            distanceL_extension= np.stack(distanceL_extension,axis=0)
            distanceL = np.concatenate([distanceL_compression,distanceL_extension],axis=0)
            pointsL = pointsL_compression+pointsL_extension
            featureL_norm = (featureL)/(featureL.std(axis=0,keepdims=True)+1e-4)
            featureL = featureL_norm/((featureL_norm**2).sum(axis=1,keepdims=True)+1e-4)**0.5
        
        from sklearn.cluster import KMeans
        
        cN = 3
        colorL  = ['r','g','b','c','m','y','k','brown']
        kmeans = KMeans(n_clusters=cN, random_state=0).fit(featureL)
        import shapefile
        shpDir = args.resDir+'shp/'
        if not os.path.exists(shpDir):
            os.makedirs(shpDir)
        
        sf_compression = shapefile.Writer(shpDir+'compression.shp', shapeType=shapefile.POLYLINE)
        sf_extension = shapefile.Writer(shpDir+'extension.shp', shapeType=shapefile.POLYLINE)
        sf_compression.field('class', "N", size=5) 
        sf_extension.field('class', "N", size=5)
        lines_compression = []
        lines_extension = []
        for i in range(len(featureL)):
            line = [[[point[0],point[1]]for point in pointsL[i]]]
            Class = kmeans.labels_[i]
            
            if i<len(pointsL_compression):
                sf = sf_compression
            else:
                sf = sf_extension
            sf.line(line)
            sf.record(Class)
        
        sf_compression.close()
        sf_extension.close()
        #tsne = PCA(n_components=2)
        tsne = TSNE(n_components=3,random_state=0)
        x = tsne.fit_transform(featureL)
        dataD = {
            'x':x,
            'label':kmeans.labels_,
            'lola':lolaL,
            'points':pointsL,
            'feature':featureL,
            'ce_label':labelL,
            'distance':distanceL
        } 
        sio.savemat(f'{args.resDir}tsne.mat',dataD)
        
        
        
        tsne = TSNE(n_components=2,random_state=0)
        x = tsne.fit_transform(featureL)
        plt.close()
        plt.plot(x[:len(featureL_compression),0],x[:len(featureL_compression),1],'.k',ms=1,alpha=0.5,label='c',mec='None')
        plt.plot(x[len(featureL_compression):len(featureL_compression)+len(featureL_extension),0],x[len(featureL_compression):len(featureL_compression)+len(featureL_extension),1],'.b',ms=1,alpha=0.5,label='e',mec='None')
        if args.job == 'show_kmean_river':
            plt.plot(x[len(featureL_compression)+len(featureL_extension):len(featureL_compression)+len(featureL_extension)+len(featureL_little_river),0],x[len(featureL_compression)+len(featureL_extension):len(featureL_compression)+len(featureL_extension)+len(featureL_little_river),1],'.r',ms=1,alpha=0.5,label='little river',mec='None')
            plt.plot(x[len(featureL_compression)+len(featureL_extension)+len(featureL_little_river):,0],x[len(featureL_compression)+len(featureL_extension)+len(featureL_little_river):,1],'.g',ms=4,alpha=0.5,label='large river',mec='None')
        
        plt.gca().set_aspect('equal')
        #plt.legend()
        plt.savefig(f'{plotDir}/tsne.jpg',dpi=600)
        plt.close()
        
        for i in np.unique(kmeans.labels_):   
            plt.plot(x[kmeans.labels_==i,0],x[kmeans.labels_==i,1],'.',ms=3,alpha=0.5,label=f'{i}',mec='None',color=colorL[i])
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.savefig(f'{plotDir}/tsne_kmean.jpg',dpi=600)
        
        x = tsne.fit_transform(featureL[:len(featureL_compression)])
        plt.close()
        for i in np.unique(kmeans.labels_):   
            plt.plot(x[kmeans.labels_[:len(featureL_compression)]==i,0],x[kmeans.labels_[:len(featureL_compression)]==i,1],'.',ms=3,alpha=0.5,label=f'{i}',mec='None',color=colorL[i])
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.savefig(f'{plotDir}/tsne_kmean_c.jpg',dpi=600)
        
        
        x = tsne.fit_transform(featureL[len(featureL_compression):])
        plt.close()
        for i in np.unique(kmeans.labels_):   
            plt.plot(x[kmeans.labels_[len(featureL_compression):]==i,0],x[kmeans.labels_[len(featureL_compression):]==i,1],'.',ms=3,alpha=0.5,label=f'{i}',mec='None',color=colorL[i])
        plt.gca().set_aspect('equal')
        plt.legend()
        plt.savefig(f'{plotDir}/tsne_kmean_e.jpg',dpi=600)
                
        plt.close()
        plt.figure(figsize=(15,10))
        #for i in range(cN):   
        #    #plt.plot(lolaL[kmeans.labels_==i,0],lolaL[kmeans.labels==i,1],'.',ms=1,alpha=0.5,label=f'{i}',mec='None',color=colorL[i])
        for i in range(len(pointsL)):
            plt.plot(pointsL[i][:,0],pointsL[i][:,1],'.',ms=1,alpha=0.5,label=f'{kmeans.labels_[i]}',mec='None',color=colorL[kmeans.labels_[i]])
        plt.gca().set_aspect('equal')
        #plt.legend()
        plt.xlim([-180,180])
        plt.ylim([-90,90])
        plt.plot(vnslo,vnsla,'k',lw=0.5)
        plt.savefig(f'{plotDir}/kmean.jpg',dpi=600)
        plt.xlim([40,65])
        plt.ylim([-35,-15])
        plt.savefig(f'{plotDir}/kmean_vns.jpg',dpi=600)
        
        plt.close()
        plt.figure(figsize=(15,10))
        for i in range(len(pointsL_compression)):
            plt.plot(pointsL_compression[i][:,0],pointsL_compression[i][:,1],'.',ms=1,alpha=0.5,label=f'{kmeans.labels_[i]}',mec='None',color=colorL[kmeans.labels_[i]])
        plt.gca().set_aspect('equal')
        plt.xlim([-180,180])
        plt.ylim([-90,90])
        plt.plot(vnslo,vnsla,'k',lw=0.5)
        plt.savefig(f'{plotDir}/kmean_c.jpg',dpi=600)
        
        plt.close()
        plt.figure(figsize=(15,10))
        for i in range(len(pointsL_extension)):
            plt.plot(pointsL_extension[i][:,0],pointsL_extension[i][:,1],'.',ms=1,alpha=0.5,label=f'{kmeans.labels_[i]}',mec='None',color=colorL[kmeans.labels_[i+len(pointsL_compression)]])   
        plt.gca().set_aspect('equal')
        plt.xlim([-180,180])
        plt.ylim([-90,90])
        plt.plot(vnslo,vnsla,'k',lw=0.5)
        plt.savefig(f'{plotDir}/kmean_e.jpg',dpi=600)
        exit()
        
        
        
    if args.job == 'downSamplingO':
        h5File_in = '/H5/all_eqA_296.h5'
        h5File = '/H5/all_eqA_296_predict.h5'
        
        k0 = 'tectonics_Extention'
        k1 = 'tectonics_Compression'  
        dDegree = 10
        factor = 16 
        size = 2048
        h5File_downsample = f'/H5/all_eqA_296_predict_ds_{factor}.h5'
        #h5File_downsample = '/NAS/jyr/mount/bak/bakdata/data/all_eqA_296_predict_ds.h5'
        #pltDir  = plotDir+'show_all/'
        #if not os.path.exists(pltDir):
        #    os.makedirs(pltDir)
        with File(h5File_in,'r') as f:
            laL = f['la'][:]
            loL = f['lo'][:]
        if True:           
            downsample_h5(h5File,h5File_downsample,k0,factor,'max')
            downsample_h5(h5File,h5File_downsample,k1,factor,'max')
            downsample_h5(h5File,h5File_downsample,'image',factor)
        with File(h5File_downsample,'a') as f:
            if 'la' in f:
                del f['la']
            if 'lo' in f:
                del f['lo']
            f['la'] = laL[::factor]/2+laL[::-factor][::-1]/2
            f['lo'] = loL[::factor]/2+loL[::-factor][::-1]/2
    if args.job == 'downSampling':
        h5File_in = '/H5/all_eqA_296.h5'
        h5File = '/H5/all_eqA_296_predict.h5'
        
        #h5File_in = '/data/jiangyr/data/all_296.h5'
        #h5File = '/data/jiangyr/data/all_296_predict.h5'
        
        k0 = 'tectonics_Extention'
        k1 = 'tectonics_Compression'  
        dDegree = 10
        factor = 16
        size = 2048
        h5File_downsample = f'/H5/all_eqA_296_predict_ds_{factor}.h5'
        #h5File_downsample = f'/H5/all_296_predict_ds_{factor}.h5'
        pltDir  = args.resDir+'show_all/'
        if not os.path.exists(pltDir):
            os.makedirs(pltDir)
        #if False:
        with File(h5File_in,'r') as f:
            laL = f['la'][:]
            loL = f['lo'][:]    
        if False :    
            downsample_h5(h5File_in,h5File_downsample,k0,factor,dataset_name_out=k0+'_o',dtype='uint8',d0=args.d0,method='max')
            downsample_h5(h5File_in,h5File_downsample,k1,factor,dataset_name_out=k1+'_o',dtype='uint8',d0=args.d0,method='max')
            #exit() 
        with File(h5File_downsample,'a') as f:
            if 'la' in f:
                del f['la']
            if 'lo' in f:
                del f['lo']
            NLa = laL.shape[0]//factor
            NLo = loL.shape[0]//factor
            f['la'] = (laL[::factor]/2+laL[::-factor][::-1]/2)[:NLa]#resample(laL,laL.shape[0]//factor)
            f['lo'] = (loL[::factor]/2+loL[::-factor][::-1]/2 )[:NLo]   #resample(loL,loL.shape[0]//factor)
            #laL[::factor]/2+laL[::-factor][::-1]/2
            #f['lo'] = loL[::factor]/2+loL[::-factor][::-1]/2      
            print(f['la'][:])
            print(f['lo'][:])
            #exit()
        downsample_h5(h5File_in,h5File_downsample,k0,factor,dataset_name_out=k0+'_o',dtype='uint8',d0=args.d0,method='max')
        downsample_h5(h5File_in,h5File_downsample,k1,factor,dataset_name_out=k1+'_o',dtype='uint8',d0=args.d0,method='max')
        downsample_h5(h5File_in,h5File_downsample,'TW',factor,dataset_name_out='image_o',)
        downsample_h5(h5File,h5File_downsample,k0,factor,'max')
        downsample_h5(h5File,h5File_downsample,k1,factor,'max')
        downsample_h5(h5File,h5File_downsample,'image',factor)
        
        #if False:
    if args.job == 'show_all_ds_compare':
        import matplotlib.colors as mcolors
        factor = 16
        h5File = f'/H5/all_eqA_296_predict_ds_{factor}.h5'
        #h5File = f'/H5/all_296_predict_ds_{factor}.h5'
        k0 = 'tectonics_Extention'
        k1 = 'tectonics_Compression'  
        dDegree = 20
        size = 2048
        pltDir  = args.resDir+'show_all_ds/'
        if not os.path.exists(pltDir):
            os.makedirs(pltDir)
        with File(h5File,'r') as f:
            la = f['la'][:]
            lo = f['lo'][:]
            print(la)
            print(lo)
            #exit()
            data0 = f[k0][:].astype('float32')
            data1 = f[k1][:].astype('float32')
            data0_o = f[k0+'_o'][:].astype('float32')
            data1_o = f[k1+'_o'][:].astype('float32')
            image = f['image_o'][:].astype('float32')/255
            image_out = f['image'][:].astype('float32')/255
            image_gray = image*0+image.mean(axis=0,keepdims=True)
            image_gray= 1-(1-image_gray)**2
            #image[:] = image.mean(axis=0,keepdims=True)#*0.5
            #print(image.shape,image.min(),image.max())  
            #image = 1-(1-image)**2  
            #exit()
            threshod = 255*args.threshold
            data0[data0<=threshod]=np.nan
            data1[data1<=threshod]=np.nan
            data0_o[data0_o<=threshod]=np.nan
            data1_o[data1_o<=threshod]=np.nan
            
            data0[data0>threshod]=255
            data1[data1>threshod]=255
            data0_o[data0_o>threshod]=255
            data1_o[data1_o>threshod]=255
            
            data0_o_ = data0_o#.copy()
            #data0_o_[::2]=np.nan
            
            data1_o_ = data1_o#.copy()
            #data1_o_[::2]=np.nan
            plt.figure(figsize=(20,10))
            
            plt.gca().pcolormesh(lo[::4],la[::4],image[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image_gray[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data0[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data0_o_[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=0.2)
            
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}ext_TW_gray_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            plt.figure(figsize=(20,10))
            plt.gca().pcolormesh(lo[::4],la[::4],image[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image_gray[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data1[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data1_o_[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=0.2)
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}comp_TW_gray_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            plt.figure(figsize=(20,10))
            plt.gca().pcolormesh(lo[::4],la[::4],image[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image_gray[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data0[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data1[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}predict_TW_gray_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            plt.figure(figsize=(20,10))
            plt.gca().pcolormesh(lo[::4],la[::4],image[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image_gray[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data0_o[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data1_o[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}target_TW_gray_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            
            
            plt.figure(figsize=(20,10))
            plt.gca().pcolormesh(lo[::4],la[::4],image[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data0[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data1[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}predict_TW_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            plt.figure(figsize=(20,10))
            plt.gca().pcolormesh(lo[::4],la[::4],image[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data0_o[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data1_o[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}target_TW_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            
            plt.figure(figsize=(20,10))
            plt.gca().pcolormesh(lo[::4],la[::4],image_out[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image_out[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data0[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data1[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}predict_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            
            plt.figure(figsize=(20,10))
            plt.gca().pcolormesh(lo[::4],la[::4],image_out[0,::4,::4]*0-10,rasterized=True,alpha=1,color=image_out[:,::4,::4].transpose([1,2,0]).reshape([-1,3])).set_zorder(0)
            plt.gca().pcolorfast(lo,la,data0_o[0][0:,0:],cmap='bwr',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().pcolorfast(lo,la,data1_o[0][0:,0:],cmap='bwr_r',vmin=0,vmax=255,rasterized=True,alpha=1)
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}target_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            
            polarIndex = np.abs(la+75).argmin()
            print(polarIndex)
            laL_polar = la[:polarIndex]
            laL_polar_rad = np.deg2rad(90+laL_polar)
            loL_polar = lo[:]
            loL_polar_rad = np.deg2rad(loL_polar)
            
            
            
            x = np.cos(loL_polar_rad).reshape([1,-1])*laL_polar_rad.reshape([-1,1])
            y = np.sin(loL_polar_rad).reshape([1,-1])*laL_polar_rad.reshape([-1,1])
            #ax.pcolormesh(x,y,image[:,:polarIndex].transpose([1,2,0]),rasterized=True,alpha=1,)
            z = image_out[:,-polarIndex:][:,::-1][:,:-1,:-1].transpose([1,2,0])
            colors_image = z.reshape([-1,3]) 
            colors_TW = f['image_o'][:,polarIndex:][:,::-1][:,:-1,:-1].transpose([1,2,0]).reshape([-1,3]).astype('float32')/255
            
            plt.figure(figsize=(10,10),dpi=600,)
            fig, ax = plt.subplots()
            ax.pcolormesh(x,y,z,rasterized=True,alpha=1,color=colors_image)
            ax.pcolorfast(x,y,data0[:,-polarIndex:][:,::-1][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr')
            ax.pcolorfast(x,y,data1[:,-polarIndex:][:,::-1][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr_r')
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}predict_polar_north_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            plt.figure(figsize=(10,10),dpi=600,)
            fig, ax = plt.subplots()
            ax.pcolormesh(x,y,z,rasterized=True,alpha=1,color=colors_TW)
            ax.pcolormesh(x,y,data0[:,-polarIndex:][:,::-1][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr')
            ax.pcolormesh(x,y,data1[:,-polarIndex:][:,::-1][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr_r')
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}predict_polar_north_TW_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            
            
            x = np.cos(loL_polar_rad).reshape([1,-1])*laL_polar_rad.reshape([-1,1])
            y = np.sin(loL_polar_rad).reshape([1,-1])*laL_polar_rad.reshape([-1,1])
            plt.figure(figsize=(10,10),dpi=600,)
            z = image_out[:,:polarIndex][:,:-1,:-1].transpose([1,2,0])
            colors_image = z.reshape([-1,3])  
            colors_TW = f['image_o'][:,:polarIndex][:,:-1,:-1].transpose([1,2,0]).reshape([-1,3]).astype('float32')/255
            
            fig, ax = plt.subplots()
            #ax.pcolormesh(x,y,image[:,:polarIndex].transpose([1,2,0]),rasterized=True,alpha=1,)
            ax.pcolormesh(x,y,z,rasterized=True,alpha=1,color=colors_image )
            ax.pcolormesh(x,y,data0[:,:polarIndex][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr')
            ax.pcolormesh(x,y,data1[:,:polarIndex][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr_r')
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}predict_polar_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            #exit()
            
            plt.figure(figsize=(10,10),dpi=600,)
            fig, ax = plt.subplots()
            ax.pcolormesh(x,y,z,rasterized=True,alpha=1,color=colors_TW)
            ax.pcolormesh(x,y,data0[:,:polarIndex][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr')
            ax.pcolormesh(x,y,data1[:,:polarIndex][0,:-1,:-1],rasterized=True,alpha=1,vmin=0,vmax=255,cmap='bwr_r')
            
            plt.gca().set_aspect('equal')
            plt.savefig(f'{pltDir}predict_polar_TW_{args.threshold:.2f}.jpg',dpi=600)
            plt.close()
            exit()
            
    if args.job == 'downSamplingIn':
        h5File = '/H5/all_eqA_296.h5'
        #h5File = '/H5/all_eqA_296_predict.h5'
        
        k0 = 'tectonics_Extention'
        k1 = 'tectonics_Compression'  
        dDegree = 10
        factor = 16 
        size = 2048
        h5File_downsample = f'/H5/all_eqA_296_ds_{factor}.h5'
        #h5File_downsample = '/NAS/jyr/mount/bak/bakdata/data/all_eqA_296_predict_ds.h5'
        #pltDir  = plotDir+'show_all/'
        #if not os.path.exists(pltDir):
        #    os.makedirs(pltDir)
        with File(h5File,'r') as f:
            laL = f['la'][:]
            loL = f['lo'][:]
        if True:           
            downsample_h5(h5File,h5File_downsample,k0,factor,'max')
            downsample_h5(h5File,h5File_downsample,k1,factor,'max')
            downsample_h5(h5File,h5File_downsample,'image',factor)
        with File(h5File_downsample,'a') as f:
            if 'la' in f:
                del f['la']
            if 'lo' in f:
                del f['lo']
            f['la'] = laL[::factor]/2+laL[::-factor][::-1]/2
            f['lo'] = loL[::factor]/2+loL[::-factor][::-1]/2
            
    if args.job=='toDis' :
        fault2dis(dataFile,'tectonics_Extention.shp',maxDis=0.4,head='Extention')
        fault2dis(dataFile,'tectonics_Compression.shp',maxDis=0.4,head='Compression')
        exit()
    
    #   ['graben', 'flow-like features', 'channel', 'river channel', 'wrinkle-like features', 'aeolian Yardang landform', 'glacial landform', 'chaotic terrain']
    if args.job=='toDis_new' :
        keys = [ 'flow-like features', 'channel', 'river channel', 'wrinkle-like features', 'aeolian Yardang landform', 'glacial landform']
        fault2dis(dataFile,'data/AI_extention_YWPick/tectonics_Extention_0.95_faults.shp',maxDis=0.4,head='others',keys=keys)
        #fault2dis(dataFile,'tectonics_Compression.shp',maxDis=0.4,head='Compression')
        #exit()
    if args.job == 'plotDis' :
        with File(dataFile,'r') as h5:
            image = h5['tectonics_Extention_0.95_faults'][::10,::10]
            image = np.where(image<0.35,image,np.nan)
            plt.imshow(image,cmap='bwr')
            plt.colorbar()
            plt.savefig('tectonics_Extention_0.95_faults.png')
            plt.show()
            exit()
            
    if args.job=='findP' :
        findPotential(dataFile,d=n)
        exit()
    if args.job=='showP' :
        showPotential(dataFile)
    if args.job=='checkP' :
        checkPotential(dataFile)
    

    # PATCH: auto-pick device so it runs on Apple Silicon (mps) / CPU, not only CUDA
    if torch.cuda.is_available():
        device = f'cuda:{args.gpuIndex}'
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print('device =', device)
    
    
    if args.modelType=='mars_sam':
        net = conv_sam(32,dropout=args.dropout,isvit=False,isalign=args.isalign)
    elif args.modelType=='mars_sam_vit':
        net = conv_sam(32,dropout=args.dropout,isvit=True,isalign=args.isalign)
    elif args.modelType=='mars_sam_re':
        net = conv_sam(32,reRandom=True,dropout=args.dropout,)
    else:
        net =UNet(32,[1,2,2,4,8],[3,4],2,0,8,7,m,n,3,dropout=args.dropout)
        
    predictFile = f'/data/jiangyr/data/all_eqA_296_predict_{args.modelType}.h5'
    predictFile = f'/H5/all_eqA_296_predict.h5'
    net.to(device)
    
    step_per_batches = args.stepPerMiniBatch
    if args.modelType in ['mars_sam','mars_sam_vit']:
        opt = torch.optim.AdamW(
        [
            {"params": net.not_sam.parameters(),'lr':args.lr0},
            {"params": net.in_sam.parameters(),'lr':args.lr0/8}, 
        ],
            lr=args.lr0
            )
        optSGD = torch.optim.SGD(
        [
            {"params": net.not_sam.parameters(),'lr':args.lr0*10},
            {"params": net.in_sam.parameters(),'lr':args.lr0}, 
        ],
            lr=args.lr0
            )
    else:
        opt = torch.optim.AdamW(
        [
            {"params": net.parameters(),'lr':args.lr0},
            
        ],
            lr=args.lr0
            )
    #scheduler = WarmupDecayScheduler(opt, warmup_steps//step_per_batches, total_steps//step_per_batches, initial_lr, final_lr, decay_type='exponential')
    if args.job=='showModel':
        net.summary()
        exit()
    
    # PATCH: use the spatially-blocked split (no leakage) if key_train.lst /
    # key_val.lst exist next to --dataset; else fall back to the original 90/10.
    _dsdir = os.path.dirname(args.dataset)
    _ftr = os.path.join(_dsdir,'key_train.lst'); _fva = os.path.join(_dsdir,'key_val.lst')
    if os.path.exists(_ftr) and os.path.exists(_fva):
        _ktr = [k for k in open(_ftr).read().split('\n') if k]
        _kva = [k for k in open(_fva).read().split('\n') if k]
        print(f'SPATIAL split -> train {len(_ktr)}  val {len(_kva)}')
        dataSetTrain = Generator_new(trainFile,_ktr+keyL_h5_added*32,m=m,n=n,training=True)
        dataSetTest  = Generator_new(trainFile,_kva,m=m,n=n,training=True)
    else:
        dataSetTrain = Generator_new(trainFile,keyL_h5[:len(keyL_h5)//10*9]+keyL_h5_added*32,m=m,n=n,training=True)
        dataSetTest = Generator_new(trainFile,keyL_h5[len(keyL_h5)//10*9:],m=m,n=n,training=True)
    
    if getattr(args, 'fixedVal', False):
        dataSetTest = Generator_new(trainFile, sorted(dataSetTest.list), m=m, n=n,
                                    training=False, fixed_crop=True)
    train_loader_train = data_utils.DataLoader(dataSetTrain,
                                        batch_size=BatchSize,
                                        shuffle=True, drop_last=True, num_workers=0,collate_fn=collate_function,)  # PATCH: 0 workers (macOS spawn-safe)
    train_loader_test = data_utils.DataLoader(dataSetTest,
                                        batch_size=BatchSize,
                                        shuffle=not getattr(args, 'fixedVal', False),
                                        drop_last=not getattr(args, 'fixedVal', False),
                                        num_workers=0,collate_fn=collate_function)

    d0 = args.d0
    dw =args.dw
    #D = 0.
    alpha = args.alpha
    resDir = args.resDir
    if not os.path.exists(resDir):
        os.makedirs(resDir)
    plotDir = resDir+'plot/'
    modelFile = f'{resDir}/model.pth'
    h5FileNew =resDir+'new.h5'

    if args.job=='showSTD':
        for j in range(10):
            image,te,tc,scale=dataSetTest[0]
            #print(image.std(axis=(0,2,3))) 
            #print(image.mean(axis=(0,2,3)))
            #print('____')
            plt.close()
            plt.figure(figsize=(10,5))
            plt.subplot(1,2,1)
            plt.pcolor(image[0,:,:])
            plt.gca().set_aspect('equal')
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.pcolor(image[1,:,:])
            plt.gca().set_aspect('equal')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f'{plotDir}{j}.jpg',dpi=300)
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)
    if args.job=='train':
        from aio2_ema import EMATeacher, save_ema_checkpoint, load_ema_checkpoint
        from training_numerics import require_finite, clip_finite_gradients
        calEnd = torch
        if args.lossFunc =='IoU':
            lossFunc = IoU
        elif args.lossFunc =='Dice':
            lossFunc = Dice
        elif args.lossFunc =='FocalTversky':
            lossFunc = FocalTversky
        write_config(args)
        lr_mul = 1
        lr_mul0 =1
        
        count_parameters_all, count_parameters_train = count_parameters(net)
        with open(f'{resDir}parameters.txt','a') as f:
            f.write(f'{count_parameters_all} {count_parameters_train}\n')
        opt.zero_grad()
        teacher = EMATeacher(net, decay=args.emaDecay) if args.emaTeacher else None
        start_epoch = 0
        if teacher is not None:
            if args.emaResume:
                start_epoch = load_ema_checkpoint(args.emaResume, net, opt, teacher, vars(args))
            teacher_dir = os.path.join(resDir, 'teacher')
            os.makedirs(teacher_dir, exist_ok=True)
            print(f'EMA teacher enabled: decay={teacher.decay}, '
                  f'updates={teacher.num_updates}, next_epoch={start_epoch + 1}')
            import json
            with open(os.path.join(resDir, 'ema_config.json'), 'w') as f:
                json.dump(vars(args), f, indent=2)
        online_enabled = getattr(args, 'onlineCorrection', False)
        fixed_validation_enabled = getattr(args, 'fixedVal', False)
        best_checkpoints = None
        if online_enabled or fixed_validation_enabled:
            from aio2_online import (pseudo_weight, temporary_targets, OnlineDiagnostics,
                                     fixed_validation, BestCheckpoints, validation_manifest)
        if fixed_validation_enabled:
            validation_signature = validation_manifest(train_loader_test, resDir)
            best_checkpoints = BestCheckpoints(
                resDir, vars(args), getattr(teacher, 'resume_training_state', None),
                signature=validation_signature)
            baseline_student, baseline_teacher = fixed_validation(
                net, teacher, train_loader_test, device, args, lossFunc, d0, dw, alpha,
                epoch=start_epoch)
            best_checkpoints.update(
                start_epoch, float(np.mean(baseline_student)),
                float(np.mean(baseline_teacher)) if baseline_teacher else None,
                net, opt, teacher, baseline=True)
            print(f'Fixed validation baseline epoch={start_epoch}: '
                  f'student={np.mean(baseline_student):.6f}, '
                  f'teacher={np.mean(baseline_teacher) if baseline_teacher else float("nan"):.6f}')
        fixed_preview = None
        if getattr(args, 'emaPreview', False):
            from aio2_preview import FixedPreview, BatchDiagnostics, preserve_rng
            preview_dir = os.path.join(resDir, 'preview')
            with preserve_rng():
                fixed_preview = FixedPreview(
                    Generator_new, trainFile, dataSetTrain.list, preview_dir, n, m, d0,
                    args.previewCount, args.previewThreshold, args.previewBuffer, args.previewMinPixels)
            print(f'Fixed preview enabled: {len(fixed_preview.samples)} training crops; preview candidates are not training labels')
        for i in range(start_epoch, args.epochs):   # total target epochs, also on resume
            epoch_loss=0
            online_weight = pseudo_weight(i + 1, args) if online_enabled else 0.
            online_diagnostics = None
            if online_enabled and i + 1 >= args.onlineStart:
                online_diagnostics = OnlineDiagnostics(
                    i + 1, online_weight, resDir,
                    capture=((i + 1) % getattr(args, 'previewEvery', 5) == 0 or
                             i + 1 == args.onlineStart or i + 1 == args.epochs))
            batch_diagnostics = BatchDiagnostics(args.previewThreshold) if fixed_preview is not None else None
            # PU discount schedule: 0 during warmup, then linear ramp to args.pu.
            # Warmup matters: the model must first learn from the clean signal
            # before unlabeled-but-confident pixels are forgiven.
            pu_now = 0.0
            if args.pu > 0 and i >= args.puWarmup:
                pu_now = args.pu * min(1.0, (i - args.puWarmup + 1) / max(1, args.puRamp))
            net.train()
            net.in_sam.eval()
            
            if i==1:
                #opt = optSGD
                #lr_mul0 = 10
                opt.param_groups[0]['lr']=args.lr0*lr_mul0/1
                if args.modelType in ['mars_sam','mars_sam_vit']:
                    opt.param_groups[1]['lr']=args.lr0*lr_mul0/8*lr_mul
            if i==2:
                opt.param_groups[0]['lr']=args.lr0*lr_mul0/2
                if args.modelType in ['mars_sam','mars_sam_vit']:
                    opt.param_groups[1]['lr']=args.lr0*lr_mul0/16*lr_mul
            if i==4:
                opt.param_groups[0]['lr']=args.lr0*lr_mul0/4
                if args.modelType in ['mars_sam','mars_sam_vit']:
                    opt.param_groups[1]['lr']=args.lr0*lr_mul0/32*lr_mul
            if i==8:
                opt.param_groups[0]['lr']=args.lr0*lr_mul0/8
                if args.modelType in ['mars_sam','mars_sam_vit']:
                    opt.param_groups[1]['lr']=args.lr0*lr_mul0/32*lr_mul
            for j, (image,target) in enumerate(train_loader_train):
                if i==0:
                    opt.param_groups[0]['lr']=args.lr0*(1-np.exp(-j/len(train_loader_train)*3))
                    opt.param_groups[1]['lr']=args.lr0*(1-np.exp(-j/len(train_loader_train)*3))/8
                if not isinstance(image,torch.Tensor):
                    image =torch.from_numpy(image).to(device)
                    target = torch.from_numpy(target).to(device)
                    #print(target.shape)
                    #exit()
                require_finite(image, 'training input', resDir, epoch=i + 1, batch=j)
                require_finite(target, 'training distances', resDir, epoch=i + 1, batch=j)
                #if args.isOther:
                target,others =target[:,:2],target[:,2:]
                if args.k0Only:
                    target = target[:,:1]   # PATCH: focus all capacity on k0 (any fault)

                if args.swMode=='single':
                    tMin = target+0
                else:
                    tMin = target.min(dim=1,keepdim=True).values
                sw = calEnd.exp(-tMin**2/dw**2)*(1-alpha)+alpha
                
                if args.isOther:
                    sw_other = (calEnd.exp(-others**2/(dw)**2)*(1-alpha)+alpha)
                    #print(sw_other.amax(),sw_other.shape)
                    sw_other = sw_other.expand_as(sw)
                    sw = torch.where(sw>sw_other,sw,sw_other)
                
                #print(tMin.dtype)
                target = calEnd.where(target<d0,1.,0.)
                valid_01 = image[:,2:4].amin(dim=1,keepdim=True)#.values
                sw = calEnd.where(valid_01<1e-9,1,sw)
                
                sw = sw/sw.mean()
                
                inputs = image
                training_target, training_sw = target, sw
                if online_diagnostics is not None:
                    teacher_probabilities = teacher.predict(inputs)
                    require_finite(teacher_probabilities, 'teacher training prediction', resDir,
                                   epoch=i + 1, batch=j)
                    training_target, training_sw, pseudo_mask, candidate_stats = temporary_targets(
                        target, sw, teacher_probabilities, inputs, online_weight, args)
                predict = net(inputs)
                require_finite(predict, 'student training prediction', resDir, epoch=i + 1, batch=j)
                require_finite(training_sw, 'training weights', resDir, epoch=i + 1, batch=j)
                if args.lossFunc=='FocalTversky':
                    loss = lossFunc(training_target,predict,sw=training_sw,pu=pu_now)
                else:
                    loss = lossFunc(training_target,predict,sw=training_sw)
                require_finite(loss, 'training loss', resDir, epoch=i + 1, batch=j)
                if online_diagnostics is not None:
                    online_diagnostics.observe(j, target, pseudo_mask, teacher_probabilities,
                                               inputs, candidate_stats, loss.item())
                    del teacher_probabilities, pseudo_mask
                epoch_loss += loss.item()
                if batch_diagnostics is not None:
                    batch_diagnostics.observe(j, target, predict, image, loss.item())
                
                print(i,len(train_loader_train),j,epoch_loss/(j+1),loss.item())
                
                loss = loss/step_per_batches
                loss.backward()
                
                if j%step_per_batches ==0:
                    clip_finite_gradients(net, 0.1, resDir, epoch=i + 1, batch=j)
                    opt.step()
                    if teacher is not None:
                        teacher.update(net)
                    opt.zero_grad()
                    #scheduler.step()
                    pass
                    #print(i,loss.item())
                    
                #if j==64:
                #    break
            epoch_loss /= (j+1)
            with torch.no_grad():
                net.eval()
                lossL =[]
                teacher_lossL = []
                if fixed_validation_enabled:
                    lossL, teacher_lossL = fixed_validation(
                        net, teacher, train_loader_test, device, args, lossFunc, d0, dw, alpha,
                        epoch=i + 1)
                else:
                    for j, (image,target) in enumerate(train_loader_test):
                        #te[image<1/512]=999
                        #tc[image<1/512]=999
                        if not isinstance(image,torch.Tensor):
                            image =torch.from_numpy(image).to(device)
                            target = torch.from_numpy(target).to(device)
                    
                        target,others = target[:,:2],target[:,2:]
                        if args.k0Only:
                            target = target[:,:1]
                        if args.swMode=='single':
                            tMin = target+0
                        else:
                            tMin = target.min(dim=1,keepdim=True).values
                        sw = calEnd.exp(-tMin**2/dw**2)*(1-alpha)+alpha
                    
                        if args.isOther:
                            sw_other = (calEnd.exp(-others**2/(dw)**2)*(1-alpha)+alpha)
                            #print(sw_other.amax(),sw_other.shape)
                            sw_other = sw_other.expand_as(sw)
                            sw = torch.where(sw>sw_other,sw,sw_other)
                        
                        target = calEnd.where(target<d0,1.,0.)
                        valid_01 = image[:,2:4].min(dim=1,keepdim=True).values
                        sw = calEnd.where(valid_01<1e-9,1,sw)
                        sw = sw/sw.mean()
                    
                    
                        outputs = net(image)
                        if args.lossFunc=='FocalTversky':
                            # pu=0 in validation: report the honest, undiscounted loss
                            loss = lossFunc(target,outputs,sw=sw,pu=0.0).detach().cpu().item()
                        else:
                            loss = lossFunc(target,outputs,sw=sw).detach().cpu().item()
                        lossL.append(loss)
                        if teacher is not None:
                            teacher_outputs = teacher.predict(image)
                            if args.lossFunc == 'FocalTversky':
                                teacher_loss = lossFunc(target, teacher_outputs, sw=sw, pu=0.0)
                            else:
                                teacher_loss = lossFunc(target, teacher_outputs, sw=sw)
                            teacher_lossL.append(teacher_loss.item())
                            del teacher_outputs, teacher_loss
                        output_image = net.outputImage(image).detach().cpu().numpy()
                        if args.isalign:
                            inputs_shift = net.align(image).detach().cpu().numpy()
                        else:
                            inputs_shift= image.detach().cpu().numpy()
                        inputs = image.detach().cpu().numpy()
                        output = outputs.detach().cpu().numpy()
                        target = target.cpu().detach().numpy()
                        sw = sw.cpu().detach().numpy()
                        sw_other = sw_other.cpu().detach().numpy() if args.isOther else sw
                        target[target<0.5]=np.nan
                        #target[target>0.5]=0.5
                        output[output<0.5]=np.nan
                        #output[output>0.5]=0.5
                        #inputs = inputs.cpu().detach().numpy()
                        if j==0:
                            for k in range(min(8,BatchSize)):
                            
                                plt.close()
                                plt.figure(figsize=(10,15))
                                plt.subplot(3,2,1)
                                plt.imshow(inputs[k,0],cmap='gray')
                                plt.imshow(target[k,0],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                                plt.title('Extention target')
                                plt.gca().set_aspect('equal')
                            
                                plt.subplot(3,2,2)
                                plt.imshow(inputs[k,1],cmap='gray')
                                # PATCH: with --k0Only target has a single channel
                                plt.imshow(target[k,min(1,target.shape[1]-1)],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                                plt.title('Compression target')
                                plt.gca().set_aspect('equal')
                            
                                plt.subplot(3,2,3)
                                plt.imshow(np.clip(inputs[k,4:].transpose([1,2,0])/2+0.5,0,1))
                                plt.imshow(output[k,0],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                                plt.title('Extention predict')
                                plt.gca().set_aspect('equal')
                                plt.subplot(3,2,4)
                                plt.imshow(np.clip(output_image[k].transpose([1,2,0]),0,1))
                                plt.imshow(output[k,1],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                                plt.title('Compression predict')
                                plt.tight_layout()
                            
                                plt.subplot(3,2,5)
                                plt.imshow(sw[k,0],cmap='gray')
                                #plt.imshow(output[k,2],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                                plt.title('weight other')
                            
                                plt.subplot(3,2,6)
                                plt.imshow(sw_other[k,0],cmap='gray')
                                #plt.imshow(output[k,2],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                                plt.title('weight ohter')
                            
                                plt.tight_layout()
                                plt.savefig(f'{plotDir}{i}_{k}.jpg',dpi=300)
                            
                                plt.close()
                            
                                plt.figure(figsize=(10,15))
                                plt.subplot(3,2,1)
                                plt.imshow(inputs_shift[k,0],cmap='gray')
                                plt.imshow(target[k,0],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                                plt.title('Extention target')
                                plt.gca().set_aspect('equal')
                            
                                plt.subplot(3,2,2)
                                plt.imshow(inputs_shift[k,1],cmap='gray')
                                plt.imshow(target[k,min(1,target.shape[1]-1)],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                                plt.title('Compression target')
                                plt.gca().set_aspect('equal')
                            
                                plt.subplot(3,2,3)
                                plt.imshow(np.clip(inputs_shift[k,4:].transpose([1,2,0])/2+0.5,0,1))
                                plt.imshow(output[k,0],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                                plt.title('Extention predict')
                                plt.gca().set_aspect('equal')
                            
                                plt.subplot(3,2,4)
                                plt.imshow(np.clip(output_image[k].transpose([1,2,0]),0,1))
                                plt.imshow(output[k,1],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                                plt.title('Compression predict')
                            
                                plt.subplot(3,2,5)
                                plt.imshow(sw[k,0],cmap='gray')
                                #plt.imshow(output[k,2],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                                plt.title('weight other')
                            
                                plt.subplot(3,2,6)
                                plt.imshow(sw_other[k,0],cmap='gray')
                                #plt.imshow(output[k,2],cmap='bwr',alpha=0.3,vmin=0,vmax=1)
                                plt.title('weight ohter')
                                plt.tight_layout()
                                plt.savefig(f'{plotDir}{i}_{k}_shift.jpg',dpi=300)
                print(i,epoch_loss,np.mean(lossL))
                with open(f'{resDir}loss.txt','a') as f:
                    f.write(f'{i} {epoch_loss} {np.mean(lossL)}\n')
                if best_checkpoints is not None:
                    best_checkpoints.update(
                        i + 1, float(np.mean(lossL)),
                        float(np.mean(teacher_lossL)) if teacher_lossL else None,
                        net, opt, teacher)
                torch.save(net.state_dict(), modelFile)
                if teacher is not None:
                    teacher_val_loss = float(np.mean(teacher_lossL))
                    print(f'EMA epoch={i + 1}, updates={teacher.num_updates}, '
                          f'val_loss={teacher_val_loss:.6f}')
                    with open(os.path.join(resDir, 'ema_loss.txt'), 'a') as f:
                        f.write(f'{i} {teacher.num_updates} {teacher_val_loss}\n')
                    # Existing evaluate.py can read this directory directly.
                    torch.save(teacher.model.state_dict(), os.path.join(teacher_dir, 'model.pth'))
                    save_ema_checkpoint(os.path.join(resDir, 'ema_checkpoint.pth'),
                                        net, opt, teacher, i + 1, vars(args),
                                        training_state=best_checkpoints.state if best_checkpoints else None)
                    checkpoint_every = getattr(args, 'checkpointEvery', 20)
                    if best_checkpoints is not None and checkpoint_every and (i + 1) % checkpoint_every == 0:
                        stage_dir = os.path.join(resDir, 'checkpoints')
                        os.makedirs(stage_dir, exist_ok=True)
                        save_ema_checkpoint(os.path.join(stage_dir, f'epoch_{i + 1:04d}.pth'),
                                            net, opt, teacher, i + 1, vars(args),
                                            training_state=best_checkpoints.state)
                if online_diagnostics is not None:
                    online_diagnostics.write()
                if fixed_preview is not None:
                    batch_diagnostics.write(preview_dir, i + 1)
                    if (i + 1) % args.previewEvery == 0 or i + 1 == args.epochs:
                        with preserve_rng():
                            preview_path = fixed_preview.write(net, teacher, i + 1)
                        print(f'Fixed preview saved: {preview_path}')
    keyL =['tectonics_Extention','tectonics_Compression']
    if  args.job=='predict_test' :
        from PIL import Image
        calEnd = torch
        net.load_state_dict(torch.load(modelFile,map_location=torch.device('cpu')))
        dataSetTest = Generator_new(trainFile,keyL_h5[len(keyL_h5)//10*9:],m=m,n=n)
    
        train_loader_test = data_utils.DataLoader(dataSetTest,
                                            batch_size=BatchSize,
                                            shuffle=False, drop_last=True, num_workers=0,collate_fn=collate_function)
        #print(len(train_loader_test))
        plotDir=resDir+'plot_test/'
        if not os.path.exists(plotDir):
            os.makedirs(plotDir)
        for j, (image,target) in enumerate(train_loader_test):
                
            if not isinstance(image,torch.Tensor):
                image =torch.from_numpy(image).to(device)
                target = torch.from_numpy(target).to(device)
                
            
            if not isinstance(image,torch.Tensor):
                        image =torch.from_numpy(image).to(device)
                        target = torch.from_numpy(target).to(device)
                    
                    
            tMin = target.min(dim=1,keepdim=True).values
            
            
            target = calEnd.where(target<d0,1.,0.)
            
            
            sw = calEnd.exp(-tMin**2/dw**2)*(1-alpha)+alpha
            sw = sw/sw.mean()
            
            outputs = net.Predict(image)
            output_image = net.outputImage(image).detach().cpu().numpy()
            inputs = image.detach().cpu().numpy()
            if args.isalign:
                inputs_shift = net.align(image).detach().cpu().numpy()
            else:
                inputs_shift = inputs
            #theta0,theta1 = net.align.get_theta(image)
            #print(theta0[0].detach().cpu().numpy())
            #print(theta1[0].detach().cpu().numpy())
            #exit()
            output = outputs
            #output = output.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            target[target<0.5]=np.nan
            #target[target>0.5]=0.5
            output[output<0.5]=np.nan
            #output[output>0.5]=0.5
            #inputs = inputs.cpu().detach().numpy()
            
            for k in range(target.shape[0]):
                
                if False:
                    print(convertImage[k].shape)
                    image = Image.fromarray(convertImage[k])
                    image.save(f'{plotDir}image_{j}_{k}.jpg')
                    plt.close()
                    #plt.figure(figsize=(5,16))
                    plt.figure(figsize=(6,6))
                    plt.subplot(2,2,1)
                    plt.imshow(inputs[k,0],cmap='jet',rasterized=True)#,vmin=0,vmax=1)
                    plt.imshow(target[k,0],cmap='jet',vmin=0,vmax=1,rasterized=True,alpha=0.4)
                    plt.title('CTX target')
                    #plt.colorbar()
                    plt.gca().set_aspect('equal')
                    plt.subplot(2,2,2)
                    
                    plt.imshow(inputs[k,1],cmap='jet',rasterized=True)
                    plt.imshow(target[k,min(1,target.shape[1]-1)],cmap='jet',vmin=0,vmax=1,rasterized=True,alpha=0.4)
                    plt.title('mola target')
                    #plt.colorbar()
                    plt.gca().set_aspect('equal')
                    plt.tight_layout()
                    plt.subplot(2,2,3)
                    plt.imshow(inputs[k,0],cmap='jet',rasterized=True)#,vmin=0,vmax=1)
                    plt.imshow(output[k,0],cmap='gray_r',vmin=0,vmax=1,alpha=0.5)
                    plt.title('CTX predict')
                    #plt.colorbar()
                    plt.gca().set_aspect('equal')
                    plt.tight_layout()
                    plt.subplot(2,2,4)
                    plt.imshow(inputs[k,1],cmap='jet',rasterized=True)#,vmin=0,vmax=1)
                    plt.imshow(output[k,1],cmap='gray_r',vmin=0,vmax=1,alpha=0.5)
                    plt.title('mola predict')
                    #plt.colorbar()
                    plt.gca().set_aspect('equal')
                    plt.tight_layout()
                    plt.savefig(f'{plotDir}predict_{j}_{k}.jpg',dpi=300)
                
                
                plt.close()
                #plt.figure(figsize=(5,16))
                plt.figure(figsize=(10,10))
                plt.subplot(2,2,1)
                plt.imshow(inputs[k,0],cmap='gray')
                plt.imshow(target[k,0],cmap='bwr',alpha=0.2,vmin=0,vmax=1)
                plt.title('Extention target')
                plt.gca().set_aspect('equal')
                #plt.colorbar()
                plt.subplot(2,2,2)
                plt.imshow(inputs[k,1],cmap='gray')
                plt.imshow(target[k,min(1,target.shape[1]-1)],cmap='bwr',alpha=0.2,vmin=0,vmax=1)
                plt.title('Compression target')
                plt.gca().set_aspect('equal')
                #plt.colorbar()
                plt.subplot(2,2,3)
                plt.imshow(np.clip(inputs[k,4:].transpose([1,2,0])/2+0.5,0,1))
                plt.imshow(output[k,0],cmap='bwr',alpha=0.2,vmin=0,vmax=1)
                plt.title('Extention predict')
                plt.gca().set_aspect('equal')
                #plt.colorbar()
                plt.subplot(2,2,4)
                image = np.clip(output_image[k].transpose([1,2,0]),0,1)
                image[:,:,:] = image.mean(axis=2)[:,:,np.newaxis]
                plt.imshow(image)
                plt.imshow(output[k,1],cmap='bwr',alpha=0.2,vmin=0,vmax=1)
                plt.title('Compression predict')
                plt.tight_layout()
                #plt.savefig(f'{plotDir}{i}_{k}.jpg',dpi=300)
                plt.savefig(f'{plotDir}predict_{j}_{k}t.jpg',dpi=300)
                plt.imsave(f'{plotDir}{j}_{k}_ctx.jpg',inputs[k,0],cmap='gray',)
                plt.imsave(f'{plotDir}{j}_{k}_mola.jpg',inputs[k,1],cmap='gray')
                plt.imsave(f'{plotDir}{j}_{k}_tw.jpg',inputs[k,-3:].transpose(1,2,0)/2+0.5)#,cmap='bwr',vmin=0,vmax=1)
                plt.imsave(f'{plotDir}{j}_{k}_ctx_shift.jpg',inputs_shift[k,0],cmap='gray')
                plt.imsave(f'{plotDir}{j}_{k}_mola_shift.jpg',inputs_shift[k,1],cmap='gray')
                plt.imsave(f'{plotDir}{j}_{k}_tw_shift.jpg',inputs_shift[k,-3:].transpose(1,2,0)/2+0.5)
                
                
                
                
    
    
            
    if args.job == 'predict_all':
        net.load_state_dict(torch.load(modelFile))
        h5File_in = '/H5/all_eqA_296.h5'
        h5File_out = '/H5/all_eqA_296_predict.h5'
        k0 = 'tectonics_Extention'
        k1 = 'tectonics_Compression'
        d_0 = 256
        
        with torch.no_grad():
            net.eval()
            with File(h5File_in,'r') as f:
                with File(h5File_out,'w') as f_out:
                    f_out.create_dataset(k0,[1,*f['CTX'].shape],'uint8',fillvalue=0)
                    f_out.create_dataset(k1,[1,*f['CTX'].shape],'uint8',fillvalue=0)
                    f_out.create_dataset('image',[3,*f['CTX'].shape],'uint8',fillvalue=0)
                    f_out.create_dataset('fill',[1,*f['CTX'].shape],'uint8',fillvalue=0)
                    data = f[k0]
                    la_N = data.shape[0]
                    lo_N = data.shape[1]
                    laL = f['la'][:]
                    loL = f['lo'][:]
                    i = np.abs(laL+85).argmin()
                    for _ in range(10000000):
                        la = laL[i]
                        la_rad = np.deg2rad(la)
                        scale = np.abs(np.cos(la_rad))
                        d = int(d_0/scale)
                        
                        if i+d>la_N:
                            break
                        
                        la=laL[i+d//2]
                        la_rad = np.deg2rad(la)
                        scale = np.abs(np.cos(la_rad))
                        d = int(d_0/scale)
                        if i+d>la_N:
                            break
                        jL = []
                        inputsL = []
                        for j in range(0,lo_N,d-d//4):
                            print(i/la_N,j,d)
                            j = min(j,lo_N-d)
                            
                            #data0 = f[k0][i:i+d,j:j+d]
                            #data1 = f[k1][i:i+d,j:j+d]
                            
                            #data0 = resize_array_with_zoom(data0,(d_0,d_0))
                            #data1 = resize_array_with_zoom(data1,(d_0,d_0))
                            CTX = f['CTX'][i:i+d,j:j+d].astype(np.float32)
                            mole = f['image'][i:i+d,j:j+d].astype(np.float32)
                            TW = f['TW'][:,i:i+d,j:j+d].astype(np.float32)
                            
                            valid_0 = np.where(CTX>-2147483647,1,0).astype(np.float32)
                            valid_1 = np.where(mole>-2147483647,1,0).astype(np.float32)
                            
                            CTX [CTX <-2147483647]=0#np.nan
                            CTX  = CTX /255
                            CTX  = (CTX -0.5)/0.5
                            
                            TW = TW/255
                            TW = (TW-0.5)/0.5
                            
                            
                            mole[mole<-2147483647]=np.nan
                            imageMean = np.nanmean(mole)
                            mole = (mole-imageMean)/1e8
                            mole= mole/0.25
                            mole[np.isnan(mole)]=0
                            
                            
                            
                            inputs = np.stack([CTX,mole,valid_0,valid_1],axis=0)
                            
                            inputs = np.concatenate([inputs,TW],axis=0)
                            
                            inputsL.append(inputs)
                            jL.append(j)
                        W = np.ones([d,d],'float32')
                        W[:d//4] = np.where(W[:d//4]<np.linspace(0,1,d//4)[:,np.newaxis],W[:d//4],np.linspace(0,1,d//4)[:,np.newaxis])  
                        W[-(d//4):] = np.where(W[-(d//4):]<np.linspace(1,0,d//4)[:,np.newaxis],W[-(d//4):],np.linspace(1,0,d//4)[:,np.newaxis])
                        W[:,:d//4] = np.where(W[:,:d//4]<np.linspace(0,1,d//4)[np.newaxis],W[:,:d//4],np.linspace(0,1,d//4)[np.newaxis])
                        W[:,-(d//4):] = np.where(W[:,-(d//4):]<np.linspace(1,0,d//4)[np.newaxis],W[:,-(d//4):],np.linspace(1,0,d//4)[np.newaxis])
                        #W[-(d//4):] = np.linspace(1,0,d//4)[:,np.newaxis]
                        #W[:,:d//4] = np.linspace(0,1,d//4)[np.newaxis]
                        #W[:,-(d//4):] = np.linspace(1,0,d//4)[np.newaxis]
                        W *= 255#//2
                        W = W.astype('uint8')
                        inputs = np.stack(inputsL,axis=0)
                        inputs = np.stack([resize_array_with_zoom3d(inputs[:,k],(d_0,d_0)) for k in range(inputs.shape[1])],axis=1)
                        inputs = torch.from_numpy(inputs).to(device)
                        output = net.Predict(inputs)*255
                        images = net.OutputImage(inputs)* 255
                        images = np.clip(images,0,255)
                        data0 = output[:,0]
                        data1 = output[:,1]
                        data0 = resize_array_with_zoom3d(data0,(d,d))
                        data1 = resize_array_with_zoom3d(data1,(d,d))
                        images = np.stack([resize_array_with_zoom3d(images[:,k],(d,d)) for k in range(3)],axis=1)
                        d_interval = 1
                        
                        for index,j in enumerate(jL):
                            data0_ = f_out[k0][0,i+d_interval:i-d_interval+d,j+d_interval:j+d-d_interval].astype('float32')
                            data1_ = f_out[k1][0,i+d_interval:i-d_interval+d,j+d_interval:j+d-d_interval].astype('float32')
                            image_ = f_out['image'][:,i+d_interval:i-d_interval+d,j+d_interval:j+d-d_interval].astype('float32')
                            fill = f_out['fill'][0,i+d_interval:i-d_interval+d,j+d_interval:j+d-d_interval].astype('float32')
                            w = fill/255
                            data0_ = w*data0_+(1-w)*data0[index,d_interval:-d_interval,d_interval:-d_interval]
                            data1_ = w*data1_+(1-w)*data1[index,d_interval:-d_interval,d_interval:-d_interval]
                            image_ = w[np.newaxis]*image_+(1-w[np.newaxis])*images[index,:,d_interval:-d_interval,d_interval:-d_interval]
                            #data0_ = np.where(fill==0,data0[index,d_interval:-d_interval,d_interval:-d_interval].astype('uint8'),data0_//2+data0[index,d_interval:-d_interval,d_interval:-d_interval].astype('uint8')//2)
                            #data1_ = np.where(fill==0,data1[index,d_interval:-d_interval,d_interval:-d_interval].astype('uint8'),data1_//2+data1[index,d_interval:-d_interval,d_interval:-d_interval].astype('uint8')//2)
                            #image_ = np.where(np.repeat(fill[np.newaxis],3,axis=0)==0,images[index,:,d_interval:-d_interval,d_interval:-d_interval].astype('uint8'),image_//2+images[index,:,d_interval:-d_interval,d_interval:-d_interval].astype('uint8')//2)
                            f_out[k0][0,i+d_interval:i-d_interval+d,j+d_interval:j+d-d_interval] = data0_.astype('uint8')
                            f_out[k1][0,i+d_interval:i-d_interval+d,j+d_interval:j+d-d_interval] = data1_.astype('uint8')
                            f_out['image'][:,i+d_interval:i-d_interval+d,j+d_interval:j+d-d_interval] = image_.astype('uint8')
                            f_out['fill'][0,i+d_interval:i-d_interval+d,j+d_interval:j+d-d_interval] = W[d_interval:-d_interval,d_interval:-d_interval]
                            if False:
                                f_out[k0][0,i+d_interval:i+d-d_interval,j+d_interval:j+d-d_interval] = data0[index,d_interval:-d_interval,d_interval:-d_interval].astype('uint8')
                                f_out[k1][0,i+d_interval:i+d-d_interval,j+d_interval:j+d-d_interval] = data1[index,d_interval:-d_interval,d_interval:-d_interval].astype('uint8')
                                f_out['image'][:,i+d_interval:i+d-d_interval,j+d_interval:j+d-d_interval] = images[index,d_interval:-d_interval,d_interval:-d_interval].astype('uint8')
                        i += d-d//4
    if args.job == 'predict_all_f':
        net.load_state_dict(torch.load(modelFile))
        h5File_in = '/H5/all_eqA_296.h5'
        h5File_out = '/H5/all_eqA_296_predict_feature.h5'
        k0 = 'feature'
        #k1 = 'tectonics_Compression'
        d_0 = 256
        keydim = 256#256#32
        downsample=8#16
        
        with torch.no_grad():
            net.eval()
            with File(h5File_in,'r') as f:
                dim1 = f['CTX'].shape[0]//downsample
                dim2 = f['CTX'].shape[1]//downsample
                data = f['tectonics_Compression']
                la_N = data.shape[0]
                lo_N = data.shape[1]
                laL = f['la'][:]
                loL = f['lo'][:]
                laL_downsample = laL[::downsample]/2+laL[::-downsample][::-1]/2
                loL_downsample = loL[::downsample]/2+loL[::-downsample][::-1]/2
                with File(h5File_out,'w') as f_out:
                    f_out.create_dataset(k0,[keydim,dim1,dim2],'float16',fillvalue=0)
                    f_out.create_dataset('fill',[dim1,dim2],'float16',fillvalue=0)
                    f_out.create_dataset('la',data=laL_downsample)
                    f_out.create_dataset('lo',data=loL_downsample)
                    
                    
                        
                    
                    CTX_min_la = 3.5
                    polar_degree = 9
                    polar_degree_max = 2**0.5*polar_degree
                    
                    i0_degree = np.abs(laL+90-polar_degree).argmin()
                    i0_degree_max = np.abs(laL+90-polar_degree_max).argmin()
                    i0_degree_CTX = np.abs(laL+90-CTX_min_la).argmin()
                    i = np.abs(laL+85).argmin()
                    #i = np.abs(laL-85).argmin()
                    print(i,la_N)
                    for _ in range(10000000):
                        la = laL[i]
                        la_rad = np.deg2rad(la)
                        scale = np.abs(np.cos(la_rad))
                        d = int(d_0/scale)
                        
                        if i+d>la_N:
                            break
                        
                        la=laL[i+d//2]
                        la_rad = np.deg2rad(la)
                        scale = np.abs(np.cos(la_rad))
                        d = int(d_0/scale)
                        if i+d>la_N:
                            break
                        jL = []
                        inputsL = []
                        for j in range(0,lo_N,d-d//4):
                            print(i/la_N,j,d)
                            j = min(j,lo_N-d)
                            
                            
                            CTX = f['CTX'][i:i+d,j:j+d].astype(np.float32)
                            mole = f['image'][i:i+d,j:j+d].astype(np.float32)
                            TW = f['TW'][:,i:i+d,j:j+d].astype(np.float32)
                            
                            valid_0 = np.where(CTX>-2147483647,1,0).astype(np.float32)
                            valid_1 = np.where(mole>-2147483647,1,0).astype(np.float32)
                            
                            CTX [CTX <-2147483647]=0#np.nan
                            CTX  = CTX /255
                            CTX  = (CTX -0.5)/0.5
                            
                            TW = TW/255
                            TW = (TW-0.5)/0.5
                            
                            
                            mole[mole<-2147483647]=np.nan
                            imageMean = np.nanmean(mole)
                            mole = (mole-imageMean)/1e8
                            mole= mole/0.25
                            mole[np.isnan(mole)]=0
                            
                            
                            
                            inputs = np.stack([CTX,mole,valid_0,valid_1],axis=0)
                            
                            inputs = np.concatenate([inputs,TW],axis=0)
                            
                            inputsL.append(inputs)
                            jL.append(j)
                        
                        
                        W = np.ones([d//downsample,d//downsample],'float32')
                        W[:d//downsample//4] = np.where(W[:d//downsample//4]<np.linspace(0,1,d//downsample//4)[:,np.newaxis],W[:d//downsample//4],np.linspace(0,1,d//downsample//4)[:,np.newaxis])
                        W[-(d//downsample//4):] = np.where(W[-(d//downsample//4):]<np.linspace(1,0,d//downsample//4)[:,np.newaxis],W[-(d//downsample//4):],np.linspace(1,0,d//downsample//4)[:,np.newaxis])
                        W[:,:d//downsample//4] = np.where(W[:,:d//downsample//4]<np.linspace(0,1,d//downsample//4)[np.newaxis],W[:,:d//downsample//4],np.linspace(0,1,d//downsample//4)[np.newaxis])
                        W[:,-(d//downsample//4):] = np.where(W[:,-(d//downsample//4):]<np.linspace(1,0,d//downsample//4)[np.newaxis],W[:,-(d//downsample//4):],np.linspace(1,0,d//downsample//4)[np.newaxis])
                        
                        inputs = np.stack(inputsL,axis=0)
                        inputs = np.stack([resize_array_with_zoom3d(inputs[:,k],(d_0,d_0)) for k in range(inputs.shape[1])],axis=1)
                        #inputs = torch.from_numpy(inputs).to(device)
                        feature = net.outputFeature(inputs,d_0//downsample,d_0//downsample,d//downsample,d//downsample,D=d_0,batch_size=16)
                        
                        
                        for index,j in enumerate(jL):
                            
                            data0_ = f_out[k0][:,i//downsample:i//downsample+d//downsample,j//downsample:j//downsample+d//downsample].astype('float32')
                            fill = f_out['fill'][i//downsample:i//downsample+d//downsample,j//downsample:j//downsample+d//downsample].astype('float32')
                            

                            w = fill
                            w = w.reshape([1,*w.shape])
                            data0_ = w*data0_+(1-w)*feature[index]
                            f_out[k0][:,i//downsample:i//downsample+d//downsample,j//downsample:j//downsample+d//downsample] = data0_
                            
                            f_out['fill'][i//downsample:i//downsample+d//downsample,j//downsample:j//downsample+d//downsample] = W
                            
                        i += d-d//4
    
        
    if args.job == 'predict_all_new':
        #print(device)
        from scipy import interpolate
        from scipy.ndimage import zoom
        #import pyinterp
        #exit()
        #net.to('cpu')
        net.load_state_dict(torch.load(modelFile))
        #net.to(device)
        h5File_in = '/H5/all_eqA_296.h5'
        h5File_out = '/H5/all_296_predict.h5'
        k0 = 'tectonics_Extention'
        k1 = 'tectonics_Compression'
        d_0 = 256
        
        with torch.no_grad():
            net.eval()
            
            with File(h5File_in,'r') as f:
                with File(h5File_out,'a') as f_out:
                    if k0 not in f_out:
                        f_out.create_dataset(k0,[1,*f['CTX'].shape],'uint8',fillvalue=0)
                        f_out.create_dataset(k1,[1,*f['CTX'].shape],'uint8',fillvalue=0)
                        f_out.create_dataset('image',[3,*f['CTX'].shape],'uint8',fillvalue=0)
                        f_out.create_dataset('fill',[1,*f['CTX'].shape],'uint8',fillvalue=0)
                    data = f[k0]
                    la_N = data.shape[0]
                    lo_N = data.shape[1]
                    laL = f['la'][:]
                    loL = f['lo'][:]
                    CTX_min_la = 3.5
                    polar_degree = 9
                    polar_degree_max = 2**0.5*polar_degree
                    
                    i0_degree = np.abs(laL+90-polar_degree).argmin()
                    i0_degree_max = np.abs(laL+90-polar_degree_max).argmin()
                    i0_degree_CTX = np.abs(laL+90-CTX_min_la).argmin()
                    #print(laL)
                    #exit()
                    print(i0_degree,i0_degree_max)
                    ##exit()
                    if True:
                        laL_degree = laL[:i0_degree]
                        loL_degree = loL
                        laL_degree_max = laL[:i0_degree_max]
                        loL_degree_max = loL
                        
                        CTX = f['CTX'][:i0_degree_max].astype(np.float32)
                        mole = f['image'][:i0_degree_max].astype(np.float32)
                        TW = f['TW'][:,:i0_degree_max].astype(np.float32)
                        
                        #mole[:i0_degree_CTX] = -2147483647-1
                        
                        valid_0 = np.where(CTX>-2147483647,1,0).astype(np.float32)
                        valid_1 = np.where(mole>-2147483647,1,0).astype(np.float32)
                        valid_1[:i0_degree_CTX] = 0
                        valid_1 = valid_1*0
                        mole = mole*0
                        
                        CTX [CTX <-2147483647]=0#np.nan
                        CTX  = CTX /255
                        CTX  = (CTX -0.5)/0.5
                        
                        TW = TW/255
                        TW = (TW-0.5)/0.5
                        
                        
                        mole[mole<-2147483647]=np.nan
                        imageMean = np.nanmean(mole)
                        mole = (mole-imageMean)/1e8
                        mole= mole/0.25
                        mole[np.isnan(mole)]=0
                        
                        inputs = np.stack([CTX,mole,valid_0,valid_1],axis=0)
                        inputs = np.concatenate([inputs,TW],axis=0)
                        
                        #np.stack([resize_array_with_zoom3d(images[:,k],(d_0,d)) for k in range(3)],axis=1)
                        
                        
                        loL_rad = np.deg2rad(loL_degree)
                        laL_rad = np.deg2rad(90+laL_degree)
                        
                        laL_rad_max = np.deg2rad(90+laL_degree_max)
                        loL_rad_max = np.deg2rad(loL_degree_max)
                        
                        x_plane = np.concatenate([-laL_rad[::-1],laL_rad])   
                        y_plane = x_plane+0
                        
                        x_grid = x_plane.reshape([1,-1])+y_plane.reshape([-1,1])*0
                        y_grid = x_plane.reshape([1,-1])*0+y_plane.reshape([-1,1])
                        
                        laL_rad_max_grid = laL_rad_max.reshape([-1,1])+loL_rad_max.reshape([1,-1])*0
                        loL_rad_max_grid = laL_rad_max.reshape([-1,1])*0+loL_rad_max.reshape([1,-1])
                        
                        N0,M0 = laL_rad_max_grid.shape
                        N1 = N0
                        M1 = N1*4
                        scale = (N1/N0,M1/M0)
                        
                        laL_rad_max_grid = zoom(laL_rad_max_grid,scale,order=3)
                        loL_rad_max_grid= zoom(loL_rad_max_grid,scale,order=3)
                        inputs = np.stack([zoom(inputs[k],scale,order=3) for k in range(inputs.shape[0])],axis=0)
                        
                        inputs[:,2]= np.where(inputs[:,2]>0.8,1,0)
                        inputs[:,3]= np.where(inputs[:,3]>0.8,1,0)
                        
                        x_rad_max_grid = laL_rad_max_grid*np.cos(loL_rad_max_grid)
                        y_rad_max_grid = laL_rad_max_grid*np.sin(loL_rad_max_grid)
                        
                        laL_rad_grid = laL_rad.reshape([-1,1])+loL_rad.reshape([1,-1])*0
                        loL_rad_grid = laL_rad.reshape([-1,1])*0+loL_rad.reshape([1,-1])
                        x_rad_grid = laL_rad_grid*np.cos(loL_rad_grid)
                        y_rad_grid = laL_rad_grid*np.sin(loL_rad_grid)
                        
                        #inputs = inputs.reshape([inputs.shape[0],-1]).transpose([1,0])
                        
                        #laL_grid_rad,loL_grid_rad =np.xy2polar(x_rad_grid,y_rad_grid)
                        
                        laL_grid_rad = np.sqrt(x_rad_grid**2+y_rad_grid**2)
                        loL_grid_rad = np.arctan2(y_rad_grid, x_rad_grid)
                        #np.x
                        #inputs_fill = inputs[:,0,0]
                        #inputs = interpolate.interpn((laL_rad_max_grid[:,0],loL_rad_max_grid[0,:]),inputs.transpose([1,2,0]),(laL_rad_grid,loL_rad_grid),method='linear',bounds_error=False,fill_value=np.nan).transpose([2,0,1])
                        #for i in range(inputs.shape[0]):
                        #    inputs[i] = np.where(np.isnan(inputs[i]),inputs_fill[i],inputs[i])
                        #print(inputs.shape,x_rad_max_grid.reshape([-1]).shape,y_rad_max_grid.reshape([-1]).shape)
                        #print(x_rad_max_grid,y_rad_max_grid)
                        #print(x_grid,y_grid)
                        
                        
                        #inputs = inputs.res
                        
                        #if False:
                        inputs = np.stack([interpolate.griddata(
                                (x_rad_max_grid.reshape([-1]),y_rad_max_grid.reshape([-1])),
                                inputs[k].reshape([-1]),
                                (x_grid,y_grid),method='nearest') for k in range(inputs.shape[0])],axis=0)
                        
                        plt.figure(figsize=(6,6))
                        ac = plt.gca()
                        ac.pcolorfast(inputs[1,:,:])
                        plt.savefig('test_south.jpg',dpi=300)
                        plt.close()
                        
                        outputs = net.predict_total(inputs)
                        
                        plt.figure(figsize=(6,6))
                        ac = plt.gca()
                        ac.pcolorfast(outputs[1,:,:],vmin=0,vmax=1,cmap='bwr')
                        plt.savefig('test_south_out_ori.jpg',dpi=300)
                        plt.close()
                        
                        #outputs = interpolate.interpn((x_plane,y_plane),outputs.transpose([1,2,0]),x_rad_grid,y_rad_grid,method='linear').transpose([2,0,1])
                        #if False:
                        images = net.predict_total_image(inputs)
                        images = np.clip(images,0,1)
                        plt.figure(figsize=(6,6))
                        ac = plt.gca()
                        ac.imshow(images[:,:].transpose([1,2,0]))
                        plt.savefig('test_south_out_ori_image.jpg',dpi=300)
                        plt.close()
                        
                        if 'south_'+k0 in f_out:
                            del f_out['south_'+k0]
                            del f_out['south_'+k1]
                            del f_out['south_image']
                            del f_out['south_la']
                            del f_out['south_lo']
                        
                        f_out['south_'+k0] = (outputs[0,]*255).astype('uint8')
                        f_out['south_'+k1] = (outputs[1,]*255).astype('uint8')
                        f_out['south_image'] = (images[0,]*255).astype('uint8')
                        f_out['south_la'] = laL_grid_rad
                        f_out['south_lo'] = loL_grid_rad
                        
                        outputs = np.stack([
                            interpolate.griddata((x_grid.reshape([-1]),y_grid.reshape([-1])),outputs[k].reshape([-1]),(x_rad_grid,y_rad_grid),method='nearest')
                            for k in range(outputs.shape[0])],axis=0)
                        
                        images = np.stack([
                            interpolate.griddata((x_grid.reshape([-1]),y_grid.reshape([-1])),images[k].reshape([-1]),(x_rad_grid,y_rad_grid),method='nearest')
                            for k in range(images.shape[0])],axis=0)
                        
                        
                        
                        plt.figure(figsize=(6,6))
                        ac = plt.gca()
                        ac.pcolorfast(outputs[1,:,:],vmin=0,vmax=1,cmap='bwr')
                        plt.savefig('test_south_out.jpg',dpi=300)
                        plt.close()
                        
                        plt.figure(figsize=(6,6))
                        ac = plt.gca()
                        ac.imshow(images[:,:].transpose([1,2,0]),aspect='auto')
                        plt.savefig('test_south_out_image.jpg',dpi=300)
                        plt.close()
                        
                        f_out[k0][0,:i0_degree,:] = (outputs[0,:i0_degree,:]*255).astype('uint8')
                        f_out[k1][0,:i0_degree,:] = (outputs[1,:i0_degree,:]*255).astype('uint8')
                        f_out['image'][:,:i0_degree,:] = (images[:,:i0_degree,:]*255).astype('uint8')
                    
                    if True:
                        laL_degree = laL[-i0_degree:]
                        loL_degree = loL
                        laL_degree_max = laL[-i0_degree_max:]
                        loL_degree_max = loL
                        
                        CTX = f['CTX'][-i0_degree_max:].astype(np.float32)
                        mole = f['image'][-i0_degree_max:].astype(np.float32)
                        TW = f['TW'][:,-i0_degree_max:].astype(np.float32)
                        #mole[-i0_degree_CTX:] = -2147483647-1
                        
                        if False:
                            print(mole.shape)
                            #exit()
                            
                            #print(fL,fL.max(),fL.min())
                            #exit()
                            moleNew = mole[:,::10]
                            moleNew[moleNew<-2147483647]=np.nan
                            moleNew = moleNew - np.nanmean(moleNew,axis=1,keepdims=True)
                            moleNew[np.isnan(moleNew)]=0
                            moleNew =moleNew/moleNew.std(axis=1,keepdims=True)
                            moleNew =moleNew/moleNew.std(axis=1,keepdims=True)
                            fL = np.fft.fftfreq(moleNew.shape[1])
                            spec = fft.fft(moleNew,axis=1)
                            plt.figure(figsize=(6,6))
                            ac = plt.gca()
                            
                            N = moleNew.shape[1]
                            fL = fL[:N//2]
                            spec = spec[:,:N//2]
                            
                            
                            #fL = fL[:-1]/2+fL[1:]/2
                            ac.pcolorfast(fL,90-laL_degree_max,np.log(np.abs(spec))[:-1,:-1],cmap='jet',vmin=-3,vmax=3)
                            #plt.gca().set_yscale('log')
                            #plt.gca().set_ylim([1,10])
                            #plt.colorbar()
                            plt.savefig('test_north_fft.jpg',dpi=300)
                            exit()
                        
                        valid_0 = np.where(CTX>-2147483647,1,0).astype(np.float32)
                        valid_1 = np.where(mole>-2147483647,1,0).astype(np.float32)
                        valid_1[-i0_degree_CTX:] = 0
                        
                        valid_1 = valid_1*0
                        mole = mole*0
                        
                        CTX [CTX <-2147483647]=0#np.nan
                        CTX  = CTX /255
                        CTX  = (CTX -0.5)/0.5
                        
                        TW = TW/255
                        TW = (TW-0.5)/0.5
                        
                        
                        mole[mole<-2147483647]=np.nan
                        imageMean = np.nanmean(mole)
                        mole = (mole-imageMean)/1e8
                        mole= mole/0.25
                        mole[np.isnan(mole)]=0
                        
                        inputs = np.stack([CTX,mole,valid_0,valid_1],axis=0)
                        inputs = np.concatenate([inputs,TW],axis=0)
                        
                        #np.stack([resize_array_with_zoom3d(images[:,k],(d_0,d)) for k in range(3)],axis=1)
                        
                        
                        loL_rad = np.deg2rad(loL_degree)
                        laL_rad = np.deg2rad(90-laL_degree)
                        
                        laL_rad_max = np.deg2rad(90-laL_degree_max)
                        loL_rad_max = np.deg2rad(loL_degree_max)
                        
                        x_plane = np.concatenate([-laL_rad,laL_rad[::-1]])   
                        y_plane = x_plane+0
                        
                        x_grid = x_plane.reshape([1,-1])+y_plane.reshape([-1,1])*0
                        y_grid = x_plane.reshape([1,-1])*0+y_plane.reshape([-1,1])
                        
                        laL_rad_max_grid = laL_rad_max.reshape([-1,1])+loL_rad_max.reshape([1,-1])*0
                        loL_rad_max_grid = laL_rad_max.reshape([-1,1])*0+loL_rad_max.reshape([1,-1])
                        
                        N0,M0 = laL_rad_max_grid.shape
                        N1 = N0
                        M1 = N1*4
                        scale = (N1/N0,M1/M0)
                        
                        laL_rad_max_grid = zoom(laL_rad_max_grid,scale,order=3)
                        loL_rad_max_grid= zoom(loL_rad_max_grid,scale,order=3)
                        
                        inputs = np.stack([zoom(inputs[k],scale,order=3) for k in range(inputs.shape[0])],axis=0)
                        inputs[:,2]= np.where(inputs[:,2]>0.8,1,0)
                        inputs[:,3]= np.where(inputs[:,3]>0.8,1,0)
                        
                        x_rad_max_grid = laL_rad_max_grid*np.cos(loL_rad_max_grid)
                        y_rad_max_grid = laL_rad_max_grid*np.sin(loL_rad_max_grid)
                        
                        laL_rad_grid = laL_rad.reshape([-1,1])+loL_rad.reshape([1,-1])*0
                        loL_rad_grid = laL_rad.reshape([-1,1])*0+loL_rad.reshape([1,-1])
                        x_rad_grid = laL_rad_grid*np.cos(loL_rad_grid)
                        y_rad_grid = laL_rad_grid*np.sin(loL_rad_grid)
                        
                        #inputs = inputs.reshape([inputs.shape[0],-1]).transpose([1,0])
                        
                        #laL_grid_rad,loL_grid_rad =np.xy2polar(x_rad_grid,y_rad_grid)
                        
                        laL_grid_rad = np.sqrt(x_rad_grid**2+y_rad_grid**2)
                        loL_grid_rad = np.arctan2(y_rad_grid, x_rad_grid)
                        #np.x
                        #inputs_fill = inputs[:,0,0]
                        #inputs = interpolate.interpn((laL_rad_max_grid[:,0],loL_rad_max_grid[0,:]),inputs.transpose([1,2,0]),(laL_rad_grid,loL_rad_grid),method='linear',bounds_error=False,fill_value=np.nan).transpose([2,0,1])
                        #for i in range(inputs.shape[0]):
                        #    inputs[i] = np.where(np.isnan(inputs[i]),inputs_fill[i],inputs[i])
                        #print(inputs.shape,x_rad_max_grid.reshape([-1]).shape,y_rad_max_grid.reshape([-1]).shape)
                        #print(x_rad_max_grid,y_rad_max_grid)
                        #print(x_grid,y_grid)
                        
                        
                        #inputs = inputs.res
                        
                        #if False:
                        inputs = np.stack([interpolate.griddata(
                                (x_rad_max_grid.reshape([-1]),y_rad_max_grid.reshape([-1])),
                                inputs[k].reshape([-1]),
                                (x_grid,y_grid),method='nearest') for k in range(inputs.shape[0])],axis=0)
                        
                        plt.figure(figsize=(6,6))
                        ac = plt.gca()
                        ac.pcolorfast(inputs[1,:,:])
                        plt.savefig('test_north.jpg',dpi=300)
                        plt.close()
                        
                        outputs = net.predict_total(inputs)
                        
                        plt.figure(figsize=(6,6))
                        ac = plt.gca()
                        ac.pcolorfast(outputs[1,:,:],vmin=0,vmax=1,cmap='bwr')
                        plt.savefig('test_north_out_ori.jpg',dpi=300)
                        plt.close()
                        
                        #outputs = interpolate.interpn((x_plane,y_plane),outputs.transpose([1,2,0]),x_rad_grid,y_rad_grid,method='linear').transpose([2,0,1])
                        #if False:
                        images = net.predict_total_image(inputs)
                        images = np.clip(images,0,1)
                        plt.figure(figsize=(6,6))
                        ac = plt.gca()
                        ac.imshow(images[:,:].transpose([1,2,0]))
                        plt.savefig('test_north_out_ori_image.jpg',dpi=300)
                        plt.close()
                        
                        
                        if 'north_'+k0 in f_out:
                            del f_out['north_'+k0]
                            del f_out['north_'+k1]
                            del f_out['north_image']
                            del f_out['north_la']
                            del f_out['north_lo']
                        
                        f_out['north_'+k0] = (outputs[0,]*255).astype('uint8')
                        f_out['north_'+k1] = (outputs[1,]*255).astype('uint8')
                        f_out['north_image'] = (images[0,]*255).astype('uint8')
                        f_out['north_la'] = laL_grid_rad
                        f_out['north_lo'] = loL_grid_rad
                        
                        
                        
                        outputs = np.stack([
                            interpolate.griddata((x_grid.reshape([-1]),y_grid.reshape([-1])),outputs[k].reshape([-1]),(x_rad_grid,y_rad_grid),method='nearest')
                            for k in range(outputs.shape[0])],axis=0)
                        
                        images = np.stack([
                            interpolate.griddata((x_grid.reshape([-1]),y_grid.reshape([-1])),images[k].reshape([-1]),(x_rad_grid,y_rad_grid),method='nearest')
                            for k in range(images.shape[0])],axis=0)
                        
                        
                        
                        plt.figure(figsize=(6,6))
                        ac = plt.gca()
                        ac.pcolorfast(outputs[1,:,:],vmin=0,vmax=1,cmap='bwr')
                        plt.savefig('test_north_out.jpg',dpi=300)
                        plt.close()
                        
                        plt.figure(figsize=(6,6))
                        ac = plt.gca()
                        ac.imshow(images[:,:].transpose([1,2,0]),aspect='auto')
                        plt.savefig('test_north_out_image.jpg',dpi=300)
                        plt.close()
                        
                        f_out[k0][0,-i0_degree:,:] = (outputs[0,:i0_degree,:]*255).astype('uint8')
                        f_out[k1][0,-i0_degree:,:] = (outputs[1,:i0_degree,:]*255).astype('uint8')
                        f_out['image'][:,-i0_degree:,:] =(images[:,:i0_degree,:]*255).astype('uint8')
                        #exit()
                    
                    exit()
                    
                    
                    i = np.abs(laL+85).argmin()
                    for _ in range(10000000):
                        la = laL[i]
                        la_rad = np.deg2rad(la)
                        scale = np.abs(np.cos(la_rad))
                        d = int(d_0/scale)
                        
                        if i+d>la_N:
                            break
                        
                        la=laL[i+d_0//2]
                        la_rad = np.deg2rad(la)
                        scale = np.abs(np.cos(la_rad))
                        d = int(d_0/scale)
                        if i+d>la_N:
                            break
                        jL = []
                        inputsL = []
                        for j in range(0,lo_N,d-d//4):
                            print(i/la_N,j,d)
                            j = min(j,lo_N-d)
                            
                            #data0 = f[k0][i:i+d,j:j+d]
                            #data1 = f[k1][i:i+d,j:j+d]
                            
                            #data0 = resize_array_with_zoom(data0,(d_0,d_0))
                            #data1 = resize_array_with_zoom(data1,(d_0,d_0))
                            CTX = f['CTX'][i:i+d_0,j:j+d].astype(np.float32)
                            mole = f['image'][i:i+d_0,j:j+d].astype(np.float32)
                            TW = f['TW'][:,i:i+d_0,j:j+d].astype(np.float32)
                            
                            valid_0 = np.where(CTX>-2147483647,1,0).astype(np.float32)
                            valid_1 = np.where(mole>-2147483647,1,0).astype(np.float32)
                            
                            CTX [CTX <-2147483647]=0#np.nan
                            CTX  = CTX /255
                            CTX  = (CTX -0.5)/0.5
                            
                            TW = TW/255
                            TW = (TW-0.5)/0.5
                            
                            
                            mole[mole<-2147483647]=np.nan
                            imageMean = np.nanmean(mole)
                            mole = (mole-imageMean)/1e8
                            mole= mole/0.25
                            mole[np.isnan(mole)]=0
                            
                            
                            
                            inputs = np.stack([CTX,mole,valid_0,valid_1],axis=0)
                            
                            inputs = np.concatenate([inputs,TW],axis=0)
                            
                            inputsL.append(inputs)
                            jL.append(j)
                        W = np.ones([d_0,d],'float32')
                        W[:d_0//4] = np.where(W[:d_0//4]<np.linspace(0,1,d_0//4)[:,np.newaxis],W[:d_0//4],np.linspace(0,1,d_0//4)[:,np.newaxis])  
                        W[-(d_0//4):] = np.where(W[-(d_0//4):]<np.linspace(1,0,d_0//4)[:,np.newaxis],W[-(d_0//4):],np.linspace(1,0,d_0//4)[:,np.newaxis])
                        W[:,:d//4] = np.where(W[:,:d//4]<np.linspace(0,1,d//4)[np.newaxis],W[:,:d//4],np.linspace(0,1,d//4)[np.newaxis])
                        W[:,-(d//4):] = np.where(W[:,-(d//4):]<np.linspace(1,0,d//4)[np.newaxis],W[:,-(d//4):],np.linspace(1,0,d//4)[np.newaxis])
                        W *= 255#//2
                        W = W.astype('uint8')
                        inputs = np.stack(inputsL,axis=0)
                        inputs = np.stack([resize_array_with_zoom3d(inputs[:,k],(d_0,d_0)) for k in range(inputs.shape[1])],axis=1)
                        inputs = torch.from_numpy(inputs).to(device)
                        output = net.Predict(inputs)*255
                        images = net.OutputImage(inputs)* 255
                        images = np.clip(images,0,255)
                        data0 = output[:,0]
                        data1 = output[:,1]
                        data0 = resize_array_with_zoom3d(data0,(d_0,d))
                        data1 = resize_array_with_zoom3d(data1,(d_0,d))
                        images = np.stack([resize_array_with_zoom3d(images[:,k],(d_0,d)) for k in range(3)],axis=1)
                        d_interval = 1
                        
                        for index,j in enumerate(jL):
                            data0_ = f_out[k0][0,i+d_interval:i-d_interval+d_0,j+d_interval:j+d-d_interval].astype('float32')
                            data1_ = f_out[k1][0,i+d_interval:i-d_interval+d_0,j+d_interval:j+d-d_interval].astype('float32')
                            #image_ = f_out['image'][:,i+d_interval:i-d_interval+d_0,j+d_interval:j+d-d_interval].astype('float32')
                            fill = f_out['fill'][0,i+d_interval:i-d_interval+d_0,j+d_interval:j+d-d_interval].astype('float32')
                            w = fill/255
                            data0_ = w*data0_+(1-w)*data0[index,d_interval:-d_interval,d_interval:-d_interval]
                            data1_ = w*data1_+(1-w)*data1[index,d_interval:-d_interval,d_interval:-d_interval]
                            image_ = images[index,:,d_interval:-d_interval,d_interval:-d_interval]#image_#w[np.newaxis]*image_+(1-w[np.newaxis])*images[index,:,d_interval:-d_interval,d_interval:-d_interval]
                            #data0_ = np.where(fill==0,data0[index,d_interval:-d_interval,d_interval:-d_interval].astype('uint8'),data0_//2+data0[index,d_interval:-d_interval,d_interval:-d_interval].astype('uint8')//2)
                            #data1_ = np.where(fill==0,data1[index,d_interval:-d_interval,d_interval:-d_interval].astype('uint8'),data1_//2+data1[index,d_interval:-d_interval,d_interval:-d_interval].astype('uint8')//2)
                            #image_ = np.where(np.repeat(fill[np.newaxis],3,axis=0)==0,images[index,:,d_interval:-d_interval,d_interval:-d_interval].astype('uint8'),image_//2+images[index,:,d_interval:-d_interval,d_interval:-d_interval].astype('uint8')//2)
                            f_out[k0][0,i+d_interval:i-d_interval+d_0,j+d_interval:j+d-d_interval] = data0_.astype('uint8')
                            f_out[k1][0,i+d_interval:i-d_interval+d_0,j+d_interval:j+d-d_interval] = data1_.astype('uint8')
                            f_out['image'][:,i+d_interval:i-d_interval+d_0,j+d_interval:j+d-d_interval] = image_.astype('uint8')
                            f_out['fill'][0,i+d_interval:i-d_interval+d_0,j+d_interval:j+d-d_interval] = W[d_interval:-d_interval,d_interval:-d_interval]
                            if False:
                                f_out[k0][0,i+d_interval:i+d-d_interval,j+d_interval:j+d-d_interval] = data0[index,d_interval:-d_interval,d_interval:-d_interval].astype('uint8')
                                f_out[k1][0,i+d_interval:i+d-d_interval,j+d_interval:j+d-d_interval] = data1[index,d_interval:-d_interval,d_interval:-d_interval].astype('uint8')
                                f_out['image'][:,i+d_interval:i+d-d_interval,j+d_interval:j+d-d_interval] = images[index,d_interval:-d_interval,d_interval:-d_interval].astype('uint8')
                        if laL[i]>86:
                            break
                        i += d_0-d_0//4
    if args.job == 'predict_all_new_f':
        #print(device)
        from scipy import interpolate
        from scipy.ndimage import zoom
        #import pyinterp
        #exit()
        #net.to('cpu')
        net.load_state_dict(torch.load(modelFile))
        #net.to(device)
        h5File_in = '/H5/all_296.h5'
        h5File_out = '/H5/all_296_predict_feature.h5'
        k0 = 'feature'
        #k1 = 'tectonics_Compression'
        d_0 = 256
        keydim = 64
        downsample=8
        with torch.no_grad():
            net.eval()
            
            with File(h5File_in,'r') as f:
                dim1 = f['CTX'].shape[0]//downsample
                dim2 = f['CTX'].shape[1]//downsample
                data = f[k0]
                la_N = data.shape[0]
                lo_N = data.shape[1]
                laL = f['la'][:]
                loL = f['lo'][:]
                laL_downsample = laL[::downsample]/2+laL[::-downsample][::-1]/2
                loL_downsample = loL[::downsample]/2+loL[::-downsample][::-1]/2
                with File(h5File_out,'a') as f_out:
                    
                    f_out.create_dataset(k0,[keydim,dim1,dim2],'float16',fillvalue=0)
                    f_out.create_dataset('fill',[dim1,dim2],'float16',fillvalue=0)
                    f_out.create_dataset('la',data=laL_downsample)
                    f_out.create_dataset('lo',data=loL_downsample)
                    
                    
                        
                    
                    CTX_min_la = 3.5
                    polar_degree = 9
                    polar_degree_max = 2**0.5*polar_degree
                    
                    i0_degree = np.abs(laL+90-polar_degree).argmin()
                    i0_degree_max = np.abs(laL+90-polar_degree_max).argmin()
                    i0_degree_CTX = np.abs(laL+90-CTX_min_la).argmin()
                    i = np.abs(laL+85).argmin()
                    for _ in range(10000000):
                        la = laL[i]
                        la_rad = np.deg2rad(la)
                        scale = np.abs(np.cos(la_rad))
                        d = int(d_0/scale)
                        
                        if i+d>la_N:
                            break
                        
                        la=laL[i+d_0//2]
                        la_rad = np.deg2rad(la)
                        scale = np.abs(np.cos(la_rad))
                        d = int(d_0/scale)
                        if i+d>la_N:
                            break
                        jL = []
                        inputsL = []
                        for j in range(0,lo_N,d-d//4):
                            print(i/la_N,j,d)
                            j = min(j,lo_N-d)
                            
                            
                            CTX = f['CTX'][i:i+d_0,j:j+d].astype(np.float32)
                            mole = f['image'][i:i+d_0,j:j+d].astype(np.float32)
                            TW = f['TW'][:,i:i+d_0,j:j+d].astype(np.float32)
                            
                            valid_0 = np.where(CTX>-2147483647,1,0).astype(np.float32)
                            valid_1 = np.where(mole>-2147483647,1,0).astype(np.float32)
                            
                            CTX [CTX <-2147483647]=0#np.nan
                            CTX  = CTX /255
                            CTX  = (CTX -0.5)/0.5
                            
                            TW = TW/255
                            TW = (TW-0.5)/0.5
                            
                            
                            mole[mole<-2147483647]=np.nan
                            imageMean = np.nanmean(mole)
                            mole = (mole-imageMean)/1e8
                            mole= mole/0.25
                            mole[np.isnan(mole)]=0
                            
                            
                            
                            inputs = np.stack([CTX,mole,valid_0,valid_1],axis=0)
                            
                            inputs = np.concatenate([inputs,TW],axis=0)
                            
                            inputsL.append(inputs)
                            jL.append(j)
                        W = np.ones([d_0//downsample,d//downsample],'float32')
                        d_0_down = d_0//downsample
                        d_down = d//downsample
                        W[:d_0_down//4] = np.where(W[:d_0_down//4]<np.linspace(0,1,d_0_down//4)[:,np.newaxis],W[:d_0_down//4],np.linspace(0,1,d_0_down//4)[:,np.newaxis])
                        W[-(d_0_down//4):] = np.where(W[-(d_0_down//4):]<np.linspace(1,0,d_0_down//4)[:,np.newaxis],W[-(d_0_down//4):],np.linspace(1,0,d_0_down//4)[:,np.newaxis])
                        W[:,:d_down//4] = np.where(W[:,:d_down//4]<np.linspace(0,1,d_down//4)[np.newaxis],W[:,:d_down//4],np.linspace(0,1,d_down//4)[np.newaxis])
                        W[:,-(d_down//4):] = np.where(W[:,-(d_down//4):]<np.linspace(1,0,d_down//4)[np.newaxis],W[:,-(d_down//4):],np.linspace(1,0,d_down//4)[np.newaxis])
                        
                        inputs = np.stack(inputsL,axis=0)
                        inputs = np.stack([resize_array_with_zoom3d(inputs[:,k],(d_0,d_0)) for k in range(inputs.shape[1])],axis=1)
                        feature = net.outputFeature(inputs,d_0//downsample,d//downsample)
                        
                        for index,j in enumerate(jL):
                            data0_ = f_out[k0][:,i//downsample:i//downsample+d_0//downsample,j//downsample:j//downsample+d//downsample].astype('float32')
                            fill = f_out['fill'][i//downsample:i//downsample+d_0//downsample,j//downsample:j//downsample+d//downsample].astype('float32')
                            
                            w = fill.reshape([1,d_0//downsample,d//downsample])  
                            data0_ = w*data0_+(1-w)*data0[index]
                            f_out[k0][:,i//downsample:i//downsample+d_0//downsample,j//downsample:j//downsample+d//downsample] = data0_.astype('float16')
                            f_out['fill'][i//downsample:i//downsample+d_0//downsample,j//downsample:j//downsample+d//downsample] = W.astype('float16')
                            
                        if laL[i]>86:
                            break
                        i += d_0-d_0//4
                        
    if args.job == 'show_all_new':
        
        net.load_state_dict(torch.load(modelFile))
        h5File_in = '/data/jiangyr/data/all_eqA_296.h5'
        h5File = '/data/jiangyr/data/all_eqA_296_predict.h5'
        k0 = 'tectonics_Extention'
        k1 = 'tectonics_Compression'  
        
        h5File_in = '/data/jiangyr/data/all_296.h5'
        h5File = '/data/jiangyr/data/all_296_predict.h5'
        k0 = 'tectonics_Extention'
        k1 = 'tectonics_Compression'
        dDegree = 10
        size = 2048
        pltDir  = plotDir+'show_all_new/'
        if not os.path.exists(pltDir):
            os.makedirs(pltDir)
        with File(h5File_in,'r') as f:
            laL = f['la'][:]
            loL = f['lo'][:]
        with File(h5File,'r') as f:
            for la in range(-85,85,dDegree):
                la_rad = np.deg2rad(la+dDegree/2)
                scale = np.abs(np.cos(la_rad))
                dDegreeLo = int(dDegree/scale)
                for lo in range(-180,180,dDegreeLo):
                    lo = min(lo,180-dDegreeLo)
                    laIndex0 = np.abs(laL-la).argmin()
                    loIndex0 = np.abs(loL-lo).argmin()
                    laIndex1 = np.abs(laL-(la+dDegree)).argmin()
                    loIndex1 = np.abs(loL-(lo+dDegreeLo)).argmin()
                    inter = int((laIndex1-laIndex0)/size)
                    interLo = int((loIndex1-loIndex0)/size)
                    data0 = f[k0][0,laIndex0:laIndex1:inter,loIndex0:loIndex1:interLo].astype('float32')
                    data1 = f[k1][0,laIndex0:laIndex1:inter,loIndex0:loIndex1:interLo].astype('float32')
                    image = f['image'][:,laIndex0:laIndex1:inter,loIndex0:loIndex1:interLo]#.astype('float32')/255
                    image = np.clip(image,0,255).astype('uint8')
                    plt.close()
                    plt.figure(figsize=(10,10))
                    plt.imshow(image.transpose([1,2,0]))
                    data0[data0<255/2]=np.nan
                    data1[data1<255/2]=np.nan
                    
                    cmap = cm.get_cmap('bwr').copy()

                    # 设置NaN值的颜色为红色
                    cmap.set_bad(color=(1, 1, 1, 0.25),)
                    
                    plt.imshow(data0,cmap=cmap,vmin=0,vmax=255,alpha=0.3,)
                    plt.ylim(plt.ylim()[::-1])
                    plt.savefig(f'{pltDir}{la:.1f}_{lo:.1f}.jpg',dpi=300)
                    plt.close()
                    
                    plt.figure(figsize=(10,10))
                    plt.imshow(image.transpose([1,2,0]))
                    plt.imshow(data1,cmap=cmap,vmin=0,vmax=255,alpha=0.3)
                    plt.ylim(plt.ylim()[::-1])
                    plt.savefig(f'{pltDir}{la:.1f}_{lo:.1f}_1.jpg',dpi=300)
                    plt.close()
    if args.job == 'show_all':
        net.load_state_dict(torch.load(modelFile))
        h5File_in = '/H5/all_eqA_296.h5'
        h5File = '/H5/all_eqA_296_predict.h5'
        k0 = 'tectonics_Extention'
        k1 = 'tectonics_Compression'  
        dDegree = 10
        size = 2048
        pltDir  = plotDir+'show_all/'
        if not os.path.exists(pltDir):
            os.makedirs(pltDir)
        with File(h5File_in,'r') as f:
            laL = f['la'][:]
            loL = f['lo'][:]
        with File(h5File,'r') as f:
            for la in range(-85,85,dDegree):
                la_rad = np.deg2rad(la+dDegree/2)
                scale = np.abs(np.cos(la_rad))
                dDegreeLo = int(dDegree/scale)
                for lo in range(-180,180,dDegreeLo):
                    lo = min(lo,180-dDegreeLo)
                    laIndex0 = np.abs(laL-la).argmin()
                    loIndex0 = np.abs(loL-lo).argmin()
                    laIndex1 = np.abs(laL-(la+dDegree)).argmin()
                    loIndex1 = np.abs(loL-(lo+dDegreeLo)).argmin()
                    inter = int((laIndex1-laIndex0)/size)
                    interLo = int((loIndex1-loIndex0)/size)
                    data0 = f[k0][0,laIndex0:laIndex1:inter,loIndex0:loIndex1:interLo].astype('float32')
                    data1 = f[k1][0,laIndex0:laIndex1:inter,loIndex0:loIndex1:interLo].astype('float32')
                    image = f['image'][:,laIndex0:laIndex1:inter,loIndex0:loIndex1:interLo]#.astype('float32')/255
                    image = np.clip(image,0,255).astype('uint8')
                    plt.close()
                    plt.figure(figsize=(10,10))
                    plt.imshow(image.transpose([1,2,0]))
                    data0[data0<255/2]=np.nan
                    data1[data1<255/2]=np.nan
                    
                    plt.imshow(data0,cmap='jet',vmin=0,vmax=255,alpha=0.3)
                    plt.ylim(plt.ylim()[::-1])
                    plt.savefig(f'{pltDir}{la:.1f}_{lo:.1f}.jpg',dpi=300)
                    plt.close()
                    
                    plt.figure(figsize=(10,10))
                    plt.imshow(image.transpose([1,2,0]))
                    plt.imshow(data1,cmap='jet',vmin=0,vmax=255,alpha=0.3)
                    plt.ylim(plt.ylim()[::-1])
                    plt.savefig(f'{pltDir}{la:.1f}_{lo:.1f}_1.jpg',dpi=300)
                    plt.close()
                    
    
       
    
    if args.job == 'show_all_ds':
        factor = 16 
        #size = 2048
        h5File = f'/H5/all_eqA_296_predict_ds_{factor}.h5'
        k0 = 'tectonics_Extention'
        k1 = 'tectonics_Compression'  
        dDegree = 20
        size = 2048
        pltDir  = resDir+'show_all_ds/'
        if not os.path.exists(pltDir):
            os.makedirs(pltDir)
        with File(h5File,'r') as f:
            laL = f['la'][:]
            loL = f['lo'][:]
            data0 = f[k0][:].astype('float32')
            data1 = f[k1][:].astype('float32')
            image = f['image'][:]
            #print(image.shape)
            #exit()
            data0[data0<255/2]=np.nan
            data1[data1<255/2]=np.nan
            plt.figure(figsize=(20,20))
            plt.imshow(image.transpose([1,2,0]))
            plt.imshow(data0[0],cmap='bwr',vmin=0,vmax=255,alpha=0.8)
            plt.ylim(plt.ylim()[::-1])
            plt.savefig(f'{pltDir}0.jpg',dpi=600)
            plt.close()
            plt.figure(figsize=(20,20))
            plt.imshow(image.transpose([1,2,0]))
            plt.imshow(data1[0],cmap='bwr',vmin=0,vmax=255,alpha=0.8)
            plt.ylim(plt.ylim()[::-1])
            plt.savefig(f'{pltDir}1.jpg',dpi=600)
            plt.close()
    
                 
    
                        
                        
                        
    if 'predict' in sys.argv:
        net.load_weights(modelFile)
        with File(h5File,'r') as h5:
            with File(h5FileNew,'w') as h5New:
                N,M = h5['image'].shape
                for key in keyL:
                    h5New.create_dataset(key,[N,M],'float32',fillvalue=0)
                    Output = {key:calEnd.zeros([N,M])for key in keyL}
                iL=[]
                jL=[]
                dataL =[]
                nnL =[]
                mmL = []
                image =   h5['image'][:]
                laL    = h5['laL'][:]
                loL    = h5['loL'][:]
                if args.dataMode=='both':
                    ctx = h5['CTX'][:]
                    
                for i in range(0,N,n//2):
                    for j in range(0,M,m//2):
                        #print(i/N,j/M)
                        la = laL[i+n//2]
                        scale = calEnd.cos(la*np.pi/180)
                        #scale = 1
                        nn = int(n/scale)
                        mm = int(mm/scale)
                        i = min(i,N-nn)
                        #j = min(j,M-m)
                        jL =np.arange(j,j+mm)%M
                        data = image[i:i+n,iL]
                        if args.dataMode=='mola':
                            dataMean = data.mean()
                            data = (data-dataMean)/1e8#(imageMax-imageMin+1e-9)
                            data = data[:,np.newaxis]
                        elif args.dataMode=='both':
                            data1 = ctx[i:i+n,jL]
                            data1=data/255-0.5
                            data = np.stack([data,data1],axis=1)
                        else:
                            data = data/255
                        la = laL[i:i+n]/180*np.pi
                        lo = loL[jL]/180*np.pi
                        if False:
                            scale = np.cos(la).reshape([-1,1])+lo.reshape([1,-1])*0
                            if scale<0.9999999 or scale>1.000001:
                                data=zoom(data,(n/nn,m/mm),order=1)
                                data=zoom(data,(n/nn,m/mm),order=1)
                                scale=zoom(scale,(n/nn,m/mm),order=1)
                        dataL.append(data)
                        iL.append(i)
                        jL.append(j)
                        nnL.append(nn)
                        mmL.append(mm)
                        if len(dataL)>0:
                            if (i+n>=N and j+m>=M) or len(iL)==100:
                                data = np.concatenate(dataL,axis=0)
                                
                                output = net.Predict(data)
                                for index in range(len(iL)):
                                    i = iL[index]
                                    j = jL[index]
                                    nn = nnL[index]
                                    mm = mmL[index]
                                    #output = outputL[index]
                                    for k,key in enumerate(keyL):
                                        data = output[index,k]
                                        if nn!=n or mm!=m:
                                            data = zoom(data,(nn/n,mm/m),order=1)
                                        
                                        Output[key][i+nn//4:i+nn-nn//4,j+mm//4:j+mm-mm//4] = data[nn//4:-nn//4,mm//4:-mm//4]
                                iL = []
                                jL =[]
                                dataL=[]
                                nnL =[]
                                mmL = []
                for key in keyL:
                    h5New[key][:]= Output[key]
    if args.job=='showPredict':
        showDir = plotDir+'predict/'
        if not os.path.exists(showDir):
            os.makedirs(showDir)
        dI =1
        factor = 16
        h5File = f'/H5/all_eqA_296_predict_ds_{factor}.h5'
        with File(h5File,'r') as h5:
            image = h5['image_o'][:,::dI,::dI].transpose([1,2,0])
            image_con = h5['image'][:,::dI,::dI].transpose([1,2,0])
            te = h5['tectonics_Extention_o'][0,::dI,::dI]/255
            tc = h5['tectonics_Compression_o'][0,::dI,::dI]/255
            te_predict = h5['tectonics_Extention'][0,::dI,::dI].astype('float32')/255
            tc_predict = h5['tectonics_Compression'][0,::dI,::dI].astype('float32')/255
            te_predict[te_predict<args.threshold]=np.nan
            tc_predict[tc_predict<args.threshold]=np.nan
            te[te<args.d0]=np.nan
            tc[tc<args.d0]=np.nan
            te =te*0+1
            tc =tc*0+1
            
            #print(te.shape)
            #print(image.shape)
            N,M = te.shape
            n,m=[N//10,M//10]
            for i in range(0,N,n):
                for j in range(0,M,m):
                    i = min(i,N-n)
                    j = min(j,M-m)
                    plt.close()
                    plt.figure(figsize=(10,15))
                    plt.subplot(3,2,1)
                    #plt.pcolor(image[i:i+n,j:j+m],rasterized=True)
                    plt.imshow(image[i:i+n,j:j+m],rasterized=True)
                    #plt.colorbar()
                    plt.title('image')
                    plt.gca().set_aspect(1)
                    
                    plt.subplot(3,2,2)
                    plt.imshow(image_con[i:i+n,j:j+m],rasterized=True)
                    plt.title('image_con')
                    plt.gca().set_aspect(1)
                    
                    plt.subplot(3,2,3)
                    plt.imshow(image[i:i+n,j:j+m],rasterized=True)
                    plt.pcolormesh(te[i:i+n,j:j+m],cmap='bwr',vmin=0,vmax=1,rasterized=True)
                    #plt.colorbar()
                    plt.title('te')
                    plt.gca().set_aspect(1)      
                    
                    plt.subplot(3,2,4)
                    plt.imshow(image[i:i+n,j:j+m],rasterized=True)
                    plt.pcolormesh(tc[i:i+n,j:j+m],cmap='bwr',vmin=0,vmax=1,rasterized=True)
                    #plt.colorbar()
                    plt.title('tc')
                    plt.gca().set_aspect(1)
                    
                    plt.subplot(3,2,5)
                    plt.imshow(image[i:i+n,j:j+m],rasterized=True)
                    plt.pcolormesh(te_predict[i:i+n,j:j+m],cmap='bwr',vmin=0,vmax=1,rasterized=True)
                    #plt.colorbar()
                    plt.title('te_output')
                    plt.gca().set_aspect(1)       
                    
                    plt.subplot(3,2,6)
                    plt.imshow(image[i:i+n,j:j+m],rasterized=True)
                    plt.pcolormesh(tc_predict[i:i+n,j:j+m],cmap='bwr',vmin=0,vmax=1,rasterized=True)
                    #plt.colorbar()
                    plt.title('tc_output')
                    plt.gca().set_aspect(1)
                    plt.savefig(showDir+f'predict_{i}_{j}.jpg',dpi=500) 
