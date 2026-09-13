#pip install pyshp
import shapefile
import numpy as np
from scipy  import interpolate
import matplotlib.pyplot as plt
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 2248656400*2
from h5py import File
from glob import glob
import numpy as np
import shapefile
import os
import sys
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import platform
from scipy.ndimage import zoom
from PIL import ImageFile
import torchvision.transforms as transforms

# PATCH: make MPI optional (not needed for single-machine training on a Mac).
try:
    from mpi4py import MPI
    WorldComm = MPI.COMM_WORLD
    WorldRank = WorldComm.Get_rank()
    WorldSize = WorldComm.Get_size()
except Exception:
    WorldComm = None
    WorldRank = 0
    WorldSize = 1
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np

class RandomScaleRotateBatch_:
    def __init__(self, scale=(0.8, 1.2), degrees=45):
        """
        初始化变换参数。

        参数:
        scale (tuple): 随机缩放范围，默认为 (0.8, 1.2)
        degrees (int): 随机旋转的角度范围，默认为 45
        """
        self.scale = scale
        self.degrees = degrees
    
    def random_scale_rotate(self, img, size):
        """
        对单张图像进行随机缩放和旋转。

        参数:
        img (PIL.Image): 输入图像
        size (tuple): 目标尺寸

        返回:
        PIL.Image: 变换后的图像
        """
        # 随机缩放
        scale_factor = np.random.uniform(self.scale[0], self.scale[1])
        height, width = int(img.height * scale_factor), int(img.width * scale_factor)
        img = F.resize(img, (height, width))
        
        # 随机旋转
        angle = np.random.uniform(-self.degrees, self.degrees)
        img = F.rotate(img, angle)
        
        # 调整回原尺寸
        img = F.resize(img, size)
        return img
    
    def __call__(self, images):
        """
        对形状为 (B, C, H, W) 的图像批量进行随机缩放和旋转，并保持尺寸不变。

        参数:
        images (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)

        返回:
        torch.Tensor: 变换后的图像张量，形状与输入相同
        """
        batch_size, channels, height, width = images.shape
        size = (height, width)
        
        to_pil_image = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        transformed_images = torch.zeros_like(images)
        for i in range(batch_size):
            img = to_pil_image(images[i])
            transformed_img = self.random_scale_rotate(img, size)
            transformed_images[i] = to_tensor(transformed_img)

        return transformed_images
import torch
import torch.nn.functional as F
import numpy as np


def point2length(points):
    lo,la = points.transpose()
    dlo = np.diff(lo)
    dla = np.diff(la)
    scale = np.cos(np.deg2rad(la[:-1]/2+la[1:]/2))
    d = np.sqrt(scale**2*dlo**2+dla**2)
    return np.sum(d)

def resize_array_with_zoom(arr, target_shape):
    """
    使用 scipy.ndimage.zoom 将二维数组调整到指定形状 (m, n)。

    参数:
    arr (numpy.ndarray): 原始二维数组，形状为 (M, N)。
    target_shape (tuple): 目标形状 (m, n)，即插值后的行数和列数。

    返回:
    numpy.ndarray: 调整后的二维数组，形状为 (m, n)。
    """
    dtype0 = arr.dtype
    M, N = arr.shape
    m, n = target_shape
    
    # 计算缩放因子
    zoom_factors = (m / M, n / N)
    
    # 使用 zoom 函数进行插值
    resized_array = zoom(arr.astype('float32'), zoom_factors, order=3)  # order=3 使用三次样条插值

    return resized_array.astype(dtype0)
def resize_array_with_zoom3d_(arr, target_shape):
    """
    使用 scipy.ndimage.zoom 将二维数组调整到指定形状 (m, n)。

    参数:
    arr (numpy.ndarray): 原始二维数组，形状为 (M, N)。
    target_shape (tuple): 目标形状 (m, n)，即插值后的行数和列数。

    返回:
    numpy.ndarray: 调整后的二维数组，形状为 (m, n)。
    """
    dtype0 = arr.dtype
    M, N = arr.shape[1:]
    m, n = target_shape
    
    # 计算缩放因子
    zoom_factors = (1,m / M, n / N)
    
    # 使用 zoom 函数进行插值
    resized_array = zoom(arr.astype('float32'), zoom_factors, order=3)  # order=3 使用三次样条插值

    return resized_array.astype(dtype0)

from concurrent.futures import ThreadPoolExecutor

def resize_array_with_zoom3d(arr, target_shape):
    """
    使用 scipy.ndimage.zoom 将二维数组调整到指定形状 (m, n)，并行化处理。

    参数:
    arr (numpy.ndarray): 原始二维数组，形状为 (M, N)。
    target_shape (tuple): 目标形状 (m, n)，即插值后的行数和列数。

    返回:
    numpy.ndarray: 调整后的二维数组，形状为 (m, n)。
    """
    dtype0 = arr.dtype
    M, N = arr.shape[1:]
    m, n = target_shape
    
    # 计算缩放因子
    zoom_factors = ( m / M, n / N)
    
    def process_slice(i):
        return zoom(arr[i].astype('float32'), zoom_factors, order=3)
    
    # 使用 ThreadPoolExecutor 进行并行处理
    with ThreadPoolExecutor() as executor:
        resized_slices = list(executor.map(process_slice, range(arr.shape[0])))
    
    # 将结果拼接回原来的形状
    resized_array = np.stack(resized_slices)
    
    return resized_array.astype(dtype0)
class RandomScaleRotateBatch:
    def __init__(self, scale=(0.8, 1.2), degrees=45):
        """
        初始化变换参数。

        参数:
        scale (tuple): 随机缩放范围，默认为 (0.8, 1.2)
        degrees (int): 随机旋转的角度范围，默认为 45
        """
        self.scale = scale
        self.degrees = degrees
    
    def random_scale_rotate(self, img, size):
        """
        对单张图像进行随机缩放和旋转。

        参数:
        img (torch.Tensor): 输入图像，形状为 (C, H, W)
        size (tuple): 目标尺寸

        返回:
        torch.Tensor: 变换后的图像，形状为 (C, H, W)
        """
        # 随机缩放
        scale_factor = np.random.uniform(self.scale[0], self.scale[1])
        _, h, w = img.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
        
        # 随机旋转
        angle = np.random.uniform(-self.degrees, self.degrees)
        angle_rad = np.deg2rad(angle)
        theta = torch.tensor([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0]
        ], dtype=torch.float).to(img.device)
        
        grid = F.affine_grid(theta.unsqueeze(0), img.unsqueeze(0).size())
        img = F.grid_sample(img.unsqueeze(0), grid).squeeze(0)
        
        # 调整回原尺寸
        img = F.interpolate(img.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)
        return img
    
    def __call__(self, images):
        """
        对形状为 (B, C, H, W) 的图像批量进行随机缩放和旋转，并保持尺寸不变。

        参数:
        images (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)

        返回:
        torch.Tensor: 变换后的图像张量，形状与输入相同
        """
        batch_size, channels, height, width = images.shape
        size = (height, width)

        transformed_images = torch.zeros_like(images)
        for i in range(batch_size):
            img = images[i]
            transformed_img = self.random_scale_rotate(img, size)
            transformed_images[i] = transformed_img

        return transformed_images

def toInt(Str):
    if Str[0]=='-':
        return -toInt(Str[1:])
    else:
        return int(Str)
def equalArea(srcH5,dstH5):
    with File(srcH5,'r') as src:
        with File(dstH5,'w') as dst:
            la = src['la']
            lo = src['lo']
            n,m= src['image'].shape
            loIndex = np.arange(m)+0.0
            loIndexMean = loIndex.mean()
            loIndex =loIndex-loIndexMean
            for key in src:
                if len(src[key].shape)==2:
                    N,M=src[key].shape
                    if N==n and M==m:
                        dataSet = np.zeros((N,M),dtype=src[key].dtype)
                        for i in range(N):
                            La = la[i]
                            print(i)
                            scale = np.cos(np.deg2rad(La))
                            data = src[key][i].astype('float')
                            dataSet[i,:] = interpolate.interp1d(loIndex*scale,data,kind='linear',bounds_error=False,fill_value=np.nan)(loIndex).astype(dataSet.dtype)
                        data = dataSet[::20,::20]
                        dst[key] = dataSet
                        plt.close()
                        plt.pcolor(data,cmap='jet')
                        plt.savefig('show/'+key+'.jpg')
                        plt.close()
                else:
                    dst.copy(src[key],key)

def equalAngle(srcH5,dstH5):
    with File(srcH5,'r') as src:
        with File(dstH5,'w') as dst:
            la = src['la']
            lo = src['lo']
            n,m= src['image'].shape
            #loIndex = np.arange(m)+0.0
            #loIndexMean = loIndex.mean()
            #loIndex =loIndex-loIndexMean
            for key in src:
                if len(src[key].shape)==2:
                    N,M=src[key].shape
                    if N==n and M==m:
                        #dataSet=dst.create_dataset(key,(N,M),src[key].dtype,fillvalue=np.nan)
                        dataSet = np.zeros((N*2+1,M),dtype=src[key].dtype)
                        dLo = (lo[1]-lo[0])/180*np.pi
                        laNew = np.arange(-N,N+1)*dLo
                        laNew  = (2*np.arctan(np.exp(laNew))-np.pi/2)/np.pi*180
                        #dataO = src[key][:]
                        inter = 1000
                        for i0 in range(0,M,inter):
                            #La = la[i]
                            print(i0)
                            i1 = min(i0+inter,M) 
                            #scale = np.cos(np.deg2rad(La))
                            data =src[key][:,i0:i1].astype('float')
                            dataSet[:,i0:i1] = interpolate.interp1d(la,data,kind='linear',bounds_error=False,fill_value=np.nan,axis=0)(laNew).astype(dataSet.dtype)
                        data = dataSet[::40,::20]
                        dst[key] = dataSet
                        plt.close()
                        plt.pcolor(data,cmap='jet')
                        plt.savefig('show/'+key+'.jpg')
                        plt.close()
                else:
                    dst.copy(src[key],key) 
            del(dst['la'])
            dst['la']= laNew
                    
def toH5(dirname,dG=4,N=11855,trans=False,isT=True,isR=False,isMola=False,laN=999999,laMin=999999,laMax=999999,loMin=999999,loMax=999999,resFile=''):
    #convert .tif to .h5
    if isinstance(dG,list):
        dGla,dGlo = dG
    else:
        dGla = dG
        dGlo = dG
    if isinstance(N,list):
        Nla,Nlo = N
    else:
        Nla = N
        Nlo = N
    filenames = glob(dirname+'/*.tif')
    
    if laMin==999999:
        laMin = 90000 
        laMax = -100000
        loMin = 90000
        loMax = -100000
        for filename in filenames:
            loStr,laStr = filename.split('/')[-1][:-4].split('_')[-3:-1]
            lo = toInt(loStr[1:])
            la = toInt(laStr[1:])
            laMin = min(laMin,la)
            laMax = max(laMax,la)  
            loMin = min(loMin,lo)
            loMax = max(loMin,lo)
            laMax = laMax+dGla
        loMax = loMax+dGlo
        
    
    laL = np.arange(laMin,laMax,1/Nla)
    loL = np.arange(loMin,loMax,1/Nlo)
    print(laMin,laMax,loMin,loMax)
    #return
    if resFile=='':
        resFile = dirname+f'all_{int(N):d}.h5'
    count = 0
    with File(resFile,'a') as f:
        if not trans:
            if 'la' not in f:
                f['la'] = laL
                f['lo'] = loL
    for filename in filenames:
        if count>-1:
            print(filename)
            loStr,laStr = filename.split('/')[-1][:-4].split('_')[-3:-1]
            lo = toInt(loStr[1:])
            la = toInt(laStr[1:])
            laIndex = int((la-laMin)*Nla)
            loIndex = int((lo-loMin)*Nlo)
            NLa = int(Nla*dGla)+1
            NLo = int(Nlo*dGlo)+1
            if trans:
                with File(resFile,'a') as f:
                    f['image'][laIndex:laIndex+NLa,loIndex:loIndex+NLo]=f['image'][laIndex:laIndex+NLa,loIndex:loIndex+NLo][::-1]
            else:
                image=Image.open(filename)
                print(image.size)
                
                if isMola:
                    #pass
                    NLo,NLa = image.size
                    image=np.array(image)
                    image=image.reshape([NLa,2,NLo//2]).transpose([0,2,1]).reshape([NLa,NLo])
                    image = Image.fromarray(image)
                NLa = int(Nla*dGla)+1
                NLo = int(Nlo*dGlo)+1
                if image.size!=(NLo,NLa):
                    print('trans',(NLo,NLa))
                    image.thumbnail((NLo,NLa))
                
                image=np.array(image)
                print(image.shape)
                if count==0:
                    with File(resFile,'a') as f:
                        if 'image' not in f:
                            f.create_dataset('image',(len(laL),len(loL)),dtype=image.dtype)
                if isT:
                    np.array(image).transpose()
                if isR:
                    image = image[::-1]
                with File(resFile,'a') as f:
                    f['image'][laIndex:laIndex+NLa,loIndex:loIndex+NLo] = image
                if count%100==0:
                    plt.figure(figsize=(10,10))
                    a = plt.gca()   
                    a.pcolorfast(image[::4,::4],cmap='jet')
                    #plt.colorbar()
                    plt.savefig(f'show/{count}.jpg')
                    plt.close()
        else:
            print('skiping',count)
        count+=1
    with File(resFile,'r') as f: 
        plt.figure(figsize=(10,10))
        a = plt.gca()   
        a.pcolorfast(f['image'][::100,::100],cmap='jet')
        #plt.colorbar()
        plt.savefig('show/all.jpg')
        plt.close()
def toTif(sourceDir,targetDir,dG=4,N=11855,):
    #convert .tif to .h5
    if isinstance(dG,list):
        dGla,dGlo = dG
    else:
        dGla = dG
        dGlo = dG
    if isinstance(N,list):
        Nla,Nlo = N
    else:
        Nla = N
        Nlo = N
    if WorldRank==0:
        if  not os.path.exists(targetDir):
            os.makedirs(targetDir)
    filenames = glob(sourceDir+'/*.tif')
    N0 = len(filenames)
    N = N0//WorldSize+1
    i0 = WorldRank*N
    i1 = min((WorldRank+1)*N,N0)
    for filename in filenames[i0:i1]:
        basename = os.path.basename(filename)
        savename = targetDir+f'{dGla}_{dGlo}'+basename
        if os.path.exists(savename):
            print('skipping',filename)
            continue
        print(filename)
        loStr,laStr = filename.split('/')[-1][:-4].split('_')[-3:-1]
        lo = toInt(loStr[1:])
        la = toInt(laStr[1:])
        try:
            image=Image.open(filename)
        except:
            print('error',filename)
            continue
        else:
            pass
        #img_array = np.array(image)
        #img_array = np.uint16(img_array * 257)  # 将0-255映射到0-65535
        image = image.convert('F')
        # 创建16位整数模式的新图像
        #image = Image.fromarray(img_array, 'F')
                
        if image.size!=(int(Nlo*dGlo),int(Nla*dGla)):
            print('trans',(Nlo*dGlo,Nla*dGla))
            #image.thumbnail((Nlo*dGlo,Nla*dGla))
            image=image.resize((int(Nlo*dGlo),int(Nla*dGla)),Image.BILINEAR)
        #print(image.mode)
        #image=image.convert('I;16')
        image.save(savename,)
                
def getPolygon(sf,i,R=[]):
    # get shapefile polygon of index i in R
    # if fault not in R, return [],[].R =[min_lat,max_lat,min_lon,max_lon ]
    # else return x_lon,y_lat
    shape = sf.shape(i)
    
    if len(R)!=0:
        Lo0,La0,Lo1,La1=shape.bbox
        la0,la1,lo0,lo1 = R
        if La1<la0 or La0>la1 or Lo1<lo0 or Lo0>lo1:
            return [],[]
    
    x_lon = np.zeros((len(shape.points)))
    y_lat = np.zeros((len(shape.points)))
    x_lon,y_lat = np.array(shape.points).transpose()
    
    return x_lon,y_lat
def fault2dis(h5file,faultfile,maxDis=0.2,head='test',keys=['']):
    faultname = os.path.basename(faultfile)[:-4]
    sf = shapefile.Reader(faultfile)
    faultN = len(sf.shapes())
    with File(h5file,'a') as f:
        laL = f['la'][:]
        loL = f['lo'][:]
        NLa = len(laL)
        NLo = len(loL)
        if faultname in f:
            print(faultname,'already exist')
            del(f[faultname])
        f.create_dataset(faultname,(len(laL),len(loL)),dtype=np.float16,fillvalue=999)
        dis = f[faultname]
        #dis[:]=99999
        R = [laL[0],laL[-1],loL[0],loL[-1]]
        #print(R)
        #exit()
        for i in range(faultN):
            if len(keys)>0:
                key = sf.record(i)[-1]
                if key not in keys:
                    continue
            x_lon,y_lat = getPolygon(sf,i,R)
            if len(x_lon)==0:
                print('*',i)
                continue
            print(i)
            for j  in range(len(x_lon)-1):
                lo0 = x_lon[j]
                la0 = y_lat[j]
                lo1 = x_lon[j+1]
                la1 = y_lat[j+1]
                scale = np.cos(np.mean([la0,la1])*np.pi/180)
                laMax = max(la0,la1)+maxDis
                loMax = max(lo0,lo1)+maxDis/scale
                laMin = min(la0,la1)-maxDis
                loMin = min(lo0,lo1)-maxDis/scale
                loIndex0 = np.searchsorted(loL,loMin)
                loIndex1 = np.searchsorted(loL,loMax)
                laIndex0 = np.searchsorted(laL,laMin)
                laIndex1 = np.searchsorted(laL,laMax)
                if laIndex0==0 or loIndex0==0 or laIndex1==NLa or loIndex1==NLo:
                    continue
                print(loIndex0,loIndex1,laIndex0,laIndex1)
                dla0 = laL[laIndex0:laIndex1].reshape(-1,1)-la0
                dla1 = laL[laIndex0:laIndex1].reshape(-1,1)-la1 
                dlo0 = loL[loIndex0:loIndex1].reshape(1,-1)-lo0
                dlo1 = loL[loIndex0:loIndex1].reshape(1,-1)-lo1
                r = ((la0-la1)**2+(lo0-lo1)**2*scale**2)**0.5
                dla01 = la1-la0
                dlo01 = lo1-lo0
                Dis = np.abs( dla0*dlo1-dla1*dlo0)*scale/r
                Dis0 = (dla0**2+dlo0**2*scale**2)**0.5
                Dis1 = (dla1**2+dlo1**2*scale**2)**0.5
                dot0 = dla0*dla01+dlo0*dlo01*scale**2
                dot1 = -dla1*dla01-dlo1*dlo01*scale**2
                Dis[dot0<0] = Dis0[dot0<0]
                Dis[dot1<0] = Dis1[dot1<0]
                dis0 = dis[laIndex0:laIndex1,loIndex0:loIndex1]
                dis0[Dis<dis0] = Dis[Dis<dis0].astype(np.float16)
                dis[laIndex0:laIndex1,loIndex0:loIndex1] = dis0
                continue
                if True and j==len(x_lon)-2:# and i%10==0:
                    laMax = y_lat.max()+maxDis
                    loMax = x_lon.max()+maxDis/scale
                    laMin = y_lat.min()-maxDis
                    loMin = x_lon.min()-maxDis/scale
                    loIndex0 = np.searchsorted(loL,loMin)
                    loIndex1 = np.searchsorted(loL,loMax)
                    laIndex0 = np.searchsorted(laL,laMin)
                    laIndex1 = np.searchsorted(laL,laMax)
                    plt.close()
                    plt.figure(figsize=(5,8))
                    plt.subplot(2,1,1)
                    plt.pcolor(loL[loIndex0:loIndex1],laL[laIndex0:laIndex1],dis[laIndex0:laIndex1,loIndex0:loIndex1],cmap='jet',rasterized=True,vmin=0,vmax=maxDis)
                    plt.xlim(plt.xlim())
                    plt.ylim(plt.ylim())
                    plt.plot(x_lon,y_lat,'r-',linewidth=1)
                    plt.title('distance')
                    plt.gca().set_aspect(1/np.cos(np.mean(y_lat)*np.pi/180))
                    
                    plt.colorbar()
                    plt.subplot(2,1,2)
                    plt.pcolor(loL[loIndex0:loIndex1],laL[laIndex0:laIndex1],f['image'][laIndex0:laIndex1,loIndex0:loIndex1],cmap='gray',rasterized=True)
                    plt.xlim(plt.xlim())
                    plt.ylim(plt.ylim())
                    plt.colorbar()
                    plt.plot(x_lon,y_lat,'r--',linewidth=1)
                    plt.gca().set_aspect(1/np.cos(np.mean(y_lat)*np.pi/180))
                    plt.title('image')
                    plt.tight_layout()
                    plt.savefig(f'show/{head}_{x_lon.mean():.2f}_{y_lat.mean():.2f}.jpg')
                    plt.close()
                    #return

def findPotential(h5file,d=1024):
    with File(h5file,'a') as f:
        for key in f:
            
            if 'tectonics' in key:
                basename = key.split('_')[1]
                name = basename+'_potential'
                N = f[key].shape[0]
                M = f[key].shape[1]
                if name in f:
                    del(f[name])
                laloL=[]
                for i in range(0,N,d):
                    for j in range(0,M,d):
                        #print(i,j)
                        if i+d>N:
                            i = N-d
                        if j+d>M:
                            j = M-d
                        laIndex = i+d//2
                        loIndex = j+d//2
                        data =f[key][i:i+d,j:j+d]
                        #data = np.nan_to_num(data,999)
                        if np.nanmin(data)<0.05:
                            print('**************',i,j,len(laloL))#,np.nanmin(data),data)
                            #exit()
                            laloL.append([laIndex,loIndex])
                f[name] = np.array(laloL)
                f[name].attrs['d']=d
def showPotential(h5file):
    plotDir =os.path.basename(h5file)[:-3]+'/'
    with File(h5file,'a') as f:
        image = f['image']
        for key in f:
            if 'tectonics' in key:
                dis = f[key]
                basename = key.split('_')[1]
                tmpDir = plotDir+basename+'/'
                if not os.path.exists(tmpDir):
                    os.makedirs(tmpDir)
                print(tmpDir)
                name = basename+'_potential'
                #faultL = f[name]
                laloL = f[name][:]
                d = f[name].attrs['d']
                print(d)
                for laIndex,loIndex in laloL:
                    I= image[laIndex-d//2:laIndex+d//2:2,loIndex-d//2:loIndex+d//2:2]
                    plt.close()
                    plt.figure(figsize=(4,8))
                    plt.subplot(2,1,1)
                    plt.pcolor(I,cmap='gray')#,vmin=0,vmax=256)
                    plt.gca().set_aspect('equal')
                    plt.subplot(2,1,2)
                    expDis = np.exp(-dis[laIndex-d//2:laIndex+d//2:2,loIndex-d//2:loIndex+d//2:2]**2/0.0005)
                    expDis[expDis<0.1]=np.nan
                    plt.pcolor(I,cmap='gray')#,vmin=0,vmax=256,alpha=0.2)
                    plt.pcolor(expDis,cmap='hot',vmin=0,vmax=1,alpha=0.3)
                    plt.gca().set_aspect('equal')
                    plt.tight_layout()
                    plt.savefig(f'{tmpDir}{laIndex}_{loIndex}.jpg',dpi=300)
def checkPotential(h5file):
    plotDir = h5file[:-3]+'_check/'
    with File(h5file,'a') as f:
        for key in f:
            if 'tectonics' in key:
                basename = key.split('_')[1]
                tmpDir = plotDir+basename+'/'
                if not os.path.exists(tmpDir):
                    os.makedirs(tmpDir)
                print(tmpDir)
                name = basename+'_potential'
                d = f[name].attrs['d']
                newName = basename+'_potential_check'
                laloL=[]
                #faultL = f[name]
                for filename in glob(tmpDir+'*.jpg'):
                    laIndexStr,loIndexStr = os.path.basename(filename)[:-4].split('_')
                    laIndex = int(laIndexStr)
                    loIndex = int(loIndexStr)
                    laloL.append([laIndex,loIndex])
                f[newName] = np.array(laloL)
                f[newName].attrs['d']=d

def count_parameters(model):
    """
    统计给定PyTorch模型的总参数量和可训练参数量

    参数:
        model (nn.Module): PyTorch模型

    返回:
        total_params (int): 模型的总参数量
        trainable_params (int): 模型的可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

class Generator(data_utils.Dataset):
    def __init__ (self,filename,m=256,n=256,maxLoop=1000000000,potentail=False,pCheck=False,inter=1,mode='mola',loMin=-180,loMax=180,isData=False,isEq=False):
        with File(filename,'r') as f:
            if isData:
                
                self.f ={ key: f[key][:] for key in f}
                self.d = int(f['Compression_potential'].attrs['d'])
                #f = self.f
            else:
                #self.f= File(filename,'r') 
                #f = self.f
                self.d = int(f['Compression_potential'].attrs['d'])
                self.f = filename
            #self.d = d
            laL = f['la'][:]
            loL = f['lo'][:]
            self.isEq = True
            self.mode = mode
            self.laL = laL
            self.loL = loL
            self.NLa = len(laL)
            self.NLo = len(loL)
            self.m = m
            self.n = n
            self.N,self.M = f['image'].shape
            self.loMin = loMin
            self.loMax = loMax
            self.potentail = potentail
            self.inter = inter
            if potentail:
                if pCheck:
                    laloL = np.concatenate([f['Extention_potential_check'][:],f['Compression_potential_check'][:]])
                    #d = f['Compression_potential_check'].attrs['d']
                else:
                    laloL = np.concatenate([f['Extention_potential'][:],f['Compression_potential'][:]])
                    #d = f['Compression_potential'].attrs['d']
                self.laloL = laloL
        #self.f = filename
        #if isinstance(self.f ,str):
        #    self.f = File(self.f,'r')  
    def __len__(self):
        if self.potentail:
            return len(self.laloL)
            return (len(self.laloL)*self.d*self.d)//(self.n*self.m)//10
        
        else:
            return self.NLa*self.NLo//self.n//self.m
    def __getitem__(self, index):
        #print(index,len(self))
        if isinstance(self.f ,str):
            self.f = File(self.f,'r')
        inter = self.inter
        for i in range(1000000):
            if self.potentail and np.random.rand()>0.2:
                index = np.random.randint(0,len(self.laloL))    
                laIndex,loIndex = self.laloL[index]
                laIndex = int(laIndex+ self.d*(np.random.random()-0.5))-self.n//2
                loIndex = int(loIndex+ self.d*(np.random.random()-0.5))-self.m//2
            else:
                laIndex = np.random.randint(0, self.NLa- self.n-1)
                loIndex = np.random.randint(0, self.NLo- self.m-1)
            laIndex=max(0,laIndex)
            loIndex = max(0,loIndex)
            laIndex = min(self.NLa-self.n,laIndex)
            loIndex = min(self.NLo-self.m,loIndex)
            la=self.laL[laIndex+self.n//2]
            lo=self.loL[loIndex+self.m//2]
            #lo = self.loL[laIndex]
            #
            #la0 = self.laL[laIndex]
            #la1 = self.laL[laIndex+self.n]
            #scale0 = np.cos(la0*np.pi/180)
            #scale1 = np.cos(la1*np.pi/180)
            #scale = (scale0+scale1)/2
            scale = np.cos(la*np.pi/180)#是否波动
            #print(scale)
            if np.random.rand()>0.5:
                scale *= 0.5+1*np.random.rand()
            #scale =1
            M =int(np.round(self.m/scale))
            N =int(np.round(self.n/scale))
            if laIndex+N>=self.NLa:
                laIndex = self.NLa-N-1
            if loIndex+M>=self.NLo:
                loIndex = self.NLo-M-1
            if self.loL[loIndex]>=self.loMin and self.loL[(loIndex+M-1)%len(self.loL)]<self.loMax:# and self.laL[laIndex]<88 and self.laL[laIndex]>-88:
                break
        if self.mode=='mola':    
            image =self.f['mola'][laIndex:laIndex+N,np.arange(loIndex,loIndex+M)%len(self.loL)]
            image=image.astype(np.float32)#/1e9
            image[image<-2147483647]=0#np.nan
            #imageMax = image.max()
            imageMean = np.nanmean(image)
            image = (image-imageMean)/1e8#(imageMax-imageMin+1e-9)
            image = image[np.newaxis]
        elif self.mode=='both':
            #tensor([0.0958, 0.1219])
            #tensor([0.1288, 0.1453])
            #tensor([-1.4521e-05,  4.8916e-01])
            #tensor([2.5586e-05, 4.8599e-01])
            image =self.f['image'][laIndex:laIndex+N,np.arange(loIndex,loIndex+M)%len(self.loL)]
            image=image.astype(np.float32)#/1e9
            image[image<-2147483647]=np.nan
            #imageMax = image.max()
            imageMean = np.nanmean(image)
            image = (image-imageMean)/1e8#/#(imageMax-imageMin+1e-9)
            image = image/0.25
            image[np.isnan(image)]=-10
            
            image1 =self.f['CTX'][laIndex:laIndex+N,np.arange(loIndex,loIndex+M)%len(self.loL)]
            image1=image1.astype(np.float32)#/1e9
            image1[image1<-2147483647]=0#np.nan
            image1 = image1/255
            image1 = (image1-0.5)/0.12
            
            image = np.stack([image,image1],axis=0)
            
            
        else:
            image =self.f['image'][laIndex:laIndex+N,loIndex:loIndex+M].astype(np.float32)/255
        te    =  self.f['tectonics_Extention'][laIndex:laIndex+N,np.arange(loIndex,loIndex+M)%len(self.loL)].astype(np.float32)
        tc    =  self.f['tectonics_Compression'][laIndex:laIndex+N,np.arange(loIndex,loIndex+M)%len(self.loL)].astype(np.float32)
        la = self.laL[laIndex:laIndex+N]/180*np.pi
        lo = self.loL[loIndex:loIndex+M]/180*np.pi
        if scale<0.9999999 or scale>1.000001:
            #print(image.shape)
            scale_Pair = ((self.n+2)/N,(self.m+2)/M)
            image = np.stack([zoom(im,scale_Pair,order=1) for im in image],axis=0)
            image = image[:,:self.n,:self.m]
            #image=zoom(image,(self.n/N,self.m/M),order=1)
            te = zoom(te,scale_Pair,order=1)
            tc = zoom(tc,scale_Pair,order=1)
            te = te[:self.n,:self.m]
            tc = tc[:self.n,:self.m]
            #scale = np.cos(la).reshape([-1,1])+lo.reshape([1,-1])*0
            #scale = zoom(scale,(self.n/N,self.m/M),order=1)
        else:
            pass
            #scale = np.cos(la).reshape([-1,1])+lo.reshape([1,-1])*0
        scale = tc*0+scale
        inter = 1
        if np.random.rand()>0.5:
            inter = -inter
        #image=np.nan_to_num(image,0)
        te[np.isnan(te)]=999
        tc[np.isnan(tc)]=999
        #te=np.nan_to_num(te,999)
        #tc=np.nan_to_num(tc,999)
        image = image.astype(np.float32)
        image = np.clip(image,-20,20)
        
        te = te.astype(np.float32)
        tc = tc.astype(np.float32)
        scale = scale.astype(np.float32)
        image = np.where(np.isnan(image),0,image)
        te = np.where(np.isnan(te),999,te)
        tc = np.where(np.isnan(tc),999,tc)
        scale = np.where(np.isnan(scale),1,scale)
        
        if np.random.rand()>0.5:
            return image[:,::inter,::inter]+0,te[::inter,::inter]+0,tc[::inter,::inter]+0,scale[::inter,::inter]+0
        else:
            return image[:,::inter,::inter].transpose(0,2,1)+0,te[::inter,::inter].transpose()+0,tc[::inter,::inter].transpose()+0,scale[::inter,::inter].transpose()+0
from scipy.ndimage import zoom
class Generator_new(data_utils.Dataset):
    def __init__ (self,filename,key_list,m=256,n=256,training=False,isOthers=False,fixed_crop=False):
        if fixed_crop and training:
            raise ValueError('fixed_crop requires training=False')
        self.fixed_crop = fixed_crop
        self.filename = filename
        #print(key_list)
        self.list = key_list
        self.m = m
        self.n = n
        self.training = training   
        self.isOthers = isOthers 
        self.dataD = {}
    def __len__(self):
        return len(self.list)
    def __getitem__(self, index):
        index = self.list[index]
        if index in self.dataD:
            CTX,mole,te,tc,valid_0,valid_1,TW,to = self.dataD[index]
        else:
            with File(self.filename,'r') as f:
                CTX = f[index]['CTX'][:].astype(np.float32)
                mole = f[index]['image'][:].astype(np.float32)
                TW  = f[index]['TW'][:].astype(np.float32)
                
                valid_0 = np.where(CTX>-2147483647,1,0)
                valid_1 = np.where(mole>-2147483647,1,0)
                
                CTX [CTX <-2147483647]=0#np.nan
                CTX  = CTX /255
                CTX  = (CTX -0.5)/0.5
                
                TW [TW <-2147483647]=0#np.nan
                TW  = TW /255
                TW  = (TW -0.5)/0.5
                
                mole[mole<-2147483647]=np.nan
                #imageMax = image.max()
                #print(mole.min())
                imageMean = np.nanmean(mole)
                # PATCH (Earth bathymetry): Mars used /1e8 which crushes metre-scale
                # bathymetry to ~0. Normalise by per-tile std instead (scale-independent).
                imageStd = np.nanstd(mole) + 1e-6
                mole = (mole - imageMean) / imageStd
                mole[np.isnan(mole)]=0
                
                te = f[index]['k0'][:].astype(np.float32)
                tc = f[index]['k1'][:].astype(np.float32)
                
                te[te<0]=999
                tc[tc<0]=999
                
                #print(tc[tc<100].max())
                #exit()
                tc[tc>0.35]=0.35001
                te[te>0.35]=0.35001
                if 'k2' in f[index]:
                    
                    to = f[index]['k2'][:].astype(np.float32)
                    #print(to.max())
                    to[to<0]=999
                    to[to>0.35]=0.35001
                else:
                    to = np.zeros_like(te)+999
            # PATCH (v6): cap the in-RAM cache — with the multi-region global
            # dataset (1000+ tiles, ~5 MB each decompressed) unbounded caching
            # would exhaust Mac unified memory. h5 re-reads are cheap.
            if len(self.dataD) < 600:
                self.dataD[index] = CTX,mole,te,tc,valid_0,valid_1,TW,to
        #te[te<0]=999
        #tc[tc<0]=999
        
        if self.training:
            scale = np.random.rand()*0.5+0.75
            te = zoom(te.astype('float32'), scale, order=1)
            tc = zoom(tc.astype('float32'), scale, order=1)
            to = zoom(to.astype('float32'), scale, order=1)
            CTX = zoom(CTX.astype('float32'), scale, order=1)
            mole = zoom(mole.astype('float32'), scale, order=1)
            valid_0 = zoom(valid_0.astype('float32'), scale, order=1)
            valid_1 = zoom(valid_1.astype('float32'), scale, order=1)
            valid_0 = np.where(valid_0>0.5,1,0).astype('float32')
            valid_1 = np.where(valid_1>0.5,1,0).astype('float32')
            #if np.random.random()<0.2:
            #    valid_1 = valid_0*valid_1
            #    mole = np.where(valid_1>0.5,mole,0).astype('float32')
            TW = np.array([zoom(TW[c].astype('float32'), scale, order=1)for c in range(3)])
            
        else:
            te = te.astype('float32')
            tc = tc.astype('float32')
            to = to.astype('float32')
            CTX = CTX.astype('float32')
            mole = mole.astype('float32')
            valid_0 = valid_0.astype('float32')
            valid_1 = valid_1.astype('float32')
            TW = np.array([TW[c].astype('float32')for c in range(3)])
        
        if False:
            H,W = valid_0.shape
            hL = np.linspace(0,1,H).reshape(-1,1)
            wL = np.linspace(0,1,W).reshape(1,-1)
            A = np.random.rand()
            B = np.random.rand()
            C = (A+B)*np.random.rand()
            if np.random.rand()<0.15:
                if np.random.rand()<0.3:
                    valid_0= np.where(A*hL+B*wL<C,0,valid_0).astype('float32')
                elif np.random.rand()<0.4:
                    valid_0= np.where(A*hL**2+B*wL<C,0,valid_0).astype('float32')
                elif np.random.rand()<0.5:
                    valid_0= np.where(A*hL+B*wL**2<C,0,valid_0).astype('float32')
                else:
                    valid_0= np.where(A*hL**2+B*wL**2<C,0,valid_0).astype('float32')
                CTX = CTX*valid_0
            elif np.random.rand()<0.15:
                if np.random.rand()<0.3:
                    valid_1= np.where(A*hL+B*wL<C,0,valid_1).astype('float32')
                elif np.random.rand()<0.4:
                    valid_1= np.where(A*hL**2+B*wL<C,0,valid_1).astype('float32')
                elif np.random.rand()<0.5:
                    valid_1= np.where(A*hL+B*wL**2<C,0,valid_1).astype('float32')
                else:
                    valid_1= np.where(A*hL**2+B*wL**2<C,0,valid_1).astype('float32')
                mole = mole*valid_1
            
            
            if np.random.random()>0.5 and self.training:
                angle = np.random.randint(-20,20)
                CTX = torch.from_numpy(CTX).unsqueeze(0)
                mole = torch.from_numpy(mole).unsqueeze(0)
                te = torch.from_numpy(te).unsqueeze(0)
                tc = torch.from_numpy(tc).unsqueeze(0)
                valid_0 = torch.from_numpy(valid_0).unsqueeze(0)
                valid_1 = torch.from_numpy(valid_1).unsqueeze(0)
                TW = torch.from_numpy(TW)
                CTX=transforms.RandomRotation([angle,angle], )(CTX).squeeze(0).numpy()
                mole=transforms.RandomRotation([angle,angle], )(mole).squeeze(0).numpy()
                te=transforms.RandomRotation([angle,angle],fill=0.35 )(te).squeeze(0).numpy()
                tc=transforms.RandomRotation([angle,angle], fill=0.35)(tc).squeeze(0).numpy()
                valid_0=transforms.RandomRotation([angle,angle], )(valid_0).squeeze(0).numpy()
                valid_1=transforms.RandomRotation([angle,angle], )(valid_1).squeeze(0).numpy()
                TW = transforms.RandomRotation([angle,angle], )(TW).numpy()
            
        if self.fixed_crop:
            if CTX.shape[0] < self.n or CTX.shape[1] < self.m:
                raise ValueError('Fixed crop is larger than the source tile')
            i0 = (CTX.shape[0] - self.n) // 2
            j0 = (CTX.shape[1] - self.m) // 2
        else:
            i0 = np.random.randint(0,CTX.shape[0]-self.n)
            j0 = np.random.randint(0,CTX.shape[1]-self.m)
        
        valid_0 = np.where(valid_0>0.5,1.,0.).astype('float32')
        valid_1 = np.where(valid_1>0.5,1.,0.).astype('float32')
        inputs = np.concatenate([
            CTX[i0:i0+self.n,j0:j0+self.m][np.newaxis,:],
            mole[i0:i0+self.n,j0:j0+self.m][np.newaxis,:],
            valid_0[i0:i0+self.n,j0:j0+self.m][np.newaxis,:],
            valid_1[i0:i0+self.n,j0:j0+self.m][np.newaxis,:],
            TW[:,i0:i0+self.n,j0:j0+self.m],
            ])
        label = np.concatenate([
            te[i0:i0+self.n,j0:j0+self.m][np.newaxis,:],
            tc[i0:i0+self.n,j0:j0+self.m][np.newaxis,:],
            to[i0:i0+self.n,j0:j0+self.m][np.newaxis,:],
            ])
        label[label>=0.35] = 999
        # PATCH: full uniform D4 augmentation (8 symmetries of the square).
        # The old chain covered only 5/8 ops with skewed probabilities
        # (rot180 got 41%, rot90/rot270/anti-transpose were unreachable).
        # Fault lines are orientation-agnostic, so all 8 are physically valid.
        if not self.training:
            return inputs,label
        k = np.random.randint(8)
        if k & 1:
            inputs = inputs[:,::-1];        label = label[:,::-1]
        if k & 2:
            inputs = inputs[:,:,::-1];      label = label[:,:,::-1]
        if k & 4:
            inputs = inputs.transpose(0,2,1); label = label.transpose(0,2,1)
        return np.ascontiguousarray(inputs), np.ascontiguousarray(label)
        if False:
            if np.random.random()>0.8 or not self.training:
                return CTX[i0:i0+self.n,j0:j0+self.m],mole[i0:i0+self.n,j0:j0+self.m],te[i0:i0+self.n,j0:j0+self.m],tc[i0:i0+self.n,j0:j0+self.m],valid_0[i0:i0+self.n,j0:j0+self.m],valid_1[i0:i0+self.n,j0:j0+self.m]
            elif np.random.random()>0.8:
                return CTX[i0:i0+self.n,j0:j0+self.m].transpose(),mole[i0:i0+self.n,j0:j0+self.m].transpose(),te[i0:i0+self.n,j0:j0+self.m].transpose(),tc[i0:i0+self.n,j0:j0+self.m].transpose(),valid_0[i0:i0+self.n,j0:j0+self.m].transpose(),valid_1[i0:i0+self.n,j0:j0+self.m].transpose()
            elif np.random.random()>0.8:
                return CTX[i0:i0+self.n,j0:j0+self.m][::-1],mole[i0:i0+self.n,j0:j0+self.m][::-1],te[i0:i0+self.n,j0:j0+self.m][::-1],tc[i0:i0+self.n,j0:j0+self.m][::-1],valid_0[i0:i0+self.n,j0:j0+self.m][::-1],valid_1[i0:i0+self.n,j0:j0+self.m][::-1]
            elif np.random.random()>0.8:
                return CTX[i0:i0+self.n,j0:j0+self.m][:,::-1],mole[i0:i0+self.n,j0:j0+self.m][:,::-1],te[i0:i0+self.n,j0:j0+self.m][:,::-1],tc[i0:i0+self.n,j0:j0+self.m][:,::-1],valid_0[i0:i0+self.n,j0:j0+self.m][:,::-1],valid_1[i0:i0+self.n,j0:j0+self.m][:,::-1]
            else:
                return CTX[i0:i0+self.n,j0:j0+self.m][::-1,::-1],mole[i0:i0+self.n,j0:j0+self.m][::-1,::-1],te[i0:i0+self.n,j0:j0+self.m][::-1,::-1],tc[i0:i0+self.n,j0:j0+self.m][::-1,::-1],valid_0[i0:i0+self.n,j0:j0+self.m][::-1,::-1],valid_1[i0:i0+self.n,j0:j0+self.m][::-1,::-1]
        
        
        
        
import torch
torch.manual_seed(1)
#torch.cuda.manual_seed_all(1)
def select_device():
    # Check the operating system
    os_type = platform.system()

    if os_type == "Linux":
        # Assuming the system is Linux, find the CUDA device with the most free memory
        most_free_memory = 0
        best_device = None
        #torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            # Get the free memory of the current device
            torch.cuda.set_device(i)
            free_memory = torch.cuda.mem_get_info()[0]

            # Update the best device if this device has more free memory
            if free_memory > most_free_memory:
                most_free_memory = free_memory
                best_device = i
        #print(most_free_memory)
        #exit()
        return torch.device(f'cuda:{best_device}' if torch.cuda.is_available() else 'cpu')
    elif os_type == "Darwin":
        # Assuming the system is macOS, use 'mps' for Apple Silicon
        return torch.device('mps' if torch.backends.mps.is_available() and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1" else 'cpu')
    else:
        return torch.device('cpu')
def collate_function(dataL):
    result =[]
    for i in range(len(dataL[0])):
        result.append(np.concatenate([data[i][np.newaxis,:] for data in dataL]))
    return result
import h5py
import numpy as np

def downsample_h5(input_file, output_file, dataset_name, factor=10, method='average',dataset_name_out='',dtype='',d0=-1):
    with h5py.File(input_file, 'r') as f_in:
        data = f_in[dataset_name]
       
        if dataset_name_out=='':
            dataset_name_out = dataset_name
        
        #print(data.dtype)
        #exit()
        data = data[:]
        if len(data.shape)==2:
            data = data[np.newaxis]
        C, H, W = data.shape
        # 计算降采样后的形状
        H_down = H // factor
        W_down = W // factor
        
        # 处理边界情况，使得降采样可以均匀处理整个数组
        H_new = H_down * factor
        W_new = W_down * factor
        # 只取前面H_new和W_new部分，确保能整除factor
        if H_new == H:
            i0 = 0
            i1 = H_new
        else:
            i0 = (H - H_new) // 2
            i1 = i0 + H_new
        if W_new >= W-2:
            j0 = 0
            j1 = W_new
        else:
            j0 = (W - W_new) // 2
            j1 = j0 + W_new
        print('reading',dataset_name)
        data_cropped = data[:, i0:i1, j0:j1]
        del data
        if d0>0:
            d_max = 0.02
            data_cropped = np.where(data_cropped<-2147483647,data_cropped[0,0,0]*0+999.,data_cropped)
            data_cropped = np.where(data_cropped<0,data_cropped[0,0,0]*0+999.,data_cropped)
            data_cropped = np.where(data_cropped>d_max,data_cropped[0,0,0]*0,data_cropped[0,0,0]*0+255.)
        if dtype=='':
            dtype = data_cropped.dtype
        
        # 重新定义形状以进行降采样
        print('downsampling',dataset_name)
        data_reshaped = data_cropped.reshape(C, H_down, factor, W_down, factor)
        
        if method == 'average':
            downsampled = data_reshaped.mean(axis=(2, 4))
        elif method == 'max':
            downsampled = data_reshaped.max(axis=(2, 4))
        else:
            raise ValueError("Invalid method. Use 'average' or 'max'.")
        
        # 将降采样后的数组保存到新的HDF5文件中
        print('writing',dataset_name)
        with h5py.File(output_file, 'a') as f_out:
            if dataset_name_out in f_out:
                del f_out[dataset_name_out]
            f_out.create_dataset(dataset_name_out, data=downsampled.astype(dtype))
        print('done')
def fitFault(points, method='cubic'):
    """
    利用插值算法计算给定点的斜率，并返回球面上的角度 theta。

    参数:
    points (numpy.ndarray): 输入点的数组，形状为 (N, 3)，每行表示一个点 (lo, la, z)。
    method (str): 插值方法，默认为 'cubic'。

    返回:
    numpy.ndarray: 每个点的斜率角度 theta，单位为弧度。
    """
    lo, la = points.transpose()
    z = np.arange(len(lo))*0.1
    
    # 对 lo 和 la 进行整体插值
    if method == 'cubic':
        interp_lo = interpolate.CubicSpline(z, lo)
        interp_la = interpolate.CubicSpline(z, la)
    elif method == 'linear':
        interp_lo = interpolate.interp1d(z, lo, kind='linear', fill_value="extrapolate")
        interp_la = interpolate.interp1d(z, la, kind='linear', fill_value="extrapolate")
    else:
        raise ValueError("Unsupported interpolation method. Use 'cubic' or 'linear'.")
    
    # 计算导数（斜率）
    d_lo_dz = interp_lo.derivative()(z)
    d_la_dz = interp_la.derivative()(z)
    
    # 根据局部的 latitude 计算 scale 因子
    scale = np.cos(np.deg2rad(la))
    
    # 计算斜率角度 theta
    # 对 theta 进行滤波，考虑其周期性
    theta = np.arctan2(d_la_dz, d_lo_dz * scale)
    theta_unwrapped = np.unwrap(theta)  # 解开周期性，避免跳跃
    # 使用更宽的滑动窗口进行平滑
    if False:
        window_size = 15  # 滑动窗口大小
        window = np.ones(window_size) / window_size
        theta_mean = np.mean(theta_unwrapped)  # 计算均值
        theta_centered = theta_unwrapped - theta_mean  # 去均值
        theta_fft = np.fft.fft(theta_centered)  # 进行快速傅里叶变换
        freqs = np.fft.fftfreq(len(theta_centered))  # 计算频率
        cutoff = 4 / len(theta_centered)  # 设置最长周期为1/4的截止频率
        theta_fft[np.abs(freqs) < cutoff] = 0  # 去除低频分量
        theta_filtered = np.fft.ifft(theta_fft).real  # 逆变换得到滤波后的信号
        theta_smoothed = np.convolve(theta_filtered, window, mode='same')  # 滑动平均滤波
        theta_smoothed += theta_mean  # 加回均值
    theta_smoothed = theta*0+theta.mean()
    theta = np.mod(theta_smoothed + np.pi, 2 * np.pi) - np.pi  # 将结果重新映射到 [-π, π]
    return theta
    
def generatePerpendicularPoints(points, width, spacing, method='cubic'):
    """
    根据给定的断层点，生成每个点垂直于断层方向的若干点。

    参数:
    points (numpy.ndarray): 输入点的数组，形状为 (N, 2)，每行表示一个点 (lo, la)。
    width (float): 垂直方向的总宽度（单位为度）。
    spacing (float): 垂直方向点的间距（单位为度）。
    method (str): 插值方法，默认为 'cubic'。

    返回:
    numpy.ndarray: 输出点的经纬度矩阵，形状为 (N, M, 2)，其中 M 为垂直方向的点数。
    """
    lo, la = points.transpose()
    theta = fitFault(points, method=method)  # 获取每个点的斜率角度

    # 计算垂直方向的角度
    perpendicular_theta = theta + np.pi / 2

    # 计算垂直方向的点数
    num_points = int(width / spacing) + 1
    offsets = np.linspace(-width / 2, width / 2, num_points)

    # 初始化结果矩阵
    result = np.zeros((len(points), num_points, 2))

    for i, (lon, lat, angle) in enumerate(zip(lo, la, perpendicular_theta)):
        # 根据垂直方向的角度计算偏移量
        d_lo = offsets * np.sin(angle) / np.cos(np.deg2rad(lat))
        d_la = offsets * np.cos(angle)

        # 生成垂直方向的点
        result[i, :, 0] = lon + d_lo
        result[i, :, 1] = lat + d_la

    return result
from matplotlib.colors import ListedColormap

if __name__ =='__main__':
    if 'CTX' in sys.argv:
        toTif('/Volumes/My Passport*/CTX_V01/','/Volumes/ZTData/data/CTX/',dG=[4,4],N=128*8)
    if 'CTXH5' in sys.argv:
        toH5('/Volumes/ZTData/data/CTX/',dG=[4,4],N=106694/360,trans=False,isT=False,isR=True,isMola=False,laMin=-90,laMax=90,loMin=-180,loMax=180,resFile = '/Volumes/ZTData/data/CTX_296.h5')
    if 'mola' in sys.argv:
        toH5('/home/jiangyr/data/mola/',dG=[180,360],N=128,trans=False,isT=False,isR=True,isMola=True)
    if 'molaF' in sys.argv:
        #fault2dis('/home/jiangyr/data/mola/all_128.h5','tectonics_Extention.shp',maxDis=0.3)
        fault2dis('/home/jiangyr/data/mola/all_128.h5','tectonics_Compression.shp',maxDis=0.3,head='Compression')
        #fault2dis(h5File,'tectonics_Compression.shp',maxDis=0.3,head='Compression')
    
    if 'molaNew' in sys.argv:
        toH5('/data/jiangyr/data/mola/',dG=[180,360],N=106694/360,trans=False,isT=False,isR=True,isMola=True)
    if 'molaNewF' in sys.argv:
        #fault2dis('/home/jiangyr/data/mola/all_128.h5','tectonics_Extention.shp',maxDis=0.3)
        fault2dis('/home/jiangyr/data/molaNew/all_128.h5','tectonics_Compression.shp',maxDis=0.3,head='Compression')
        #fault2dis(h5File,'tectonics_Compression.shp',maxDis=0.3,head='Compression')
    if 'molaEq' in sys.argv:
        equalArea('/home/jiangyr/data/molaNew/all_296.h5','/home/jiangyr/data/molaNew/all_eq_296.h5')
    if 'molaEqA' in sys.argv:
        equalAngle('/home/jiangyr/data/molaNew/all_296.h5','/home/jiangyr/data/molaNew/all_eqA_296.h5')
    
    if 'CTXEqA' in sys.argv:
        equalAngle('/Volumes/ZTData/data/CTX_296.h5','/Volumes/ZTData/data/CTX_eqA_296.h5')
    
    if 'merge' in sys.argv:
        with File('/Volumes/ZTData/data/all_eqA_296.h5','a') as f:
            with File('/Volumes/ZTData/data/CTX_eqA_296.h5','r') as g:
                if 'CTX' in f:
                    del(f['CTX'])
                print(g['image'].dtype)
                f['CTX'] = g['image'][:].astype(np.float16)
    if 'toDis' in sys.argv:
        dataFile  = '/data/jiangyr/data/mola/all_296.h5' 
        fault2dis(dataFile,'data/tectonics_Extention.shp',maxDis=0.4,head='Extention')
        fault2dis(dataFile,'data/tectonics_Compression.shp',maxDis=0.4,head='Compression')
    if 'merge296' in sys.argv:
        with File('/data/jiangyr/data/mola/all_296.h5','a') as f:
            with File('/data/jiangyr/data/CTX_296.h5','r') as g:
                if 'CTX' in f:
                    del(f['CTX'])
                print(g['image'].dtype)
                f['CTX'] = g['image'][:].astype(np.float16)
    if 'addTiff' in sys.argv:
        h5File_in = '/data/jiangyr/data/all_296.h5'
        tiffFile = '/home/jiangyr/HX1_GRAS_MoRIC_DOM_076m_Global_00N00E_A.tif'
        import rasterio
        from scipy import signal 
        with rasterio.open(tiffFile) as src:
            dataset = src
            if False:
                print(f"Width: {dataset.width}, Height: {dataset.height}")
        
                # 打印图像的仿射变换（表示每个像素的大小及其位置）
                print(f"Affine: {dataset.transform}")
                
                # 获取图像的地理范围（边界框）
                bounds = dataset.bounds
                print(f"Bounds: {bounds}")
                
                # 获取图像的坐标参考系统（CRS）
                print(f"CRS: {dataset.crs}")

                # 读取某个像素值（例如读取左上角第一个像素的值）
                pixel_value = dataset.read(1)[0, 0]
                print(f"Value of pixel (0, 0): {pixel_value}")
            with File(h5File_in,'a') as f:
                laL = f['la'][:]
                loL = f['lo'][:]
                if 'TW' in f:
                    del f['TW']
                f.create_dataset('TW',[3,*f['tectonics_Extention'].shape],'uint8',fillvalue=0)
                laL_tiff = 90-np.arange(140387)/140386*180
                loL_tiff = np.arange(280775)/280774*360-180
                for i in range(len(laL)):
                    
                    la = laL[i]
                    laIndex_tiff = np.abs(laL_tiff-la).argmin()
                    print(i/len(laL),i,la,laIndex_tiff)
                    
                    line = src.read([1, 2, 3],window=((laIndex_tiff,laIndex_tiff+1),(0,280775)))[:,0].astype('float32')
                    #print(line.shape)
                    #exit()
                    line = signal.resample(line,len(loL),axis=-1).astype('uint8')
                    #print(line.shape)
                    #exit()
                    f['TW'][:,i] = line
                if False:
                    window = ((100, 500), (200, 600))
                    a = 140387//8, 140387//4
                    b = 280775//8, 280775//4
                    window =(a,b)
                    block = src.read(window=window) 
                    print(block.shape)
                    plt.imshow(block.transpose([1,2,0]))
                    plt.savefig(f'test.jpg',dpi=300)