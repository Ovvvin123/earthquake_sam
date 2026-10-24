
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from sam2.build_sam import build_sam2
from sam2.utils.transforms import SAM2Transforms
from sam2.sam2_image_predictor import SAM2ImagePredictor

model_cfg = "sam2_hiera_l.yaml"
sam2_checkpoint = 'sam2_hiera_large.pt'
model_cfg = "configs/sam2/sam2_hiera_s.yaml"
sam2_checkpoint = 'sam2_hiera_small.pt'
#sam2_model = build_sam2(model_cfg, sam2_checkpoint)
sam2_model_build = lambda: build_sam2(model_cfg, sam2_checkpoint,image_size=(256,256),device='cpu')

import timm
vit_build = lambda: timm.create_model("hf_hub:timm/vit_medium_patch16_gap_256.sw_in12k_ft_in1k" )


class SpatialAwareAffineAlign(nn.Module):
    def __init__(self, in_channels: int, feature_channels: int = 16, final_channels: int = 128):
        super().__init__()
        # Feature extraction with configurable channel sizes
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, stride=1),
            ResBlock(feature_channels, feature_channels*2, dropout=0),
            nn.AvgPool2d(2, 2, 1),
            ResBlock(feature_channels*2, feature_channels*2, dropout=0),
            nn.AvgPool2d(2, 2, 1),
            ResBlock(feature_channels*2, feature_channels*4, dropout=0),
            nn.AvgPool2d(2, 2, 1),
            ResBlock(feature_channels*4, feature_channels*4, dropout=0),
            nn.AvgPool2d(2, 2, 1),
            ResBlock(feature_channels*4, feature_channels*4, dropout=0),
            nn.AvgPool2d(2, 2, 1),
            ResBlock(feature_channels*4, final_channels, dropout=0),
        )
       
        # MLP head with adaptive layer sizes
        self.mlp_head = nn.Sequential(
            nn.Linear(final_channels, final_channels * 2),
            nn.ReLU(inplace=True),  # Save memory with inplace
            nn.Linear(final_channels * 2, 12)
        )

        # Initialize with identity transformation
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.constant_(self.mlp_head[-1].weight, 0.0)
        identity = torch.tensor([1, 0, 0, 0, 1, 0] * 2, dtype=torch.float)
        self.mlp_head[-1].bias.data.copy_(identity)

    def get_theta(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract transformation parameters from input tensor"""
        # Global average pooling
        feat = self.feature_extractor(x).mean(dim=[-2, -1])  # [B, C]
        theta = self.mlp_head(feat).view(-1, 2, 2, 3)  # [B, 2, 2, 3]
        return theta[:, 0], theta[:, 1]  # Split into two transformations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial transformations to input tensor"""
        B, C, H, W = x.shape

        # Get transformation parameters
        theta_0, theta_1 = self.get_theta(x)
        
        # Create grids for each transformation
        grid0 = F.affine_grid(theta_0, [B, 1, H, W], align_corners=False)
        grid1 = F.affine_grid(theta_1, [B, 3, H, W], align_corners=False)

        # Apply transformations
        x0 = F.grid_sample(x[:, :1], grid0, align_corners=False)
        x1 = F.grid_sample(x[:, 2:3], grid0, align_corners=False)
        x3 = F.grid_sample(x[:, -3:], grid1, align_corners=False)

        # Concatenate results with original channels
        return torch.cat([x0, x[:, 1:2], x1, x[:, 3:4], x3], dim=1)
    
class Reconstruct(nn.Module):
    def __init__(self, in_Channel,out_Channel,dropout=0,ch=32,head_num=1):
        super().__init__()
        #self.conv_in = nn.Conv2d(in_Channel,16,3,1)
        self.convert = nn.Sequential(
            nn.Conv2d(in_Channel, ch//2, kernel_size=3, stride=1, padding=1),
            ResBlock(
                    in_ch=ch//2, out_ch=ch,
                    dropout=dropout, attn=False,head_num=head_num,dilation=1),
            ResBlock(
                    in_ch=ch, out_ch=ch,
                    dropout=dropout, attn=False,head_num=head_num,dilation=2),
            ResBlock(
                    in_ch=ch, out_ch=ch*2,
                    dropout=dropout, attn=False,head_num=head_num,dilation=4),
        )
        self.conv_out = nn.Conv2d(ch*2,out_Channel)
    

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch,head_num=1):
        super().__init__()
        self.group_norm = nn.GroupNorm(8, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj0 = nn.Conv2d(in_ch, in_ch*2, 1, stride=1, padding=0)
        self.proj1 = nn.Conv2d(in_ch*2, in_ch, 1, stride=1, padding=0)
        self.emb = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.ac = nn.ReLU()
        self.initialize()
        self.head_num = head_num

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj1, self.proj0, self.emb]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj0.weight, gain=1e-5)
        init.xavier_uniform_(self.proj1.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        
        wL = torch.arange(W, device=x.device).float()
        hL = torch.arange(H, device=x.device).float()
        maxT =512#8*max(H,W)
        #fL = torch.arange(1,C//4+1, device=x.device).float()/maxT
        fL = maxT**(-torch.linspace(0,1,C//4, device=x.device).float())
        #if self.training:
        #    fL = fL*(0.75+0.5*torch.rand(1).to(x.device ))
        #fL_reversed = fL.flip(0)
        hf = hL[None, None, :, None] * fL[None,:, None, None]
        wf = wL[None, None, None, :] * fL[None,:,None, None]
        
        wf = wf.expand(1, C//4, H,W)*2*math.pi
        hf = hf.expand(1, C//4, H,W)*2*math.pi
        
        emb = torch.cat([torch.sin(wf), torch.cos(wf), torch.sin(hf), torch.cos(hf)], dim=1)
        emb = self.emb(emb)
        #h = x
        
        h = self.group_norm(x)+emb
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        B_cal = B*self.head_num
        C_cal = C//self.head_num
        k = k.reshape(B_cal ,C_cal, H * W)
        q = q.reshape(B_cal ,C_cal, H * W)
        v = v.reshape(B_cal ,C_cal, H * W)
        
        q = q.permute(0, 2,  1).reshape(B_cal, H * W, C_cal)
        w = torch.bmm(q, k) * (int(C_cal) ** (-0.5))
        #q = q.reshape(B, H * W, C//self.head_num,self.head_num)
        #k = k.reshape(B, C//self.head_num,self.head_num, H * W)
        assert list(w.shape) == [B_cal, H * W, H * W]
        w = F.softmax(w, dim=-1)
        #print(w.shape)

        v = v.permute(0, 2, 1).reshape(B_cal, H * W, C_cal)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B_cal, H * W, C_cal]
        #print(h.shape,H,W,C_cal)
        h = h.permute(0, 2, 1,).reshape(B, C, H,W)
        h = self.proj0(h)
        h = self.ac(h)
        h = self.proj1(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout, attn=False,head_num=1,dilation=1):
        super().__init__()
        padding = ((3-1)*dilation)//2
        if dropout > 0:
            Dropout = nn.Dropout(dropout)
        else:
            Dropout = nn.Identity()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=padding,dilation=dilation),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            Swish(),
            Dropout,
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=padding,dilation=dilation),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch,head_num=head_num)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x):
        h = self.block1(x)
        #h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h

class Clip(nn.Module):
    def __init__(self, min_val, max_val):
        super(Clip, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        #x = 
        return torch.sigmoid(x)*(self.max_val-self.min_val)+self.min_val
class conv_sam(nn.Module):
    def __init__(self, ch, sam_build=sam2_model_build,head_num=4,dropout=0,in_Channel=7,out_Channel=3,mask_threshold=0.0,max_hole_area=0.0,max_sprinkle_area=0.0,n=256,m=256,reRandom=False,isvit=True,isalign=False,adaptation=None):
        self.n = n
        self.m = m
        self.out_Channel = out_Channel
        self.isvit = isvit
        
        super().__init__()
        self.sam = sam_build()
        self.isalign = isalign
        if isalign:
            self.align = SpatialAwareAffineAlign(in_Channel)
        if isvit:
            self.vit = vit_build()
            vitChannel = 512
            self.convvit = nn.Conv2d(vitChannel, ch*8, kernel_size=1, stride=1, padding=0)
            
            self.input_vit = nn.Sequential(
                nn.Conv2d(ch*2, 3, kernel_size=3, stride=1, padding=1),)
                #Clip(-1,1),)
        
        if reRandom:
            print('reRandom')
            exit()
            for param in self.sam.parameters():
                if param.requires_grad:
                    #if len(param.shape)>1:
                    init.xavier_uniform_(param)
        
        self.head = nn.Sequential(
            nn.Conv2d(in_Channel, ch//2, kernel_size=3, stride=1, padding=1),
            ResBlock(
                    in_ch=ch//2, out_ch=ch,
                    dropout=dropout, attn=False,head_num=head_num,dilation=1),
            ResBlock(
                    in_ch=ch, out_ch=ch,
                    dropout=dropout, attn=False,head_num=head_num,dilation=2),
            ResBlock(
                    in_ch=ch, out_ch=ch*2,
                    dropout=dropout, attn=False,head_num=head_num,dilation=4),
        )
        self.input_sam = nn.Sequential(
                nn.Conv2d(ch*2, 3, kernel_size=3, stride=1, padding=1))
        
        #if isvit:
            
        self.conv0 = nn.Sequential(
            nn.Conv2d(ch*2, ch*8, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(16, stride=16),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch*2, ch*2, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(8, stride=8),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch*2, ch, kernel_size=1, stride=1, padding=0),
            nn.AvgPool2d(4, stride=4),  
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch*2, ch, kernel_size=1, stride=1, padding=0),
        )
        self.conv_sam_0 = nn.Conv2d(ch*8, ch*8, kernel_size=1, stride=1, padding=0)
        self.conv_sam_1 = nn.Conv2d(ch*2, ch*2, kernel_size=1, stride=1, padding=0)
        self.conv_sam_2 = nn.Conv2d(ch, ch, kernel_size=1, stride=1, padding=0)
        self.upblock0 = nn.Sequential(
        ResBlock(
                    in_ch=ch*8, out_ch=ch*2,
                    dropout=dropout, attn=True,head_num=head_num),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.upblock1 = nn.Sequential(
        ResBlock(
                    in_ch=ch*2, out_ch=ch,
                    dropout=dropout, attn=True,head_num=head_num//2),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.upblock2 = nn.Sequential(
            ResBlock(
                        in_ch=ch, out_ch=ch,
                        dropout=dropout, attn=False,head_num=head_num),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            )
        self.tail = nn.Sequential(
            ResBlock(
                    in_ch=ch, out_ch=ch,
                    dropout=dropout, attn=False,head_num=head_num),
            nn.Conv2d(ch, out_Channel, 3, stride=1, padding=1),
            nn.Softmax(dim=1)
        )
    # Common modules for both cases
        base_not_sam = [
            'head', 'input_sam', 'upblock0', 'upblock1', 'upblock2', 'tail',
            'conv0', 'conv1', 'conv2', 'conv3', 'conv_sam_0', 'conv_sam_1', 'conv_sam_2'
        ]

       
        if self.isvit:
            base_not_sam = base_not_sam[:-3]+['convvit','input_vit']+base_not_sam[-3:]
        # Create in_sam ModuleList
        # Create ModuleList based on configuration
        self.not_sam = nn.ModuleList([getattr(self, name) for name in base_not_sam])
        self.in_sam = nn.ModuleList([self.sam])
        if self.isalign:
            self.in_sam.append(self.align)
        if self.isvit:
            self.in_sam.append(self.vit)
        from sam2_lora import configure_sam
        configure_sam(self, adaptation)

    def _output(self, features, return_logits=False):
        logits = self.tail[1](self.tail[0](features))
        return logits if return_logits else self.tail[2](logits)

    def forward(self, x, feature_masks=None, return_logits=False):
        if feature_masks is not None:
            return self._forward_with_feature_masks(x, feature_masks, return_logits=return_logits)
        if self.isalign:
            x = self.align(x)
        h = self.head(x)
        
        sam_h = self.sam.forward_image(self.input_sam(h))
        
        if self.isvit:
            vit_h = self.vit.forward_features(self.input_vit(h)).view(x.shape[0],16,16,-1).permute(0,3,1,2)
            
            sam_h0 = self.upblock0(self.conv_sam_0(sam_h['vision_features'])+self.conv0(h)+self.convvit(vit_h))
        else:
            sam_h0 = self.upblock0(self.conv_sam_0(sam_h['vision_features'])+self.conv0(h))
        sam_h1 = sam_h0+self.conv_sam_1(sam_h['backbone_fpn'][1])+self.conv1(h)
        sam_h1 = self.upblock1(sam_h1)
        sam_h2 = sam_h1+self.conv_sam_2(sam_h['backbone_fpn'][0])+self.conv2(h)
        sam_h = self.upblock2(sam_h2)
        h = self.conv3(h)+sam_h
        return self._output(h, return_logits=return_logits)

    def complementary_feature_channels(self):
        # Include the local high-resolution skip so it cannot bypass perturbation.
        return (self.conv_sam_0.out_channels, self.conv_sam_1.out_channels,
                self.conv_sam_2.out_channels, self.conv3[0].out_channels)

    def _forward_with_feature_masks(self, x, masks, return_logits=False):
        if not self.training:
            raise ValueError('Feature masks are training-only; inference uses normal forward.')
        channels = self.complementary_feature_channels()
        if len(masks) != len(channels):
            raise ValueError('Expected one feature mask per decoder input scale.')
        for mask, channel in zip(masks, channels):
            if mask.shape != (x.shape[0], channel, 1, 1) or mask.device != x.device:
                raise ValueError('Feature mask batch/channel/device mismatch.')
        if self.isalign:
            x = self.align(x)
        h = self.head(x)
        sam_h = self.sam.forward_image(self.input_sam(h))
        deepest = self.conv_sam_0(sam_h['vision_features']) + self.conv0(h)
        if self.isvit:
            vit_h = self.vit.forward_features(self.input_vit(h)).view(
                x.shape[0], 16, 16, -1).permute(0, 3, 1, 2)
            deepest = deepest + self.convvit(vit_h)
        decoded = self.upblock0(deepest * masks[0])
        decoded = self.upblock1(
            decoded + (self.conv_sam_1(sam_h['backbone_fpn'][1])
                       + self.conv1(h)) * masks[1])
        decoded = self.upblock2(
            decoded + (self.conv_sam_2(sam_h['backbone_fpn'][0])
                       + self.conv2(h)) * masks[2])
        return self._output(self.conv3(h) * masks[3] + decoded, return_logits=return_logits)

    @torch.no_grad()
    def OutputFeature(self, x):
        if self.isalign:
            x = self.align(x)
        h = self.head(x)
        
        sam_h = self.sam.forward_image(self.input_sam(h))#['backbone_fpn'][1]
        
        if self.isvit:
            vit_h = self.vit.forward_features(self.input_vit(h)).view(x.shape[0],16,16,-1).permute(0,3,1,2)
            
            return self.conv_sam_0(sam_h['vision_features'])+self.conv0(h)+self.convvit(vit_h)
            
        else:
            return self.conv_sam_0(sam_h['vision_features'])+self.conv0(h)
            
    def outputFeature(self,x, d0,d1,D1=0,D2=0,D=512,batch_size=4):
        featureL = []
        for i in range(0,len(x),batch_size):
            X = x[i:i+batch_size]
            X = torch.from_numpy(X).float()
            if X.shape[-2] > D or X.shape[-1] > D:
                X = torch.nn.functional.adaptive_avg_pool2d(X,(D,D))
            X = X.to(self.head[0].weight.device)
            feature = self.OutputFeature(X)
            if X.shape[-2]>d0 or X.shape[-1]>d1:
                feature = torch.nn.functional.adaptive_avg_pool2d(feature,(d0,d1))
            if D1>0:
                feature = torch.nn.functional.interpolate(feature,(D1,D2))
            
            featureL.append(feature.cpu().detach().numpy())
        return np.concatenate(featureL)
    @torch.no_grad()
    def outputImage(self, x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        x = x.clone()
        mole = x[:,1]
        vaild1 = x[:,3]
        w = vaild1/(vaild1.sum(dim=(1,2),keepdim=True)+1e-4)
        mole_mean = (mole*w).mean(dim=(1,2),keepdim=True)
        mole = mole - mole_mean
        mole = torch.where(vaild1>0,mole,torch.zeros_like(mole))
        x[:,1] = mole
        x = x.to(self.head[0].weight.device)
        if self.isalign:
            x = self.align(x)
        h= self.head(x)
        sam_h = self.input_sam(h)
        sam_h = torch.nn.functional.adaptive_avg_pool2d(sam_h,(x.shape[-2],x.shape[-1]))
        for i in range(len(mean)):
            sam_h[:,i] = sam_h[:,i]*std[i]+mean[i]
        return sam_h
    def OutputImage(self, x):
        hL = []
        for i in range(0,len(x),8):
            i1 = min(i+8,len(x))
            hL.append(self.outputImage(x[i:i1]).cpu().detach().numpy())
        return np.concatenate(hL)
    def Predict_(self,x):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).float()
            x = x.clone()
            mole = x[:,1]
            vaild1 = x[:,3]
            w = vaild1/(vaild1.mean(dim=(1,2),keepdim=True)+1e-4)
            mole_mean = (mole*w).mean(dim=(1,2),keepdim=True)
            mole = mole - mole_mean
            mole = torch.where(vaild1>0,mole,torch.zeros_like(mole))
            x[:,1] = mole
            x = x.to(self.head[0].weight.device)
            h = np.zeros((len(x),self.out_Channel,*x.shape[-2:]))
            w = np.zeros((1,1,self.n,self.m))+0.1
            w[0,0,self.n//4:-self.n//4,self.m//4:-self.m//4] = 1
            h_count = h+0
            for i in range(0,x.shape[-2],self.n//2):
                for j in range(0,x.shape[-1],self.m//2):
                    i0 = min(i,x.shape[-2]-self.n)
                    j0 = min(j,x.shape[-1]-self.m)
                    h[:,:,i0:i0+self.n,j0:j0+self.m]+=self(x[:,:,i0:i0+self.n,j0:j0+self.m]).cpu().numpy()*w
                    h_count[:,:,i0:i0+self.n,j0:j0+self.m] += w
            h = h/h_count
        return h
    @torch.no_grad()
    def Predict(self,x):
        h =np.zeros((len(x),self.out_Channel,*x.shape[-2:]))
        for i in range(0,len(x),8):
            i1 = min(i+8,len(x))
            h[i:i1] = self.Predict_(x[i:i1])
        return h
    def predict_total(self,x):
        inputsL = []
        for i in range(0,x.shape[1],self.n//2):
            i1 = min(i+self.n,x.shape[1])
            i = i1-self.n
            for j in range(0,x.shape[2],self.m//2):
                j1 = min(j+self.m,x.shape[2])
                j=j1-self.m
                inputsL.append(x[:,i:i1,j:j1])
            #inputsL.append(x[i:i1]) 
        inputs = np.array(inputsL)
        outputs = self.Predict(inputs)
        print(inputs.shape,outputs.shape)
        x_out = x[:3]*0
        x_count = x[:3]*0
        count= 0
        for i in range(0,x.shape[1],self.n//2):
            i1 = min(i+self.n,x.shape[1])
            i = i1-self.n
            for j in range(0,x.shape[2],self.m//2):
                j1 = min(j+self.m,x.shape[2])
                j=j1-self.m
                x_out[:,i:i1,j:j1] += outputs[count]
                x_count[:,i:i1,j:j1] += 1
                count += 1
        x_out = x_out/x_count
        return x_out
    def predict_total_image(self,x):
        inputsL = []
        for i in range(0,x.shape[1],self.n//2):
            i1 = min(i+self.n,x.shape[1])
            i = i1-self.n
            for j in range(0,x.shape[2],self.m//2):
                j1 = min(j+self.m,x.shape[2])
                j=j1-self.m
                inputsL.append(x[:,i:i1,j:j1])
        inputs = np.array(inputsL)
        outputs = self.OutputImage(inputs)
        print(inputs.shape,outputs.shape)
        x_out = x[:3]*0
        x_count = x[:3]*0
        count= 0
        for i in range(0,x.shape[1],self.n//2):
            i1 = min(i+self.n,x.shape[1])
            i = i1-self.n
            for j in range(0,x.shape[2],self.m//2):
                j1 = min(j+self.m,x.shape[2])
                j=j1-self.m
                x_out[:,i:i1,j:j1] += outputs[count]
                x_count[:,i:i1,j:j1] += 1
                count += 1
            #count += 1
        x_out = x_out/x_count
        return x_out
    def predict(self,x):
        return self.Predict(x)

class UNet(nn.Module):
    def __init__(self, ch, ch_mult, attn, num_res_blocks, dropout,head_num=1,in_Channel=3,n=3,m=3,out_Channel=3,In=lambda x:x,Out=lambda x:x):
        self.In = In
        self.Out = Out
        self.m = m
        self.n = n
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'

        self.head = nn.Conv2d(in_Channel, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch] 
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch,
                    dropout=dropout, attn=(i in attn),head_num=head_num))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, dropout, attn=True,head_num=head_num),
            ResBlock(now_ch, now_ch, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch,
                    dropout=dropout, attn=(i in attn),head_num=head_num))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(8, now_ch),
            Swish(),
            nn.Conv2d(now_ch, out_Channel, 3, stride=1, padding=1),
            nn.Softmax(dim=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-2].weight, gain=1e-5)
        init.zeros_(self.tail[-2].bias)

    def forward(self, x):
        h = self.In(x)
        h = self.head(h)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h)
        h = self.tail(h)

        assert len(hs) == 0
        h = self.Out(h)
        #h = h*std
        
        return h
    def Predict_(self,x):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).float()
            x = x.to(self.head.weight.device)
            h = self(x)
            h = h.cpu().numpy()
        return h
    def Predict(self,x):
        if not isinstance(x, torch.Tensor):
            h = x*0
        else:
            h = x.cpu().numpy() *0
        for i in range(0,len(x),32):
            i1 = min(i+32,len(x))
            h[i:i1] = self.Predict_(x[i:i1])
        return h
    def predict_total(self,x):
        inputsL = []
        for i in range(0,len(x),self.n//2):
            i1 = min(i+self.n,len(x))
            for j in range(0,len(x[0]),self.m):
                j1 = min(j+self.m,len(x[0]))
                inputsL.append(x[i:i1,j:j1])
            inputsL.append(x[i:i1]) 
        inputs = np.array(inputsL)
        outputs = self.Predict(inputs)
        x_out = x*0
        x_count = x*0
        count= 0
        for i in range(0,len(x),self.n//2):
            i1 = min(i+self.n,len(x))
            for j in range(0,len(x[0]),self.m):
                j1 = min(j+self.m,len(x[0]))
                x_out[i:i1,j:j1] += outputs[count]
                x_count[i:i1,j:j1] += 1
                count += 1
            x_out[i:i1] += outputs[count]
            x_count[i:i1] += 1
            count += 1
        x_out = x_out/x_count
        return x_out
    def predict_total_image(self,x):
        inputsL = []
        for i in range(0,len(x),self.n//2):
            i1 = min(i+self.n,len(x))
            for j in range(0,len(x[0]),self.m):
                j1 = min(j+self.m,len(x[0]))
                inputsL.append(x[i:i1,j:j1])
            inputsL.append(x[i:i1]) 
        inputs = np.array(inputsL)
        outputs = self.Predict(inputs)
        x_out = x*0
        x_count = x*0
        count= 0
        for i in range(0,len(x),self.n//2):
            i1 = min(i+self.n,len(x))
            for j in range(0,len(x[0]),self.m):
                j1 = min(j+self.m,len(x[0]))
                x_out[i:i1,j:j1] += outputs[count]
                x_count[i:i1,j:j1] += 1
                count += 1
            x_out[i:i1] += outputs[count]
            x_count[i:i1] += 1
            count += 1
        x_out = x_out/x_count
        return x_out
    def predict(self,x):
        return self.Predict(x)
    @torch.no_grad()
    def outputImage(self, x):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(2).to(x.device)
        std = torch.tensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(2).to(x.device)
        return x * std + mean
    def OutputImage(self, x):
        hL = []
        for i in range(0,len(x),8):
            i1 = min(i+8,len(x))
            hL.append(self.outputImage(x[i:i1]).cpu().detach().numpy())
        return np.concatenate(hL)



def norm2(x):
    return x/torch.sqrt((torch.abs(x)+1e-25))

def denorm2(x):
    return x*torch.abs(x)

def huber_loss_(x, rela_delta=3.0):
    absx = torch.abs(x)
    delta = rela_delta*torch.sqrt(torch.mean(x*x))
    return torch.where(absx < delta, 0.5 * x ** 2, delta * (absx - 0.5 * delta)).mean()
def huber_loss(x0,xP, delta=1e-1,trace_weight=True):
    x = x0-xP
    absx = torch.abs(x)
    if trace_weight:
        STD = torch.std(x0,dim=2,keepdim=True)
        weight = torch.where(STD>1e-8,1,1e-8).detach()
    else:
        weight = 1
    return (torch.where(absx < delta, 0.5 * x ** 2, delta * (absx - 0.5 * delta))*weight).mean()

def IoU___(y_true, y_pred,sw=1,delta=1/20):#,eps=1e-1):
    
    y_pred = y_pred[:,:y_true.shape[1]]
    dy = y_true - y_pred
    sw = sw/sw.mean()
    #loss = 
    return (dy**2*sw).mean()
def IoU(y_true, y_pred,sw=1,delta=1/20):#,eps=1e-1):
    
    y_pred = y_pred[:,:y_true.shape[1]]
    eps = 1e-5
    union=torch.where(y_true>y_pred,y_true,y_pred)
    union = torch.mean(union*sw)
    overlap = torch.where(y_true<y_pred,y_true,y_pred)
    overlap = torch.mean(overlap*sw)
    mean = (torch.mean(y_true))
    loss = (1-(overlap+eps)/(union+eps))*mean
    return loss
def Dice(y_true, y_pred, sw=1, delta=1/20):
    eps = 1e-5
    
    y_true = y_true.float()
    y_pred = y_pred.float()

    # 避免 shape 不一致问题
    y_pred = y_pred[:, :y_true.shape[1]]

    # 计算带权重的 Dice 损失
    intersection = torch.mean(sw * y_true * y_pred)
    union = torch.mean(sw * y_true) + torch.mean(sw * y_pred)
    mean = (torch.mean(y_true))
    dice = (2. * intersection + eps) / (union + eps)
    loss = (1 - dice)*mean

    return loss
def FocalTversky(y_true, y_pred, sw=1, delta=1/20,
                 tv_alpha=0.3, tv_beta=0.7, gamma=0.75, pu=0.0):
    """Focal-Tversky loss (FIX of Dice above):
    - NO `*mean` scaling: the old Dice multiplied the whole loss by the
      positive-pixel fraction (~0.09), shrinking gradients ~10x and pushing
      the model into the under-confident 'output tiny values everywhere'
      local optimum (best_threshold ~0.01 in v1/v3).
    - tv_beta > tv_alpha penalises false negatives more than false positives
      (recall-oriented; faults are thin and the catalogue is incomplete).
    - gamma < 1 focuses gradient on hard examples near convergence.
    - pu (0..1): positive-unlabeled discount for the INCOMPLETE fault
      catalogue ("unlabeled != no fault"). The false-positive penalty of a
      pixel is multiplied by (1 - pu*confidence): pixels the model is
      confident are faults but the catalogue does not label are partially
      forgiven instead of fully punished (AIO2-style, soft variant).
      Confidence is detached so the model cannot lower the loss by simply
      inflating predictions. Use pu=0 for validation/plain training.
    """
    eps = 1e-5
    y_true = y_true.float()
    y_pred = y_pred[:, :y_true.shape[1]].float()
    if not torch.is_tensor(sw):
        sw = y_true.new_tensor(float(sw))
    fp_w = sw
    if pu > 0:
        # HARD CAP pu<=0.6: verified numerically that with pu=1 the model can
        # zero the FP penalty by saturating all predictions (all-ones gives
        # loss 0 = degenerate optimum). At pu<=0.6 cheating still costs >2x
        # an honest solution, so the discount only helps genuinely
        # unlabeled-but-fault-like pixels.
        pu = min(float(pu), 0.6)
        conf = y_pred.detach().clamp(0., 1.)
        fp_w = sw * (1.0 - pu * conf)
    tp = torch.sum(sw  * y_true        * y_pred)
    fn = torch.sum(sw  * y_true        * (1. - y_pred))
    fp = torch.sum(fp_w * (1. - y_true) * y_pred)
    tversky = (tp + eps) / (tp + tv_beta * fn + tv_alpha * fp + eps)
    # For gamma < 1, x**gamma has an infinite derivative at x=0.
    # Float32 can round Tversky to exactly 1 on almost-perfect/empty crops;
    # backward then produces NaN even while the displayed loss is zero.
    # Bound only this singular neighborhood; ordinary loss values are unchanged.
    error = (1. - tversky).clamp_min(torch.finfo(tversky.dtype).eps)
    return error ** gamma
def IoU__(y_true, y_pred,sw=1,delta=1/20):#,eps=1e-1):
    y_pred = y_pred[:,:y_true.shape[1]]
    eps = 1e-5
    
    union = torch.mean((y_true+y_pred)*sw)*0.5
    overlap = torch.mean((y_true*y_pred)*sw)
    mean = torch.mean(y_true)*10
    loss = (1-(overlap+eps)/(union+eps))*mean
    return loss
def IoU_(y_true, y_pred,sw=1,delta=1/20):#,eps=1e-1):
    eps = 1e-5
    union = torch.mean((y_true+y_pred)*sw)*0.5
    overlap = torch.mean((y_true*y_pred)*sw)
    loss = (1-(overlap+eps)/(union+eps))#*(torch.mean(y_true)+eps)
    return loss
class WarmupDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, initial_lr, final_lr, decay_type='linear', last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.decay_type = decay_type
        self.current_step = 0
        super(WarmupDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            warmup_lr = self.initial_lr + (self.final_lr - self.initial_lr) * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress,1)
            if self.decay_type == 'linear':
                warmup_lr = self.final_lr * (1 - progress)
            elif self.decay_type == 'exponential':
                warmup_lr = self.final_lr * (0.1 ** progress)
            else:
                raise ValueError(f"Unknown decay type: {self.decay_type}")

        self.current_step += 1
        return [warmup_lr for _ in self.optimizer.param_groups]

if __name__ == '__main__':
    batch_size = 1
    in_Channel = 7
    out_Channel = 1
    model = SpatialAwareAffineAlign(in_Channel)
    model.to('cuda:0')
    x = torch.randn(batch_size, in_Channel, 128, 128, device='cuda:0')
    y = model(x)
    theta_0,theat_1 = model.get_theta(x)
    print(theat_1)
    
