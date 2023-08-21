from chain_img_processor import ChainImgProcessor, ChainImgPlugin
import os
from numpy import asarray

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import torch.nn.utils.spectral_norm as SpectralNorm
from torchvision.ops import roi_align

from math import sqrt
import os

import cv2 
import os
from torchvision.transforms.functional import normalize
import copy
import threading

modname = os.path.basename(__file__)[:-3] # calculating modname

oDMDNet = None
device = None

THREAD_LOCK_DMDNET = threading.Lock()



# start function
def start(core:ChainImgProcessor):
    manifest = { # plugin settings
        "name": "DMDNet", # name
        "version": "1.0", # version

        "default_options": {},
        "img_processor": {
            "dmdnet": DMDNETPlugin
        }
    }
    return manifest

def start_with_options(core:ChainImgProcessor, manifest:dict):
    pass


class DMDNETPlugin(ChainImgPlugin):

    # https://stackoverflow.com/a/67174339
    def landmarks106_to_68(self, pt106):
        map106to68=[1,10,12,14,16,3,5,7,0,23,21,19,32,30,28,26,17,
                        43,48,49,51,50,
                        102,103,104,105,101,
                        72,73,74,86,78,79,80,85,84,
                        35,41,42,39,37,36,
                        89,95,96,93,91,90,
                        52,64,63,71,67,68,61,58,59,53,56,55,65,66,62,70,69,57,60,54
                        ]
      
        pt68 = []
        for i in range(68):
            index = map106to68[i]
            pt68.append(pt106[index])
        return pt68

    def init_plugin(self):
        global create
        
        if oDMDNet == None:
            create(self.device)


    def process(self, frame, params:dict):
        if "face_detected" in params:
            if not params["face_detected"]:
                return frame

        temp_frame = copy.copy(frame)
        if "processed_faces" in params:
            for face in params["processed_faces"]:
                start_x, start_y, end_x, end_y = map(int, face['bbox'])
                temp_face, start_x, start_y, end_x, end_y = self.cutout(temp_frame, start_x, start_y, end_x, end_y, 0.5)
                if temp_face.size:
                    temp_face = self.enhance_face(params["input_face_datas"][0], temp_face, face, start_x, start_y)
                    temp_face = cv2.resize(temp_face, (end_x - start_x,end_y - start_y), interpolation = cv2.INTER_LANCZOS4)
                    temp_frame = self.paste_into(temp_face, temp_frame, start_x, start_y, end_x, end_y, False)


        if not "blend_ratio" in params: 
            return temp_frame
        
        blend_ratio = params["blend_ratio"]
        return cv2.addWeighted(temp_frame, blend_ratio, frame, 1.0 - blend_ratio,0)

    def check_bbox(self, imgs, boxes):
        boxes = boxes.view(-1, 4, 4)
        colors = [(0, 255, 0), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
        i = 0
        for img, box in zip(imgs, boxes):
            img = (img + 1)/2 * 255
            img2 = img.permute(1, 2, 0).float().cpu().flip(2).numpy().copy()
            for idx, point in enumerate(box):
                cv2.rectangle(img2, (int(point[0]), int(point[1])), (int(point[2]), int(point[3])), color=colors[idx], thickness=2)
            cv2.imwrite('dmdnet_{:02d}.png'.format(i), img2)
            i += 1

    # clip: cutout face: face struct
    def enhance_face(self, ref_face, clip, face, x_offs, y_offs):
        global device

        lm106 = face.landmark_2d_106
        lq_landmarks = asarray(self.landmarks106_to_68(lm106))

        sub = np.full_like(lq_landmarks, [x_offs, y_offs])
        lq_landmarks = lq_landmarks - sub

        lq,lq_landmarks = read_img_tensor(clip, lq_landmarks)

        LQLocs = get_component_location(lq_landmarks)
        #self.check_bbox(lq, LQLocs.unsqueeze(0))

        # specific
        # start_x, start_y, end_x, end_y = map(int, ref_face['bbox'])
        # temp_face = temp_frame[start_y:end_y, start_x:end_x]
        # if temp_face.size:
        #     SpecificImgs = []
        #     SpecificLocs = []
        #     lm106 = ref_face.landmark_2d_106
        #     ref_landmarks = asarray(self.landmarks106_to_68(lm106))
        #     ref_tensor, ref_landmarks, _ = read_img_tensor(clip, ref_landmarks)
        #     SpecificImgs.append(ref_tensor)
        #     ref_locs = get_component_location(ref_landmarks)
        #     SpecificLocs.append(ref_locs.unsqueeze(0))

        #     SpecificImgs = torch.cat(SpecificImgs, dim=0)
        #     SpecificLocs = torch.cat(SpecificLocs, dim=0)
        #     # check_bbox(SpecificImgs, SpecificLocs)
        #     SpMem256, SpMem128, SpMem64 = DMDNet.generate_specific_dictionary(sp_imgs = SpecificImgs.to(device), sp_locs = SpecificLocs)
        #     SpMem256Para = {}
        #     SpMem128Para = {}
        #     SpMem64Para = {}
        #     for k, v in SpMem256.items():
        #         SpMem256Para[k] = v
        #     for k, v in SpMem128.items():
        #         SpMem128Para[k] = v
        #     for k, v in SpMem64.items():
        #         SpMem64Para[k] = v

        # generic
        SpMem256Para, SpMem128Para, SpMem64Para = None, None, None

        with torch.no_grad():
            with THREAD_LOCK_DMDNET:
                try:
                    GenericResult, SpecificResult = oDMDNet(lq = lq.to(device), loc = LQLocs.unsqueeze(0), sp_256 = SpMem256Para, sp_128 = SpMem128Para, sp_64 = SpMem64Para)
                except Exception as e:
                    print(f'Error {e} there may be something wrong with the detected component locations.')
                    return clip
        save_generic = GenericResult * 0.5 + 0.5
        save_generic = save_generic.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
        save_generic = np.clip(save_generic.float().cpu().numpy(), 0, 1) * 255.0

        check_lq = lq * 0.5 + 0.5
        check_lq = check_lq.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
        check_lq = np.clip(check_lq.float().cpu().numpy(), 0, 1) * 255.0
        enhanced_img =  np.hstack((check_lq, save_generic))
        temp_frame =  save_generic.astype("uint8")
        # temp_frame = save_generic.astype("uint8")
        return temp_frame
    

def create(devicename):
    global device, oDMDNet

    test = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(devicename)
    oDMDNet = DMDNet().to(device)
    weights = torch.load('./models/DMDNet.pth') 
    oDMDNet.load_state_dict(weights, strict=True)

    oDMDNet.eval()
    num_params = 0
    for param in oDMDNet.parameters():
        num_params += param.numel()

    # print('{:>8s} : {}'.format('Using device', device))
    # print('{:>8s} : {:.2f}M'.format('Model params', num_params/1e6))



def read_img_tensor(Img=None, landmarks=None): #rgb -1~1 
#    Img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # BGR or G
    if Img.ndim == 2:
        Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)  # GGG
    else:
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)  # RGB
    
    if Img.shape[0] != 512 or Img.shape[1] != 512:
        y_factor = 512 / Img.shape[0]
        x_factor = 512 / Img.shape[1]
        mult = np.full_like(landmarks, [x_factor, y_factor])
        landmarks = landmarks * mult
        Img = cv2.resize(Img, (512,512), interpolation = cv2.INTER_AREA)


    # ImgForLands = Img.copy()

    Img = Img.transpose((2, 0, 1))/255.0
    Img = torch.from_numpy(Img).float()
    normalize(Img, [0.5,0.5,0.5], [0.5,0.5,0.5], inplace=True)
    ImgTensor = Img.unsqueeze(0)
    return ImgTensor, landmarks


def get_component_location(Landmarks, re_read=False):
    if re_read:
        ReadLandmark = []
        with open(Landmarks,'r') as f:
            for line in f:
                tmp = [float(i) for i in line.split(' ') if i != '\n']
                ReadLandmark.append(tmp)
        ReadLandmark = np.array(ReadLandmark) #
        Landmarks = np.reshape(ReadLandmark, [-1, 2]) # 68*2
    Map_LE_B = list(np.hstack((range(17,22), range(36,42))))
    Map_RE_B = list(np.hstack((range(22,27), range(42,48))))
    Map_LE = list(range(36,42))
    Map_RE = list(range(42,48))
    Map_NO = list(range(29,36))
    Map_MO = list(range(48,68))

    Landmarks[Landmarks>504]=504
    Landmarks[Landmarks<8]=8
    
    #left eye
    Mean_LE = np.mean(Landmarks[Map_LE],0)
    L_LE1 = Mean_LE[1] - np.min(Landmarks[Map_LE_B,1])
    L_LE1 = L_LE1 * 1.3
    L_LE2 = L_LE1 / 1.9
    L_LE_xy = L_LE1 + L_LE2
    L_LE_lt = [L_LE_xy/2, L_LE1]
    L_LE_rb = [L_LE_xy/2, L_LE2]
    Location_LE = np.hstack((Mean_LE - L_LE_lt + 1, Mean_LE + L_LE_rb)).astype(int)

    #right eye
    Mean_RE = np.mean(Landmarks[Map_RE],0)
    L_RE1 = Mean_RE[1] - np.min(Landmarks[Map_RE_B,1])
    L_RE1 = L_RE1 * 1.3
    L_RE2 = L_RE1 / 1.9
    L_RE_xy = L_RE1 + L_RE2
    L_RE_lt = [L_RE_xy/2, L_RE1]
    L_RE_rb = [L_RE_xy/2, L_RE2]
    Location_RE = np.hstack((Mean_RE - L_RE_lt + 1, Mean_RE + L_RE_rb)).astype(int)

    #nose
    Mean_NO = np.mean(Landmarks[Map_NO],0)
    L_NO1 =( np.max([Mean_NO[0] - Landmarks[31][0], Landmarks[35][0] - Mean_NO[0]])) * 1.25
    L_NO2 = (Landmarks[33][1] - Mean_NO[1]) * 1.1
    L_NO_xy = L_NO1 * 2
    L_NO_lt = [L_NO_xy/2, L_NO_xy - L_NO2]
    L_NO_rb = [L_NO_xy/2, L_NO2]
    Location_NO = np.hstack((Mean_NO - L_NO_lt + 1, Mean_NO + L_NO_rb)).astype(int)
    
    #mouth
    Mean_MO = np.mean(Landmarks[Map_MO],0)
    L_MO = np.max((np.max(np.max(Landmarks[Map_MO],0) - np.min(Landmarks[Map_MO],0))/2,16)) * 1.1
    MO_O = Mean_MO - L_MO + 1
    MO_T = Mean_MO + L_MO
    MO_T[MO_T>510]=510
    Location_MO = np.hstack((MO_O, MO_T)).astype(int)
    return torch.cat([torch.FloatTensor(Location_LE).unsqueeze(0), torch.FloatTensor(Location_RE).unsqueeze(0), torch.FloatTensor(Location_NO).unsqueeze(0), torch.FloatTensor(Location_MO).unsqueeze(0)], dim=0)




def calc_mean_std_4D(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization_4D(content_feat, style_feat): # content_feat is ref feature, style is degradate feature
    size = content_feat.size()
    style_mean, style_std = calc_mean_std_4D(style_feat)
    content_mean, content_std = calc_mean_std_4D(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def convU(in_channels, out_channels,conv_layer, norm_layer, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        SpectralNorm(conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias)),
        nn.LeakyReLU(0.2),
        SpectralNorm(conv_layer(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias)),
    )
    

class MSDilateBlock(nn.Module):
    def __init__(self, in_channels,conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, kernel_size=3, dilation=[1,1,1,1], bias=True):
        super(MSDilateBlock, self).__init__()
        self.conv1 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[0], bias=bias)
        self.conv2 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[1], bias=bias)
        self.conv3 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[2], bias=bias)
        self.conv4 =  convU(in_channels, in_channels,conv_layer, norm_layer, kernel_size,dilation=dilation[3], bias=bias)
        self.convi =  SpectralNorm(conv_layer(in_channels*4, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias))
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat  = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.convi(cat) + x
        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)

    def forward(self, input, style):
        style_mean, style_std = calc_mean_std_4D(style)
        out = self.norm(input)
        size = input.size()
        out = style_std.expand(size) * out + style_mean.expand(size)
        return out

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))
    def forward(self, image, noise):
        if noise is None:
            b, c, h, w = image.shape
            noise = image.new_empty(b, 1, h, w).normal_()
        return image + self.weight * noise

class StyledUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1,upsample=False, noise_inject=False):
        super().__init__()

        self.noise_inject = noise_inject
        if upsample:
            self.conv1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)),
                nn.LeakyReLU(0.2),
            )
        else:
            self.conv1 = nn.Sequential(
                SpectralNorm(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)),
                nn.LeakyReLU(0.2),
                SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)),
            )
        self.convup = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)),
                nn.LeakyReLU(0.2),
                SpectralNorm(nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding)),
            )
        if self.noise_inject:
            self.noise1 = NoiseInjection(out_channel)

        self.lrelu1 = nn.LeakyReLU(0.2)

        self.ScaleModel1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel,out_channel,3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1))
        )
        self.ShiftModel1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channel,out_channel,3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(out_channel, out_channel, 3, 1, 1)),
        )
       
    def forward(self, input, style):
        out = self.conv1(input)
        out = self.lrelu1(out)
        Shift1 = self.ShiftModel1(style)
        Scale1 = self.ScaleModel1(style)
        out = out * Scale1 + Shift1
        if self.noise_inject:
            out = self.noise1(out, noise=None)
        outup = self.convup(out)
        return outup


####################################################################
###############Face Dictionary Generator
####################################################################
def AttentionBlock(in_channel):
    return nn.Sequential(
        SpectralNorm(nn.Conv2d(in_channel, in_channel, 3, 1, 1)),
        nn.LeakyReLU(0.2),
        SpectralNorm(nn.Conv2d(in_channel, in_channel, 3, 1, 1)),
    )

class DilateResBlock(nn.Module):
    def __init__(self, dim, dilation=[5,3] ):
        super(DilateResBlock, self).__init__()
        self.Res = nn.Sequential(
            SpectralNorm(nn.Conv2d(dim, dim, 3, 1, ((3-1)//2)*dilation[0], dilation[0])),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(dim, dim, 3, 1, ((3-1)//2)*dilation[1], dilation[1])),
        )
    def forward(self, x):
        out = x + self.Res(x)
        return out


class KeyValue(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Sequential(
            SpectralNorm(nn.Conv2d(indim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(keydim, keydim, kernel_size=(3,3), padding=(1,1), stride=1)),
        )
        self.Value = nn.Sequential(
            SpectralNorm(nn.Conv2d(indim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(valdim, valdim, kernel_size=(3,3), padding=(1,1), stride=1)),
        )
    def forward(self, x):  
        return self.Key(x), self.Value(x)

class MaskAttention(nn.Module):
    def __init__(self, indim):
        super(MaskAttention, self).__init__()
        self.conv1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(indim, indim//3, kernel_size=(3,3), padding=(1,1), stride=1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(indim//3, indim//3, kernel_size=(3,3), padding=(1,1), stride=1)),
        )
        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(indim, indim//3, kernel_size=(3,3), padding=(1,1), stride=1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(indim//3, indim//3, kernel_size=(3,3), padding=(1,1), stride=1)),
        )
        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(indim, indim//3, kernel_size=(3,3), padding=(1,1), stride=1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(indim//3, indim//3, kernel_size=(3,3), padding=(1,1), stride=1)),
        )
        self.convCat = nn.Sequential(
            SpectralNorm(nn.Conv2d(indim//3 * 3, indim, kernel_size=(3,3), padding=(1,1), stride=1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(indim, indim, kernel_size=(3,3), padding=(1,1), stride=1)),
        )
    def forward(self, x, y, z):
        c1 = self.conv1(x)
        c2 = self.conv2(y)
        c3 = self.conv3(z)
        return self.convCat(torch.cat([c1,c2,c3], dim=1))

class Query(nn.Module):
    def __init__(self, indim, quedim):
        super(Query, self).__init__()
        self.Query = nn.Sequential(
            SpectralNorm(nn.Conv2d(indim, quedim, kernel_size=(3,3), padding=(1,1), stride=1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(quedim, quedim, kernel_size=(3,3), padding=(1,1), stride=1)),
        )
    def forward(self, x):
        return self.Query(x)

def roi_align_self(input, location, target_size):
    return torch.cat([F.interpolate(input[i:i+1,:,location[i,1]:location[i,3],location[i,0]:location[i,2]],(target_size,target_size),mode='bilinear',align_corners=False) for i in range(input.size(0))],0)

class FeatureExtractor(nn.Module):
    def __init__(self, ngf = 64, key_scale = 4):#
        super().__init__()

        self.key_scale = 4
        self.part_sizes = np.array([80,80,50,110]) #
        self.feature_sizes = np.array([256,128,64]) # 

        self.conv1 = nn.Sequential(
                SpectralNorm(nn.Conv2d(3, ngf, 3, 2, 1)),
                nn.LeakyReLU(0.2),
                SpectralNorm(nn.Conv2d(ngf, ngf, 3, 1, 1)),
            )
        self.conv2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf, ngf, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(ngf, ngf, 3, 1, 1))
        )
        self.res1 = DilateResBlock(ngf, [5,3])
        self.res2 = DilateResBlock(ngf, [5,3])

        
        self.conv3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf, ngf*2, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(ngf*2, ngf*2, 3, 1, 1)),
            )
        self.conv4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf*2, ngf*2, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(ngf*2, ngf*2, 3, 1, 1))
        )
        self.res3 = DilateResBlock(ngf*2, [3,1])
        self.res4 = DilateResBlock(ngf*2, [3,1])

        self.conv5 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf*2, ngf*4, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(ngf*4, ngf*4, 3, 1, 1)),
        )
        self.conv6 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ngf*4, ngf*4, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Conv2d(ngf*4, ngf*4, 3, 1, 1))
        )
        self.res5 = DilateResBlock(ngf*4, [1,1])
        self.res6 = DilateResBlock(ngf*4, [1,1])

        self.LE_256_Q = Query(ngf, ngf // self.key_scale)
        self.RE_256_Q = Query(ngf, ngf // self.key_scale)
        self.MO_256_Q = Query(ngf, ngf // self.key_scale)
        self.LE_128_Q = Query(ngf * 2, ngf * 2 // self.key_scale)
        self.RE_128_Q = Query(ngf * 2, ngf * 2 // self.key_scale)
        self.MO_128_Q = Query(ngf * 2, ngf * 2 // self.key_scale)
        self.LE_64_Q = Query(ngf * 4, ngf * 4 // self.key_scale)
        self.RE_64_Q = Query(ngf * 4, ngf * 4 // self.key_scale)
        self.MO_64_Q = Query(ngf * 4, ngf * 4 // self.key_scale)


    def forward(self, img, locs):
        le_location = locs[:,0,:].int().cpu().numpy()
        re_location = locs[:,1,:].int().cpu().numpy()
        no_location = locs[:,2,:].int().cpu().numpy()
        mo_location = locs[:,3,:].int().cpu().numpy()
        

        f1_0 = self.conv1(img) 
        f1_1 = self.res1(f1_0)
        f2_0 = self.conv2(f1_1)
        f2_1 = self.res2(f2_0)

        f3_0 = self.conv3(f2_1) 
        f3_1 = self.res3(f3_0)
        f4_0 = self.conv4(f3_1)
        f4_1 = self.res4(f4_0)

        f5_0 = self.conv5(f4_1) 
        f5_1 = self.res5(f5_0)
        f6_0 = self.conv6(f5_1)
        f6_1 = self.res6(f6_0)


        ####ROI Align
        le_part_256 = roi_align_self(f2_1.clone(), le_location//2, self.part_sizes[0]//2)
        re_part_256 = roi_align_self(f2_1.clone(), re_location//2, self.part_sizes[1]//2)
        mo_part_256 = roi_align_self(f2_1.clone(), mo_location//2, self.part_sizes[3]//2)

        le_part_128 = roi_align_self(f4_1.clone(), le_location//4, self.part_sizes[0]//4)
        re_part_128 = roi_align_self(f4_1.clone(), re_location//4, self.part_sizes[1]//4)
        mo_part_128 = roi_align_self(f4_1.clone(), mo_location//4, self.part_sizes[3]//4)

        le_part_64 = roi_align_self(f6_1.clone(), le_location//8, self.part_sizes[0]//8)
        re_part_64 = roi_align_self(f6_1.clone(), re_location//8, self.part_sizes[1]//8)
        mo_part_64 = roi_align_self(f6_1.clone(), mo_location//8, self.part_sizes[3]//8)


        le_256_q = self.LE_256_Q(le_part_256)
        re_256_q = self.RE_256_Q(re_part_256)
        mo_256_q = self.MO_256_Q(mo_part_256)

        le_128_q = self.LE_128_Q(le_part_128)
        re_128_q = self.RE_128_Q(re_part_128)
        mo_128_q = self.MO_128_Q(mo_part_128)

        le_64_q = self.LE_64_Q(le_part_64)
        re_64_q = self.RE_64_Q(re_part_64)
        mo_64_q = self.MO_64_Q(mo_part_64)

        return {'f256': f2_1, 'f128': f4_1, 'f64': f6_1,\
            'le256': le_part_256, 're256': re_part_256, 'mo256': mo_part_256, \
            'le128': le_part_128, 're128': re_part_128, 'mo128': mo_part_128, \
            'le64': le_part_64, 're64': re_part_64, 'mo64': mo_part_64, \
            'le_256_q': le_256_q, 're_256_q': re_256_q, 'mo_256_q': mo_256_q,\
            'le_128_q': le_128_q, 're_128_q': re_128_q, 'mo_128_q': mo_128_q,\
            'le_64_q': le_64_q, 're_64_q': re_64_q, 'mo_64_q': mo_64_q}


class DMDNet(nn.Module):
    def __init__(self, ngf = 64, banks_num = 128):
        super().__init__()
        self.part_sizes = np.array([80,80,50,110]) # size for 512
        self.feature_sizes = np.array([256,128,64]) # size for 512

        self.banks_num = banks_num
        self.key_scale = 4

        self.E_lq = FeatureExtractor(key_scale = self.key_scale)
        self.E_hq = FeatureExtractor(key_scale = self.key_scale)

        self.LE_256_KV = KeyValue(ngf, ngf // self.key_scale, ngf)
        self.RE_256_KV = KeyValue(ngf, ngf // self.key_scale, ngf)
        self.MO_256_KV = KeyValue(ngf, ngf // self.key_scale, ngf)

        self.LE_128_KV = KeyValue(ngf * 2 , ngf * 2 // self.key_scale, ngf * 2)
        self.RE_128_KV = KeyValue(ngf * 2 , ngf * 2 // self.key_scale, ngf * 2)
        self.MO_128_KV = KeyValue(ngf * 2 , ngf * 2 // self.key_scale, ngf * 2)

        self.LE_64_KV = KeyValue(ngf * 4 , ngf * 4 // self.key_scale, ngf * 4)
        self.RE_64_KV = KeyValue(ngf * 4 , ngf * 4 // self.key_scale, ngf * 4)
        self.MO_64_KV = KeyValue(ngf * 4 , ngf * 4 // self.key_scale, ngf * 4)


        self.LE_256_Attention = AttentionBlock(64)
        self.RE_256_Attention = AttentionBlock(64)
        self.MO_256_Attention = AttentionBlock(64)

        self.LE_128_Attention = AttentionBlock(128)
        self.RE_128_Attention = AttentionBlock(128)
        self.MO_128_Attention = AttentionBlock(128)

        self.LE_64_Attention = AttentionBlock(256)
        self.RE_64_Attention = AttentionBlock(256)
        self.MO_64_Attention = AttentionBlock(256)

        self.LE_256_Mask = MaskAttention(64)
        self.RE_256_Mask = MaskAttention(64)
        self.MO_256_Mask = MaskAttention(64)

        self.LE_128_Mask = MaskAttention(128)
        self.RE_128_Mask = MaskAttention(128)
        self.MO_128_Mask = MaskAttention(128)

        self.LE_64_Mask = MaskAttention(256)
        self.RE_64_Mask = MaskAttention(256)
        self.MO_64_Mask = MaskAttention(256)

        self.MSDilate = MSDilateBlock(ngf*4, dilation = [4,3,2,1])

        self.up1 = StyledUpBlock(ngf*4, ngf*2, noise_inject=False) #
        self.up2 = StyledUpBlock(ngf*2, ngf, noise_inject=False) #
        self.up3 = StyledUpBlock(ngf, ngf, noise_inject=False) #
        self.up4 = nn.Sequential( 
            SpectralNorm(nn.Conv2d(ngf, ngf, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            UpResBlock(ngf),
            UpResBlock(ngf),
            SpectralNorm(nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1)),
            nn.Tanh()
        )
 
        # define generic memory, revise register_buffer to register_parameter for backward update
        self.register_buffer('le_256_mem_key', torch.randn(128,16,40,40))
        self.register_buffer('re_256_mem_key', torch.randn(128,16,40,40))
        self.register_buffer('mo_256_mem_key', torch.randn(128,16,55,55))
        self.register_buffer('le_256_mem_value', torch.randn(128,64,40,40))
        self.register_buffer('re_256_mem_value', torch.randn(128,64,40,40))
        self.register_buffer('mo_256_mem_value', torch.randn(128,64,55,55))
        

        self.register_buffer('le_128_mem_key', torch.randn(128,32,20,20))
        self.register_buffer('re_128_mem_key', torch.randn(128,32,20,20))
        self.register_buffer('mo_128_mem_key', torch.randn(128,32,27,27))
        self.register_buffer('le_128_mem_value', torch.randn(128,128,20,20))
        self.register_buffer('re_128_mem_value', torch.randn(128,128,20,20))
        self.register_buffer('mo_128_mem_value', torch.randn(128,128,27,27))

        self.register_buffer('le_64_mem_key', torch.randn(128,64,10,10))
        self.register_buffer('re_64_mem_key', torch.randn(128,64,10,10))
        self.register_buffer('mo_64_mem_key', torch.randn(128,64,13,13))
        self.register_buffer('le_64_mem_value', torch.randn(128,256,10,10))
        self.register_buffer('re_64_mem_value', torch.randn(128,256,10,10))
        self.register_buffer('mo_64_mem_value', torch.randn(128,256,13,13))

    
    def readMem(self, k, v, q):
        sim = F.conv2d(q, k)
        score = F.softmax(sim/sqrt(sim.size(1)), dim=1) #B * S * 1 * 1 6*128
        sb,sn,sw,sh = score.size()
        s_m = score.view(sb, -1).unsqueeze(1)#2*1*M
        vb,vn,vw,vh = v.size()
        v_in = v.view(vb, -1).repeat(sb,1,1)#2*M*(c*w*h)
        mem_out = torch.bmm(s_m, v_in).squeeze(1).view(sb, vn, vw,vh)
        max_inds = torch.argmax(score, dim=1).squeeze()
        return mem_out, max_inds
    

    def memorize(self, img, locs):
        fs = self.E_hq(img, locs)
        LE256_key, LE256_value = self.LE_256_KV(fs['le256'])
        RE256_key, RE256_value = self.RE_256_KV(fs['re256'])
        MO256_key, MO256_value = self.MO_256_KV(fs['mo256'])

        LE128_key, LE128_value = self.LE_128_KV(fs['le128'])
        RE128_key, RE128_value = self.RE_128_KV(fs['re128'])
        MO128_key, MO128_value = self.MO_128_KV(fs['mo128'])

        LE64_key, LE64_value = self.LE_64_KV(fs['le64'])
        RE64_key, RE64_value = self.RE_64_KV(fs['re64'])
        MO64_key, MO64_value = self.MO_64_KV(fs['mo64'])

        Mem256 = {'LE256Key': LE256_key, 'LE256Value': LE256_value, 'RE256Key': RE256_key, 'RE256Value': RE256_value,'MO256Key': MO256_key, 'MO256Value': MO256_value}
        Mem128 = {'LE128Key': LE128_key, 'LE128Value': LE128_value, 'RE128Key': RE128_key, 'RE128Value': RE128_value,'MO128Key': MO128_key, 'MO128Value': MO128_value}
        Mem64 = {'LE64Key': LE64_key, 'LE64Value': LE64_value, 'RE64Key': RE64_key, 'RE64Value': RE64_value,'MO64Key': MO64_key, 'MO64Value': MO64_value}
 
        FS256 = {'LE256F':fs['le256'], 'RE256F':fs['re256'], 'MO256F':fs['mo256']}
        FS128 = {'LE128F':fs['le128'], 'RE128F':fs['re128'], 'MO128F':fs['mo128']}
        FS64 = {'LE64F':fs['le64'], 'RE64F':fs['re64'], 'MO64F':fs['mo64']}
        
        return Mem256, Mem128, Mem64

    def enhancer(self, fs_in, sp_256=None, sp_128=None, sp_64=None):
        le_256_q = fs_in['le_256_q']
        re_256_q = fs_in['re_256_q']
        mo_256_q = fs_in['mo_256_q']

        le_128_q = fs_in['le_128_q']
        re_128_q = fs_in['re_128_q']
        mo_128_q = fs_in['mo_128_q']

        le_64_q = fs_in['le_64_q']
        re_64_q = fs_in['re_64_q']
        mo_64_q = fs_in['mo_64_q']

        
        ####for 256
        le_256_mem_g, le_256_inds = self.readMem(self.le_256_mem_key, self.le_256_mem_value, le_256_q)
        re_256_mem_g, re_256_inds = self.readMem(self.re_256_mem_key, self.re_256_mem_value, re_256_q)
        mo_256_mem_g, mo_256_inds = self.readMem(self.mo_256_mem_key, self.mo_256_mem_value, mo_256_q)

        le_128_mem_g, le_128_inds = self.readMem(self.le_128_mem_key, self.le_128_mem_value, le_128_q)
        re_128_mem_g, re_128_inds = self.readMem(self.re_128_mem_key, self.re_128_mem_value, re_128_q)
        mo_128_mem_g, mo_128_inds = self.readMem(self.mo_128_mem_key, self.mo_128_mem_value, mo_128_q)

        le_64_mem_g, le_64_inds = self.readMem(self.le_64_mem_key, self.le_64_mem_value, le_64_q)
        re_64_mem_g, re_64_inds = self.readMem(self.re_64_mem_key, self.re_64_mem_value, re_64_q)
        mo_64_mem_g, mo_64_inds = self.readMem(self.mo_64_mem_key, self.mo_64_mem_value, mo_64_q)

        if sp_256 is not None and sp_128 is not None and sp_64 is not None:
            le_256_mem_s, _ = self.readMem(sp_256['LE256Key'], sp_256['LE256Value'], le_256_q)
            re_256_mem_s, _ = self.readMem(sp_256['RE256Key'], sp_256['RE256Value'], re_256_q)
            mo_256_mem_s, _ = self.readMem(sp_256['MO256Key'], sp_256['MO256Value'], mo_256_q)
            le_256_mask = self.LE_256_Mask(fs_in['le256'],le_256_mem_s,le_256_mem_g)
            le_256_mem = le_256_mask*le_256_mem_s + (1-le_256_mask)*le_256_mem_g
            re_256_mask = self.RE_256_Mask(fs_in['re256'],re_256_mem_s,re_256_mem_g)
            re_256_mem = re_256_mask*re_256_mem_s + (1-re_256_mask)*re_256_mem_g
            mo_256_mask = self.MO_256_Mask(fs_in['mo256'],mo_256_mem_s,mo_256_mem_g)
            mo_256_mem = mo_256_mask*mo_256_mem_s + (1-mo_256_mask)*mo_256_mem_g

            le_128_mem_s, _ = self.readMem(sp_128['LE128Key'], sp_128['LE128Value'], le_128_q)
            re_128_mem_s, _ = self.readMem(sp_128['RE128Key'], sp_128['RE128Value'], re_128_q)
            mo_128_mem_s, _ = self.readMem(sp_128['MO128Key'], sp_128['MO128Value'], mo_128_q)
            le_128_mask = self.LE_128_Mask(fs_in['le128'],le_128_mem_s,le_128_mem_g)
            le_128_mem = le_128_mask*le_128_mem_s + (1-le_128_mask)*le_128_mem_g
            re_128_mask = self.RE_128_Mask(fs_in['re128'],re_128_mem_s,re_128_mem_g)
            re_128_mem = re_128_mask*re_128_mem_s + (1-re_128_mask)*re_128_mem_g
            mo_128_mask = self.MO_128_Mask(fs_in['mo128'],mo_128_mem_s,mo_128_mem_g)
            mo_128_mem = mo_128_mask*mo_128_mem_s + (1-mo_128_mask)*mo_128_mem_g

            le_64_mem_s, _ = self.readMem(sp_64['LE64Key'], sp_64['LE64Value'], le_64_q)
            re_64_mem_s, _ = self.readMem(sp_64['RE64Key'], sp_64['RE64Value'], re_64_q)
            mo_64_mem_s, _ = self.readMem(sp_64['MO64Key'], sp_64['MO64Value'], mo_64_q)
            le_64_mask = self.LE_64_Mask(fs_in['le64'],le_64_mem_s,le_64_mem_g)
            le_64_mem = le_64_mask*le_64_mem_s + (1-le_64_mask)*le_64_mem_g
            re_64_mask = self.RE_64_Mask(fs_in['re64'],re_64_mem_s,re_64_mem_g)
            re_64_mem = re_64_mask*re_64_mem_s + (1-re_64_mask)*re_64_mem_g
            mo_64_mask = self.MO_64_Mask(fs_in['mo64'],mo_64_mem_s,mo_64_mem_g)
            mo_64_mem = mo_64_mask*mo_64_mem_s + (1-mo_64_mask)*mo_64_mem_g
        else:
            le_256_mem = le_256_mem_g
            re_256_mem = re_256_mem_g
            mo_256_mem = mo_256_mem_g
            le_128_mem = le_128_mem_g
            re_128_mem = re_128_mem_g
            mo_128_mem = mo_128_mem_g
            le_64_mem = le_64_mem_g
            re_64_mem = re_64_mem_g
            mo_64_mem = mo_64_mem_g

        le_256_mem_norm = adaptive_instance_normalization_4D(le_256_mem, fs_in['le256'])
        re_256_mem_norm = adaptive_instance_normalization_4D(re_256_mem, fs_in['re256'])
        mo_256_mem_norm = adaptive_instance_normalization_4D(mo_256_mem, fs_in['mo256'])
        
        ####for 128
        le_128_mem_norm = adaptive_instance_normalization_4D(le_128_mem, fs_in['le128'])
        re_128_mem_norm = adaptive_instance_normalization_4D(re_128_mem, fs_in['re128'])
        mo_128_mem_norm = adaptive_instance_normalization_4D(mo_128_mem, fs_in['mo128'])
        
        ####for 64
        le_64_mem_norm = adaptive_instance_normalization_4D(le_64_mem, fs_in['le64'])
        re_64_mem_norm = adaptive_instance_normalization_4D(re_64_mem, fs_in['re64'])
        mo_64_mem_norm = adaptive_instance_normalization_4D(mo_64_mem, fs_in['mo64'])
    

        EnMem256 = {'LE256Norm': le_256_mem_norm, 'RE256Norm': re_256_mem_norm, 'MO256Norm': mo_256_mem_norm}
        EnMem128 = {'LE128Norm': le_128_mem_norm, 'RE128Norm': re_128_mem_norm, 'MO128Norm': mo_128_mem_norm}
        EnMem64 = {'LE64Norm': le_64_mem_norm, 'RE64Norm': re_64_mem_norm, 'MO64Norm': mo_64_mem_norm}
        Ind256 = {'LE': le_256_inds, 'RE': re_256_inds, 'MO': mo_256_inds}
        Ind128 = {'LE': le_128_inds, 'RE': re_128_inds, 'MO': mo_128_inds}
        Ind64 = {'LE': le_64_inds, 'RE': re_64_inds, 'MO': mo_64_inds}
        return EnMem256, EnMem128, EnMem64, Ind256, Ind128, Ind64

    def reconstruct(self, fs_in, locs, memstar):
        le_256_mem_norm, re_256_mem_norm, mo_256_mem_norm = memstar[0]['LE256Norm'], memstar[0]['RE256Norm'], memstar[0]['MO256Norm']
        le_128_mem_norm, re_128_mem_norm, mo_128_mem_norm = memstar[1]['LE128Norm'], memstar[1]['RE128Norm'], memstar[1]['MO128Norm']
        le_64_mem_norm, re_64_mem_norm, mo_64_mem_norm = memstar[2]['LE64Norm'], memstar[2]['RE64Norm'], memstar[2]['MO64Norm']

        le_256_final = self.LE_256_Attention(le_256_mem_norm - fs_in['le256']) * le_256_mem_norm + fs_in['le256']
        re_256_final = self.RE_256_Attention(re_256_mem_norm - fs_in['re256']) * re_256_mem_norm + fs_in['re256']
        mo_256_final = self.MO_256_Attention(mo_256_mem_norm - fs_in['mo256']) * mo_256_mem_norm + fs_in['mo256']
        
        le_128_final = self.LE_128_Attention(le_128_mem_norm - fs_in['le128']) * le_128_mem_norm + fs_in['le128']
        re_128_final = self.RE_128_Attention(re_128_mem_norm - fs_in['re128']) * re_128_mem_norm + fs_in['re128']
        mo_128_final = self.MO_128_Attention(mo_128_mem_norm - fs_in['mo128']) * mo_128_mem_norm + fs_in['mo128']
        
        le_64_final = self.LE_64_Attention(le_64_mem_norm - fs_in['le64']) * le_64_mem_norm + fs_in['le64']
        re_64_final = self.RE_64_Attention(re_64_mem_norm - fs_in['re64']) * re_64_mem_norm + fs_in['re64']
        mo_64_final = self.MO_64_Attention(mo_64_mem_norm - fs_in['mo64']) * mo_64_mem_norm + fs_in['mo64']


        le_location = locs[:,0,:]
        re_location = locs[:,1,:]
        mo_location = locs[:,3,:]
        le_location = le_location.cpu().int().numpy()
        re_location = re_location.cpu().int().numpy()
        mo_location = mo_location.cpu().int().numpy()

        up_in_256 = fs_in['f256'].clone()# * 0
        up_in_128 = fs_in['f128'].clone()# * 0
        up_in_64 = fs_in['f64'].clone()# * 0

        for i in range(fs_in['f256'].size(0)):
            up_in_256[i:i+1,:,le_location[i,1]//2:le_location[i,3]//2,le_location[i,0]//2:le_location[i,2]//2] = F.interpolate(le_256_final[i:i+1,:,:,:].clone(), (le_location[i,3]//2-le_location[i,1]//2,le_location[i,2]//2-le_location[i,0]//2),mode='bilinear',align_corners=False)
            up_in_256[i:i+1,:,re_location[i,1]//2:re_location[i,3]//2,re_location[i,0]//2:re_location[i,2]//2] = F.interpolate(re_256_final[i:i+1,:,:,:].clone(), (re_location[i,3]//2-re_location[i,1]//2,re_location[i,2]//2-re_location[i,0]//2),mode='bilinear',align_corners=False)
            up_in_256[i:i+1,:,mo_location[i,1]//2:mo_location[i,3]//2,mo_location[i,0]//2:mo_location[i,2]//2] = F.interpolate(mo_256_final[i:i+1,:,:,:].clone(), (mo_location[i,3]//2-mo_location[i,1]//2,mo_location[i,2]//2-mo_location[i,0]//2),mode='bilinear',align_corners=False)
            
            up_in_128[i:i+1,:,le_location[i,1]//4:le_location[i,3]//4,le_location[i,0]//4:le_location[i,2]//4] = F.interpolate(le_128_final[i:i+1,:,:,:].clone(), (le_location[i,3]//4-le_location[i,1]//4,le_location[i,2]//4-le_location[i,0]//4),mode='bilinear',align_corners=False)
            up_in_128[i:i+1,:,re_location[i,1]//4:re_location[i,3]//4,re_location[i,0]//4:re_location[i,2]//4] = F.interpolate(re_128_final[i:i+1,:,:,:].clone(), (re_location[i,3]//4-re_location[i,1]//4,re_location[i,2]//4-re_location[i,0]//4),mode='bilinear',align_corners=False)
            up_in_128[i:i+1,:,mo_location[i,1]//4:mo_location[i,3]//4,mo_location[i,0]//4:mo_location[i,2]//4] = F.interpolate(mo_128_final[i:i+1,:,:,:].clone(), (mo_location[i,3]//4-mo_location[i,1]//4,mo_location[i,2]//4-mo_location[i,0]//4),mode='bilinear',align_corners=False)

            up_in_64[i:i+1,:,le_location[i,1]//8:le_location[i,3]//8,le_location[i,0]//8:le_location[i,2]//8] = F.interpolate(le_64_final[i:i+1,:,:,:].clone(), (le_location[i,3]//8-le_location[i,1]//8,le_location[i,2]//8-le_location[i,0]//8),mode='bilinear',align_corners=False)
            up_in_64[i:i+1,:,re_location[i,1]//8:re_location[i,3]//8,re_location[i,0]//8:re_location[i,2]//8] = F.interpolate(re_64_final[i:i+1,:,:,:].clone(), (re_location[i,3]//8-re_location[i,1]//8,re_location[i,2]//8-re_location[i,0]//8),mode='bilinear',align_corners=False)
            up_in_64[i:i+1,:,mo_location[i,1]//8:mo_location[i,3]//8,mo_location[i,0]//8:mo_location[i,2]//8] = F.interpolate(mo_64_final[i:i+1,:,:,:].clone(), (mo_location[i,3]//8-mo_location[i,1]//8,mo_location[i,2]//8-mo_location[i,0]//8),mode='bilinear',align_corners=False)
        
        ms_in_64 = self.MSDilate(fs_in['f64'].clone())
        fea_up1 = self.up1(ms_in_64, up_in_64)
        fea_up2 = self.up2(fea_up1, up_in_128) #
        fea_up3 = self.up3(fea_up2, up_in_256) #
        output = self.up4(fea_up3) #
        return output

    def generate_specific_dictionary(self, sp_imgs=None, sp_locs=None):
        return self.memorize(sp_imgs, sp_locs)

    def forward(self, lq=None, loc=None, sp_256 = None, sp_128 = None, sp_64 = None):
        fs_in = self.E_lq(lq, loc) # low quality images
        GeMemNorm256, GeMemNorm128, GeMemNorm64, Ind256, Ind128, Ind64 = self.enhancer(fs_in)
        GeOut = self.reconstruct(fs_in, loc, memstar = [GeMemNorm256, GeMemNorm128, GeMemNorm64])
        if sp_256 is not None and sp_128 is not None and sp_64 is not None:
            GSMemNorm256, GSMemNorm128, GSMemNorm64, _, _, _ = self.enhancer(fs_in, sp_256, sp_128, sp_64)
            GSOut = self.reconstruct(fs_in, loc, memstar = [GSMemNorm256, GSMemNorm128, GSMemNorm64])
        else:
            GSOut = None
        return GeOut, GSOut

class UpResBlock(nn.Module):
    def __init__(self, dim, conv_layer = nn.Conv2d, norm_layer = nn.BatchNorm2d):
        super(UpResBlock, self).__init__()
        self.Model = nn.Sequential(
            SpectralNorm(conv_layer(dim, dim, 3, 1, 1)),
            nn.LeakyReLU(0.2),
            SpectralNorm(conv_layer(dim, dim, 3, 1, 1)),
        )
    def forward(self, x):
        out = x + self.Model(x)
        return out

