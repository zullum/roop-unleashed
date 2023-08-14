import os
import cv2
import numpy as np
import torch
import threading
from chain_img_processor import ChainImgProcessor, ChainImgPlugin
from torchvision import transforms
from clip.clipseg import CLIPDensePredT
from numpy import asarray


THREAD_LOCK_CLIP = threading.Lock()

modname = os.path.basename(__file__)[:-3] # calculating modname

model_clip = None

   


# start function
def start(core:ChainImgProcessor):
    manifest = { # plugin settings
        "name": "Text2Clip", # name
        "version": "1.0", # version

        "default_options": {
        },
        "img_processor": {
            "txt2clip": Text2Clip
        }
    }
    return manifest

def start_with_options(core:ChainImgProcessor, manifest:dict):
    pass



class Text2Clip(ChainImgPlugin):

    def load_clip_model(self):
        global model_clip

        if model_clip is None:
            model_clip = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
            model_clip.eval();
            model_clip.load_state_dict(torch.load('models/CLIP/rd64-uni-refined.pth', map_location=torch.device('cpu')), strict=False)

        device = torch.device(super().device)
        model_clip.to(device)    


    def init_plugin(self):
        self.load_clip_model()

    def process(self, frame, params:dict):
        if "face_detected" in params:
            if not params["face_detected"]:
                return frame
       
        return self.mask_original(params["original_frame"], frame, params["clip_prompt"])
        
    def unload(self):
        model_clip.to('cpu')


    def mask_original(self, img1, img2, keywords):
        global model_clip

        source_image_small = cv2.resize(img1, (256,256))
        
        img_mask = np.full((source_image_small.shape[0],source_image_small.shape[1]), 0, dtype=np.float32)
        mask_border = 1
        l = 0
        t = 0
        r = 1
        b = 1
        
        mask_blur = 5
        clip_blur = 5
        
        img_mask = cv2.rectangle(img_mask, (mask_border+int(l), mask_border+int(t)), 
                                (256 - mask_border-int(r), 256-mask_border-int(b)), (255, 255, 255), -1)    
        img_mask = cv2.GaussianBlur(img_mask, (mask_blur*2+1,mask_blur*2+1), 0)    
        img_mask /= 255

        
        input_image = source_image_small

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((256, 256)),
        ])
        img = transform(input_image).unsqueeze(0)

        thresh = 0.5
        prompts = keywords.split(',')
        with THREAD_LOCK_CLIP:
            with torch.no_grad():
                preds = model_clip(img.repeat(len(prompts),1,1,1), prompts)[0]
        clip_mask = torch.sigmoid(preds[0][0])
        for i in range(len(prompts)-1):
            clip_mask += torch.sigmoid(preds[i+1][0])
           
        clip_mask = clip_mask.data.cpu().numpy()
        np.clip(clip_mask, 0, 1)
        
        clip_mask[clip_mask>thresh] = 1.0
        clip_mask[clip_mask<=thresh] = 0.0
        kernel = np.ones((5, 5), np.float32)
        clip_mask = cv2.dilate(clip_mask, kernel, iterations=1)
        clip_mask = cv2.GaussianBlur(clip_mask, (clip_blur*2+1,clip_blur*2+1), 0)
       
        img_mask *= clip_mask
        img_mask[img_mask<0.0] = 0.0
       
        img_mask = cv2.resize(img_mask, (img2.shape[1], img2.shape[0]))
        img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
       
        target = img2.astype(np.float32)
        result = (1-img_mask) * target
        result += img_mask * img1.astype(np.float32)
        return np.uint8(result)
       
