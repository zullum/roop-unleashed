# Codeformer enchance plugin
# author: Vladislav Janvarev

# CountFloyd 20230717, extended to blend original/destination images

from chain_img_processor import ChainImgProcessor, ChainImgPlugin
import os
import cv2
from numpy import asarray

modname = os.path.basename(__file__)[:-3] # calculating modname

# start function
def start(core:ChainImgProcessor):
    manifest = { # plugin settings
        "name": "Codeformer", # name
        "version": "3.0", # version

        "default_options": {
            "background_enhance": True,  #
            "face_upsample": True,  #
            "upscale": 2,  #
            "codeformer_fidelity": 0.8,
            "skip_if_no_face":False,

        },

        "img_processor": {
            "codeformer": PluginCodeformer # 1 function - init, 2 - process
        }
    }
    return manifest

def start_with_options(core:ChainImgProcessor, manifest:dict):
    pass

class PluginCodeformer(ChainImgPlugin):
    def init_plugin(self):
        import plugins.codeformer_app_cv2
        pass

    def process(self, img, params:dict):
        import copy
        
        # params can be used to transfer some img info to next processors
        from plugins.codeformer_app_cv2 import inference_app
        options = self.core.plugin_options(modname)

        if "face_detected" in params:
            if not params["face_detected"]:
                return img

        # don't touch original    
        temp_frame = copy.copy(img)
        if "processed_faces" in params:
            for face in params["processed_faces"]:
                start_x, start_y, end_x, end_y = map(int, face['bbox'])
                temp_face, start_x, start_y, end_x, end_y = self.cutout(temp_frame, start_x, start_y, end_x, end_y, 0.5)
                if temp_face.size:
                    temp_face = inference_app(temp_face, options.get("background_enhance"), options.get("face_upsample"),
                                        options.get("upscale"), options.get("codeformer_fidelity"),
                                        options.get("skip_if_no_face"))
                    temp_frame = self.paste_into(temp_face, temp_frame, start_x, start_y, end_x, end_y, False)
        else:
            temp_frame = inference_app(temp_frame, options.get("background_enhance"), options.get("face_upsample"),
                options.get("upscale"), options.get("codeformer_fidelity"),
                options.get("skip_if_no_face"))

        if not "blend_ratio" in params: 
            return temp_frame
        
        blend_ratio = params["blend_ratio"]
        return cv2.addWeighted(temp_frame, blend_ratio, img, 1.0 - blend_ratio,0)

