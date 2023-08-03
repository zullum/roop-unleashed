# Codeformer enchance plugin
# author: Vladislav Janvarev

# CountFloyd 20230717, extended to blend original/destination images

from chain_img_processor import ChainImgProcessor, ChainImgPlugin
import os
from PIL import Image
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
                padding_x = int((end_x - start_x) * 0.5)
                padding_y = int((end_y - start_y) * 0.5)
                start_x = max(0, start_x - padding_x)
                start_y = max(0, start_y - padding_y)
                end_x = max(0, end_x + padding_x)
                end_y = max(0, end_y + padding_y)
                temp_face = temp_frame[start_y:end_y, start_x:end_x]
                if temp_face.size:
                    temp_face = inference_app(temp_face, options.get("background_enhance"), options.get("face_upsample"),
                                        options.get("upscale"), options.get("codeformer_fidelity"),
                                        options.get("skip_if_no_face"))
                    temp_frame[start_y:end_y, start_x:end_x] = temp_face
        else:
            temp_frame = inference_app(temp_frame, options.get("background_enhance"), options.get("face_upsample"),
                options.get("upscale"), options.get("codeformer_fidelity"),
                options.get("skip_if_no_face"))

        

        if not "blend_ratio" in params: 
            return temp_frame


        temp_frame = Image.blend(Image.fromarray(img), Image.fromarray(temp_frame), params["blend_ratio"])
        return asarray(temp_frame)

