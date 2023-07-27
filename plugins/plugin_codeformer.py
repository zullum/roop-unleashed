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
        # params can be used to transfer some img info to next processors
        from plugins.codeformer_app_cv2 import inference_app
        options = self.core.plugin_options(modname)

        if "face_detected" in params:
            if not params["face_detected"]:
                return img

        temp_frame = inference_app(img, options.get("background_enhance"), options.get("face_upsample"),
                              options.get("upscale"), options.get("codeformer_fidelity"),
                              options.get("skip_if_no_face"))
        

        if not "blend_ratio" in params: 
            return temp_frame


        temp_frame = Image.blend(Image.fromarray(img), Image.fromarray(temp_frame), params["blend_ratio"])
        return asarray(temp_frame)

