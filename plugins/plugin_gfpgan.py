from chain_img_processor import ChainImgProcessor, ChainImgPlugin
import os
import gfpgan
import threading
from PIL import Image
from numpy import asarray

from roop.utilities import resolve_relative_path, conditional_download
modname = os.path.basename(__file__)[:-3] # calculating modname

model_gfpgan = None
THREAD_LOCK_GFPGAN = threading.Lock()


# start function
def start(core:ChainImgProcessor):
    manifest = { # plugin settings
        "name": "GFPGAN", # name
        "version": "1.4", # version

        "default_options": {},
        "img_processor": {
            "gfpgan": GFPGAN
        }
    }
    return manifest

def start_with_options(core:ChainImgProcessor, manifest:dict):
    pass


class GFPGAN(ChainImgPlugin):

    def init_plugin(self):
        global model_gfpgan

        if model_gfpgan is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
            model_gfpgan = gfpgan.GFPGANer(model_path=model_path, upscale=1) # type: ignore[attr-defined]



    def process(self, frame, params:dict):
        global model_gfpgan

        if model_gfpgan is None:
            return frame 
        
        if "face_detected" in params:
            if not params["face_detected"]:
                return frame

        with THREAD_LOCK_GFPGAN:
            _, _, temp_frame = model_gfpgan.enhance(
                    frame,
                    paste_back=True
                )

        if not "blend_ratio" in params: 
            return temp_frame

        temp_frame = Image.blend(Image.fromarray(frame), Image.fromarray(temp_frame), params["blend_ratio"])
        return asarray(temp_frame)
