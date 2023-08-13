from chain_img_processor import ChainImgProcessor, ChainImgPlugin
import os
import gfpgan
import threading
from PIL import Image
from numpy import asarray
import cv2

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
            model_gfpgan = gfpgan.GFPGANer(model_path=model_path, upscale=1, device=self.device) # type: ignore[attr-defined]


    def process(self, frame, params:dict):
        import copy

        global model_gfpgan

        if model_gfpgan is None:
            return frame 
        
        if "face_detected" in params:
            if not params["face_detected"]:
                return frame
        # don't touch original    
        temp_frame = copy.copy(frame)
        if "processed_faces" in params:
            for face in params["processed_faces"]:
                start_x, start_y, end_x, end_y = map(int, face['bbox'])
                temp_face, start_x, start_y, end_x, end_y = self.cutout(temp_frame, start_x, start_y, end_x, end_y, 0.5)
                if temp_face.size:
                    with THREAD_LOCK_GFPGAN:
                        _, _, temp_face = model_gfpgan.enhance(
                                temp_face,
                                paste_back=True
                            )
                    temp_frame = self.paste_into(temp_face, temp_frame, start_x, start_y, end_x, end_y, False)
        else:
            with THREAD_LOCK_GFPGAN:
                _, _, temp_frame = model_gfpgan.enhance(
                        temp_frame,
                        paste_back=True
                    )

        if not "blend_ratio" in params: 
            return temp_frame

        temp_frame = Image.blend(Image.fromarray(frame), Image.fromarray(temp_frame), params["blend_ratio"])
        return asarray(temp_frame)
