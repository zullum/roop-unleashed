import cv2
import numpy as np
import onnxruntime
from chain_img_processor import ChainImgProcessor, ChainImgPlugin
import os
from numpy import asarray

from roop.utilities import resolve_relative_path
import roop.globals

modname = os.path.basename(__file__)[:-3] # calculating modname

# onnx gfpgan code & model ported from
# https://github.com/fAIseh00d/refacer/tree/main

model_gfpgan = None


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
    name = None

    def init_plugin(self):
        global model_gfpgan

        if model_gfpgan is None:
            model_path = resolve_relative_path('../models/GFPGANv1.4.onnx')
            model_gfpgan = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)

        self.name = model_gfpgan.get_inputs()[0].name


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
                    temp_face = self.enhance(temp_face)
                    temp_frame = self.paste_into(temp_face, temp_frame, start_x, start_y, end_x, end_y, False)
        else:
            temp_frame = self.enhance(temp_frame)

        if not "blend_ratio" in params: 
            return temp_frame
        
        blend_ratio = params["blend_ratio"]
        return cv2.addWeighted(temp_frame, blend_ratio, frame, 1.0 - blend_ratio,0)


    def enhance(self, image_array):
        input_shape = image_array.shape
        image_array = self.pre_process(image_array)
        ort_inputs = {self.name: image_array}
        result = model_gfpgan.run(None, ort_inputs)[0][0]
        result = self.post_process(result)
        result = cv2.resize(result, (input_shape[1], input_shape[0]))
        return result


    def pre_process(self, image_array):
        image_array = cv2.resize(image_array, (512, 512))
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image_array = image_array.astype('float32') / 255.0
        image_array = (image_array - 0.5) / 0.5
        image_array = np.expand_dims(image_array, axis=0).transpose(0, 3, 1, 2)
        return image_array

    def post_process(self, result):
        result = np.clip(result, -1, 1)
        result = (result + 1) / 2
        result = result.transpose(1, 2, 0) * 255.0
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        return result.astype(np.uint8)

