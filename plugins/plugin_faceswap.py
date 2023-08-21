from chain_img_processor import ChainImgProcessor, ChainImgPlugin
from roop.face_util import get_first_face, get_all_faces, swap_face
import os
from roop.utilities import compute_cosine_distance

modname = os.path.basename(__file__)[:-3] # calculating modname

# start function
def start(core:ChainImgProcessor):
    manifest = { # plugin settings
        "name": "Faceswap", # name
        "version": "1.0", # version

        "default_options": {
            "swap_mode": "selected",
            "max_distance": 0.65, # max distance to detect face similarity
        },
        "img_processor": {
            "faceswap": Faceswap
        }
    }
    return manifest

def start_with_options(core:ChainImgProcessor, manifest:dict):
    pass


class Faceswap(ChainImgPlugin):

    def init_plugin(self):
        pass


    def process(self, frame, params:dict):
        if not "input_face_datas" in params or len(params["input_face_datas"]) < 1:
            params["face_detected"] = False
            return  frame
        
        temp_frame = frame
        params["face_detected"] = True
        params["processed_faces"] = []

        face_index = 0
        if "selected_index" in params:
            face_index = params["selected_index"]
        if face_index >= len(params["input_face_datas"]):
            face_index = 0


        if params["swap_mode"] == "first":
            face = get_first_face(frame)
            if face is None:
                params["face_detected"] = False
                return frame
            params["processed_faces"].append(face)
            frame = swap_face(params["input_face_datas"][face_index], face, frame) 
            return frame

        else:
            faces = get_all_faces(frame)
            if faces is None:
                params["face_detected"] = False
                return frame
            
            dist_threshold = params["face_distance_threshold"]

            if params["swap_mode"] == "all":
                for face in faces:
                    params["processed_faces"].append(face)
                    temp_frame = swap_face(params["input_face_datas"][face_index], face, temp_frame)
                return temp_frame
            
            elif params["swap_mode"] == "selected":
                for i,tf in enumerate(params["target_face_datas"]):
                    for face in faces:
                        if compute_cosine_distance(tf.embedding, face.embedding) <= dist_threshold:
                            if(i < len(params["input_face_datas"])):
                                temp_frame = swap_face(params["input_face_datas"][i], face, temp_frame)
                                params["processed_faces"].append(face)
                            break

            elif params["swap_mode"] == "all_female" or params["swap_mode"] == "all_male":
                gender = 'F' if params["swap_mode"] == "all_female" else 'M'
                face_found = False
                for face in faces:
                    if face.sex == gender:
                        face_found = True
                    if face_found:
                        params["processed_faces"].append(face)
                        temp_frame = swap_face(params["input_face_datas"][face_index], face, temp_frame)
                        face_found = False

        return temp_frame
