import threading
from typing import Any
import insightface

import roop.globals
from roop.typing import Frame, Face

import cv2
import numpy as np
from roop.capturer import get_video_frame
from roop.utilities import resolve_relative_path, conditional_download

FACE_ANALYSER = None
THREAD_LOCK_ANALYSER = threading.Lock()
THREAD_LOCK_SWAPPER = threading.Lock()
FACE_SWAPPER = None


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK_ANALYSER:
        if FACE_ANALYSER is None:
            if roop.globals.CFG.force_cpu:
                print('Forcing CPU for Face Analysis')
                FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            else:
                FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640) if roop.globals.default_det_size else (320,320))
    return FACE_ANALYSER


def get_first_face(frame: Frame) -> Any:
    try:
        faces = get_face_analyser().get(frame)
        return min(faces, key=lambda x: x.bbox[0])
    #   return sorted(faces, reverse=True, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))[0]
    except:
        return None


def get_all_faces(frame: Frame) -> Any:
    try:
        faces = get_face_analyser().get(frame)
        return sorted(faces, key = lambda x : x.bbox[0])
    except:
        return None


def extract_face_images(source_filename, video_info):
    face_data = []
    source_image = None
    
    if video_info[0]:
        frame = get_video_frame(source_filename, video_info[1])
        if frame is not None:
            source_image = frame
        else:
            return face_data
    else:
        source_image = cv2.imread(source_filename)

        
    faces = get_all_faces(source_image)
    if faces is None:
        return face_data

    i = 0
    for face in faces:
        (startX, startY, endX, endY) = face['bbox'].astype("int")
        face_temp = source_image[startY:endY, startX:endX]
        if face_temp.size < 1:
            continue
        i += 1
        face_data.append([face, face_temp])
    return face_data




def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK_SWAPPER:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)

def face_offset_top(face: Face, offset):
    smallestmin = np.min(face.landmark_2d_106, 1)
    smallest = smallestmin[1]
    face['bbox'][1] += offset
    face['bbox'][3] += offset
    lm106 = face.landmark_2d_106
    add = np.full_like(lm106, [0, offset])
    face['landmark_2d_106'] = lm106 + add
    return face
