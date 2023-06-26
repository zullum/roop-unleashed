import threading
from typing import Any
import insightface

import roop.globals
from roop.typing import Frame
import cv2
from PIL import Image
from roop.capturer import get_video_frame

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER


def get_one_face(frame: Frame) -> Any:
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(frame: Frame) -> Any:
    try:
        faces = get_face_analyser().get(frame)
        return sorted(faces, key = lambda x : x.bbox[0])
    except IndexError:
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

        
    faces = get_many_faces(source_image)

    i = 0
    for face in faces:
        (startX, startY, endX, endY) = face['bbox'].astype("int")
        face_temp = source_image[startY:endY, startX:endX]
        if face_temp.size < 1:
            continue
        i += 1
        face_data.append([face, face_temp])
    return face_data