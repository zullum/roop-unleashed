from typing import Any, List, Callable
import cv2
import threading

import roop.globals
import roop.processors.frame.core
import torch

from enhancer.DMDNet import DMDNet

from roop.core import update_status
from roop.face_analyser import get_one_face
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video, get_destfilename_from_path
from PIL import Image
from numpy import asarray
from enhancer.GFPGAN import enhance_GFPGAN
from enhancer.Codeformer import enhance_Codeformer
from enhancer.DMDNet import enhance_DMDNet

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'



def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/henryruhs/roop/resolve/main/GFPGANv1.4.pth'])
    conditional_download(download_directory_path, ['https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth'])
    conditional_download(download_directory_path, ['https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    FACE_ENHANCER = None


def enhance_face(temp_frame: Frame) -> Frame:

    with THREAD_SEMAPHORE:
        temp_frame_original = Image.fromarray(temp_frame)
        with THREAD_LOCK:
            if roop.globals.selected_enhancer == "DMDNet":
                return enhance_DMDNet(temp_frame)
            elif roop.globals.selected_enhancer == "Codeformer":
                temp_frame = enhance_Codeformer(temp_frame)
            elif roop.globals.selected_enhancer == "GFPGAN":
                temp_frame = enhance_GFPGAN(temp_frame)  
            else:
                return temp_frame

        temp_frame = Image.blend(temp_frame_original, Image.fromarray(temp_frame), 0.5)
    return asarray(temp_frame)

def process_frame(source_path: str, temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame)
    return temp_frame


def process_frames(is_batch: bool, source_face: Face, target_face: Face, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is not None:
            result = process_frame(None, temp_frame)
        if result is not None:
            if is_batch:
                tf = get_destfilename_from_path(temp_frame_path, roop.globals.output_path, '_fake.png')
                cv2.imwrite(tf, result)
            else:
                cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_face: Face, target_face: Face, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    if target_frame is not None:
        result = process_frame(None, target_frame)
    if result is not None:
        cv2.imwrite(output_path, result)


def process_video(source_face: Any, target_face: Any, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(source_face, target_face, temp_frame_paths, process_frames)


def process_batch_images(source_face: Any, target_face: Any, temp_frame_paths: List[str]) -> None:
    global DIST_THRESHOLD

    DIST_THRESHOLD = 0.85
    roop.processors.frame.core.process_video(source_face, target_face, temp_frame_paths, process_frames)
