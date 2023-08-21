#!/usr/bin/env python3

import os
import sys
import shutil
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

import warnings
from typing import List
import platform
import signal
import argparse
import torch
import onnxruntime
#import tensorflow

from time import time

import roop.globals
import roop.metadata
import roop.utilities as util
import roop.ui as ui
from settings import Settings
from roop.face_util import extract_face_images
from chain_img_processor import ChainImgProcessor, ChainVideoProcessor, ChainBatchImageProcessor, ChainVideoImageProcessor

clip_text = None

call_display_ui = None



if 'ROCMExecutionProvider' in roop.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-s', '--source', help='select a source image', dest='source_path')
    program.add_argument('-t', '--target', help='select a target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('-f', '--folder', help='select a target folder with images or videos to batch process', dest='target_folder_path')
    program.add_argument('--frame-processor', help='frame processors (choices: face_swapper, face_enhancer, ...)', dest='frame_processor', default=['face_swapper'], nargs='+')
    program.add_argument('--keep-fps', help='keep target fps', dest='keep_fps', action='store_true')
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true')
    program.add_argument('--skip-audio', help='skip target audio', dest='skip_audio', action='store_true')
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true')
    program.add_argument('--source-face_index', help='index position of source face in image', dest='source_face_index', type=int, default=0)
    program.add_argument('--target-face_index', help='index position of target face in image', dest='target_face_index', type=int, default=0)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, ...)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    args = program.parse_args()

    roop.globals.source_path = args.source_path
    roop.globals.target_path = args.target_path
    roop.globals.output_path = util.normalize_output_path(roop.globals.source_path, roop.globals.target_path, args.output_path)
    roop.globals.target_folder_path = args.target_folder_path
    roop.globals.headless = args.source_path or args.target_path or args.output_path
    # Always enable all processors when using GUI
    if not roop.globals.headless:
        roop.globals.frame_processors = ['face_swapper', 'face_enhancer']
    else:
        roop.globals.frame_processors = args.frame_processor

    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_frames = args.keep_frames
    roop.globals.skip_audio = args.skip_audio
    roop.globals.many_faces = args.many_faces
    roop.globals.source_face_index = args.source_face_index
    roop.globals.target_face_index = args.target_face_index
    roop.globals.video_encoder = args.video_encoder
    roop.globals.video_quality = args.video_quality
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = decode_execution_providers(args.execution_provider)
    roop.globals.execution_threads = args.execution_threads


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]

def set_execution_provider(execution_provider: str):
    longname = decode_execution_providers([execution_provider])[0]
    allproviders = onnxruntime.get_available_providers()
    unsupported = ['TensorrtExecutionProvider']
    filtered = filter(lambda i: i not in unsupported, allproviders)
    allproviders = list(filtered)
    if allproviders[0] != longname:
        allproviders.remove(longname)
        allproviders.insert(0, longname)
    roop.globals.execution_providers = allproviders

def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in roop.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in roop.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # prevent tensorflow memory leak
    # gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
    #         tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
    #     ])
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))



def release_resources() -> None:
    import gc

    gc.collect()
    if 'CUDAExecutionProvider' in roop.globals.execution_providers and torch.cuda.is_available():
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    
    download_directory_path = util.resolve_relative_path('../models')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'])
    # util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.pth'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx'])
    util.conditional_download(download_directory_path, ['https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth'])
    download_directory_path = util.resolve_relative_path('../models/CLIP')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth'])
    download_directory_path = util.resolve_relative_path('../models/CodeFormer')
    util.conditional_download(download_directory_path, ['https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'])
    download_directory_path = util.resolve_relative_path('../models/CodeFormer/facelib')
    util.conditional_download(download_directory_path, ['https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth'])
    util.conditional_download(download_directory_path, ['https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth'])
    download_directory_path = util.resolve_relative_path('../models/CodeFormer/realesrgan')
    util.conditional_download(download_directory_path, ['https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth'])

    if not shutil.which('ffmpeg'):
       update_status('ffmpeg is not installed.')
    return True

def set_display_ui(function):
    global call_display_ui

    call_display_ui = function


def update_status(message: str) -> None:
    global call_display_ui

    print(message)
    if call_display_ui is not None:
        call_display_ui(message)




def start() -> None:
    if roop.globals.headless:
        faces = extract_face_images(roop.globals.source_path,  (False, 0))
        roop.globals.INPUT_FACES.append(faces[roop.globals.source_face_index])
        faces = extract_face_images(roop.globals.target_path,  (False, util.has_image_extension(roop.globals.target_path)))
        roop.globals.TARGET_FACES.append(faces[roop.globals.target_face_index])
        if 'face_enhancer' in roop.globals.frame_processors:
            roop.globals.selected_enhancer = 'GFPGAN'
       
    batch_process(None, False, None)


def InitPlugins():
    if not roop.globals.IMAGE_CHAIN_PROCESSOR:
        roop.globals.IMAGE_CHAIN_PROCESSOR = ChainImgProcessor()
        roop.globals.BATCH_IMAGE_CHAIN_PROCESSOR = ChainBatchImageProcessor()
        # roop.globals.VIDEO_CHAIN_PROCESSOR = ChainVideoProcessor()
        roop.globals.VIDEO_CHAIN_PROCESSOR = ChainVideoImageProcessor()
        roop.globals.IMAGE_CHAIN_PROCESSOR.init_with_plugins()
        roop.globals.BATCH_IMAGE_CHAIN_PROCESSOR.init_with_plugins()
        roop.globals.VIDEO_CHAIN_PROCESSOR.init_with_plugins()


def get_processing_plugins(use_clip):
    processors = "faceswap"
    if use_clip:
        processors += ",txt2clip"
    
    if roop.globals.selected_enhancer == 'GFPGAN':
        processors += ",gfpgan"
    elif roop.globals.selected_enhancer == 'Codeformer':
        processors += ",codeformer"
    elif roop.globals.selected_enhancer == 'DMDNet':
        processors += ",dmdnet"
    
    return processors


def live_swap(frame, swap_mode, use_clip, clip_text, selected_index = 0):
    if frame is None:
        return frame

    InitPlugins()
    processors = get_processing_plugins(use_clip)


    temp_frame, _ = roop.globals.IMAGE_CHAIN_PROCESSOR.run_chain(frame,  
                                                    {"swap_mode": swap_mode,
                                                        "original_frame": frame,
                                                        "blend_ratio": roop.globals.blend_ratio,
                                                        "selected_index": selected_index,
                                                        "face_distance_threshold": roop.globals.distance_threshold,
                                                        "input_face_datas": roop.globals.INPUT_FACES, "target_face_datas": roop.globals.TARGET_FACES,
                                                        "clip_prompt": clip_text},
                                                        processors)
    return temp_frame
    
def preview_mask(frame, clip_text):
    import numpy as np
    
    maskimage = np.zeros((frame.shape), np.uint8)
    processors = "txt2clip"
    
    InitPlugins()

    temp_frame, _ = roop.globals.IMAGE_CHAIN_PROCESSOR.run_chain(maskimage,  
                                                    {"original_frame": frame, "clip_prompt": clip_text}, processors)
    return temp_frame




def params_gen_func(proc, frame):
    global clip_text

    return {"original_frame": frame, "blend_ratio": roop.globals.blend_ratio,
             "swap_mode": roop.globals.face_swap_mode, "face_distance_threshold": roop.globals.distance_threshold, 
             "input_face_datas": roop.globals.INPUT_FACES, "target_face_datas": roop.globals.TARGET_FACES,
             "clip_prompt": clip_text}

def batch_process(files, use_clip, new_clip_text, use_new_method) -> None:
    global clip_text

    roop.globals.processing = True
    InitPlugins()
    processors = get_processing_plugins(use_clip)
    release_resources()
    limit_resources()

    clip_text = new_clip_text

    imagefiles = []
    imagefinalnames = []
    videofiles = []
    videofinalnames = []
    need_join = False

    if files is None:
        need_join = True
        if roop.globals.target_folder_path is None:
            roop.globals.target_folder_path = os.path.dirname(roop.globals.target_path)
            files = [os.path.basename(roop.globals.target_path)]
            roop.globals.output_path = os.path.dirname(roop.globals.output_path)
        else:
            files = [f for f in os.listdir(roop.globals.target_folder_path) if os.path.isfile(os.path.join(roop.globals.target_folder_path, f))]
            
        update_status('Sorting videos/images')


    for f in files:
        if need_join:
            fullname = os.path.join(roop.globals.target_folder_path, f)
        else:
            fullname = f
        if util.has_image_extension(fullname):
            imagefiles.append(fullname)
            imagefinalnames.append(util.get_destfilename_from_path(fullname, roop.globals.output_path, f'_fake.{roop.globals.CFG.output_image_format}'))
        elif util.is_video(fullname) or util.has_extension(fullname, ['gif']):
            videofiles.append(fullname)
            videofinalnames.append(util.get_destfilename_from_path(fullname, roop.globals.output_path, f'_fake.{roop.globals.CFG.output_video_format}'))


    if(len(imagefiles) > 0):
        update_status('Processing image(s)')
        roop.globals.BATCH_IMAGE_CHAIN_PROCESSOR.run_batch_chain(imagefiles, imagefinalnames, roop.globals.execution_threads, processors, params_gen_func)
    if(len(videofiles) > 0):
        for index,v in enumerate(videofiles):
            if not roop.globals.processing:
                end_processing('Processing stopped!')
                return

            start_processing = time()

            fps = util.detect_fps(v)
            update_status(f'Creating {os.path.basename(videofinalnames[index])} with {fps} FPS...')
            if roop.globals.keep_frames or not use_new_method:
                util.create_temp(v)
                update_status('Extracting frames...')
                util.extract_frames(v)
                if not roop.globals.processing:
                    end_processing('Processing stopped!')
                    return

                temp_frame_paths = util.get_temp_frame_paths(v)
                roop.globals.BATCH_IMAGE_CHAIN_PROCESSOR.run_batch_chain(temp_frame_paths, temp_frame_paths, roop.globals.execution_threads, processors, params_gen_func)
                if not roop.globals.processing:
                    end_processing('Processing stopped!')
                    return
                
                util.create_video(v, videofinalnames[index], fps)
                if not roop.globals.keep_frames:
                    util.delete_temp_frames(temp_frame_paths[0])
            else:
                roop.globals.VIDEO_CHAIN_PROCESSOR.run_batch_chain(v, videofinalnames[index], fps,
                                                                    roop.globals.execution_threads, roop.globals.CFG.frame_buffer_size,
                                                                      processors, params_gen_func)
                
            if not roop.globals.processing:
                end_processing('Processing stopped!')
                return
            
            if os.path.isfile(videofinalnames[index]):
                if util.has_extension(v, ['gif']):
                    gifname = util.get_destfilename_from_path(v, './output', '_fake.gif')
                    update_status('Creating final GIF')
                    util.create_gif_from_video(videofinalnames[index], gifname)
                elif not roop.globals.skip_audio:
                    finalname = util.get_destfilename_from_path(videofinalnames[index], roop.globals.output_path, f'_final.{roop.globals.CFG.output_video_format}')
                    util.restore_audio(videofinalnames[index], v, finalname)
                    if os.path.isfile(finalname):
                        os.remove(videofinalnames[index])
                update_status(f'\nProcessing {os.path.basename(videofinalnames[index])} took {time() - start_processing} secs')

            else:
                update_status(f'Failed processing {os.path.basename(videofinalnames[index])}!')
            release_resources()
    end_processing('Finished')


def end_processing(msg:str):
    update_status(msg)
    roop.globals.target_folder_path = None
    release_resources()


def destroy() -> None:
    if roop.globals.target_path:
        util.clean_temp(roop.globals.target_path)
    release_resources()        
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    roop.globals.CFG = Settings('config.yaml')
    if roop.globals.headless:
        start()
    else:
        ui.run()
