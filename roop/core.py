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

import roop.globals
import roop.metadata
import roop.ui as ui
from settings import Settings
from roop.utilities import has_image_extension, is_video, detect_fps, create_video, extract_frames, create_gif_from_video, get_temp_frame_paths, restore_audio, create_temp, clean_temp, normalize_output_path, has_extension, get_destfilename_from_path, resolve_relative_path, conditional_download
from roop.face_helper import extract_face_images
from chain_img_processor import ChainImgProcessor, ChainVideoProcessor, ChainBatchImageProcessor

clip_text = None


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
    roop.globals.output_path = normalize_output_path(roop.globals.source_path, roop.globals.target_path, args.output_path)
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
        # tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            # tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        # ])
    # limit memory usage
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'])
    conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.pth'])
    # conditional_download(download_directory_path, ['https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth'])
    download_directory_path = resolve_relative_path('../models/CLIP')
    conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth'])
    download_directory_path = resolve_relative_path('../models/CodeFormer')
    conditional_download(download_directory_path, ['https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'])
    download_directory_path = resolve_relative_path('../models/CodeFormer/facelib')
    conditional_download(download_directory_path, ['https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth'])
    conditional_download(download_directory_path, ['https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth'])
    download_directory_path = resolve_relative_path('../models/CodeFormer/realesrgan')
    conditional_download(download_directory_path, ['https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth'])

    if not shutil.which('ffmpeg'):
       update_status('ffmpeg is not installed.')
    return True


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')
    # if not roop.globals.headless:
        # ui.update_status(message)



def start() -> None:
    if roop.globals.headless:
        faces = extract_face_images(roop.globals.source_path,  (False, 0))
        roop.globals.SELECTED_FACE_DATA_INPUT = faces[roop.globals.source_face_index]
        faces = extract_face_images(roop.globals.target_path,  (False, has_image_extension(roop.globals.target_path)))
        roop.globals.SELECTED_FACE_DATA_OUTPUT = faces[roop.globals.target_face_index]
        if 'face_enhancer' in roop.globals.frame_processors:
            roop.globals.selected_enhancer = 'GFPGAN'
       
    batch_process(None, False, None)


def InitPlugins():
    if not roop.globals.IMAGE_CHAIN_PROCESSOR:
        roop.globals.IMAGE_CHAIN_PROCESSOR = ChainImgProcessor()
        roop.globals.BATCH_IMAGE_CHAIN_PROCESSOR = ChainBatchImageProcessor()
        roop.globals.VIDEO_CHAIN_PROCESSOR = ChainVideoProcessor()
        roop.globals.IMAGE_CHAIN_PROCESSOR.init_with_plugins()
        roop.globals.BATCH_IMAGE_CHAIN_PROCESSOR.init_with_plugins()
        roop.globals.VIDEO_CHAIN_PROCESSOR.init_with_plugins()


def live_swap(frame, swap_mode, use_clip, clip_text):
    if frame is None:
        return frame

    InitPlugins()

    processors = "faceswap"
    if use_clip:
        processors += ",txt2clip"
    if roop.globals.selected_enhancer == 'GFPGAN':
        processors += ",gfpgan"
    elif roop.globals.selected_enhancer == 'Codeformer':
        processors += ",codeformer"

    temp_frame, _ = roop.globals.IMAGE_CHAIN_PROCESSOR.run_chain(frame,  
                                                    {"swap_mode": swap_mode,
                                                        "original_frame": frame,
                                                        "face_distance_threshold": roop.globals.distance_threshold,
                                                        "input_face_datas": [roop.globals.SELECTED_FACE_DATA_INPUT], "target_face_datas": [roop.globals.SELECTED_FACE_DATA_OUTPUT],
                                                        "clip_prompt": clip_text},
                                                        processors)
    return temp_frame



def params_gen_func(proc, frame):
    global clip_text

    return {"original_frame": frame, "blend_ratio": roop.globals.blend_ratio,
             "swap_mode": roop.globals.face_swap_mode, "face_distance_threshold": roop.globals.distance_threshold, 
             "input_face_datas": [roop.globals.SELECTED_FACE_DATA_INPUT], "target_face_datas": [roop.globals.SELECTED_FACE_DATA_OUTPUT],
             "clip_prompt": clip_text}

def batch_process(files, use_clip, new_clip_text) -> None:
    global clip_text

    InitPlugins()
    
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
        if has_image_extension(fullname):
            imagefiles.append(fullname)
            imagefinalnames.append(get_destfilename_from_path(fullname, roop.globals.output_path, f'_fake.{roop.globals.CFG.output_image_format}'))
        elif is_video(fullname) or has_extension(fullname, ['gif']):
            videofiles.append(fullname)
            videofinalnames.append(get_destfilename_from_path(fullname, roop.globals.output_path, f'_fake.{roop.globals.CFG.output_video_format}'))

    processors = "faceswap"
    if use_clip:
        processors += ",txt2clip"
    if roop.globals.selected_enhancer == 'GFPGAN':
        processors += ",gfpgan"
    elif roop.globals.selected_enhancer == 'Codeformer':
        processors += ",codeformer"

    if(len(imagefiles) > 0):
        update_status('Processing image(s)')
        roop.globals.BATCH_IMAGE_CHAIN_PROCESSOR.run_batch_chain(imagefiles, imagefinalnames, 8, processors, params_gen_func)
    if(len(videofiles) > 0):
        for index,v in enumerate(videofiles):
            update_status(f'Processing video {v}')
            fps = detect_fps(v)
            if roop.globals.keep_frames:
                update_status('Creating temp resources...')
                create_temp(v)
                update_status('Extracting frames...')
                extract_frames(v)
                temp_frame_paths = get_temp_frame_paths(v)
                roop.globals.BATCH_IMAGE_CHAIN_PROCESSOR.run_batch_chain(temp_frame_paths, temp_frame_paths, 8, processors, params_gen_func)
                update_status(f'Creating video with {fps} FPS...')
                create_video(v, videofinalnames[index], fps)
            else:
                update_status(f'Creating video with {fps} FPS...')
                roop.globals.VIDEO_CHAIN_PROCESSOR.run_video_chain(v,videofinalnames[index], fps, 8, processors, params_gen_func, roop.globals.target_path)
            if os.path.isfile(videofinalnames[index]):
                if has_extension(v, ['gif']):
                    gifname = get_destfilename_from_path(v, './output', '_fake.gif')
                    update_status('Creating final GIF')
                    create_gif_from_video(videofinalnames[index], gifname)
                elif not roop.globals.skip_audio:
                    finalname = get_destfilename_from_path(videofinalnames[index], roop.globals.output_path, f'_final.{roop.globals.CFG.output_video_format}')
                    restore_audio(videofinalnames[index], v, finalname)
                    if os.path.isfile(videofinalnames[index]):
                        os.remove(videofinalnames[index])
            else:
                update_status('Failed!')

            
    update_status('Finished')
    roop.globals.target_folder_path = None


def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    limit_resources()
    roop.globals.CFG = Settings('config.yaml')
    if roop.globals.headless:
        start()
    else:
        ui.run()
