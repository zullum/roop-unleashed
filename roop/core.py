#!/usr/bin/env python3

import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
#import tensorflow

import roop.globals
import roop.metadata
import roop.ui as ui
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path, has_extension
from roop.face_analyser import extract_face_images

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
    if roop.globals.headless:
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
    #if not shutil.which('ffmpeg'):
    #    update_status('ffmpeg is not installed.')
    #    return False
    return True


def update_status(message: str, scope: str = 'ROOP.CORE') -> None:
    print(f'[{scope}] {message}')
    if not roop.globals.headless:
        ui.update_status(message)



def start() -> None:
    if roop.globals.headless:
        faces = extract_face_images(roop.globals.source_path,  (False, 0))
        roop.globals.SELECTED_FACE_DATA_INPUT = faces[roop.globals.source_face_index]
        faces = extract_face_images(roop.globals.target_path,  (False, has_image_extension(roop.globals.target_path)))
        roop.globals.SELECTED_FACE_DATA_OUTPUT = faces[roop.globals.target_face_index]
        if 'face_enhancer' in roop.globals.frame_processors:
            roop.globals.selected_enhancer = 'GFPGAN'

    if roop.globals.target_folder_path is not None:
        batch_process()

    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_start():
            return

    current_target = roop.globals.target_path

    # process image to image
    if has_image_extension(current_target):
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            target = current_target
            if frame_processor.NAME == 'ROOP.FACE-ENHANCER':
                if roop.globals.selected_enhancer == None or roop.globals.selected_enhancer == 'None':
                    continue
                target = roop.globals.output_path

            update_status(f'{frame_processor.NAME} in progress...')
            frame_processor.process_image(roop.globals.SELECTED_FACE_DATA_INPUT, roop.globals.SELECTED_FACE_DATA_OUTPUT, target, roop.globals.output_path)
            frame_processor.post_process()
            release_resources()
        if is_image(current_target):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return

    update_status('Creating temp resources...')
    create_temp(current_target)
    update_status('Extracting frames...')
    extract_frames(current_target)
    temp_frame_paths = get_temp_frame_paths(current_target)

    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if frame_processor.NAME == 'ROOP.FACE-ENHANCER' and roop.globals.selected_enhancer == 'None':
            continue

        update_status(f'{frame_processor.NAME} in progress...')
        frame_processor.process_video(roop.globals.SELECTED_FACE_DATA_INPUT, roop.globals.SELECTED_FACE_DATA_OUTPUT, temp_frame_paths)
        frame_processor.post_process()
        release_resources()
    # handles fps
    if roop.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Creating video with {fps} fps...')
        create_video(current_target, fps)
    else:
        update_status('Creating video with 30.0 fps...')
        create_video(current_target)
    # handle audio
    if roop.globals.skip_audio or has_extension(current_target, ['gif']):
        move_temp(current_target, roop.globals.output_path)
        update_status('Skipping audio...')
    else:
        if roop.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(current_target, roop.globals.output_path)
    # clean and validate
    clean_temp(current_target)
    if is_video(roop.globals.output_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')


def batch_process() -> None:
    files = [f for f in os.listdir(roop.globals.target_folder_path) if os.path.isfile(os.path.join(roop.globals.target_folder_path, f))]
    update_status('Sorting videos/images')

    imagefiles = []
    videofiles = []

    for f in files:
        if has_image_extension(os.path.join(roop.globals.target_folder_path, f)):
            imagefiles.append(os.path.join(roop.globals.target_folder_path, f))
        elif is_video(os.path.join(roop.globals.target_folder_path, f)):
            videofiles.append(os.path.join(roop.globals.target_folder_path, f))

    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if frame_processor.NAME == 'ROOP.FACE-ENHANCER' and roop.globals.selected_enhancer == 'None':
            continue

        update_status(f'{frame_processor.NAME} in progress...')
        frame_processor.process_batch_images(roop.globals.SELECTED_FACE_DATA_INPUT, roop.globals.SELECTED_FACE_DATA_OUTPUT, imagefiles)

    if len(videofiles) > 0:
        for video in videofiles:
            update_status(f'Processing {video}')
            update_status('Creating temp resources...')
            create_temp(video)
            update_status('Extracting frames...')
            extract_frames(video)
            temp_frame_paths = get_temp_frame_paths(video)
            for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
                if frame_processor.NAME == 'ROOP.FACE-ENHANCER' and roop.globals.selected_enhancer == 'None':
                    continue

                update_status(f'{frame_processor.NAME} in progress...')
                frame_processor.process_video(roop.globals.SELECTED_FACE_DATA_INPUT, roop.globals.SELECTED_FACE_DATA_OUTPUT, temp_frame_paths)
                frame_processor.post_process()
                release_resources()
            # handles fps
            if roop.globals.keep_fps:
                update_status('Detecting fps...')
                fps = detect_fps(video)
                update_status(f'Creating video with {fps} fps...')
                create_video(video, fps)
            else:
                update_status('Creating video with 30.0 fps...')
                create_video(video)
            # handle audio
            if roop.globals.skip_audio:
                move_temp(video, roop.globals.output_path)
                update_status('Skipping audio...')
            else:
                if roop.globals.keep_fps:
                    update_status('Restoring audio...')
                else:
                    update_status('Restoring audio might cause issues as fps are not kept...')
                restore_audio(video, roop.globals.output_path)
            clean_temp(video)






def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    sys.exit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if roop.globals.headless:
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()
