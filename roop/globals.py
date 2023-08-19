from settings import Settings
from typing import List

source_path = None
target_path = None
output_path = None
target_folder_path = None

frame_processors: List[str] = []
keep_fps = None
keep_frames = None
skip_audio = None
many_faces = None
use_batch = None
source_face_index = 0
target_face_index = 0
face_position = None
video_encoder = None
video_quality = None
max_memory = None
execution_providers: List[str] = []
execution_threads = None
headless = None
log_level = 'error'
selected_enhancer = None
face_swap_mode = None
blend_ratio = 0.5
distance_threshold = 0.65
default_det_size = True

processing = False 

FACE_ENHANCER = None

INPUT_FACES = []
TARGET_FACES = []

IMAGE_CHAIN_PROCESSOR = None
VIDEO_CHAIN_PROCESSOR = None
BATCH_IMAGE_CHAIN_PROCESSOR = None

CFG: Settings = None


