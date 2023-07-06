import cv2
import torch
import threading
from tqdm import tqdm
from torchvision.transforms.functional import normalize
from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from codeformer.basicsr.utils.registry import ARCH_REGISTRY
from codeformer.basicsr.utils import img2tensor, tensor2img

from roop.utilities import resolve_relative_path, get_device

# if 'ROCMExecutionProvider' in roop.globals.execution_providers:
    # del torch

CODE_FORMER = None
FACE_HELPER = None


def create():
    global CODE_FORMER, FACE_HELPER
    
    model_path = resolve_relative_path('../models/codeformer.pth')
    model = torch.load(model_path)['params_ema']
    device = torch.device(get_device())
    CODE_FORMER = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=['32', '64', '128', '256'],
    ).to(device)
    CODE_FORMER.load_state_dict(model)
    CODE_FORMER.eval()
    
    FACE_HELPER = FaceRestoreHelper(
            upscale_factor = int(1),
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=get_device()
        )


def enhance_Codeformer(temp_frame):
    global CODE_FORMER

    if CODE_FORMER == None:
        create()

    FACE_HELPER.clean_all()

    try:
        FACE_HELPER.read_image(temp_frame)
        # get face landmarks for each face
        FACE_HELPER.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        # align and warp each face
        FACE_HELPER.align_warp_face()
        for idx, cropped_face in enumerate(FACE_HELPER.cropped_faces):
            face_t = data_preprocess(cropped_face)
            face_enhanced = restore_face(face_t)
            FACE_HELPER.add_restored_face(face_enhanced)

        FACE_HELPER.get_inverse_affine()
        enhanced_img = FACE_HELPER.paste_faces_to_input_image()
        return enhanced_img

    except RuntimeError as error:
        print(f"Failed inference for CodeFormer: {error}")


def data_preprocess(frame):
    frame_t = img2tensor(frame / 255.0, bgr2rgb=True, float32=True)
    normalize(frame_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    return frame_t.unsqueeze(0).to(get_device())


def generate_output(frame_t, codeformer_fidelity = 0.6):
    with torch.no_grad():
        output = CODE_FORMER(frame_t, w=codeformer_fidelity, adain=True)[0]
    return output


def restore_face(face_t):
    try:
        output = generate_output(face_t)
        restored_face = postprocess_output(output)
        del output
    except RuntimeError as error:
        print(f"Failed inference for CodeFormer: {error}")
        restored_face = postprocess_output(face_t)
    return restored_face


def postprocess_output(output):
    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
    return restored_face.astype("uint8")