"""
Modified version from codeformer-pip project

S-Lab License 1.0

Copyright 2022 S-Lab

https://github.com/kadirnar/codeformer-pip/blob/main/LICENSE
"""

import os

import cv2
import torch
from codeformer.facelib.detection import init_detection_model
from codeformer.facelib.parsing import init_parsing_model
from torchvision.transforms.functional import normalize

from codeformer.basicsr.archs.rrdbnet_arch import RRDBNet
from codeformer.basicsr.utils import img2tensor, imwrite, tensor2img
from codeformer.basicsr.utils.download_util import load_file_from_url
from codeformer.basicsr.utils.realesrgan_utils import RealESRGANer
from codeformer.basicsr.utils.registry import ARCH_REGISTRY
from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from codeformer.facelib.utils.misc import is_gray
import threading

from plugins.codeformer_face_helper_cv2 import FaceRestoreHelperOptimized

THREAD_LOCK_FACE_HELPER = threading.Lock()
THREAD_LOCK_FACE_HELPER_CREATE = threading.Lock()
THREAD_LOCK_FACE_HELPER_PROCERSSING = threading.Lock()
THREAD_LOCK_CODEFORMER_NET = threading.Lock()
THREAD_LOCK_CODEFORMER_NET_CREATE = threading.Lock()
THREAD_LOCK_BGUPSAMPLER = threading.Lock()

pretrain_model_url = {
    "codeformer": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    "detection": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
    "parsing": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
    "realesrgan": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
}

# download weights
if not os.path.exists("models/CodeFormer/codeformer.pth"):
    load_file_from_url(
        url=pretrain_model_url["codeformer"], model_dir="models/CodeFormer/", progress=True, file_name=None
    )
if not os.path.exists("models/CodeFormer/facelib/detection_Resnet50_Final.pth"):
    load_file_from_url(
        url=pretrain_model_url["detection"], model_dir="models/CodeFormer/facelib", progress=True, file_name=None
    )
if not os.path.exists("models/CodeFormer/facelib/parsing_parsenet.pth"):
    load_file_from_url(
        url=pretrain_model_url["parsing"], model_dir="models/CodeFormer/facelib", progress=True, file_name=None
    )
if not os.path.exists("models/CodeFormer/realesrgan/RealESRGAN_x2plus.pth"):
    load_file_from_url(
        url=pretrain_model_url["realesrgan"], model_dir="models/CodeFormer/realesrgan", progress=True, file_name=None
    )


def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# set enhancer with RealESRGAN
def set_realesrgan():
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="models/CodeFormer/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler


upsampler = set_realesrgan()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

codeformers_cache = []

def get_codeformer():
    if len(codeformers_cache) > 0:
        with THREAD_LOCK_CODEFORMER_NET:
            if len(codeformers_cache) > 0:
                return codeformers_cache.pop()

    with THREAD_LOCK_CODEFORMER_NET_CREATE:
        codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)
        ckpt_path = "models/CodeFormer/codeformer.pth"
        checkpoint = torch.load(ckpt_path)["params_ema"]
        codeformer_net.load_state_dict(checkpoint)
        codeformer_net.eval()
        return codeformer_net



def release_codeformer(codeformer):
    with THREAD_LOCK_CODEFORMER_NET:
        codeformers_cache.append(codeformer)

#os.makedirs("output", exist_ok=True)

# ------- face restore thread cache ----------

face_restore_helper_cache = []

detection_model = "retinaface_resnet50"

inited_face_restore_helper_nn = False

import time

def get_face_restore_helper(upscale):
    global inited_face_restore_helper_nn
    with THREAD_LOCK_FACE_HELPER:
        face_helper = FaceRestoreHelperOptimized(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
            device=device,
        )
        #return face_helper

        if inited_face_restore_helper_nn:
            while len(face_restore_helper_cache) == 0:
                time.sleep(0.05)
            face_detector, face_parse = face_restore_helper_cache.pop()
            face_helper.face_detector = face_detector
            face_helper.face_parse = face_parse
            return face_helper
        else:
            inited_face_restore_helper_nn = True
            face_helper.face_detector = init_detection_model(detection_model, half=False, device=face_helper.device)
            face_helper.face_parse = init_parsing_model(model_name="parsenet", device=face_helper.device)
            return face_helper

def get_face_restore_helper2(upscale): # still not work well!!!
    face_helper = FaceRestoreHelperOptimized(
        upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=detection_model,
        save_ext="png",
        use_parse=True,
        device=device,
    )
    #return face_helper

    if len(face_restore_helper_cache) > 0:
        with THREAD_LOCK_FACE_HELPER:
            if len(face_restore_helper_cache) > 0:
                face_detector, face_parse = face_restore_helper_cache.pop()
                face_helper.face_detector = face_detector
                face_helper.face_parse = face_parse
                return face_helper

    with THREAD_LOCK_FACE_HELPER_CREATE:
        face_helper.face_detector = init_detection_model(detection_model, half=False, device=face_helper.device)
        face_helper.face_parse = init_parsing_model(model_name="parsenet", device=face_helper.device)
        return face_helper

def release_face_restore_helper(face_helper):
    #return
    #with THREAD_LOCK_FACE_HELPER:
    face_restore_helper_cache.append((face_helper.face_detector, face_helper.face_parse))
    #pass

def inference_app(image, background_enhance, face_upsample, upscale, codeformer_fidelity, skip_if_no_face = False):
    # take the default setting for the demo
    has_aligned = False
    only_center_face = False
    draw_box = False

    #print("Inp:", image, background_enhance, face_upsample, upscale, codeformer_fidelity)
    if isinstance(image, str):
        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
    else:
        img = image
    #print("\timage size:", img.shape)

    upscale = int(upscale)  # convert type to int
    if upscale > 4:  # avoid memory exceeded due to too large upscale
        upscale = 4
    if upscale > 2 and max(img.shape[:2]) > 1000:  # avoid memory exceeded due to too large img resolution
        upscale = 2
    if max(img.shape[:2]) > 1500:  # avoid memory exceeded due to too large img resolution
        upscale = 1
        background_enhance = False
        #face_upsample = False

    face_helper = get_face_restore_helper(upscale)

    bg_upsampler = upsampler if background_enhance else None
    face_upsampler = upsampler if face_upsample else None

    if has_aligned:
        # the input faces are already cropped and aligned
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.is_gray = is_gray(img, threshold=5)
        if face_helper.is_gray:
            print("\tgrayscale input: True")
        face_helper.cropped_faces = [img]
    else:
        with THREAD_LOCK_FACE_HELPER_PROCERSSING:
            face_helper.read_image(img)
            # get face landmarks for each face

            num_det_faces = face_helper.get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
        #print(f"\tdetect {num_det_faces} faces")

        if num_det_faces == 0 and skip_if_no_face:
            release_face_restore_helper(face_helper)
            return img

        # align and warp each face
        face_helper.align_warp_face()



    # face restoration for each cropped face
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        codeformer_net = get_codeformer()
        try:
            with torch.no_grad():
                output = codeformer_net(cropped_face_t, w=codeformer_fidelity, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
        except RuntimeError as error:
            print(f"Failed inference for CodeFormer: {error}")
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
        release_codeformer(codeformer_net)

        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face)

    # paste_back
    if not has_aligned:
        # upsample the background
        if bg_upsampler is not None:
            with THREAD_LOCK_BGUPSAMPLER:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
        else:
            bg_img = None
        face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        if face_upsample and face_upsampler is not None:
            restored_img = face_helper.paste_faces_to_input_image(
                upsample_img=bg_img,
                draw_box=draw_box,
                face_upsampler=face_upsampler,
            )
        else:
            restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=draw_box)

    if image.shape != restored_img.shape:
        h, w, _ = image.shape
        restored_img = cv2.resize(restored_img, (w, h), interpolation=cv2.INTER_LINEAR)


    release_face_restore_helper(face_helper)
    # save restored img
    if isinstance(image, str):
        save_path = f"output/out.png"
        imwrite(restored_img, str(save_path))
        return save_path
    else:
        return restored_img
