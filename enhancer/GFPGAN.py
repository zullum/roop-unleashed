import gfpgan
import roop.globals

from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

myself = None


def create():
    global myself

    model_path = resolve_relative_path('../models/GFPGANv1.4.pth')
    myself = gfpgan.GFPGANer(model_path=model_path, upscale=1) # type: ignore[attr-defined]


def enhance_GFPGAN(temp_frame):
    global myself

    if myself == None:
        create()

    _, _, temp_frame = myself.enhance(
            temp_frame,
            paste_back=True
        )
    return temp_frame


