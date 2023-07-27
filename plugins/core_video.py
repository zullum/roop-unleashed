# Core plugin
# author: Vladislav Janvarev

from chain_img_processor import ChainImgProcessor, ChainVideoProcessor

# start function
def start(core:ChainImgProcessor):
    manifest = {
        "name": "Core video plugin",
        "version": "2.0",

        "default_options": {
            "video_save_codec": "libx264", # default codec to save
            "video_save_crf": 14, # default crf to save
        },

    }
    return manifest

def start_with_options(core:ChainVideoProcessor, manifest:dict):
    options = manifest["options"]

    core.video_save_codec = options["video_save_codec"]
    core.video_save_crf = options["video_save_crf"]

    return manifest
