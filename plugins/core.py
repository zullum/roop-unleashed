# Core plugin
# author: Vladislav Janvarev

from chain_img_processor import ChainImgProcessor

# start function
def start(core:ChainImgProcessor):
    manifest = {
        "name": "Core plugin",
        "version": "2.0",

        "default_options": {
            "default_chain": "faceswap", # default chain to run
            "init_on_start": "faceswap,txt2clip,gfpgan,codeformer", # init these processors on start
            "is_demo_row_render": False,
        },

    }
    return manifest

def start_with_options(core:ChainImgProcessor, manifest:dict):
    options = manifest["options"]

    core.default_chain = options["default_chain"]
    core.init_on_start = options["init_on_start"]

    core.is_demo_row_render= options["is_demo_row_render"]

    return manifest
