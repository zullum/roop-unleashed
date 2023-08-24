import os
import time
import gradio as gr
import cv2
import pathlib
import shutil
import roop.globals
import roop.metadata
import roop.utilities as util

from roop.face_util import extract_face_images
from roop.capturer import get_video_frame, get_video_frame_total, get_image_frame

restart_server = False
live_cam_active = False

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

SELECTION_FACES_DATA = None

last_image = None

input_thumbs = []
target_thumbs = []


IS_INPUT = True
SELECTED_FACE_INDEX = 0

SELECTED_INPUT_FACE_INDEX = 0
SELECTED_TARGET_FACE_INDEX = 0

roop.globals.keep_fps = None
roop.globals.keep_frames = None
roop.globals.skip_audio = None
roop.globals.use_batch = None

input_faces = None
target_faces = None
face_selection = None
fake_cam_image = None

current_cam_image = None
cam_swapping = False

selected_preview_index = 0

is_processing = False            



def prepare_environment():
    roop.globals.output_path = os.path.abspath(os.path.join(os.getcwd(), "output"))
    os.makedirs(roop.globals.output_path, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = os.path.abspath(os.path.join(os.getcwd(), "temp"))
    os.makedirs(os.environ["TEMP"], exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = os.environ["TEMP"]


def run():
    from roop.core import suggest_execution_providers, decode_execution_providers, set_display_ui, set_execution_provider
    global input_faces, target_faces, face_selection, fake_cam_image, restart_server, live_cam_active, on_settings_changed

    prepare_environment()

    available_themes = ["Default", "gradio/glass", "gradio/monochrome", "gradio/seafoam", "gradio/soft", "gstaff/xkcd", "freddyaboulton/dracula_revamped", "ysharma/steampunk"]
    image_formats = ['jpg','png', 'webp']
    video_formats = ['avi','mkv', 'mp4', 'webm']
    video_codecs = ['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc']
    providerlist = suggest_execution_providers()
    
    settings_controls = []

    live_cam_active = roop.globals.CFG.live_cam_start_active
    set_display_ui(show_msg)
    roop.globals.execution_providers = decode_execution_providers([roop.globals.CFG.provider])
    print(f'Using provider {roop.globals.execution_providers} - Device:{util.get_device()}')    
    #set_execution_provider(roop.globals.CFG.provider)
    #print(f'Available providers {roop.globals.execution_providers}, using {roop.globals.execution_providers[0]} - Device:{util.get_device()}')
    
    run_server = True
    mycss = """
        span {color: var(--block-info-text-color)}
        #filelist {
            max-height: 238.4px;
            overflow-y: auto !important;
        }
"""

    while run_server:
        server_name = roop.globals.CFG.server_name
        if server_name is None or len(server_name) < 1:
            server_name = None
        server_port = roop.globals.CFG.server_port
        if server_port <= 0:
            server_port = None
        ssl_verify = False if server_name == '0.0.0.0' else True
        with gr.Blocks(title=f'{roop.metadata.name} {roop.metadata.version}', theme=roop.globals.CFG.selected_theme, css=mycss) as ui:
            with gr.Row(variant='panel'):
                    gr.Markdown(f"### [{roop.metadata.name} {roop.metadata.version}](https://github.com/C0untFloyd/roop-unleashed)")
                    gr.HTML(util.create_version_html(), elem_id="versions")
            with gr.Tab("Face Swap"):
                with gr.Row():
                    with gr.Column():
                        input_faces = gr.Gallery(label="Input faces", allow_preview=True, preview=True, height=128, object_fit="scale-down")
                        with gr.Row():
                                bt_remove_selected_input_face = gr.Button("Remove selected")
                                bt_clear_input_faces = gr.Button("Clear all", variant='stop')
                        bt_srcimg = gr.Image(label='Source Face Image', type='filepath', tool=None)
                    with gr.Column():
                        target_faces = gr.Gallery(label="Target faces", allow_preview=True, preview=True, height=128, object_fit="scale-down")
                        with gr.Row():
                                bt_remove_selected_target_face = gr.Button("Remove selected")
                        bt_destfiles = gr.Files(label='Target File(s)', file_count="multiple", elem_id='filelist')
                with gr.Row():
                    with gr.Column(visible=False) as dynamic_face_selection:
                        face_selection = gr.Gallery(label="Detected faces", allow_preview=True, preview=True, height=256, object_fit="scale-down")
                        with gr.Row():
                            bt_faceselect = gr.Button("Use selected face")
                            bt_cancelfaceselect = gr.Button("Done")
            
                with gr.Row():
                    with gr.Column(scale=1):
                        selected_face_detection = gr.Dropdown(["First found", "All faces", "Selected face", "All female", "All male"], value="First found", label="Select face selection for swapping")
                        max_face_distance = gr.Slider(0.01, 1.0, value=0.65, label="Max Face Similarity Threshold")
                    with gr.Column(scale=1):
                        video_swapping_method = gr.Dropdown(["Extract Frames to media","In-Memory processing"], value="In-Memory", label="Select video processing method", interactive=True)
                        chk_det_size = gr.Checkbox(label="Use default Det-Size", value=True, elem_id='default_det_size', interactive=True)
                    with gr.Column(scale=2):
                        roop.globals.keep_fps = gr.Checkbox(label="Keep FPS", value=True)
                        roop.globals.keep_frames = gr.Checkbox(label="Keep Frames (relevant only when extracting frames)", value=False)
                        roop.globals.skip_audio = gr.Checkbox(label="Skip audio", value=False)
                with gr.Row():
                    with gr.Column():
                        selected_enhancer = gr.Dropdown(["None", "Codeformer", "DMDNet", "GFPGAN"], value="None", label="Select post-processing")
                    with gr.Column():
                        blend_ratio = gr.Slider(0.0, 1.0, value=0.65, label="Original/Enhanced image blend ratio")
                with gr.Row():
                    with gr.Accordion(label="Masking", open=False):
                        with gr.Row():
                            with gr.Column(scale=1):
                                chk_useclip = gr.Checkbox(label="Use Text Masking", value=False)
                                clip_text = gr.Textbox(label="List of objects to mask and restore back on fake image", placeholder="cup,hands,hair,banana")
                                gr.Dropdown(["Clip2Seg"], value="Clip2Seg", label="Engine")
                            with gr.Column(scale=1):
                                bt_preview_mask = gr.Button("Show Mask Preview", variant='secondary')
                            with gr.Column(scale=2):
                                maskpreview = gr.Image(label="Preview Mask", shape=(None,512), interactive=False)
                            
                with gr.Row(variant='panel'):
                    with gr.Column():
                        bt_start = gr.Button("Start", variant='primary')
                    with gr.Column():
                        bt_stop = gr.Button("Stop", variant='secondary')
                    with gr.Column():
                        fake_preview = gr.Checkbox(label="Face swap frames", value=False)
                    with gr.Column():
                        bt_refresh_preview = gr.Button("Refresh", variant='secondary')
                with gr.Row(variant='panel'):
                    with gr.Column():
                        with gr.Accordion(label="Results", open=True):
                            gr.Button("Open Output Folder", size='sm').click(fn=lambda: util.open_folder(util.resolve_relative_path('../output/')))
                            resultfiles = gr.Files(label='Processed File(s)', interactive=False)
                            resultimage = gr.Image(type='filepath', interactive=False)
                    with gr.Column():
                        with gr.Accordion(label="Preview Original/Fake Frame", open=True):
                            previewimage = gr.Image(label="Preview Image", interactive=False)
                            with gr.Row(variant='panel'):
                                with gr.Column():
                                    preview_frame_num = gr.Slider(0, 0, value=0, label="Frame Number", step=1.0, interactive=True)
                                with gr.Column():
                                    bt_use_face_from_preview = gr.Button("Use Face from this Frame", variant='primary')
                        
            with gr.Tab("Live Cam"):
                cam_toggle = gr.Checkbox(label='Activate', value=live_cam_active)
                if live_cam_active:
                    with gr.Row():
                        with gr.Column():
                            cam = gr.Webcam(label='Camera', source='webcam', mirror_webcam=True, interactive=True, streaming=False)
                        with gr.Column():
                            fake_cam_image = gr.Image(label='Fake Camera Output', interactive=False)


            with gr.Tab("Extras"):
                with gr.Row():
                    files_to_process = gr.Files(label='File(s) to process', file_count="multiple")
                # with gr.Row(variant='panel'):
                #     with gr.Accordion(label="Post process", open=False):
                #         with gr.Column():
                #             selected_post_enhancer = gr.Dropdown(["None", "Codeformer", "GFPGAN"], value="None", label="Select post-processing")
                #         with gr.Column():
                #             gr.Button("Start").click(fn=lambda: gr.Info('Not yet implemented...'))
                with gr.Row(variant='panel'):
                    with gr.Accordion(label="Video/GIF", open=False):
                        with gr.Row(variant='panel'):
                            with gr.Column():
                                gr.Markdown("""
                                            # Cut video
                                            Be aware that this means re-encoding the video which might take a longer time.
                                            Encoding uses your configuration from the Settings Tab.
    """)
                            with gr.Column():
                                cut_start_time = gr.Slider(0, 1000000, value=0, label="Start Frame", step=1.0, interactive=True)
                            with gr.Column():
                                cut_end_time = gr.Slider(1, 1000000, value=1, label="End Frame", step=1.0, interactive=True)
                            with gr.Column():
                                start_cut_video = gr.Button("Start")

    #                     with gr.Row(variant='panel'):
    #                         with gr.Column():
    #                             gr.Markdown("""
    #                                         # Join videos
    #                                         This also re-encodes the videos like cutting above.
    # """)
    #                         with gr.Column():
    #                             start_join_videos = gr.Button("Start")
                        with gr.Row(variant='panel'):
                            gr.Markdown("Extract frames from video")
                            start_extract_frames = gr.Button("Start")
                        with gr.Row(variant='panel'):
                            gr.Markdown("Create video from image files")
                            gr.Button("Start").click(fn=lambda: gr.Info('Not yet implemented...'))
                        with gr.Row(variant='panel'):
                            gr.Markdown("Create GIF from video")
                            start_create_gif = gr.Button("Create GIF")
                with gr.Row():
                    extra_files_output = gr.Files(label='Resulting output files', file_count="multiple")
                        
            
            with gr.Tab("Settings"):
                with gr.Row():
                    with gr.Column():
                        themes = gr.Dropdown(available_themes, label="Theme", info="Change needs complete restart", value=roop.globals.CFG.selected_theme)
                    with gr.Column():
                        settings_controls.append(gr.Checkbox(label="Public Server", value=roop.globals.CFG.server_share, elem_id='server_share', interactive=True))
                        settings_controls.append(gr.Checkbox(label='Clear output folder before each run', value=roop.globals.CFG.clear_output, elem_id='clear_output', interactive=True))
                    with gr.Column():
                        input_server_name = gr.Textbox(label="Server Name", lines=1, info="Leave blank to run locally", value=roop.globals.CFG.server_name)
                    with gr.Column():
                        input_server_port = gr.Number(label="Server Port", precision=0, info="Leave at 0 to use default", value=roop.globals.CFG.server_port)
                with gr.Row():
                    with gr.Column():
                        settings_controls.append(gr.Dropdown(providerlist, label="Provider", value=roop.globals.CFG.provider, elem_id='provider', interactive=True))
                        settings_controls.append(gr.Checkbox(label="Force CPU for Face Analyser", value=roop.globals.CFG.force_cpu, elem_id='force_cpu', interactive=True))
                        max_threads = gr.Slider(1, 64, value=roop.globals.CFG.max_threads, label="Max. Number of Threads", info='default: 8', step=1.0, interactive=True)
                    with gr.Column():
                        memory_limit = gr.Slider(0, 128, value=roop.globals.CFG.memory_limit, label="Max. Memory to use (Gb)", info='0 meaning no limit', step=1.0, interactive=True)
                        frame_buffer_size = gr.Slider(1, 512, value=roop.globals.CFG.frame_buffer_size, label="Frame Buffer Size", info='Num. Images to preload for each thread', step=1.0, interactive=True)
                        settings_controls.append(gr.Dropdown(image_formats, label="Image Output Format", info='default: png', value=roop.globals.CFG.output_image_format, elem_id='output_image_format', interactive=True))
                    with gr.Column():
                        settings_controls.append(gr.Dropdown(video_codecs, label="Video Codec", info='default: libx264', value=roop.globals.CFG.output_video_codec, elem_id='output_video_codec', interactive=True))
                        settings_controls.append(gr.Dropdown(video_formats, label="Video Output Format", info='default: mp4', value=roop.globals.CFG.output_video_format, elem_id='output_video_format', interactive=True))
                        video_quality = gr.Slider(0, 100, value=roop.globals.CFG.video_quality, label="Video Quality (crf)", info='default: 14', step=1.0, interactive=True)
                    with gr.Column():
                        button_apply_restart = gr.Button("Restart Server", variant='primary')
                        settings_controls.append(gr.Checkbox(label='Start with active live cam', value=roop.globals.CFG.live_cam_start_active, elem_id='live_cam_start_active', interactive=True))
                        button_clean_temp = gr.Button("Clean temp folder")
                        button_apply_settings = gr.Button("Apply Settings")

            previewinputs = [preview_frame_num, bt_destfiles, fake_preview, selected_enhancer, selected_face_detection,
                                max_face_distance, blend_ratio, chk_useclip, clip_text] 
            input_faces.select(on_select_input_face, None, None).then(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage])
            bt_remove_selected_input_face.click(fn=remove_selected_input_face, outputs=[input_faces])
            bt_srcimg.change(fn=on_srcimg_changed, show_progress='full', inputs=bt_srcimg, outputs=[dynamic_face_selection, face_selection, input_faces])


            target_faces.select(on_select_target_face, None, None)
            bt_remove_selected_target_face.click(fn=remove_selected_target_face, outputs=[target_faces])

            previewinputs = [preview_frame_num, bt_destfiles, fake_preview, selected_enhancer, selected_face_detection,
                                max_face_distance, blend_ratio, chk_useclip, clip_text] 

            bt_destfiles.change(fn=on_destfiles_changed, inputs=[bt_destfiles], outputs=[preview_frame_num]).then(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage])
            bt_destfiles.select(fn=on_destfiles_selected, inputs=[bt_destfiles], outputs=[preview_frame_num]).then(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage])
            bt_destfiles.clear(fn=on_clear_destfiles, outputs=[target_faces])
            resultfiles.select(fn=on_resultfiles_selected, inputs=[resultfiles], outputs=[resultimage])

            face_selection.select(on_select_face, None, None)
            bt_faceselect.click(fn=on_selected_face, outputs=[input_faces, target_faces, selected_face_detection])
            bt_cancelfaceselect.click(fn=on_end_face_selection, outputs=[dynamic_face_selection, face_selection])
            
            bt_clear_input_faces.click(fn=on_clear_input_faces, outputs=[input_faces])

            chk_det_size.select(fn=on_option_changed)

            bt_preview_mask.click(fn=on_preview_mask, inputs=[preview_frame_num, bt_destfiles, clip_text], outputs=[maskpreview]) 

            start_event = bt_start.click(fn=start_swap, 
                inputs=[selected_enhancer, selected_face_detection, roop.globals.keep_fps, roop.globals.keep_frames,
                         roop.globals.skip_audio, max_face_distance, blend_ratio, bt_destfiles, chk_useclip, clip_text,video_swapping_method],
                outputs=[bt_start, resultfiles, resultimage])
            
            bt_stop.click(fn=stop_swap, cancels=[start_event])
            
            bt_refresh_preview.click(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage])            
            fake_preview.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage])
            preview_frame_num.change(fn=on_preview_frame_changed, inputs=previewinputs, outputs=[previewimage], show_progress='hidden')
            bt_use_face_from_preview.click(fn=on_use_face_from_selected, show_progress='full', inputs=[bt_destfiles, preview_frame_num], outputs=[dynamic_face_selection, face_selection, target_faces, selected_face_detection])

            
            # Live Cam
            cam_toggle.change(fn=on_cam_toggle, inputs=[cam_toggle])
            if live_cam_active:
                cam.stream(on_stream_swap_cam, inputs=[cam, selected_enhancer, blend_ratio], outputs=[fake_cam_image], preprocess=True, postprocess=True, show_progress="hidden")

            # Extras
            start_cut_video.click(fn=on_cut_video, inputs=[files_to_process, cut_start_time, cut_end_time], outputs=[extra_files_output])
            # start_join_videos.click(fn=on_join_videos, inputs=[files_to_process], outputs=[extra_files_output])
            start_extract_frames.click(fn=on_extract_frames, inputs=[files_to_process], outputs=[extra_files_output])
            start_create_gif.click(fn=on_create_gif, inputs=[files_to_process], outputs=[extra_files_output])

            # Settings
            for s in settings_controls:
                s.select(fn=on_settings_changed)
            max_threads.input(fn=lambda a,b='max_threads':on_settings_changed_misc(a,b), inputs=[max_threads])
            memory_limit.input(fn=lambda a,b='memory_limit':on_settings_changed_misc(a,b), inputs=[memory_limit])
            frame_buffer_size.input(fn=lambda a,b='frame_buffer_size':on_settings_changed_misc(a,b), inputs=[frame_buffer_size])
            video_quality.input(fn=lambda a,b='video_quality':on_settings_changed_misc(a,b), inputs=[video_quality])

            button_clean_temp.click(fn=clean_temp, outputs=[bt_srcimg, input_faces, target_faces, bt_destfiles])
            button_apply_settings.click(apply_settings, inputs=[themes, input_server_name, input_server_port])
            button_apply_restart.click(restart)



        restart_server = False
        try:
            ui.queue().launch(inbrowser=True, server_name=server_name, server_port=server_port, share=roop.globals.CFG.server_share, ssl_verify=ssl_verify, prevent_thread_lock=False, show_error=True)
        except:
            restart_server = True
            run_server = False
        try:
            while restart_server == False:
                time.sleep(5.0)
        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            run_server = False
        ui.close()



def on_option_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(roop.globals, attribname):
            setattr(roop.globals, attribname, evt.selected)
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(roop.globals, attribname):
            setattr(roop.globals, attribname, evt.value)
            return
    raise gr.Error(f'Unhandled Setting for {evt.target}')


def on_settings_changed_misc(new_val, attribname):
    if hasattr(roop.globals.CFG, attribname):
        setattr(roop.globals.CFG, attribname, new_val)
    else:
        print("Didn't find attrib!")
        


def on_settings_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(roop.globals.CFG, attribname):
            setattr(roop.globals.CFG, attribname, evt.selected)
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(roop.globals.CFG, attribname):
            setattr(roop.globals.CFG, attribname, evt.value)
            return
            
    raise gr.Error(f'Unhandled Setting for {evt.target}')



def on_srcimg_changed(imgsrc, progress=gr.Progress()):
    global RECENT_DIRECTORY_SOURCE, SELECTION_FACES_DATA, IS_INPUT, input_faces, face_selection, input_thumbs, last_image
    
    IS_INPUT = True

    if imgsrc == None or last_image == imgsrc:
        return gr.Column.update(visible=False), None, input_thumbs
    
    last_image = imgsrc
    
    progress(0, desc="Retrieving faces from image", )      
    source_path = imgsrc
    thumbs = []
    if util.is_image(source_path):
        roop.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(roop.globals.source_path)
        SELECTION_FACES_DATA = extract_face_images(roop.globals.source_path,  (False, 0))
        progress(0.5, desc="Retrieving faces from image")      
        for f in SELECTION_FACES_DATA:
            image = convert_to_gradio(f[1])
            thumbs.append(image)
            
    progress(1.0, desc="Retrieving faces from image")      
    if len(thumbs) < 1:
        raise gr.Error('No faces detected!')

    if len(thumbs) == 1:
        roop.globals.INPUT_FACES.append(SELECTION_FACES_DATA[0][0])
        input_thumbs.append(thumbs[0])
        return gr.Column.update(visible=False), None, input_thumbs
       
    return gr.Column.update(visible=True), thumbs, gr.Gallery.update(visible=True)

def on_select_input_face(evt: gr.SelectData):
    global SELECTED_INPUT_FACE_INDEX

    SELECTED_INPUT_FACE_INDEX = evt.index

def remove_selected_input_face():
    global input_thumbs, SELECTED_INPUT_FACE_INDEX

    if len(roop.globals.INPUT_FACES) > SELECTED_INPUT_FACE_INDEX:
        f = roop.globals.INPUT_FACES.pop(SELECTED_INPUT_FACE_INDEX)
        del f
    if len(input_thumbs) > SELECTED_INPUT_FACE_INDEX:
        f = input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)
        del f

    return input_thumbs

def on_select_target_face(evt: gr.SelectData):
    global SELECTED_TARGET_FACE_INDEX

    SELECTED_TARGET_FACE_INDEX = evt.index

def remove_selected_target_face():
    global target_thumbs, SELECTED_TARGET_FACE_INDEX

    if len(roop.globals.TARGET_FACES) > SELECTED_TARGET_FACE_INDEX:
        f = roop.globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    if len(target_thumbs) > SELECTED_TARGET_FACE_INDEX:
        f = target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)
        del f
    return target_thumbs





def on_use_face_from_selected(files, frame_num):
    global IS_INPUT, SELECTION_FACES_DATA

    IS_INPUT = False
    thumbs = []
    
    roop.globals.target_path = files[selected_preview_index].name
    if util.is_image(roop.globals.target_path) and not roop.globals.target_path.lower().endswith(('gif')):
        SELECTION_FACES_DATA = extract_face_images(roop.globals.target_path,  (False, 0))
        if len(SELECTION_FACES_DATA) > 0:
            for f in SELECTION_FACES_DATA:
                image = convert_to_gradio(f[1])
                thumbs.append(image)
        else:
            gr.Info('No faces detected!')
            roop.globals.target_path = None
                
    elif util.is_video(roop.globals.target_path) or roop.globals.target_path.lower().endswith(('gif')):
        selected_frame = frame_num
        SELECTION_FACES_DATA = extract_face_images(roop.globals.target_path, (True, selected_frame))
        if len(SELECTION_FACES_DATA) > 0:
            for f in SELECTION_FACES_DATA:
                image = convert_to_gradio(f[1])
                thumbs.append(image)
        else:
            gr.Info('No faces detected!')
            roop.globals.target_path = None

    if len(thumbs) == 1:
        roop.globals.TARGET_FACES.append(SELECTION_FACES_DATA[0][0])
        target_thumbs.append(thumbs[0])
        return gr.Row.update(visible=False), None, target_thumbs, gr.Dropdown.update(value='Selected face')

    return gr.Row.update(visible=True), thumbs, gr.Gallery.update(visible=True), gr.Dropdown.update(visible=True)



def on_select_face(evt: gr.SelectData):  # SelectData is a subclass of EventData
    global SELECTED_FACE_INDEX
    SELECTED_FACE_INDEX = evt.index
    

def on_selected_face():
    global IS_INPUT, SELECTED_FACE_INDEX, SELECTION_FACES_DATA, input_thumbs, target_thumbs
    
    fd = SELECTION_FACES_DATA[SELECTED_FACE_INDEX]
    image = convert_to_gradio(fd[1])
    if IS_INPUT:
        roop.globals.INPUT_FACES.append(fd[0])
        input_thumbs.append(image)
        return input_thumbs, gr.Gallery.update(visible=True), gr.Dropdown.update(visible=True)
    else:
        roop.globals.TARGET_FACES.append(fd[0])
        target_thumbs.append(image)
        return gr.Gallery.update(visible=True), target_thumbs, gr.Dropdown.update(value='Selected face')
    
#        bt_faceselect.click(fn=on_selected_face, outputs=[dynamic_face_selection, face_selection, input_faces, target_faces])

def on_end_face_selection():
    return gr.Column.update(visible=False), None


def on_preview_frame_changed(frame_num, files, fake_preview, enhancer, detection, face_distance, blend_ratio, use_clip, clip_text):
    global SELECTED_INPUT_FACE_INDEX, is_processing

    from roop.core import live_swap

    if is_processing or files is None or selected_preview_index >= len(files) or frame_num is None:
        return None

    filename = files[selected_preview_index].name
    if util.is_video(filename) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filename, frame_num)
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None:
        return None 

    if not fake_preview or len(roop.globals.INPUT_FACES) < 1:
        return convert_to_gradio(current_frame)

    roop.globals.face_swap_mode = translate_swap_mode(detection)
    roop.globals.selected_enhancer = enhancer
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio

    if use_clip and clip_text is None or len(clip_text) < 1:
        use_clip = False

    roop.globals.execution_threads = roop.globals.CFG.max_threads
    current_frame = live_swap(current_frame, roop.globals.face_swap_mode, use_clip, clip_text, SELECTED_INPUT_FACE_INDEX)
    if current_frame is None:
        return None 
    return convert_to_gradio(current_frame)


def on_preview_mask(frame_num, files, clip_text):
    from roop.core import preview_mask
    global is_processing

    if is_processing:
        return None
        
    filename = files[selected_preview_index].name
    if util.is_video(filename) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filename, frame_num)
    else:
        current_frame = get_image_frame(filename)
    if current_frame is None:
        return None

    current_frame = preview_mask(current_frame, clip_text)
    return convert_to_gradio(current_frame)


def on_clear_input_faces():
    global input_thumbs
    
    input_thumbs.clear()
    roop.globals.INPUT_FACES.clear()
    return input_thumbs

def on_clear_destfiles():
    global target_thumbs

    roop.globals.TARGET_FACES.clear()
    target_thumbs.clear()
    return target_thumbs    



def translate_swap_mode(dropdown_text):
    if dropdown_text == "Selected face":
        return "selected"
    elif dropdown_text == "First found":
        return "first"
    elif dropdown_text == "All female":
        return "all_female"
    elif dropdown_text == "All male":
        return "all_male"
    
    return "all"
        


def start_swap( enhancer, detection, keep_fps, keep_frames, skip_audio, face_distance, blend_ratio,
                target_files, use_clip, clip_text, processing_method, progress=gr.Progress(track_tqdm=True)):
    from roop.core import batch_process
    global is_processing

    if target_files is None or len(target_files) <= 0:
        return gr.Button.update(variant="primary"), None, None
    
    if roop.globals.CFG.clear_output:
        shutil.rmtree(roop.globals.output_path)


    prepare_environment()

    roop.globals.selected_enhancer = enhancer
    roop.globals.target_path = None
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.keep_fps = keep_fps
    roop.globals.keep_frames = keep_frames
    roop.globals.skip_audio = skip_audio
    roop.globals.face_swap_mode = translate_swap_mode(detection)
    if use_clip and clip_text is None or len(clip_text) < 1:
        use_clip = False
    
    if roop.globals.face_swap_mode == 'selected':
        if len(roop.globals.TARGET_FACES) < 1:
            gr.Error('No Target Face selected!')
            return gr.Button.update(variant="primary"),None, None

    is_processing = True            
    yield gr.Button.update(variant="secondary"), None, None
    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None

    batch_process([file.name for file in target_files], use_clip, clip_text, processing_method == "In-Memory")
    is_processing = False
    outdir = pathlib.Path(roop.globals.output_path)
    outfiles = [item for item in outdir.iterdir() if item.is_file()]
    if len(outfiles) > 0:
        yield gr.Button.update(variant="primary"),gr.Files.update(value=outfiles), gr.Image.update(value=outfiles[0])
    else:
        yield gr.Button.update(variant="primary"),None, None


def stop_swap():
    roop.globals.processing = False
    gr.Info('Aborting processing - please wait for the remaining threads to be stopped')

   
def on_destfiles_changed(destfiles):
    global selected_preview_index

    if destfiles is None or len(destfiles) < 1:
        return gr.Slider.update(value=0, maximum=0)
    
    selected_preview_index = 0
    filename = destfiles[selected_preview_index].name
    if util.is_video(filename) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filename)
    else:
        total_frames = 0
    
    return gr.Slider.update(value=0, maximum=total_frames)



def on_destfiles_selected(evt: gr.SelectData, target_files):
    global selected_preview_index

    if evt is not None:
        selected_preview_index = evt.index
    filename = target_files[selected_preview_index].name
    if util.is_video(filename) or filename.lower().endswith('gif'):
        total_frames = get_video_frame_total(filename)
    else:
        total_frames = 0

    return gr.Slider.update(value=0, maximum=total_frames)
    

def on_resultfiles_selected(evt: gr.SelectData, files):
    selected_index = evt.index
    filename = files[selected_index].name
    if util.is_video(filename) or filename.lower().endswith('gif'):
        current_frame = get_video_frame(filename, 0)
    else:
        current_frame = get_image_frame(filename)
    return convert_to_gradio(current_frame)

    
        
def on_cam_toggle(state):
    global live_cam_active, restart_server

    live_cam_active = state
    gr.Warning('Server will be restarted for this change!')
    restart_server = True


def on_stream_swap_cam(camimage, enhancer, blend_ratio):
    from roop.core import live_swap
    global current_cam_image, cam_counter, cam_swapping, fake_cam_image, SELECTED_INPUT_FACE_INDEX

    roop.globals.selected_enhancer = enhancer
    roop.globals.blend_ratio = blend_ratio

    if not cam_swapping and len(roop.globals.INPUT_FACES) > 0:
        cam_swapping = True
        current_cam_image = live_swap(camimage, "all", False, None, SELECTED_INPUT_FACE_INDEX)
        cam_swapping = False
    return current_cam_image


def on_cut_video(files, cut_start_frame, cut_end_frame):
    if files is None:
        return None
    
    resultfiles = []
    for tf in files:
        f = tf.name
        # destfile = get_destfilename_from_path(f, resolve_relative_path('./output'), '_cut')
        destfile = util.get_destfilename_from_path(f, './output', '_cut')
        util.cut_video(f, destfile, cut_start_frame, cut_end_frame)
        if os.path.isfile(destfile):
            resultfiles.append(destfile)
        else:
            gr.Error('Cutting video failed!')
    return resultfiles

def on_join_videos(files):
    if files is None:
        return None
    
    filenames = []
    for f in files:
        filenames.append(f.name)
    destfile = util.get_destfilename_from_path(filenames[0], './output', '_join')        
    util.join_videos(filenames, destfile)
    resultfiles = []
    if os.path.isfile(destfile):
        resultfiles.append(destfile)
    else:
        gr.Error('Joining videos failed!')
    return resultfiles




def on_extract_frames(files):
    if files is None:
        return None
    
    resultfiles = []
    for tf in files:
        f = tf.name
        resfolder = util.extract_frames(f)
        for file in os.listdir(resfolder):
            outfile = os.path.join(resfolder, file)
            if os.path.isfile(outfile):
                resultfiles.append(outfile)
    return resultfiles


def on_create_gif(files):
    if files is None:
        return None
    
    for tf in files:
        f = tf.name
        gifname = util.get_destfilename_from_path(f, './output', '.gif')
        util.create_gif_from_video(f, gifname)
    return gifname





def clean_temp():
    global input_thumbs, target_thumbs
    
    shutil.rmtree(os.environ["TEMP"])
    prepare_environment()
   
    input_thumbs.clear()
    roop.globals.INPUT_FACES.clear()
    roop.globals.TARGET_FACES.clear()
    target_thumbs = []
    gr.Info('Temp Files removed')
    return None,None,None,None


def apply_settings(themes, input_server_name, input_server_port):
    roop.globals.CFG.selected_theme = themes
    roop.globals.CFG.server_name = input_server_name
    roop.globals.CFG.server_port = input_server_port
    roop.globals.CFG.save()
    show_msg('Settings saved')

def restart():
    global restart_server
    restart_server = True


def show_msg(msg: str):
    gr.Info(msg)



# Gradio wants Images in RGB
def convert_to_gradio(image):
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
