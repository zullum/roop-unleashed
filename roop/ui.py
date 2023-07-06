import os
import customtkinter as ctk
import webbrowser
import cv2
import roop.globals
import roop.metadata

from typing import Callable, Tuple
from PIL import Image, ImageOps
from roop.face_analyser import get_many_faces, get_one_face, extract_face_images
from roop.capturer import get_video_frame, get_video_frame_total
#from roop.predicter import predict_frame
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import is_image, is_video, resolve_relative_path, open_with_default_app, compute_cosine_distance

ROOT = None
ROOT_HEIGHT = 550
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
IMAGE_BUTTON_WIDTH = 200
IMAGE_BUTTON_HEIGHT = 200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None
FACE_BUTTONS = []
INPUT_FACES_DATA = None
OUTPUT_FACES_DATA = None


def init(start: Callable, destroy: Callable) -> ctk.CTk:
    global ROOT, PREVIEW, FACE_SELECT

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)
    FACE_SELECT = create_select_faces_win(ROOT)
    return ROOT



def create_root(start: Callable, destroy: Callable) -> ctk.CTk:
    global source_button, target_button, status_label

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))
    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(f'{roop.metadata.name} {roop.metadata.version}')
    root.configure()
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())

    base_x1 = 0.075
    base_x2 = 0.575
    base_y = 0.635
    
    source_button = ctk.CTkButton(root, text='Select image with face(s)', width=IMAGE_BUTTON_WIDTH, height=IMAGE_BUTTON_HEIGHT, compound='top', anchor='center', command=lambda: select_source_path())
    source_button.place(relx=base_x1, rely=0.05)

    target_button = ctk.CTkButton(root, text='Select target image/video', width=IMAGE_BUTTON_WIDTH, height=IMAGE_BUTTON_HEIGHT, compound='top', anchor='center', command=lambda: select_target_path())
    target_button.place(relx=base_x2, rely=0.05)

    enhance_label = ctk.CTkLabel(root, text='Select face enhancement engine', anchor='w')
    enhance_label.place(relx=base_x1, rely=0.49)
    enhance_label.configure(text_color=ctk.ThemeManager.theme.get('RoopDonate').get('text_color'))
    
    enhancer_cb = ctk.CTkComboBox(root, values=["None", "Codeformer", "DMDNet", "GFPGAN"], width=IMAGE_BUTTON_WIDTH, command=select_enhancer)
    enhancer_cb.set("None")
    enhancer_cb.place(relx=base_x1, rely=0.532)
    
    keep_fps_value = ctk.BooleanVar(value=roop.globals.keep_fps)
    keep_fps_checkbox = ctk.CTkSwitch(root, text='Keep fps', variable=keep_fps_value, command=lambda: setattr(roop.globals, 'keep_fps', not roop.globals.keep_fps))
    keep_fps_checkbox.place(relx=base_x1, rely=base_y)

    keep_frames_value = ctk.BooleanVar(value=roop.globals.keep_frames)
    keep_frames_switch = ctk.CTkSwitch(root, text='Keep frames', variable=keep_frames_value, command=lambda: setattr(roop.globals, 'keep_frames', keep_frames_value.get()))
    keep_frames_switch.place(relx=base_x1, rely=0.67)

    keep_audio_value = ctk.BooleanVar(value=roop.globals.keep_audio)
    keep_audio_switch = ctk.CTkSwitch(root, text='Keep audio', variable=keep_audio_value, command=lambda: setattr(roop.globals, 'keep_audio', keep_audio_value.get()))
    keep_audio_switch.place(relx=base_x2, rely=base_y)

    many_faces_value = ctk.BooleanVar(value=roop.globals.many_faces)
    many_faces_switch = ctk.CTkSwitch(root, text='Many faces', variable=many_faces_value, command=lambda: setattr(roop.globals, 'many_faces', many_faces_value.get()))
    many_faces_switch.place(relx=base_x2, rely=0.67)

    base_y = 0.8
  
    start_button = ctk.CTkButton(root, text='Start', command=lambda: select_output_path(start))
    start_button.place(relx=base_x1, rely=base_y, relwidth=0.15, relheight=0.05)

    stop_button = ctk.CTkButton(root, text='Destroy', command=lambda: destroy())
    stop_button.place(relx=0.35, rely=base_y, relwidth=0.15, relheight=0.05)

    preview_button = ctk.CTkButton(root, text='Preview', command=lambda: toggle_preview())
    preview_button.place(relx=0.55, rely=base_y, relwidth=0.15, relheight=0.05)

    result_button = ctk.CTkButton(root, text='Show Result', command=lambda: show_result())
    result_button.place(relx=0.75, rely=base_y, relwidth=0.15, relheight=0.05)

    status_label = ctk.CTkLabel(root, text=None, justify='center')
    status_label.place(relx=base_x1, rely=0.9, relwidth=0.8)

    donate_label = ctk.CTkLabel(root, text='Visit the Github Page', justify='center', cursor='hand2')
    donate_label.place(relx=0.1, rely=0.95, relwidth=0.8)
    donate_label.configure(text_color=ctk.ThemeManager.theme.get('RoopDonate').get('text_color'))
    donate_label.bind('<Button>', lambda event: webbrowser.open('https://github.com/C0untFloyd/roop-unleashed'))

    return root

def create_preview(parent) -> ctk.CTkToplevel:
    global preview_label, preview_slider

def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title('Preview')
    preview.configure()
    preview.protocol('WM_DELETE_WINDOW', lambda: toggle_preview())
    preview.resizable(width=False, height=False)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill='both', expand=True)

    preview_slider = ctk.CTkSlider(preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value))

    return preview

def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()

def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, INPUT_FACES_DATA

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(title='Select source image', initialdir=RECENT_DIRECTORY_SOURCE)
    image = None
    if is_image(source_path):
        roop.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(roop.globals.source_path)
        INPUT_FACES_DATA = extract_face_images(roop.globals.source_path,  (False, 0))
        if len(INPUT_FACES_DATA) > 0:
            if len(INPUT_FACES_DATA) == 1:
                image = render_face_from_frame(INPUT_FACES_DATA[0][1], (IMAGE_BUTTON_WIDTH, IMAGE_BUTTON_HEIGHT))
                roop.globals.SELECTED_FACE_DATA_INPUT = INPUT_FACES_DATA[0][0]
            else:
                show_face_selection(INPUT_FACES_DATA, True)
        else:
            roop.globals.source_path = None
    else:
        roop.globals.source_path = None
    source_button.configure(image=image)
    source_button._draw()


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET, OUTPUT_FACES_DATA

    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(title='Select target image or video', initialdir=RECENT_DIRECTORY_TARGET)
    image = None
    if is_image(target_path) and not target_path.lower().endswith(('gif')):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(roop.globals.target_path)
        if roop.globals.many_faces:
            roop.globals.SELECTED_FACE_DATA_OUTPUT = None
            image = render_image_preview(target_path, (IMAGE_BUTTON_WIDTH, IMAGE_BUTTON_HEIGHT))
        else:
            OUTPUT_FACES_DATA = extract_face_images(roop.globals.target_path, (False, 0))
            if len(OUTPUT_FACES_DATA) > 0:
                if len(OUTPUT_FACES_DATA) == 1:
                    image = render_face_from_frame(OUTPUT_FACES_DATA[0][1], (IMAGE_BUTTON_WIDTH, IMAGE_BUTTON_HEIGHT))
                    roop.globals.SELECTED_FACE_DATA_OUTPUT = OUTPUT_FACES_DATA[0][0]
                    if roop.globals.SELECTED_FACE_DATA_INPUT is not None:
                        emb1 = roop.globals.SELECTED_FACE_DATA_INPUT['embedding']
                        emb2 = roop.globals.SELECTED_FACE_DATA_OUTPUT['embedding']
                        dist = compute_cosine_distance(emb1, emb2)
                        print(f'Similarity Distance between Source->Target={dist}')
                else:
                    show_face_selection(OUTPUT_FACES_DATA, False)
            else:
                roop.globals.target_path = None

    elif is_video(target_path) or target_path.lower().endswith(('gif')):
        roop.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(roop.globals.target_path)
        if roop.globals.many_faces:
            roop.globals.SELECTED_FACE_DATA_OUTPUT = None
            image = render_video_preview(target_path, (IMAGE_BUTTON_WIDTH, IMAGE_BUTTON_HEIGHT))
        else:
            max_frame = get_video_frame_total(roop.globals.target_path)
            dialog = ctk.CTkInputDialog(text=f"Please input frame number with target face (1 - {max_frame})", title="Extract Face from Video")
            selected_frame = dialog.get_input()
            try:
                selected_frame = int(selected_frame)
            except:
                selected_frame = 1
            
            selected_frame = max(selected_frame, 1)
            selected_frame = min(selected_frame, max_frame)
            OUTPUT_FACES_DATA = extract_face_images(roop.globals.target_path, (True, selected_frame))
            if len(OUTPUT_FACES_DATA) > 0:
                if len(OUTPUT_FACES_DATA) == 1:
                    image = render_face_from_frame(OUTPUT_FACES_DATA[0][1], (IMAGE_BUTTON_WIDTH, IMAGE_BUTTON_HEIGHT))
                    roop.globals.SELECTED_FACE_DATA_OUTPUT = OUTPUT_FACES_DATA[0][0]
                else:
                    show_face_selection(OUTPUT_FACES_DATA, False)
            else:
                roop.globals.target_path = None
        
    else:
        roop.globals.target_path = None

    target_button.configure(image=image)
    target_button._draw()

def select_output_path(start):
    global RECENT_DIRECTORY_OUTPUT


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT

    if is_image(roop.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='Save image output file', defaultextension='.png', initialfile='output.png', initialdir=RECENT_DIRECTORY_OUTPUT)
    elif is_video(roop.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(title='Save video output file', defaultextension='.mp4', initialfile='output.mp4', initialdir=RECENT_DIRECTORY_OUTPUT)
    else:
        output_path = None
    if output_path:
        roop.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(roop.globals.output_path)
        start()


def select_enhancer(choice):
    roop.globals.selected_enhancer = choice


def show_result():
    open_with_default_app(roop.globals.output_path)
    


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_face_from_frame(face, size: Tuple[int, int] = None) -> ctk.CTkImage:
    image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    image = ImageOps.fit(image, (IMAGE_BUTTON_WIDTH, IMAGE_BUTTON_HEIGHT), Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(video_path: str, size: Tuple[int, int], frame_number: int = 0) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == 'normal':
        PREVIEW.withdraw()
    elif roop.globals.source_path and roop.globals.target_path:
        init_preview()
        update_preview()
        PREVIEW.deiconify()


def init_preview() -> None:
    if is_image(roop.globals.target_path):
        preview_slider.pack_forget()
    if is_video(roop.globals.target_path):
        video_frame_total = get_video_frame_total(roop.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill='x')
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if roop.globals.source_path and roop.globals.target_path:
        temp_frame = get_video_frame(roop.globals.target_path, frame_number)

        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            if frame_processor.NAME == 'ROOP.FACE-ENHANCER':
                continue

            temp_frame = frame_processor.process_frame(source_face=roop.globals.SELECTED_FACE_DATA_INPUT, target_face=roop.globals.SELECTED_FACE_DATA_OUTPUT, temp_frame=temp_frame)
            image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
            image = ImageOps.contain(image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS)
            image = ctk.CTkImage(image, size=image.size)
            preview_label.configure(image=image)


def create_select_faces_win(parent) -> ctk.CTkToplevel:
    global scrollable_frame

    face_win = ctk.CTkToplevel(parent)
    face_win.minsize(800, 400)
    face_win.title('Select Face(s)')
    face_win.configure()
    face_win.withdraw()
    face_win.protocol('WM_DELETE_WINDOW', lambda: cancel_face_selection())
    scrollable_frame = ctk.CTkScrollableFrame(face_win, orientation='horizontal', label_text='Choose face by clicking on it', width=(IMAGE_BUTTON_WIDTH + 40)*3, height=IMAGE_BUTTON_HEIGHT+32)
    scrollable_frame.grid(row=0, column=0, padx=20, pady=20)
    scrollable_frame.place(relx=0.05, rely=0.05)
    cancel_button = ctk.CTkButton(face_win, text='Cancel', command=lambda: cancel_face_selection())
    cancel_button.place(relx=0.05, rely=0.85, relwidth=0.075, relheight=0.075)

    return face_win

def cancel_face_selection() -> None:
    toggle_face_selection();
    ROOT.wm_attributes('-disabled', False)
    ROOT.focus()

def select_face(index, is_input) -> None:
    global source_button, target_button, INPUT_FACES_DATA, OUTPUT_FACES_DATA

    if is_input:
        roop.globals.SELECTED_FACE_DATA_INPUT = INPUT_FACES_DATA[index][0]
        image = render_face_from_frame(INPUT_FACES_DATA[index][1], (IMAGE_BUTTON_WIDTH, IMAGE_BUTTON_HEIGHT))
        source_button.configure(image=image)
        source_button._draw()
    else:
        roop.globals.SELECTED_FACE_DATA_OUTPUT = OUTPUT_FACES_DATA[index][0]
        image = render_face_from_frame(OUTPUT_FACES_DATA[index][1], (IMAGE_BUTTON_WIDTH, IMAGE_BUTTON_HEIGHT))
        target_button.configure(image=image)
        target_button._draw()
        if roop.globals.SELECTED_FACE_DATA_INPUT is not None:
            emb1 = roop.globals.SELECTED_FACE_DATA_INPUT['embedding']
            emb2 = roop.globals.SELECTED_FACE_DATA_OUTPUT['embedding']
            dist = compute_cosine_distance(emb1, emb2)
            print(f'Similarity Distance between Source->Target={dist}')

    toggle_face_selection();
    ROOT.wm_attributes('-disabled', False)
    ROOT.focus()



def toggle_face_selection() -> None:
    if FACE_SELECT.state() == 'normal':
        FACE_SELECT.withdraw()
    else:
        FACE_SELECT.deiconify()


def show_face_selection(faces, is_input):
    global FACE_BUTTONS, scrollable_frame

    ROOT.wm_attributes('-disabled', True)

    if len(FACE_BUTTONS) > 0:
        for b in FACE_BUTTONS:
            try:
                # b.place_forget()
                b.destroy()
            except:
                continue
        FACE_BUTTONS.clear()

    i = 0
    for face in faces:
        image = render_face_from_frame(face[1], (128, 128))
        score = face[0]['det_score']
        age = face[0]['age']
        button_text = f'Score: {score} - Sex: {face[0].sex} - Age: {age}'
        face_button = ctk.CTkButton(scrollable_frame, text=button_text, width=128, height=128, compound='top', anchor='center', command=lambda faceindex=i: select_face(index=faceindex, is_input=is_input))
        face_button.grid(row=0, column=i, pady=5, padx=5)
        face_button.configure(image=image)
        face_button._draw()
        FACE_BUTTONS.append(face_button)
        i += 1

    FACE_SELECT.deiconify()

