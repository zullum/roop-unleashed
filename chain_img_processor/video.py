from typing import Any, List, Callable
from roop.typing import Frame
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from threading import Thread, Lock
from time import sleep, time
from queue import Queue
from .image import ChainImgProcessor
from tqdm import tqdm
import cv2
from chain_img_processor.ffmpeg_writer import FFMPEG_VideoWriter # ffmpeg install needed
import roop.globals


class ChainVideoImageProcessor(ChainImgProcessor):
    chain = None
    func_params_gen = None
    num_threads = 1
    current_index = 0
    processing_threads = 1
    reading_frames = False
    buffer_wait_time = 0.1
    loadbuffersize = 0 

    lock = Lock()

    frames_queue = None
    processed_queue = None

    last_index = -1

    def __init__(self):
        ChainImgProcessor.__init__(self)


    def init_with_plugins(self):
        self.init_plugins(["core"])
        self.display_init_info()

        init_on_start_arr = self.init_on_start.split(",")
        for proc_id in init_on_start_arr:
            self.init_processor(proc_id)


    def update_progress(self, progress: Any = None) -> None:
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
        progress.set_postfix({
            'memory_usage': '{:.2f}'.format(memory_usage).zfill(5) + 'GB',
            'execution_threads': self.num_threads
        })
        progress.update(1)
        progress.refresh()


    def read_frames_thread(self, cap, num_threads):
        self.reading_frames = True

        num_frame = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            num_frame += 1
            self.frames_queue.put((num_frame,frame))

        self.reading_frames = False

    def process_frames(self, progress) -> None:
        my_list = []
        while True:
            while len(my_list) > 0:
                index,frame = my_list[0]
                myframe = False

                with self.lock:
                    if self.last_index + 1 == index:
                        self.last_index = index
                        myframe = True
                
                if myframe:
                    self.processed_queue.put(my_list.pop(0))
                else:
                    break

            try:
                frametuple = self.frames_queue.get_nowait()
                index,frame = frametuple
                if self.func_params_gen:
                    params = self.func_params_gen(None, frame)
                else:
                    params = {}
                resimg, _ = self.run_chain(frame, params, self.chain)
                with self.lock:
                    if self.last_index + 1 == index:
                        self.processed_queue.put((index,resimg))
                        self.last_index = index
                    else:
                        my_list.append((index,resimg))
                progress()
            except:
                if not self.reading_frames and len(my_list) < 1:
                    self.processing_threads -= 1
                    return



    def write_frames_thread(self, target_video, width, height, fps, total):
        with FFMPEG_VideoWriter(target_video, (width, height), fps, codec=roop.globals.video_encoder, crf=roop.globals.video_quality, audiofile=None) as output_video_ff:
            lastindex = 0
            while True:
                try:
                    frametuple = self.processed_queue.get_nowait()        
                    index, frame = frametuple
                    if index < lastindex:
                        print('Frame order corrupted!')
                    # assert index >= lastindex
                    output_video_ff.write_frame(frame)
                except:
                    if self.processing_threads > 0:
                        sleep(0.1)
                    else:
                        return
            


    def run_batch_chain(self, source_video, target_video, fps, threads:int = 1, buffersize=32, chain = None, params_frame_gen_func = None):
        cap = cv2.VideoCapture(source_video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.chain = chain
        self.func_params_gen = params_frame_gen_func
        progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        total = frame_count
        self.num_threads = threads
        self.frames_queue = Queue()
        self.processed_queue = Queue()
        self.last_index = -1

        if buffersize < 1:
            buffersize = 1
        self.loadbuffersize = buffersize
        self.processing_threads = self.num_threads

        readthread = Thread(target=self.read_frames_thread, args=(cap, threads))
        readthread.start()


        # preload buffer
        preload_size = min(frame_count, self.loadbuffersize * self.num_threads)
        while self.frames_queue.qsize() < preload_size:
            sleep(0.1)

        writethread = Thread(target=self.write_frames_thread, args=(target_video, width, height, fps, total))
        writethread.start()

        start_processing = time()

        with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
            with ThreadPoolExecutor(thread_name_prefix='swap_proc', max_workers=self.num_threads) as executor:
                futures = []
                
                for threadindex in range(threads):
                    future = executor.submit(self.process_frames, lambda: self.update_progress(progress))
                    futures.append(future)
                
                for future in as_completed(futures):
                    future.result()
        # wait for the task to complete
        readthread.join()
        writethread.join()

        print(f'\nProcessing took {time() - start_processing} secs')


