from typing import Any, List, Callable
from roop.typing import Frame
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from threading import Thread
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
    processing_frames = 1
    reading_frames = False
    buffer_wait_time = 0.1
    loadbuffersize = 0 

    savedict =	{}

    framequeues = []

    def __init__(self):
        ChainImgProcessor.__init__(self)

    def pick_from_queue(self) -> List[str]:
        entry = None
        while self.queue.empty():
            if self.reading_frames:
                sleep(self.buffer_wait_time)
            else:
                return None

        entry = (self.current_index, self.queue.get())
        self.current_index += 1
        return entry


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
        i = 0
        num_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            count = 0
            while(self.framequeues[i].qsize() > self.loadbuffersize):
                i += 1
                count += 1
                i %= num_threads
                if count == num_threads:
                    sleep(0.1)

            self.framequeues[i].put((num_frame,frame))
            num_frame += 1
            i += 1
            i %= num_threads
        self.reading_frames = False




    def write_frames_thread(self, target_video, width, height, fps, total):
        i = 0
        with FFMPEG_VideoWriter(target_video, (width, height), fps, codec=roop.globals.video_encoder, crf=roop.globals.video_quality, audiofile=None) as output_video_ff:
            while i < total:
                if i in self.savedict:
                    frame = self.savedict.pop(i)
                    if frame is not None:
                        output_video_ff.write_frame(frame)
                    # del frame
                    i += 1
                    sleep(0.1)
                else:
                    if self.processing_frames > 0:
                        sleep(0.5)
                    else:
                        if(len(self.savedict) < 1):
                            print(f'Write Videoframe {i}: No more frames!')
                            break

           


    def process_frames(self, threadindex, progress) -> None:
        copy_queue = Queue()
        while True:
            while self.framequeues[threadindex].empty():
                if self.reading_frames:
                    sleep(0.1)
                else:
                    self.processing_frames -= 1
                    return
            i = 0
            while i < self.loadbuffersize and not self.framequeues[threadindex].empty():
                copy_queue.put(self.framequeues[threadindex].get())
                i += 1

            while not copy_queue.empty(): 
                frametuple = copy_queue.get()
                if frametuple is None:
                    return
                index,frame = frametuple
                if self.func_params_gen:
                    params = self.func_params_gen(None, frame)
                else:
                    params = {}
                resimg, _ = self.run_chain(frame, params, self.chain)
                # print(f'Hello from {threadindex} -> {str(self.framequeues[threadindex].qsize())} - Reading: {self.reading_frames}')
                self.savedict[index] = resimg
                progress()



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
        self.queue = Queue()
        self.current_index = 0

        start_processing = time()

        for _ in range(threads):
            self.framequeues.append(Queue())
        self.loadbuffersize = buffersize
        self.processing_frames = self.num_threads

        readthread = Thread(target=self.read_frames_thread, args=(cap, threads))
        readthread.start()

        # preload buffer
        preload_size = min(frame_count, self.loadbuffersize * self.num_threads)
        while True:
            size = 0
            for i in range(self.num_threads):
                if not self.framequeues[i].empty():
                    size += self.framequeues[i].qsize()
            if size < preload_size:
                sleep(0.1)
            else:
                break


        writethread = Thread(target=self.write_frames_thread, args=(target_video, width, height, fps, total))
        writethread.start()


        with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
            with ThreadPoolExecutor(thread_name_prefix='swap_proc', max_workers=self.num_threads) as executor:
                futures = []
                
                for threadindex in range(threads):
                    future = executor.submit(self.process_frames, threadindex, lambda: self.update_progress(progress))
                    futures.append(future)
                
                for future in as_completed(futures):
                    future.result()
        # wait for the task to complete
        readthread.join()
        writethread.join()

        print(f'\nProcessing took {time() - start_processing} secs')
        self.savedict.clear()


