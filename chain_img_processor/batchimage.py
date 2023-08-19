from typing import Any, List, Callable
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from .image import ChainImgProcessor
from tqdm import tqdm
import cv2
import roop.globals

def create_queue(temp_frame_paths: List[str]) -> Queue[str]:
    queue: Queue[str] = Queue()
    for frame_path in temp_frame_paths:
        queue.put(frame_path)
    return queue


def pick_queue(queue: Queue[str], queue_per_future: int) -> List[str]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues



class ChainBatchImageProcessor(ChainImgProcessor):
    chain = None
    func_params_gen = None
    num_threads = 1

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
        progress.refresh()
        progress.update(1)


    def process_frames(self, source_files: List[str], target_files: List[str], current_files, update: Callable[[], None]) -> None:
        for f in current_files:
            if not roop.globals.processing:
                return
            
            temp_frame = cv2.imread(f)
            if temp_frame is not None:
                if self.func_params_gen:
                    params = self.func_params_gen(None, temp_frame)
                else:
                    params = {}
                resimg, _ = self.run_chain(temp_frame, params, self.chain)
                if resimg is not None:
                    i = source_files.index(f)
                    cv2.imwrite(target_files[i], resimg)
            if update:
                update()


    def run_batch_chain(self, source_files, target_files, threads:int = 1, chain = None, params_frame_gen_func = None):
        self.chain = chain
        self.func_params_gen = params_frame_gen_func
        progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        total = len(source_files)
        self.num_threads = threads
        with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
            with ThreadPoolExecutor(max_workers=threads) as executor:
                futures = []
                queue = create_queue(source_files)
                queue_per_future = max(len(source_files) // threads, 1)
                while not queue.empty():
                    future = executor.submit(self.process_frames, source_files, target_files, pick_queue(queue, queue_per_future), lambda: self.update_progress(progress))
                    futures.append(future)
                for future in as_completed(futures):
                    future.result()

