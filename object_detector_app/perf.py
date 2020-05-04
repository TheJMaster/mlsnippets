import timeit
from object_detection_app import draw_worker, detect_worker, track_worker
from multiprocessing import Queue, Process
import numpy as np
import cv2

# Benchmark constants
NUM_TRIALS = 100
FRAME_PATH = "frame.png"
FRAME_DATA = None

# Draw process setup
DRAW_INPUT_QUEUE = Queue(maxsize=1)
DRAW_OUTPUT_QUEUE = Queue(maxsize=1)

# Detect process setup
DETECT_INPUT_QUEUE = Queue(maxsize=1)
DETECT_OUTPUT_QUEUE = Queue(maxsize=1)
DETECT_GPU_ID = 0


# Track process setup
TRACK_INPUT_QUEUE = Queue(maxsize=1)
TRACK_OUTPUT_QUEUE = Queue(maxsize=1)


def benchmark_draw():
    DRAW_INPUT_QUEUE.put(FRAME_DATA)
    DRAW_OUTPUT_QUEUE.get()


def benchmark_detect():
    DETECT_INPUT_QUEUE.put(FRAME_DATA)
    DETECT_OUTPUT_QUEUE.get() 


def benchmark_detect_bs_2():
    DETECT_INPUT_QUEUE.put((FRAME_DATA, FRAME_DATA))
    DETECT_OUTPUT_QUEUE.get()


def benchmark_detect_bs_3():
    DETECT_INPUT_QUEUE.put((FRAME_DATA, FRAME_DATA, FRAME_DATA))
    DETECT_OUTPUT_QUEUE.get()


def benchmark_track():
    TRACK_INPUT_QUEUE.put((FRAME_DATA, [], None))
    TRACK_OUTPUT_QUEUE.get()


def timeit_setup(benchmark_name):
    return "from __main__ import {}".format(benchmark_name)


def run_benchmark_suite():
    global FRAME_DATA
    
    # Setup frame
    FRAME_DATA = cv2.imread(FRAME_PATH)

    # Run draw benchmark
    draw_proc = Process(target=draw_worker, args=(DRAW_INPUT_QUEUE, DRAW_OUTPUT_QUEUE,))
    draw_proc.start()
    benchmark_draw()  # warm up
    draw_benchmark_res = timeit.timeit("benchmark_draw()", setup=timeit_setup("benchmark_draw"), number=NUM_TRIALS)
    draw_proc.terminate()
    draw_proc.join()
    print("DRAW BENCHMARK: {}ms".format(draw_benchmark_res / NUM_TRIALS * 1000))

    #detect_proc = Process(target=detect_worker, args=(DETECT_INPUT_QUEUE, DETECT_OUTPUT_QUEUE, DETECT_GPU_ID,))
    #detect_proc.start()
    #benchmark_detect()  # warm up

    #detect_benchmark_res = timeit.timeit("benchmark_detect()", setup=timeit_setup("benchmark_detect"), number=NUM_TRIALS)
    #print("DETECT BENCHMARK (BATCH SIZE = 1): {}ms".format(detect_benchmark_res / NUM_TRIALS * 1000))
    
    #detect_benchmark_res_2 = timeit.timeit("benchmark_detect_bs_2()", setup=timeit_setup("benchmark_detect_bs_2"), number=NUM_TRIALS)
    #print("DETECT BENCHMARK (BATCH SIZE = 2): {}ms".format(detect_benchmark_res_2 / NUM_TRIALS * 1000))
    #detect_benchmark_res_3 = timeit.timeit("benchmark_detect_bs_3()", setup=timeit_setup("benchmark_detect_bs_3"), number=NUM_TRIALS)
    #print("DETECT BENCHMARK (BATCH SIZE = 3): {}ms".format(detect_benchmark_res_3 / NUM_TRIALS * 1000))
    
    #detect_proc.terminate()
    #detect_proc.join()

    track_proc = Process(target=track_worker, args=(TRACK_INPUT_QUEUE, TRACK_OUTPUT_QUEUE,))
    track_proc.start()
    benchmark_track()  # warm up
    track_benchmark_res = timeit.timeit("benchmark_track()", setup=timeit_setup("benchmark_track"), number=NUM_TRIALS)
    print("TRACK BENCHMARK: {}ms".format(track_benchmark_res / NUM_TRIALS * 1000))
    track_proc.terminate()
    track_proc.join()


if __name__ == "__main__":
    run_benchmark_suite()
