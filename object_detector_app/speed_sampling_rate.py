from object_detection_app import draw_worker
from multiprocessing import Queue, Process
import cv2
import os
import numpy as np
import csv
import glob
import time

# Path to input png file
IMAGES_PATH = "test_input/kitti_single/training/image_2/*.png"

# Configs to benchmark speed.
# Configs are stored as the map {var-name -> value}
CONFIGS = [
        {"detect_rate": 1},
        {"detect_rate": 5},
        {"detect_rate": 10},
        {"detect_rate": 30},
        {"detect_rate": 60}]

# Track and detect variables that are being kept constant.
NUM_DETECT_WORKERS = 1
X_SPLIT = 1
Y_SPLIT = 1
TRACK_GPU_ID = 0

# Path to write results CSV.
RESULTS_CSV_PATH = "speed_sr_out.csv"



def get_images():
    image_paths = sorted(glob.glob(IMAGES_PATH))[0:505]
    images = []
    for idx, image_path in enumerate(image_paths):
        print("processing image: {} / {}".format(idx + 1, len(image_paths)), end="\r")
        images.append(cv2.imread(image_path))
    return images


def current_time_millis():
    return int(round(time.time() * 1000))


def run_speed_test():
    images = get_images()

    with open(RESULTS_CSV_PATH, 'w') as results_file:
        writer = csv.writer(results_file)

        for config in CONFIGS:
            draw_input_queue = Queue(maxsize=1)
            draw_output_queue = Queue(maxsize=1)
            draw_proc = Process(target=draw_worker, args=(draw_input_queue,
                                                        draw_output_queue,
                                                        NUM_DETECT_WORKERS,
                                                        TRACK_GPU_ID,
                                                        X_SPLIT,
                                                        Y_SPLIT,
                                                        config["detect_rate"]))
            draw_proc.start()

            fps = []
            last_sec = time.time()
            count = 0
            for idx, image in enumerate(images):
                print("running detection on image: {} / {}".format(idx + 1, len(images)), end="\r")
                draw_input_queue.put(image)
                draw_output_queue.get()
                if idx < 5:
                    continue  # Skip the first few to get a tighter bound
                curr_time = int(time.time())
                if curr_time < last_sec + 1:
                    count = count + 1
                else:
                    last_sec = curr_time
                    fps.append(count)
                    count = 1

            draw_proc.terminate()
            draw_proc.join()

            fps = np.array(fps)
            writer.writerow([config["detect_rate"], np.mean(fps), np.std(fps)])


if __name__ == "__main__":
    run_speed_test()
