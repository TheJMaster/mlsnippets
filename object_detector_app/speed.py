from object_detection_app import draw_worker
from multiprocessing import Queue, Process
import cv2
import os
import numpy as np
import csv
import glob
import time

# Path to input png file
IMAGES_PATH = "kitti-object-detection/kitti_single/training/image_2/*.png"

# Configs to benchmark speed.
# Configs are stored as the map {var-name -> value}
CONFIGS = [
        {"num_detect_workers": 1, "x_split": 1, "y_split": 1},
        {"num_detect_workers": 2, "x_split": 1, "y_split": 1},
        {"num_detect_workers": 3, "x_split": 1, "y_split": 1},
        {"num_detect_workers": 1, "x_split": 1, "y_split": 1},
        {"num_detect_workers": 1, "x_split": 2, "y_split": 1},
        {"num_detect_workers": 1, "x_split": 3, "y_split": 1},
        {"num_detect_workers": 1, "x_split": 1, "y_split": 1},
        {"num_detect_workers": 1, "x_split": 1, "y_split": 2},
        {"num_detect_workers": 1, "x_split": 1, "y_split": 3}]

# Track and detect variables that are being kept constant.
TRACK_GPU_ID = 0
DETECT_RATE = 1

# Path to write results CSV.
RESULTS_CSV_PATH = "speed_out.csv"


# Return greatest multiple of 6 less than n.
# Note: 6 is chosen because the dimensions of x and y need to be split into 2 and 3.
def greatest_multiple_of_six_less_than(n):
    factor = 0
    while factor * 6 <= n:
        factor = factor + 1
    return factor * 6


def get_scaled_images():
    image_paths = sorted(glob.glob(IMAGES_PATH))[0:505]
    images = []
    for idx, image_path in enumerate(image_paths):
        print("processing image: {} / {}".format(idx + 1, len(image_paths)), end="\r")
        image = cv2.imread(image_path)

        # Image dims should atleast be even.
        new_height = greatest_multiple_of_six_less_than(image.shape[0])
        new_width = greatest_multiple_of_six_less_than(image.shape[1])
        images.append(cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR))
    return images


def current_time_millis():
    return int(round(time.time() * 1000))


def run_speed_test():
    images = get_scaled_images()

    with open(RESULTS_CSV_PATH, 'w') as results_file:
        writer = csv.writer(results_file)

        for config in CONFIGS:
            draw_input_queue = Queue(maxsize=1)
            draw_output_queue = Queue(maxsize=1)
            draw_proc = Process(target=draw_worker, args=(draw_input_queue,
                                                        draw_output_queue,
                                                        config["num_detect_workers"],
                                                        TRACK_GPU_ID,
                                                        config["x_split"],
                                                        config["y_split"],
                                                        DETECT_RATE))
            draw_proc.start()

            durations = []
            for idx, image in enumerate(images):
                print("running detection on image: {} / {}".format(idx + 1, len(images)), end="\r")
                before = current_time_millis()
                draw_input_queue.put(image)
                draw_output_queue.get()
                if idx < 5:
                    continue  # Skip the first few to get a tighter bound
                durations.append(current_time_millis() - before)

            draw_proc.terminate()
            draw_proc.join()

            durations = np.array(durations)
            writer.writerow([config["num_detect_workers"], config["x_split"], config["y_split"], np.mean(durations), np.std(durations)])


if __name__ == "__main__":
    run_speed_test()
