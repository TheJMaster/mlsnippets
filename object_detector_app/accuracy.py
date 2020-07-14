from object_detection_app import draw_worker
from multiprocessing import Queue, Process
import cv2
import os
import numpy as np
import csv
import glob

# Path to input png file
IMAGES_PATH = "test_input/kitti_single/training/image_2/*.png"

# Path to output txt file
LABELS_PATH = "test_input/kitti_single/training/label_2/*.txt"

# Configs to benchmark accuracy.
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
RESULTS_CSV_PATH = "accuracy_out.csv"

# Path to write annotated images (used for debugging).
RESULTS_IMAGES_PATH = "img_out"


# Return greatest multiple of 6 less than n.
# Note: 6 is chosen because the dimensions of x and y need to be split into 2 and 3.
def greatest_multiple_of_six_less_than(n):
    factor = 0
    while factor * 6 <= n:
        factor = factor + 1
    return factor * 6


# Return the scaled bounding boxes used for testing.
def get_scaled_images_and_bounding_boxes():
    image_paths = sorted(glob.glob(IMAGES_PATH))[0:500]
    label_paths = sorted(glob.glob(LABELS_PATH))[0:500]

    scaled_images = []
    scaled_bounding_boxes = []
    count = 1
    total = len(image_paths)
    for image_path, label_path in zip(image_paths, label_paths):
        print("processing image: {} / {}".format(count, total), end="\r")
        count = count + 1
        image = cv2.imread(image_path)

        # Image dims should atleast be even.
        new_height = greatest_multiple_of_six_less_than(image.shape[0])
        new_width = greatest_multiple_of_six_less_than(image.shape[1])
        scaled_images.append(cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR))

        label_file = open(label_path)
        label_lines = label_file.readlines()
        bounding_boxes = []
        for label_line in label_lines:
            label_parts = label_line.split()
            if label_parts[0] != "Car":
                continue  # Only worry about detecting cars for this process
            if label_parts[1] != "0.00":
                continue  # Only worry about detecting bounding boxes that arn't truncated
            if label_parts[2] != "0":
                continue  # Only worry about detecting cars that are fully visible
            bounding_boxes.append([float(label_parts[4]) / image.shape[1] * new_width,
                                   float(label_parts[5]) / image.shape[0] * new_height,
                                   float(label_parts[6]) / image.shape[1] * new_width,
                                   float(label_parts[7]) / image.shape[0] * new_height])
        scaled_bounding_boxes.append(bounding_boxes)
    return scaled_images, scaled_bounding_boxes


# Compute the intersection over union for two different bounding boxes.
# Taken from: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def run_accuracy_test():
    all_images, all_target_bounding_boxes = get_scaled_images_and_bounding_boxes()

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

            total_found = 0
            total_expected = 0
            sum_iou_for_all_images = 0
            count = 0
            for image, target_bounding_boxes in zip(all_images, all_target_bounding_boxes):
                print("running detection on image: {} / {}".format(count, len(all_images)), end="\r")
                draw_input_queue.put(image)
                annotated_img, found_bounding_boxes = draw_output_queue.get()

                found = 0
                ious = []
                for found_bounding_box_idx, found_bounding_box in enumerate(found_bounding_boxes):
                    max_iou = 0
                    for target_bounding_box in target_bounding_boxes:
                        iou = intersection_over_union(found_bounding_box, target_bounding_box)
                        if iou > max_iou:
                            max_iou = iou
                    if max_iou > 0.5:
                        ious.append(max_iou)
                        found = found + 1
                total_found = total_found + found
                total_expected = total_expected + len(target_bounding_boxes)
                sum_iou_for_all_images = sum_iou_for_all_images + np.sum(np.array(ious))

                # Write correct bounding boxes to image, save results.
                for bounding_box in target_bounding_boxes:
                    cv2.rectangle(annotated_img,
                      (int(bounding_box[0]), int(bounding_box[1])),
                      (int(bounding_box[2]), int(bounding_box[3])),
                      [0, 0, 0], 2)
                cv2.imwrite(os.path.join(RESULTS_IMAGES_PATH, str(count) + ".png"), annotated_img)
                count = count + 1
            draw_proc.terminate()
            draw_proc.join()

            writer.writerow([config["num_detect_workers"], config["x_split"], config["y_split"], total_found, total_expected, sum_iou_for_all_images / total_found])


if __name__ == "__main__":
    run_accuracy_test()
