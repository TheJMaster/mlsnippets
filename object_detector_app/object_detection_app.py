#!/usr/bin/python3
"""Bus detection and tracking application."""

import argparse
from multiprocessing import Queue, Process
import os
import random
import sys
import time
from functools import partial

import numpy as np
from termcolor import cprint

import cv2
from utils.app_utils import FPS, LocalVideoStream, HLSVideoStream

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, 'models/ssd_mobilenet/v1_coco/frozen_inference_graph.pb')

NEXT_BOX_ID = 0
TRACKED_BOX_IDS = []
UNDETECTED_COUNTS = []
COLORS = []

UNDETECTED_THRESHOLD = 15 # Number of frames to allow for undetected.
EQUALITY_THRESHOLD = 30 # How close we want the edges to be for two boxes to be considered the same.

MIN_BOX_DIM = 10
MAX_BOX_DIM = 300

DETECT_GPU_IDS = [0, 1, 2]
TRACK_GPU_ID = 0

LOG=False


def log(info):
    """Log information to console."""
    if LOG:
        cprint("[INFO] {}".format(info), 'grey', 'on_white')


def approx_eq(num_one, num_two):
    """Returns true iff the two numbers are within an equality threshold of each other."""
    return abs(num_one-num_two) < EQUALITY_THRESHOLD


def contains_box(box, boxes):
    """Returns true iff the list of boxes (approx) contains the provided box."""
    for candidate_box in boxes:
        if sum([approx_eq(box[i], candidate_box[i]) for i in range(len(box))]) >= len(box) / 2:
            return True
    return False


def exclusive_boxes(tracked_boxes, detected_boxes):
    """Returns detected but untracked bounding boxes, undetected but tracked bounding boxes."""
    detected_untracked_boxes = []
    for box in detected_boxes:
        if not contains_box(box, tracked_boxes):
            detected_untracked_boxes.append(box)

    undetected_tracked_boxes = []
    for box in tracked_boxes:
        if not contains_box(box, detected_boxes):
            undetected_tracked_boxes.append(box)

    return (np.array(detected_untracked_boxes), np.array(undetected_tracked_boxes))


def validate_boxes(boxes):
    """Filters out invalid boxes."""
    valid_boxes = []
    for box in boxes:
        valid = True
        x_dim = abs(box[2] - box[0])
        y_dim = abs(box[3] - box[1])
        # Ensure each dimension is at least the min dim
        if x_dim < MIN_BOX_DIM or y_dim < MIN_BOX_DIM:
            valid = False
        # Ensure each dimension is at most the max dim
        if x_dim > MAX_BOX_DIM or y_dim > MAX_BOX_DIM:
            valid = False
        if valid:
            valid_boxes.append(box)
    return valid_boxes


def common_boxes(all_boxes):
    """Returns bounding boxes present in atleast 2 of 3 sets."""
    boxes = []
    for box in all_boxes[0]:
        if contains_box(box, all_boxes[1]) or contains_box(box, all_boxes[2]):
            boxes.append(box)
    for box in all_boxes[1]:
        if not contains_box(box, all_boxes[0]) and contains_box(box, all_boxes[2]):
            boxes.append(box)
    return boxes


def detect_worker(input_queue, output_queue, gpu_id):
    # pylint: disable-msg=too-many-locals
    """Runs object detection on the provided numpy image using a frozen detection model."""
    import tensorflow as tf  # pylint: disable-msg=import-outside-toplevel
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Setup tesnorflow session
    detection_graph = tf.Graph()
    with detection_graph.as_default():  # pylint: disable-msg=not-context-manager
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)

    while True:
        # Expect images for batch request.
        images = input_queue.get()
       
        # Expand dimensions to create batch since the model expects images to
        # have shape: [1, Height, Width, 3]
        batch = None
        if type(images) is not tuple:
            batch = np.expand_dims(images, axis=0)
        else:
            images_expanded = [np.expand_dims(image, axis=0) for image in images]
            batch = np.concatenate(tuple(images_expanded))
       
        # Grab the tensor to populate with the image batch.
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each class of the objects.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Run initial detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: batch})

        batch_results = []
        for (image, boxes, scores, classes) in zip(images, boxes, scores, classes):
            # Flatten all results for filtering.
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)
            
            # Filter for a particular class.
            # TODO(justin): change this to classes == 6 (bus)
            indicies = np.argwhere(classes == 3)
            boxes = np.squeeze(boxes[indicies])
            scores = np.squeeze(scores[indicies])
            
            # Remove all instances of classes with a low score (< 0.5).
            indicies = np.argwhere(scores > 0.5)
            boxes = np.squeeze(boxes[indicies])

            # Convert all of the boxes to match style from RE-3 tracker.
            height = image.shape[0]
            width = image.shape[1]
            detected_boxes = []
            if np.isnan(boxes).any():  # Don't handle non listable arrays.
                for box in list(boxes):
                    if isinstance(box, np.ndarray):
                        detected_boxes.append([box[1]*width, box[0]*height, box[3]*width, box[2]*height])
            batch_results.append(np.array(detected_boxes))

        # Return bounding boxes for batch via output queue.
        output_queue.put(np.array(batch_results))

    # Close session if loop is killed.
    sess.close()


def track_worker(input_queue, output_queue):
    """Runs Re3 tracker on images from input queue, passing bounding boxes to output queue."""
    sys.path.insert(1, '/home/jtjohn24/mlsnippets/object_detector_app/re3-tensorflow')
    from tracker import re3_tracker  # pylint: disable-msg=import-outside-toplevel, import-error

    tracker = re3_tracker.Re3Tracker(TRACK_GPU_ID)
    while True:
        image_np, box_ids, init_bounding_boxes = input_queue.get()
        if init_bounding_boxes is not None:
            log("init_bounding_boxes length: {}".format(len(init_bounding_boxes)))
        tracked_boxes = np.empty(0)
        if len(box_ids) > 1:
            tracked_boxes = tracker.multi_track(box_ids, image_np, init_bounding_boxes)
        elif len(box_ids) == 1:
            init_bounding_box = None
            if init_bounding_boxes is not None:
                init_bounding_box = init_bounding_boxes[box_ids[0]]
            tracked_boxes = tracker.track(box_ids[0], image_np, init_bounding_box)
        output_queue.put(tracked_boxes)


def add_all_not_present(source, target):
    for box in source:
        if not contains_box(box, target):
            target.append(box)
    return target


# Resize all boxes to become the target height and width
def resize_all(boxes, orig_h, orig_w, new_h, new_w):
    res = []
    if len(boxes) == 0 or not isinstance(boxes, np.ndarray):
        return np.array([])
    for box in boxes:
        if isinstance(box, np.ndarray):
            res.append([box[1]/orig_w*new_w, box[0]/orig_h*new*h, box[3]/orig_w*new_w, box[2]/orig_h*new_h])
    return np.array(res)


def run_detect_batch(detect_input_queues, detect_output_queues, batch):
    for input_queue in detect_input_queues:
        input_queue.put(batch)
    detect_results = []
    for output_queue in detect_output_queues:
        detect_results.append(output_queue.get())
    return detect_results


def find_objects(image_np, detect_input_queues, detect_output_queues, track_input_queue,
                 track_output_queue):
    # pylint: disable-msg=too-many-locals
    """Run bus detection using on image, tracking existing buses using input/output queues."""
    global NEXT_BOX_ID  # pylint: disable-msg=global-statement

    # Split images into foreground and background for additional checking.
    sub_imgs = np.split(image_np, 2)
    background_img_orig = sub_imgs[0]
    foreground_img_orig = sub_imgs[1]

    # Resize images to SSD expectations (300 x 300)
    whole_img = cv2.resize(image_np, dsize=(300, 300), interpolation=cv2.INTER_LINEAR)
    background_img = cv2.resize(background_img_orig, dsize=(300, 300), interpolation=cv2.INTER_LINEAR)
    foreground_img = cv2.resize(foreground_img_orig, dsize=(300, 300), interpolation=cv2.INTER_LINEAR)

    # Run foreground/background detection batch.
    detect_results = run_detect_batch(detect_input_queues, detect_output_queues, (background_img, foreground_img, whole_img))
    background_detected_boxes = common_boxes([boxes[0] for boxes in detect_results])
    foreground_detected_boxes = common_boxes([boxes[1] for boxes in detect_results])
    whole_detected_boxes = common_boxes([boxes[2] for boxes in detect_results])

    # Resize bounding boxes to match original image sizes.
    background_detected_boxes = resize_all(background_detected_boxes, 300, 300, background_img_orig.shape[0], background_img_orig[1])
    foreground_detected_boxes = resize_all(foreground_detected_boxes, 300, 300, foreground_img_orig.shape[0], background_img_orig[1])
    whole_detected_boxes = resize_all(whole_detected_boxes, 300, 300, image_np.shape[0], image_np[1])

    # Shift y dim of foreground boxes to position relative to the entire image.
    for box in foreground_detected_boxes:
        box[1] = box[1] + background_img.shape[0]
        box[3] = box[3] + background_img.shape[0]

    # Find unique boxes from all detection runs.
    detected_boxes = common_boxes([background_detected_boxes, foreground_detected_boxes, whole_detected_boxes])
    detected_boxes = add_all_not_present(background_detected_boxes, detected_boxes)
    detected_boxes = add_all_not_present(foreground_detected_boxes, detected_boxes)
    detected_boxes = add_all_not_present(whole_detected_boxes, detected_boxes)

    # Find all of the already tracked bounding boxes.
    track_input_queue.put((image_np, TRACKED_BOX_IDS, None))
    tracked_boxes = track_output_queue.get()
    log("tracker found {} boxes".format(len(tracked_boxes)))

    # Find any boxes that have been detected but not tracked, and vce versa.
    detected_untracked_boxes, undetected_tracked_boxes = exclusive_boxes(tracked_boxes,
                                                                         detected_boxes)

    log("{} boxes were detected but untracked".format(len(detected_untracked_boxes)))
    log("{} boxes were undetected but tracked".format(len(undetected_tracked_boxes)))

    # Remove tracked but undetected boxes from tracking.
    for i, box in reversed(list(enumerate(tracked_boxes))):
        if np.transpose(box) in undetected_tracked_boxes:
            UNDETECTED_COUNTS[i] = UNDETECTED_COUNTS[i] + 1
            if UNDETECTED_COUNTS[i] == UNDETECTED_THRESHOLD:
                del TRACKED_BOX_IDS[i]
                del COLORS[i]
                del UNDETECTED_COUNTS[i]
                log("deleted box {}".format(i))
        else:
            UNDETECTED_COUNTS[i] = 0

    # Add untracked but detected boxes to be detected.
    init_bounding_boxes = {}
    for box in detected_untracked_boxes:
        box_id = "box_{}".format(NEXT_BOX_ID)
        NEXT_BOX_ID = NEXT_BOX_ID + 1
        TRACKED_BOX_IDS.append(box_id)
        init_bounding_boxes[box_id] = box

        hue = random.uniform(0, 1) * 255
        color = cv2.cvtColor(np.uint8([[[hue, 128, 200]]]), cv2.COLOR_HSV2RGB).squeeze().tolist()
        COLORS.append(color)

        UNDETECTED_COUNTS.append(0)
        log("added bounding box for {}".format(box_id))

    # Run tracker again for all objects to get final bounding boxes.
    track_input_queue.put((image_np, TRACKED_BOX_IDS, init_bounding_boxes))
    bounding_boxes = track_output_queue.get()
    log("{} bounding boxes found by final tracker".format(len(bounding_boxes)))

    # Validate boxes before plotting.
    detected_boxes = validate_boxes(detected_boxes)
    bounding_boxes = validate_boxes(bounding_boxes)
    log("validated boxes")

    # Display detected boxes in gray.
    for detected_box in detected_boxes:
        cv2.rectangle(image_np,
                      (int(detected_box[0]), int(detected_box[1])),
                      (int(detected_box[2]), int(detected_box[3])),
                      [204, 204, 204], 2)
    log("added detected boxes to image")

    # Display tracked boxes on the image each in a different color.
    for idx, bounding_box in enumerate(bounding_boxes):
        cv2.rectangle(image_np,
                      (int(bounding_box[0]), int(bounding_box[1])),
                      (int(bounding_box[2]), int(bounding_box[3])),
                      COLORS[idx], 2)
    log("added tracked boxes to image")

    return image_np


def draw_worker(input_q, output_q):
    """Detect and track buses from input and save image annotated with bounding boxes to output."""
    # Start detection processes.
    detect_worker_input_queues = [Queue(maxsize=3)]*3
    detect_worker_output_queues = [Queue(maxsize=3)]*3
    for worker_id in range(3):
        Process(target=detect_worker, args=(detect_worker_input_queues[worker_id],
                                            detect_worker_output_queues[worker_id],
                                            DETECT_GPU_IDS[worker_id])).start()

    # Start tracking process.
    track_input_queue = Queue(maxsize=3)
    track_output_queue = Queue(maxsize=3)
    track = Process(target=track_worker, args=(track_input_queue, track_output_queue,))
    track.start()

    # Annotate all new frames.
    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        if np.shape(frame) == ():
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(find_objects(frame_rgb, detect_worker_input_queues,
                                  detect_worker_output_queues, track_input_queue,
                                  track_output_queue))
    fps.stop()
    track.join()


def main(args):
    """Sets up object detection according to the provided args."""
    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    draw_proc = Process(target=draw_worker, args=(input_q, output_q,))
    draw_proc.start()

    if args.stream:
        print('Reading from hls stream.')
        video_capture = HLSVideoStream(src=args.stream).start()
    elif args.video_path:
        print('Reading from local video.')
        video_capture = LocalVideoStream(src=args.video_path,
                                         width=args.width,
                                         height=args.height).start()
    else:
        print('Reading from webcam.')
        video_capture = LocalVideoStream(src=args.video_source,
                                         width=args.width,
                                         height=args.height).start()

    fps = FPS().start()
    while True:  # fps._numFrames < 120
        frame = video_capture.read()
        input_q.put(frame)
        start_time = time.time()

        output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        # cv2.imshow('Video', output_rgb)
        fps.update()

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - start_time))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    draw_proc.join()
    video_capture.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-str', '--stream', dest="stream", action='store', type=str, default=None)
    PARSER.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    PARSER.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    PARSER.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    PARSER.add_argument('-p', '--path', dest="video_path", type=str, default=None)
    PARSER.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=1, help='Size of the queue.')
    main(PARSER.parse_args())
