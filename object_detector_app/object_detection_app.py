#!/usr/bin/python3
"""Bus detection and tracking application."""

import argparse
from multiprocessing import Queue, Pool, Process
import os
import random
import sys
import time

import numpy as np
from termcolor import cprint

import cv2
from utils.app_utils import FPS, LocalVideoStream, HLSVideoStream

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

NEXT_BOX_ID = 0
TRACKED_BOX_IDS = []
UNDETECTED_COUNTS = []
COLORS = []

UNDETECTED_THRESHOLD = 10 # Number of frames to allow for undetected.
EQUALITY_THRESHOLD = 30 # How close we want the edges to be for two boxes to be considered the same.

def log(info):
    """Log information to console."""
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


def filter_boxes(tracked_boxes, detected_boxes):
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


def detect_worker(image_np):
    # pylint: disable-msg=too-many-locals
    """Runs object detection on the provided numpy image using a frozen detection model."""
    import tensorflow as tf  # pylint: disable-msg=import-outside-toplevel

    # Setup tesnorflow session
    detection_graph = tf.Graph()
    with detection_graph.as_default():  # pylint: disable-msg=not-context-manager
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)

    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
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
        feed_dict={image_tensor: image_np_expanded})

    # Flatten all results for filtering.
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)

    # Filter for a particular class.
    # TODO(justin): change this to classes == 6 (bus)
    indicies = np.argwhere(classes == 3) # class 3 == cars
    boxes = np.squeeze(boxes[indicies])
    classes = np.squeeze(classes[indicies]).astype(np.int32)
    scores = np.squeeze(scores[indicies])

    # Remove all instances of classes with a low score (< 0.5).
    indicies = np.argwhere(scores > 0.5)
    boxes = np.squeeze(boxes[indicies])
    classes = np.squeeze(classes[indicies])
    scores = np.squeeze(scores[indicies])

    # Convert all of the boxes to match style from RE-3 tracker.
    height = image_np.shape[0]
    width = image_np.shape[1]
    detected_boxes = []
    for box in boxes:
        if isinstance(box, np.ndarray):
            detected_boxes.append([box[1]*width, box[0]*height, box[3]*width, box[2]*height])
    detected_boxes = np.array(detected_boxes)
    log("detection found {} boxes".format(len(detected_boxes)))

    sess.close()
    return detected_boxes


def track_worker(input_queue, output_queue):
    """Runs Re3 tracker on images from input queue, passing bounding boxes to output queue."""
    sys.path.insert(1, '/home/jtjohn24/mlsnippets/object_detector_app/re3-tensorflow')
    from tracker import re3_tracker  # pylint: disable-msg=import-outside-toplevel, import-error

    tracker = re3_tracker.Re3Tracker(os.getenv('CUDA_VISIBLE_DEVICES'))
    while True:
        image_np, init_bounding_boxes = input_queue.get()
        tracked_boxes = np.empty(0)
        if len(TRACKED_BOX_IDS) > 1:
            tracked_boxes = tracker.multi_track(TRACKED_BOX_IDS, image_np, init_bounding_boxes)
        elif len(TRACKED_BOX_IDS) == 1:
            init_bounding_box = None
            if init_bounding_boxes is not None:
                init_bounding_box = init_bounding_boxes[TRACKED_BOX_IDS[0]]
            tracked_boxes = tracker.track(TRACKED_BOX_IDS[0], image_np, init_bounding_box)
        output_queue.put(tracked_boxes)


def find_objects(image_np, tracker_input_q, tracker_output_q):
    # pylint: disable-msg=too-many-locals
    """Run bus detection using on image, tracking existing buses using input/output queues."""
    global NEXT_BOX_ID  # pylint: disable-msg=global-statement

    # Run three seperate object detectors.
    all_detected_boxes = Pool(processes=3).map(detect_worker, [image_np]*3)
    detected_boxes = common_boxes(all_detected_boxes)

    # Find all of the already tracked bounding boxes.
    tracker_input_q.put((image_np, None))
    tracked_boxes = tracker_output_q.get()
    log("tracker found {} boxes".format(len(tracked_boxes)))

    # Find any boxes that have been detected but not tracked, and vce versa.
    detected_untracked_boxes, undetected_tracked_boxes = filter_boxes(tracked_boxes, detected_boxes)
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
    tracker_input_q.put((image_np, init_bounding_boxes))
    bounding_boxes = tracker_output_q.get()
    log("{} bounding boxes found by final tracker".format(len(bounding_boxes)))

    # Display detected boxes in gray.
    for detected_box in detected_boxes:
        cv2.rectangle(image_np,
                      (int(detected_box[0]), int(detected_box[1])),
                      (int(detected_box[2]), int(detected_box[3])),
                      [204, 204, 204], 2)

    # Display tracked boxes on the image each in a different color.
    for idx, bounding_box in enumerate(bounding_boxes):
        cv2.rectangle(image_np,
                      (int(bounding_box[0]), int(bounding_box[1])),
                      (int(bounding_box[2]), int(bounding_box[3])),
                      COLORS[idx], 2)

    return image_np


def draw_worker(input_q, output_q):
    """Detect and track buses from input and save image annotated with bounding boxes to output."""
    tracker_input_queue = Queue(maxsize=3)
    tracker_output_queue = Queue(maxsize=3)
    tracker = Process(target=track_worker, args=(tracker_input_queue, tracker_output_queue,))
    tracker.start()
    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        if np.shape(frame) == ():
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(find_objects(frame_rgb, tracker_input_queue, tracker_output_queue))
    fps.stop()
    tracker.join()


def main(args):
    """Sets up object detection according to the provided args."""
    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    drawer_proc = Process(target=draw_worker, args=(input_q, output_q,))
    drawer_proc.start()

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
        cv2.imshow('Video', output_rgb)
        fps.update()

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - start_time))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    drawer_proc.join()
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
