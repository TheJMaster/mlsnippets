import sys
import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import random

from utils.app_utils import FPS, LocalVideoStream, HLSVideoStream
from multiprocessing import Queue, Pool, Process
from termcolor import cprint

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

# Returns true iff the two numbers are within an equality threshold of each other.
def approx_eq(x, y):
    return abs(x-y) < EQUALITY_THRESHOLD


# Returns true iff the list of boxes contains the provided box. We considered two boxes
# equal if any two of their sides are within some threshold distance of each other.
def contains_box(box, boxes):
    for candidate_box in boxes:
        if sum([approx_eq(box[i], candidate_box[i]) for i in range(len(box))]) >= len(box) / 2:
            return True
    return False


# Returns (detected but untracked boxes, undetected but tracked boxes)
def filter_boxes(tracked_boxes, detected_boxes):
    detected_untracked_boxes = []
    for box in detected_boxes:
        if not contains_box(box, tracked_boxes):
            detected_untracked_boxes.append(box)

    undetected_tracked_boxes = []
    for box in tracked_boxes:
        if not contains_box(box, detected_boxes):
            undetected_tracked_boxes.append(box)

    return (np.array(detected_untracked_boxes), np.array(undetected_tracked_boxes))


def detect_init():
    global tf, sess, detection_graph
    import tensorflow as tf

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)


def detection_worker(image_np):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Run initial detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Remove all instances of non-bus classes.
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)

    # Filter for a particular class.
    # TODO(justin): Change this to classes == 6 (bus)
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


def tracker_worker(image_input_q, output_q):
    sys.path.insert(1, '/home/jtjohn24/mlsnippets/object_detector_app/re3-tensorflow')
    from tracker import re3_tracker

    while True:
        image_np, init_bounding_boxes = image_input_q.get()
        tracked_boxes = np.empty(0)
        if len(TRACKED_BOX_IDS) > 1:
            tracked_boxes = tracker.multi_track(TRACKED_BOX_IDS, image_np, init_bounding_boxes)
        elif len(TRACKED_BOX_IDS) == 1:
            init_bounding_box = None
            if init_bounding_boxes is not None:
                init_bounding_box = init_bounding_boxes[TRACKED_BOX_IDS[0]]
            tracked_boxes = tracker.track(TRACKED_BOX_IDS[0], image_np, init_bounding_box)
        output_q.put(tracked_boxes)


# Common boxes returns only the bounding boxes present in 2 of 3 sets.
def common_boxes(all_boxes):
    boxes = []
    for box in all_boxes[0]:
        if contains_box(box, all_boxes[1]) or contains_box(box, all_boxes[2]):
            boxes.append(box)
    for box in all_boxes[1]:
        if not contains_box(box, all_boxes[0]) and contains_box(box, all_boxes[2]):
            boxes.append(box)
    return boxes

def log(s):
    cprint("[INFO] {}".format(s), 'grey', 'on_white')

# Sess is used to detect new busses, tracker is used to track existing ones.
def detect_objects(image_np, tracker_input_q, tracker_output_q):
    # Run and collect multiple detections.
    pool = multiprocessing.Pool(processes=3, initializer=detect_init)
    all_detected_boxes = pool.map(detection_worker, [image_np]*3)
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

        r = random.uniform(0, 1) * 255
        color = cv2.cvtColor(np.uint8([[[r, 128, 200]]]), cv2.COLOR_HSV2RGB).squeeze().tolist()
        COLORS.append(color)

        UNDETECTED_COUNTS.append(0)
        log("added bounding box for {}".format(box_id))

    # Run tracker again for all boxes.
    tracker_input_q.put((image_np, init_bounding_boxes))
    bboxes = tracker_output_q.get()
    log("{} bounding boxes found by final tracker".format(len(bboxes)))

    # Display detected boxes (helpful for debugging) in gray.
    for bb, bbox in enumerate(detected_boxes):
        cv2.rectangle(image_np,
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])),
                      [204, 204, 204], 2)

    # Display tracked boxes on the image (each in a different color).
    for bb, bbox in enumerate(bboxes):
        cv2.rectangle(image_np,
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[2]), int(bbox[3])),
                      COLORS[bb], 2)

    return image_np


def track_worker(input_q, output_q):
    # TODO: Add a flag so we can set the CUDA_VISIBLE_DEVICES flag (as they do in re3_tracker.py)
    # TODO: Do we really need any size > 1 here?
    t_input_q = Queue(maxsize=3)
    t_output_q = Queue(maxsize=3)
    t_worker = Process(target=tracker_worker, args=(t_input_q, t_output_q,))
    t_worker.start()
    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        if np.shape(frame) == ():
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, t_input_q, t_output_q))
    fps.stop()
    t_worker.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-str', '--stream', dest="stream", action='store', type=str, default=None)
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=360, help='Height of the frames in the video stream.')
    parser.add_argument('-p', '--path', dest="video_path", type=str, default=None)
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=1, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    tracker_proc = Process(target=track_worker, args=(input_q, output_q,))
    tracker_proc.start()

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
        t = time.time()

        output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', output_rgb)
        fps.update()

        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    tracker_proc.join()
    video_capture.stop()
    cv2.destroyAllWindows()
