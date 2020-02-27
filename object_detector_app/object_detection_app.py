import sys
import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
import random


from utils.app_utils import FPS, LocalVideoStream, HLSVideoStream
from multiprocessing import Queue, Pool
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

sys.path.insert(1, '/home/jtjohn24/mlsnippets/object_detector_app/re3-tensorflow') 
from tracker import re3_tracker

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

next_box_id = 0
tracked_box_ids = []
undetected_counts = []
colors = []

undetected_threshold = 10 # Number of frames to allow for undetected.
equality_threshold = 30 # How close we want the edges to be for two boxes to be considered the same.

# Returns true iff the two numbers are within an equality threshold of each other.
def approx_eq(x, y):
    return abs(x-y) < equality_threshold

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

# Sess is used to detect new busses, tracker is used to track existing ones.
def detect_objects(image_np, sess, detection_graph, tracker):
    global next_box_id, tracked_box_ids
    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
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
    print("[INFO] Detection found {} boxes".format(len(detected_boxes)))

    # Find all of the already tracked bounding boxes.
    tracked_boxes = np.empty(0)
    if len(tracked_box_ids) > 1:
        tracked_boxes = tracker.multi_track(tracked_box_ids, image_np)
    elif len(tracked_box_ids) == 1:
        tracked_boxes = tracker.track(tracked_box_ids[0], image_np)
    print("[INFO] Tracker found {} boxes".format(len(tracked_boxes)))

    # Find any boxes that have been detected but not tracked, and vce versa.
    detected_untracked_boxes, undetected_tracked_boxes = filter_boxes(tracked_boxes, detected_boxes)
    print("[INFO] {} boxes were detected but untracked".format(len(detected_untracked_boxes)))
    print("[INFO] {} boxes were undetected but tracked".format(len(undetected_tracked_boxes)))

    # Remove tracked but undetected boxes from tracking.
    for i, box in reversed(list(enumerate(tracked_boxes))):
        if np.transpose(box) in undetected_tracked_boxes:
            undetected_counts[i] = undetected_counts[i] + 1
            if undetected_counts[i] == undetected_threshold:
                del tracked_box_ids[i]
                del colors[i]
                del undetected_counts[i]
                print("[INFO] Deleted box {}".format(i))
        else:
            undetected_counts[i] = 0

    # Add untracked but detected boxes to be detected.    
    init_bounding_boxes = {}
    for box in detected_untracked_boxes:
        box_id = "box_{}".format(next_box_id)
        next_box_id = next_box_id + 1
        
        tracked_box_ids.append(box_id)
        init_bounding_boxes[box_id] = box
        
        r = random.uniform(0, 1) * 255
        color = cv2.cvtColor(np.uint8([[[r, 128, 200]]]), cv2.COLOR_HSV2RGB).squeeze().tolist()
        colors.append(color)

        undetected_counts.append(0)
        print("[INFO] Added bounding box for {}".format(box_id))

    # Run tracker again for all boxes.
    bboxes = np.empty(0)
    if len(tracked_box_ids) > 1:
        bboxes = tracker.multi_track(tracked_box_ids, image_np, init_bounding_boxes)
    elif len(tracked_box_ids) == 1:
        box_id = tracked_box_ids[0]
        bboxes = tracker.track(box_id, image_np, init_bounding_boxes[box_id])
    print("[INFO] {} bounding boxes found by final tracker".format(len(bboxes)))

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
            colors[bb], 2)

    return image_np


def worker(input_q, output_q):
    tracker = re3_tracker.Re3Tracker(-1)
    
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        if np.shape(frame) == ():
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph, tracker))

    fps.stop()
    sess.close()


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

    # TODO: Change back the default number of workers to 2.
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=1, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=1, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))


    if (args.stream):
        print('Reading from hls stream.')
        video_capture = HLSVideoStream(src=args.stream).start()
    elif (args.video_path):
        print('Reading from local video.')
        video_capture = LocalVideoStream(src=args.video_path, width=args.width, height=args.height).start() 
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

    # pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
