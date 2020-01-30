import sys
import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf


from utils.app_utils import FPS, WebcamVideoStream, HLSVideoStream
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

track_next_bus = True

# Sess is used to detect new busses, tracker is used to track existing ones.
def detect_objects(image_np, sess, detection_graph, tracker):
    global track_next_bus

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

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Remove all instances of non-bus classes.
    # TODO(justin): Change this to classes == 6
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    scores = np.squeeze(scores)

    indicies = np.argwhere(classes == 3)
    boxes = np.squeeze(boxes[indicies])
    classes = np.squeeze(classes[indicies]).astype(np.int32)
    scores = np.squeeze(scores[indicies])

    # Remove all instances of classes with a low score (< 0.5).
    indicies = np.argwhere(scores > 0.5)
    boxes = np.squeeze(boxes[indicies])
    classes = np.squeeze(classes[indicies])
    scores = np.squeeze(scores[indicies])

    height = image_np.shape[0]
    width = image_np.shape[1]

    # Visualize trakcer on array.
    img = image_np.copy()
    boxToDraw = None
    if len(boxes) > 0:
        if track_next_bus:
            if isinstance(boxes[0], np.ndarray):
                h = image_np.shape[0]
                w = image_np.shape[1]
                b = boxes[0]
                box = [b[1]*w, b[0]*h, b[3]*w, b[2]*h]
                print("box: ", box)
                boxToDraw = tracker.track('bus', img, box)
                track_next_bus = False
        else:
            boxToDraw = tracker.track('bus', img)
       
        # If I found a box, draw it.
        if boxToDraw is not None:
            cv2.rectangle(image_np,
                (int(boxToDraw[0]), int(boxToDraw[1])),
                (int(boxToDraw[2]), int(boxToDraw[3])),
                [0, 0, 255], 2)

            # If this car left the screen, pick a new one to track.
            if boxToDraw[0] < 5 or boxToDraw[0] > 1075 or boxToDraw[3] > 715 or boxToDraw[3] < 5 or boxToDraw[3] - boxToDraw[1] < 20 or boxToDraw[2] - boxToDraw[0] < 20:
                track_next_bus = True
    elif not track_next_bus:
        track_next_bus = True
    
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
    parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
                        default=2, help='Number of workers.')
    parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
                        default=5, help='Size of the queue.')
    args = parser.parse_args()

    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    pool = Pool(args.num_workers, worker, (input_q, output_q))


    if (args.stream):
        print('Reading from hls stream.')
        video_capture = HLSVideoStream(src=args.stream).start()
    else:
        print('Reading from webcam.')
        video_capture = WebcamVideoStream(src=args.video_source,
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

    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
