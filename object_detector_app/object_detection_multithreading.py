import os
import cv2
import time
import argparse
import numpy as np
import subprocess as sp
import json
import tensorflow as tf
from utils.constants import IMAGE_SCALE_FACTOR
from queue import Queue
from threading import Thread
from utils.app_utils import FPS, HLSVideoStream, WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util

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


def detect_objects(image_np, sess, detection_graph):
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

    # Visualization of the results of a detection.
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.5
    )
    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors, frame=image_np)


def worker(input_q, output_q):
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

    # i = -1

    while True:
        # i += 1
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # if i % 10 == 0:
            # data = detect_objects(frame_rgb, sess, detection_graph)
            # i = 0
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))
        

    fps.stop()
    sess.close()

def boxes_are_similar(point1, point2):
    for item in ('xmin', 'xmax', 'ymin', 'ymax'):
        if abs(point1[item] - point2[item]) > 0.2:
            return False
    return True

def add_found(point):
    point['found'] = True
    return point

def set_all_not_found(points):
    for point in points:
        point['found'] = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-strin', '--stream-input', dest="stream_in", action='store', type=str, default=None)
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')
    parser.add_argument('-strout','--stream-output', dest="stream_out", help='The URL to send the livestreamed object detection to.')
    args = parser.parse_args()

    input_q = Queue(1)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    if (args.stream_in):
        print('Reading from hls stream.')
        video_capture = HLSVideoStream(src=args.stream_in).start()
    else:
        print('Reading from webcam.')
        video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()


    args.width = int(video_capture.WIDTH * IMAGE_SCALE_FACTOR)
    args.height = int(video_capture.HEIGHT * IMAGE_SCALE_FACTOR)

    print("args width" + str(args.width))
    print("args height" + str(args.height))

    last_bus_points = []

    while True:

        frame = video_capture.read()
        print(str(frame.shape))
        input_q.put(frame)

        t = time.time()

        if output_q.empty():
            pass  # fill up queue
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            data = output_q.get()

            set_all_not_found(last_bus_points)
            
            rec_points = data['rect_points']
            class_names = data['class_names']
            class_colors = data['class_colors']
            frame = data['frame']
            for point, name, color in zip(rec_points, class_names, class_colors):
                cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                              (int(point['xmax'] * args.width), int(point['ymax'] * args.height)), color, 3)
                cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                              (int(point['xmin'] * args.width) + len(name[0]) * 6,
                               int(point['ymin'] * args.height) - 10), color, -1, cv2.LINE_AA)
                cv2.putText(frame, name[0], (int(point['xmin'] * args.width), int(point['ymin'] * args.height)), font,
                            0.3, (0, 0, 0), 1)

                if name[0].split(":")[0].lower() in ('bus', 'train'):
                    similar_point_found = False
                    for i, point2 in enumerate(last_bus_points):
                        if boxes_are_similar(point, point2):
                            last_bus_points[i] = add_found(point2)
                            # print("modified point2: " + str(point2))
                            similar_point_found = True
                            break
                    if not similar_point_found:
                        last_bus_points.append(add_found(point))
                        print("new bus detected, name: " + name[0])
                        cv2.imshow('new bus', frame[(int(point['ymin'] * args.height)):(int(point['ymax'] * args.height)),
                                                    (int(point['xmin'] * args.width)):(int(point['xmax'] * args.width))])
                        print("point: " + str(point))
   
            last_bus_points = list(filter(lambda item: item['found'], last_bus_points))
            if args.stream_out:
                print('Streaming elsewhere!')
            else:
                cv2.imshow('Video', frame)

        fps.update()

        # print('[INFO] change in time: {:.2f}'.format(time.time() - t))

        # time.sleep(0.02)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    video_capture.stop()
    cv2.destroyAllWindows()