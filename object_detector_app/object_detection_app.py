#!/usr/bin/python3
"""Object detection and tracking application."""

import argparse
from multiprocessing import Queue, Process
import os
import random
import sys
import time

import numpy as np

import cv2
from utils.app_utils import FPS, LocalVideoStream, HLSVideoStream

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH, 'models/yolov4/yolov4_320_norm.pb')

# Path to Re3 tracker library. This is the model used for tracking objects after detection.
PATH_TO_RE3 = os.path.join(CWD_PATH, 're3-tensorflow')

# Dimensions required/encourage by the YOLOv4 model.
DETECT_IMG_DIMS = (320, 320)

# Max queue size for detection and tracking input/output threads.
MAX_QUEUE_SIZE = 3

UNDETECTED_THRESHOLD = 30 # Number of frames to allow for undetected.
EQUALITY_THRESHOLD = 10 # Equality threshold for distance between bounding box edges.

OUTPUT_FRAME_RATE = 30  # FPS of output video.


# UTILITY FUNCTIONS


def approx_eq(num_one, num_two):
    """Returns true iff the two numbers are within an equality threshold of each other."""
    return abs(num_one-num_two) < EQUALITY_THRESHOLD


def contains_box(box, boxes):
    """Returns true iff the list of boxes (approx) contains the provided box.
       Two boxes are considered equal iff two edges are within EQUALITY_THRESHOLD
       distance of each other. """
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

    return (detected_untracked_boxes, undetected_tracked_boxes)


def deduplicate_boxes(boxes):
    """ Remove duplicate bounding boxes from list. """
    res_boxes = []
    for box in boxes:
        if not contains_box(box, res_boxes):
            res_boxes.append(box)
    return res_boxes


def common_boxes(all_boxes):
    """Returns bounding boxes present in a majority of sets."""
    res_boxes = []
    for boxes in all_boxes:
        for box in boxes:
            if not contains_box(box, res_boxes):
                count = 0
                for other_boxes in all_boxes:
                    if contains_box(box, other_boxes):
                        count = count + 1
                if count > (len(all_boxes) / 2):
                    res_boxes.append(box)
    return res_boxes


def add_all_not_present(source, target):
    """ Adds bounding boxes from source not in target. """
    for box in source:
        if not contains_box(box, target):
            target.append(box)
    return target


def resize_all(boxes, orig_h, orig_w, new_h, new_w):
    """ Resize all boxes to become the target height and width. """
    res = []
    if boxes == []:
        return np.array([])
    for box in boxes:
        if isinstance(box, np.ndarray):
            res.append([box[0]/orig_w*new_w, box[1]/orig_h*new_h,
                        box[2]/orig_w*new_w, box[3]/orig_h*new_h])
    return np.array(res)


def run_detect_batch(detect_input_queues, detect_output_queues, batch):
    """ Runs detection on the batch of images across all threads. """
    for input_queue in detect_input_queues:
        input_queue.put(batch)
    detect_results = []
    for output_queue in detect_output_queues:
        detect_results.append(output_queue.get())
    return detect_results


def is_valid_coord(coord, max_coord):
    """ Returns true iff coord >= 0 and coord <= max_coord  """
    return coord >= 0 and coord <= max_coord


def compute_iou(box, boxes, box_area, boxes_area):
    # this is the iou of the box against all other boxes
    assert boxes.shape[0] == boxes_area.shape[0]
    # get all the origin-ys
    # push up all the lower origin-xs, while keeping the higher origin-xs
    ys1 = np.maximum(box[0], boxes[:, 0])
    # get all the origin-xs
    # push right all the lower origin-xs, while keeping higher origin-xs
    xs1 = np.maximum(box[1], boxes[:, 1])
    # get all the target-ys
    # pull down all the higher target-ys, while keeping lower origin-ys
    ys2 = np.minimum(box[2], boxes[:, 2])
    # get all the target-xs
    # pull left all the higher target-xs, while keeping lower target-xs
    xs2 = np.minimum(box[3], boxes[:, 3])
    # each intersection area is calculated by the
    # pulled target-x minus the pushed origin-x
    # multiplying
    # pulled target-y minus the pushed origin-y
    # we ignore areas where the intersection side would be negative
    # this is done by using maxing the side length by 0
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    # each union is then the box area
    # added to each other box area minusing their intersection calculated above
    unions = box_area + boxes_area - intersections
    # element wise division
    # if the intersection is 0, then their ratio is 0
    ious = intersections / unions
    return ious


def non_max_suppression(boxes, scores, threshold):
    assert boxes.shape[0] == scores.shape[0]

    # bottom-left origin
    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]
    # top-right s
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]
    # box coordinate ranges are inclusive-inclusive
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []

    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                           areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])
        # if there are no more scores_index
        # then we should pop it
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index)


def process_single_output(boxes, scores, score_threshold=0.4, iou_threshold=0.5, num_classes=80):
    mask = scores >= 0.4

    boxes_ = []
    scores_ = []
    classes_ = []

    # normalize boxes to [0-1] range
    boxes /= float(320) + 0.5

    for c in range(num_classes):
        class_boxes = boxes[mask[:,c]]
        class_box_scores = scores[:, c][mask[:,c]]
        nms_index = non_max_suppression(class_boxes, class_box_scores, iou_threshold)
        if len(nms_index) > 0:
            class_boxes = class_boxes[nms_index]
            class_box_scores = class_box_scores[nms_index]
            classes = np.ones_like(class_box_scores, dtype=np.int32) * c
            boxes_.extend(class_boxes)
            scores_.extend(class_box_scores)
            classes_.extend(classes)

    return boxes_, scores_, classes_


# WORKER FUNCTIONS


def detect_worker(input_queue, output_queue, gpu_id):
    """Runs object detection on the provided numpy image using a frozen detection model."""
    import tensorflow as tf
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Setup tesnorflow session
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=detection_graph)

    while True:
        images = input_queue.get()

        # Expand dimensions to create batch since the model expects images to
        # have shape: [1, Height, Width, 3]
        batch = None
        if not isinstance(images, tuple):
            batch = np.expand_dims(images, axis=0)
        else:
            images_expanded = [np.expand_dims(image, axis=0) for image in images]
            batch = np.concatenate(tuple(images_expanded))

        # Grab the tensor to populate with the image batch.
        input_tensor = detection_graph.get_tensor_by_name('inputs:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('boxes:0')

        # Each score represent how level of confidence for each class of the objects.
        scores = detection_graph.get_tensor_by_name('scores:0')

        # Run initial detection.
        boxes, scores = sess.run([boxes, scores], feed_dict={input_tensor: batch})

        batch_results = []
        for (image, boxes, scores) in zip(batch, boxes, scores):
            boxes_, scores_, classes_ = process_single_output(boxes, scores)
            h, w = image.shape[:2]
            detected_boxes = []
            for box, score, class_idx in zip(boxes_, scores_, classes_):
                top, left, bottom, right = box

                top = int(top * h)
                left = int(left * w)
                bottom = int(bottom * h)
                right = int(right * w)

                detected_boxes.append([left, top, right, bottom])
            batch_results.append(np.array(detected_boxes))

        # Return bounding boxes for batch via output queue.
        output_queue.put(np.array(batch_results))

    # Close session if loop is killed.
    sess.close()


def track_worker(input_queue, output_queue, gpu_id):
    """Runs Re3 tracker on images from input queue, passing bounding boxes to output queue."""
    sys.path.insert(1, PATH_TO_RE3)
    from tracker import re3_tracker

    tracker = re3_tracker.Re3Tracker(gpu_id)
    while True:
        image_np, box_ids, init_bounding_boxes = input_queue.get()
        tracked_boxes = np.empty(0)
        if len(box_ids) > 1:
            tracked_boxes = tracker.multi_track(box_ids, image_np, init_bounding_boxes)
        elif len(box_ids) == 1:
            init_bounding_box = None
            if init_bounding_boxes is not None:
                init_bounding_box = init_bounding_boxes[box_ids[0]]
            tracked_boxes = tracker.track(box_ids[0], image_np, init_bounding_box)

        if tracked_boxes.ndim == 0 or tracked_boxes.size == 0:
            output_queue.put([])
        elif tracked_boxes.ndim == 1:
            # If only a single box is detected, Re3 returns only that box. Convert to two dim
            # result to match.
            output_queue.put([tracked_boxes])
        else:
            output_queue.put(tracked_boxes)


def find_objects(image_np, detect_input_queues, detect_output_queues, track_input_queue,
                 track_output_queue, rows, cols, tracked_box_ids, undetected_counts,
                 box_colors, next_box_id, tracker_only):
    """Run bus detection using on image, tracking existing buses using input/output queues.
       Returns annotated image and detected box list through output queue. """

    # Check if we can just run the tracker on this frame.
    if tracker_only:
        track_input_queue.put((image_np, tracked_box_ids, None))
        tracked_boxes = track_output_queue.get()

        for idx, bounding_box in enumerate(tracked_boxes):
            cv2.rectangle(image_np,
                          (int(bounding_box[0]), int(bounding_box[1])),
                          (int(bounding_box[2]), int(bounding_box[3])),
                          box_colors[idx], 2)

        return image_np, []

    # Check that splitting frame with current parameters is safe.
    if image_np.shape[0] % rows != 0:
        print("ERROR: {} image width not divisible by x-split {}".format(image_np.shape[0],
                                                                         rows))
        return image_np, []
    if image_np.shape[1] % cols != 0:
        print("ERROR: {} image width not divisible by y-split {}".format(image_np.shape[1],
                                                                         cols))
        return image_np, []

    # Split image along x axis the correct number of times.
    x_sub_imgs = np.split(image_np, rows)
    y_sub_imgs = np.array([np.hsplit(img, cols) for img in x_sub_imgs])

    # reshape to [num sub imgs, height, width, 3] to match model input shape.
    y_sub_imgs = y_sub_imgs.reshape(rows*cols, int(image_np.shape[0]/rows),
                                    int(image_np.shape[1]/cols), 3)
    y_sub_imgs = list(y_sub_imgs)
    all_imgs = y_sub_imgs + [image_np]

    # Resize images to SSD expected img size
    resized_imgs = [cv2.resize(img, dsize=DETECT_IMG_DIMS,
                               interpolation=cv2.INTER_LINEAR) for img in all_imgs]

    # Run foreground/background detection batch.
    detect_results = run_detect_batch(detect_input_queues, detect_output_queues,
                                      tuple(resized_imgs))

    # Combine images from different detectors.
    detect_result_boxes = [[] for _ in range(len(all_imgs))]
    for detect_proc_result in detect_results:  # Results from a single detection processs
        for image_idx, image_results in enumerate(detect_proc_result):  # Results for a single image
            detect_result_boxes[image_idx].extend(image_results)

    # Resize bounding boxes to match original image sizes.
    detect_result_boxes = [deduplicate_boxes(boxes) for boxes in detect_result_boxes]
    detect_result_boxes = [resize_all(detect_result_boxes[i], DETECT_IMG_DIMS[0],
                                      DETECT_IMG_DIMS[1], all_imgs[i].shape[0],
                                      all_imgs[i].shape[1]) for i in range(len(all_imgs))]

    # Shift dims of boxes to position relative to the entire image.
    x_shift = all_imgs[0].shape[1]
    y_shift = all_imgs[0].shape[0]
    for boxes_idx in range(len(detect_result_boxes)-1):  # boxes_idx corresponds to a split
        for box_idx in range(len(detect_result_boxes[boxes_idx])):  # box_idx to a box in a split
            x_delta = (boxes_idx % cols) * x_shift
            y_delta = (int(boxes_idx / cols)) * y_shift
            detect_result_boxes[boxes_idx][box_idx][0] += x_delta
            detect_result_boxes[boxes_idx][box_idx][1] += y_delta
            detect_result_boxes[boxes_idx][box_idx][2] += x_delta
            detect_result_boxes[boxes_idx][box_idx][3] += y_delta

    # Find unique boxes from all images in the batch.
    detected_boxes = common_boxes(detect_result_boxes)
    for boxes in detect_result_boxes:
        detected_boxes = add_all_not_present(boxes, detected_boxes)

    # Find all of the already tracked bounding boxes.
    track_input_queue.put((image_np, tracked_box_ids, None))
    tracked_boxes = track_output_queue.get()

    # Find any boxes that have been detected but not tracked, and vice versa.
    detected_untracked_boxes, undetected_tracked_boxes = exclusive_boxes(tracked_boxes,
                                                                         detected_boxes)
    detected_untracked_boxes = np.array(detected_untracked_boxes)
    undetected_tracked_boxes = np.array(undetected_tracked_boxes)

    # Remove tracked but undetected boxes from tracking.
    for i, box in reversed(list(enumerate(tracked_boxes))):
        if np.transpose(box) in undetected_tracked_boxes:
            undetected_counts[i] += 1
            if undetected_counts[i] == UNDETECTED_THRESHOLD:
                del tracked_box_ids[i], undetected_counts[i], box_colors[i]
        else:
            undetected_counts[i] = 0

    # Add untracked but detected boxes to be detected.
    init_bounding_boxes = {}
    for box in detected_untracked_boxes:
        box_id = "box_{}".format(next_box_id)
        next_box_id += 1
        tracked_box_ids.append(box_id)
        init_bounding_boxes[box_id] = box

        hue = random.uniform(0, 1) * 255
        color = cv2.cvtColor(np.uint8([[[hue, 128, 200]]]), cv2.COLOR_HSV2RGB).squeeze().tolist()
        box_colors.append(color)

        undetected_counts.append(0)

    if init_bounding_boxes == {}:
        init_bounding_boxes = None

    # Run tracker again for all objects to get final bounding boxes.
    track_input_queue.put((image_np, tracked_box_ids, init_bounding_boxes))
    bounding_boxes = track_output_queue.get()

    # Display detected boxes in gray.
    for detected_box in detected_boxes:
        cv2.rectangle(image_np,
                      (int(detected_box[0]), int(detected_box[1])),
                      (int(detected_box[2]), int(detected_box[3])),
                      [204, 204, 204], 2)

    # Display tracked boxes on the image each in a different color.
    for idx, bounding_box in enumerate(bounding_boxes):
        # Tracker on occasion returns negative or out of bounds coordinate numbers.
        # Be defensive against such bounding boxes here by filtering out erroneous boxes.
        if (not is_valid_coord(bounding_box[0], image_np.shape[0]) or
                not is_valid_coord(bounding_box[1], image_np.shape[1]) or
                not is_valid_coord(bounding_box[2], image_np.shape[0]) or
                not is_valid_coord(bounding_box[3], image_np.shape[1])):
            continue

        cv2.rectangle(image_np,
                      (int(bounding_box[0]), int(bounding_box[1])),
                      (int(bounding_box[2]), int(bounding_box[3])),
                      box_colors[idx], 2)

    return (image_np, detected_boxes, next_box_id)


def draw_worker(input_q, output_q, num_detect_workers, track_gpu_id, rows, cols, detect_rate):
    """Detect and track buses from input and save image annotated with bounding boxes to output."""

    # Start detection processes.
    detect_worker_input_queues = [Queue(maxsize=MAX_QUEUE_SIZE)]*num_detect_workers
    detect_worker_output_queues = [Queue(maxsize=MAX_QUEUE_SIZE)]*num_detect_workers
    for worker_id in range(num_detect_workers):
        # TODO: Consider adding support for the case where worker_id != GPU_ID.
        Process(target=detect_worker, args=(detect_worker_input_queues[worker_id],
                                            detect_worker_output_queues[worker_id],
                                            worker_id)).start()

    # Start tracking process.
    track_input_queue = Queue(maxsize=MAX_QUEUE_SIZE)
    track_output_queue = Queue(maxsize=MAX_QUEUE_SIZE)
    track = Process(target=track_worker, args=(track_input_queue, track_output_queue, track_gpu_id))
    track.start()

    # Annotate all new frames.
    fps = FPS().start()
    frame_num = -1
    tracked_box_ids = []
    undetected_counts = []
    box_colors = []
    next_box_id = 0
    while True:
        fps.update()
        frame = input_q.get()
        frame_num += 1
        if np.shape(frame) == ():
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracker_only = (frame_num % detect_rate) != 0
        img, boxes, next_box_id = find_objects(frame_rgb,
                                               detect_worker_input_queues, detect_worker_output_queues,
                                               track_input_queue, track_output_queue,
                                               rows, cols,
                                               tracked_box_ids,
                                               undetected_counts,
                                               box_colors,
                                               next_box_id,
                                               tracker_only)
        output_q.put((img, boxes))
    fps.stop()
    track.join()


# MAIN FUNCTIONS


def main(args):
    """Sets up object detection according to the provided args."""

    # If no number of workers are specified, use all available GPUs
    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    draw_proc = Process(target=draw_worker, args=(input_q, output_q, args.detect_workers,
                                                  args.track_gpu_id, args.rows, args.cols,
                                                  args.detect_rate,))
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

    video_out = None
    if args.video_out_fname is not None:
        video_out = cv2.VideoWriter(args.video_out_fname,
                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                    OUTPUT_FRAME_RATE,
                                    (video_capture.WIDTH, video_capture.HEIGHT))

    fps = FPS().start()
    while True:  # fps._numFrames < 120
        try:
            frame = video_capture.read()
            input_q.put(frame)
            start_time = time.time()

            output_rgb = cv2.cvtColor(output_q.get()[0], cv2.COLOR_RGB2BGR)
            if args.show_frame:
                cv2.imshow('Video', output_rgb)
            if video_out is not None:
                video_out.write(output_rgb)
            fps.update()

            print('[INFO] elapsed time: {:.2f}'.format(time.time() - start_time))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except (KeyboardInterrupt, SystemExit):
            if video_out is not None:
                video_out.release()
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    if video_out is not None:
        video_out.release()
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
    PARSER.add_argument('-w', '--workers', dest="detect_workers", type=int, default=1,
                        help='Number of detection workers')
    PARSER.add_argument('-tracker-gpu-id', dest="track_gpu_id", type=int, default=0,
                        help='GPU ID to use for tracker')
    PARSER.add_argument('-rows', dest="rows", type=int, default=1,
                        help='Number of frame rows to create before running detection')
    PARSER.add_argument('-cols', dest="cols", type=int, default=1,
                        help='Number of frame columns to create before running detection')
    PARSER.add_argument('-dr', '-detect-rate', dest="detect_rate", type=int, default=1,
                        help='Run detection every detect rate frames.')
    PARSER.add_argument('-out', '-video-out', dest="video_out_fname", type=str, default=None,
                        help='Name of video out file')
    PARSER.add_argument('-show-frame', dest="show_frame", type=bool, default=False,
                        help='Specifies whether the application should show the annotated frame')
    main(PARSER.parse_args())
