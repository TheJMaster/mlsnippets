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
PATH_TO_CKPT = os.path.join(CWD_PATH, 'models/ssd_mobilenet/v1_coco/frozen_inference_graph.pb')

# Dimensions required/encourage by Tensorflow Object Detection API model.
DETECT_IMG_DIMS = (300, 300)

# Max queue size for detection and tracking input/output threads.
MAX_QUEUE_SIZE = 3

NEXT_BOX_ID = 0  # ID to use for next new bounding box.

# Note that each index of the following list refers to the same bounding box.
TRACKED_BOX_IDS = []  # IDs of bounding boxes being tracked in the current frame.
UNDETECTED_COUNTS = []  # Counts of how many times a tracked box has gone undetected.
COLORS = []  # Colors for the tracked bounding box.

UNDETECTED_THRESHOLD = 20 # Number of frames to allow for undetected.
EQUALITY_THRESHOLD = 20 # Equality threshold for distance between bounding box edges.

# If specified, output video results.
OUTPUT_FRAME_RATE = 30
OUTPUT_DIMS = (720, 480)


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
        for (image, boxes, scores, classes) in zip(batch, boxes, scores, classes):
            # Flatten all results for filtering.
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)

            # Filter for a particular class.
            # Note: classes == 3 or classes == 6 (bus)
            indicies = np.argwhere(classes == 3)
            boxes = np.squeeze(boxes[indicies])
            scores = np.squeeze(scores[indicies])

            # Remove all instances of classes with a low confidence score (< 0.5).
            indicies = np.argwhere(scores > 0.5)
            boxes = np.squeeze(boxes[indicies])

            # Set array to empty if ndim == 0
            if boxes.ndim == 0:
                boxes = np.array([])

            # Convert all of the boxes to match style from RE-3 tracker.
            height = image.shape[0]
            width = image.shape[1]
            detected_boxes = []
            for box in list(boxes):
                if isinstance(box, np.ndarray):
                    detected_boxes.append([box[1]*width, box[0]*height,
                                           box[3]*width, box[2]*height])
            batch_results.append(np.array(detected_boxes))

        # Return bounding boxes for batch via output queue.
        output_queue.put(np.array(batch_results))

    # Close session if loop is killed.
    sess.close()


def track_worker(input_queue, output_queue, gpu_id):
    """Runs Re3 tracker on images from input queue, passing bounding boxes to output queue."""
    sys.path.insert(1, '/home/jtjohn24/mlsnippets/object_detector_app/re3-tensorflow')
    from tracker import re3_tracker  # pylint: disable-msg=import-outside-toplevel, import-error

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
            output_queue.put([tracked_boxes])
        else:
            output_queue.put(tracked_boxes)


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


def find_objects(image_np, detect_input_queues, detect_output_queues, track_input_queue,
                 track_output_queue, x_split, y_split, tracker_only):
    # pylint: disable-msg=too-many-locals
    """Run bus detection using on image, tracking existing buses using input/output queues.
       Returns annotated image and detected box list through output queue. """
    global NEXT_BOX_ID  # pylint: disable-msg=global-statement

    # Check if we can just run the tracker on this frame.
    if tracker_only:
        track_input_queue.put((image_np, TRACKED_BOX_IDS, None))
        tracked_boxes = track_output_queue.get()

        for idx, bounding_box in enumerate(tracked_boxes):
            cv2.rectangle(image_np,
                          (int(bounding_box[0]), int(bounding_box[1])),
                          (int(bounding_box[2]), int(bounding_box[3])),
                          COLORS[idx], 2)

        return image_np, []

    # Check that splitting frame with current parameters is safe.
    if image_np.shape[0] % x_split != 0:
        print("ERROR: {} image width not divisible by x-split {}".format(image_np.shape[0],
                                                                         x_split))
        return image_np, []
    if image_np.shape[1] % y_split != 0:
        print("ERROR: {} image width not divisible by y-split {}".format(image_np.shape[1],
                                                                         y_split))
        return image_np, []

    # Split image along x axis the correct number of times.
    x_sub_imgs = np.split(image_np, x_split)
    y_sub_imgs = np.array([np.hsplit(img, y_split) for img in x_sub_imgs])
    y_sub_imgs = y_sub_imgs.reshape(x_split*y_split, int(image_np.shape[0]/x_split),
                                    int(image_np.shape[1]/y_split), 3)
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
            x_delta = (boxes_idx % y_split) * x_shift
            y_delta = ((int(boxes_idx / x_split)) % y_split) * y_shift
            detect_result_boxes[boxes_idx][box_idx][0] += x_delta
            detect_result_boxes[boxes_idx][box_idx][1] += y_delta
            detect_result_boxes[boxes_idx][box_idx][2] += x_delta
            detect_result_boxes[boxes_idx][box_idx][3] += y_delta

    # Find unique boxes from all images in the batch.
    detected_boxes = common_boxes(detect_result_boxes)
    for boxes in detect_result_boxes:
        detected_boxes = add_all_not_present(boxes, detected_boxes)

    # Find all of the already tracked bounding boxes.
    track_input_queue.put((image_np, TRACKED_BOX_IDS, None))
    tracked_boxes = track_output_queue.get()

    # Find any boxes that have been detected but not tracked, and vice versa.
    detected_untracked_boxes, undetected_tracked_boxes = exclusive_boxes(tracked_boxes,
                                                                         detected_boxes)
    detected_untracked_boxes = np.array(detected_untracked_boxes)
    undetected_tracked_boxes = np.array(undetected_tracked_boxes)

    # Remove tracked but undetected boxes from tracking.
    for i, box in reversed(list(enumerate(tracked_boxes))):
        if np.transpose(box) in undetected_tracked_boxes:
            UNDETECTED_COUNTS[i] = UNDETECTED_COUNTS[i] + 1
            if UNDETECTED_COUNTS[i] == UNDETECTED_THRESHOLD:
                del TRACKED_BOX_IDS[i]
                del COLORS[i]
                del UNDETECTED_COUNTS[i]
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

    if init_bounding_boxes == {}:
        init_bounding_boxes = None

    # Run tracker again for all objects to get final bounding boxes.
    track_input_queue.put((image_np, TRACKED_BOX_IDS, init_bounding_boxes))
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
                      COLORS[idx], 2)

    return (image_np, detected_boxes)


def draw_worker(input_q, output_q, num_detect_workers, track_gpu_id, x_split, y_split, detect_rate):
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
    while True:
        fps.update()
        frame = input_q.get()
        frame_num += 1
        if np.shape(frame) == ():
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracker_only = (frame_num % detect_rate) != 0
        img, boxes = find_objects(frame_rgb, detect_worker_input_queues,
                                  detect_worker_output_queues, track_input_queue,
                                  track_output_queue, x_split, y_split, tracker_only)
        output_q.put((img, boxes))
    fps.stop()
    track.join()


def main(args):
    """Sets up object detection according to the provided args."""

    # If no number of workers are specified, use all available GPUs
    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)
    draw_proc = Process(target=draw_worker, args=(input_q, output_q, args.detect_workers,
                                                  args.track_gpu_id, args.x_split, args.y_split,
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
                                    OUTPUT_DIMS)

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
    PARSER.add_argument('-x-split', dest="x_split", type=int, default=1,
                        help='Number of frame columns to create before running detection')
    PARSER.add_argument('-y-split', dest="y_split", type=int, default=1,
                        help='Number of frame rows to create before running detection')
    PARSER.add_argument('-dr', '-detect-rate', dest="detect_rate", type=int, default=1,
                        help='Run detection every detect rate frames.')
    PARSER.add_argument('-out', '-video-out', dest="video_out_fname", type=str, default=None,
                        help='Name of video out file')
    PARSER.add_argument('-show-frame', dest="show_frame", type=bool, default=False,
                        help='Specifies whether the application should show the annotated frame')
    main(PARSER.parse_args())
