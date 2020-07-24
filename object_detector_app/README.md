# Object-Detector-App

A real-time object recognition and tracking application using [Google's TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), [OpenCV](http://opencv.org/) and [Re3-tracker](https://github.com/danielgordon10/re3-tensorflow).

## Getting Started
1. `conda env create -f environment.yml`
2.  run `./models/yolov4/download_models.sh` to download YOLOv4 models.
2. `python3 object_detection_app.py`
    Optional arguments (default value):
    * Device index of the camera `--source=0`
    * Width of the frames in the video stream `--width=480`
    * Height of the frames in the video stream `--height=360`
    * Number of workers `--num-workers=1`
    * Size of the queues `--queue-size=1`
    * Get video from HLS stream rather than webcam `--stream-input=http://somertmpserver.com/hls/live.m3u8`
    * Send stream to livestreaming server `--stream-output=--stream=http://somertmpserver.com/hls/live.m3u8`
    * Get video from local file `--path=None`
    * Set GPU ID for Re3-tracker `--tracker-gpu-id=0`
    * Set number of rows to split frame into `--rows=1`
    * Set number of cols to split frame into `--cols=1`
    * Detect new objects every X frames `--detect-rate=1`
    * Save video annotated with bounding boxes `--video-out=None`
    * Show annotated frames in real time `--show-frame`

## Benchmarking
1. Download test input and save in root directory.
2. Update globals in `speed.py`, `speed_sampling_rate.py`, and `accuracy.py` to point to test data.
3. Run `python3 <benchmark>.py` and open `<benchmark>_out.csv` for results.

## Requirements
- [Anaconda / Python 3.5](https://www.continuum.io/downloads)
- [TensorFlow 1.2](https://www.tensorflow.org/)
- [OpenCV 3.0](http://opencv.org/)

## Notes
- OpenCV 3.1 might crash on OSX after a while, so that's why I had to switch to version 3.0. See open issue and solution [here](https://github.com/opencv/opencv/issues/5874).
- Moving the `.read()` part of the video stream in a multiple child processes did not work. However, it was possible to move it to a separate thread.

## Copyright
See [LICENSE](LICENSE) for details.

Original work inspired by [Dat Tran](http://www.dat-tran.com/).
