import numpy
import cv2
import random
import time
import subprocess as sp
import json
from threading import Thread
from streamurlgenerator import get_camera_urls

"""

https://58cc2dce193dd.streamlock.net/live/12_NE_50_EW.stream/chunklist_w403604642.m3u8
https://58cc2dce193dd.streamlock.net/live/Broadway_E_Pike_EW.stream/chunklist_w880989541.m3u8

"""

class HLSVideoStream:
    def __init__(self, src):
        # initialize the video camera stream and read the first frame
        # from the stream

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

        FFMPEG_BIN = "ffmpeg"

        metadata = {}

        while "streams" not in metadata.keys():
            
            print('ERROR: Could not access stream. Trying again.')

            info = sp.Popen(["ffprobe", 
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams", src],
            stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
            out, err = info.communicate(b"ffprobe -v quiet -print_format json -show_format -show_streams http://52.91.28.88:8080/hls/live.m3u8")

            metadata = json.loads(out.decode('utf-8'))
            time.sleep(5)


        print('SUCCESS: Retrieved stream metadata.')

        # [print('%s --> %s' % (str(k), str(v)) for k,v in metadata['streams'][0])]
        print(metadata["streams"][0])

        # TODO: Figure out why this is 1 and not 0
        self.WIDTH = metadata["streams"][1]["width"]
        self.HEIGHT = metadata["streams"][1]["height"]

        self.pipe = sp.Popen([ FFMPEG_BIN, "-i", src,
                 "-loglevel", "quiet", # no text output
                 "-an",   # disable audio
                 "-f", "image2pipe",
                 "-pix_fmt", "bgr24",
                 "-vcodec", "rawvideo", "-"],
                 stdin = sp.PIPE, stdout = sp.PIPE)
        print('WIDTH: ', self.WIDTH)

        raw_image = self.pipe.stdout.read(self.WIDTH*self.HEIGHT*3) # read 432*240*3 bytes (= 1 frame)
        self.frame =  numpy.fromstring(raw_image, dtype='uint8').reshape((self.HEIGHT,self.WIDTH,3))
        self.grabbed = self.frame is not None


    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        # if the thread indicator variable is set, stop the thread

        while True:
            if self.stopped:
                return

            raw_image = self.pipe.stdout.read(self.WIDTH*self.HEIGHT*3) # read 432*240*3 bytes (= 1 frame)
            self.frame =  numpy.fromstring(raw_image, dtype='uint8').reshape((self.HEIGHT,self.WIDTH,3))
            self.grabbed = self.frame is not None

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

def main():
    urls = get_camera_urls()
    # stream = HLSVideoStream(src="https://58cc2dce193dd.streamlock.net/live/1_Seneca_EW.stream/playlist.m3u8").start()
    # stream = HLSVideoStream(src="https://58cc2dce193dd.streamlock.net/live/12_S_Boren_NS.stream/playlist.m3u8").start()
    print(random.choice(urls))
    stream = HLSVideoStream(src=random.choice(urls)).start()

    print('hi')

    while True:
        frame = stream.read()
        output_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video', output_rgb)

        # time.sleep(0.3)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()