import numpy as np
import cv2 as cv

cap = cv.VideoCapture("https://58cc2dce193dd.streamlock.net/live/Westlake_N_Dexter_NS.stream/playlist.m3u8")

while True:
    ret, img = cap.read()

    if not ret:
        break

    cv.imshow('img', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
# import cv2
# import numpy
# from subprocess import Popen, PIPE

# VIDEO_URL = "https://58cc2dce193dd.streamlock.net/live/Westlake_N_Dexter_NS.stream/playlist.m3u8"


# pipe = Popen([ "ffmpeg", "-i", VIDEO_URL,
#            "-loglevel", "quiet", # no text output
#            "-an",   # disable audio
#            "-f", "image2pipe",
#            "-pix_fmt", "bgr24",
#            "-vcodec", "rawvideo", "-"],
#            stdin = PIPE, stdout = PIPE)
# while True:
#     raw_image = pipe.stdout.read(432*240*3) # read 432*240*3 bytes (= 1 frame)
#     image =  numpy.fromstring(raw_image, dtype='uint8').reshape((240,432,3))
#     cv2.imshow("GoPro",image)
#     if cv2.waitKey(5) == 27:
#         break
# cv2.destroyAllWindows()
