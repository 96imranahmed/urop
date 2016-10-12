import cv2
import sys
import numpy as np
if sys.version[0] == 2:
    from future import __division__
def main(argv):
    vid_in = cv2.VideoCapture("dock_camimage.mp4")
    print(vid_in.get(cv2.CAP_PROP_FPS))
    ret = True
    while (ret == True):
        ret, frame = vid_in.read()
        cv2.imshow('Video Stream', frame)
        cv2.waitKey(10)
    vid_in.release()
    cv2.destroyAllWindows()