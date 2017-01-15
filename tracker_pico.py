from __future__ import division
import sys
import cv2
import numpy as np
import time
from cypico import detect_frontal_faces, remove_overlap


def main(argv):
    vid_in = cv2.VideoCapture(0)
    ret = True
    cur_time = time.time()
    loop_lim = 10
    cur_loop = 0
    while (ret == True):
        cur_loop+=1
        if cur_loop > loop_lim:
            cur_loop = 0
            chk = time.time() - cur_time
            cur_time = time.time()
            print(loop_lim/chk)
        ret, frame = vid_in.read()
        r = 640 / frame.shape[1]
        dim = (640, int(frame.shape[0] * r))
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_LANCZOS4)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        data = np.asarray(gray, dtype='uint8')
        det = detect_frontal_faces(data, confidence_cutoff=4.0, orientations = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
                                    scale_factor=1.2, stride_factor=0.1, min_size=40)
        chk = remove_overlap(det)
        for cur in chk:
            cv2.circle(resized, (int(cur[1][1]), int(cur[1][0])), int(cur[2]/2), (0,0,255), 3)
        cv2.imshow('Webcam', resized)
        cv2.waitKey(10)
    vid_in.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
