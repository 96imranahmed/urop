from __future__ import division
import sys
import cv2
import numpy as np
import time
import argparse
from cypico import detect_frontal_faces, remove_overlap
from primesense import openni2

open_ni = True
parser = argparse.ArgumentParser()
parser.add_argument("--openni", help="Enable read from Asus Action Cam", action='store_true', default=False)
args = parser.parse_args()

def main(args_in):
    vid_in = None
    open_ni = args_in.openni
    #OpenCV inits
    color_stream = None
    if open_ni:
        openni2.initialize('./redist/')
# try:
        dev = openni2.Device.open_any()
        color_stream = dev.create_color_stream()
        color_stream.start()
# except:
#     print('No valid device connected, switching to default camera...')
#     open_ni = False
#     vid_in = cv2.VideoCapture(0)

        # vid_in = cv2.VideoCapture(cv2.CAP_OPENNI2_ASUS + 1)
    else:
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
        frame = None
        if open_ni:
            frame_full = color_stream.read_frame()
            frame = np.ctypeslib.as_array(frame_full.get_buffer_as_triplet())
            frame = frame.reshape((frame_full.height,frame_full.width,3)) 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            _, frame = vid_in.read()
        if frame is None: 
            print('Error reading frame') 
            return
        r = 640.0 / frame.shape[1]
        try: 
            if r < 1:
                dim = (640, int(frame.shape[0] * r))
                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_LANCZOS4)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except:
            print('Error with resize/color conversion - make sure to unplug action cam before running without openni flag')
        data = np.asarray(gray, dtype='uint8')
        det = detect_frontal_faces(data, confidence_cutoff=3.5, orientations = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
                                    scale_factor=1.5, stride_factor=0.1, min_size=40)
        chk = remove_overlap(det)
        for cur in chk:
            cv2.circle(frame, (int(cur[1][1]), int(cur[1][0])), int(cur[2]/2), (0,0,255), 3)
        cv2.imshow('Webcam', frame)
        cv2.waitKey(10)
    vid_in.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(args)
