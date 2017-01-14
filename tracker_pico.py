import sys
import cv2
import numpy as np
from cypico import detect_frontal_faces, remove_overlap

def main(argv):
    vid_in = cv2.VideoCapture(0)
    ret = True
    while (ret == True):
        ret, frame = vid_in.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        data = np.asarray(gray, dtype='uint8')
        det = detect_frontal_faces(data, confidence_cutoff=4.0, orientations = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
        chk = remove_overlap(det)
        for cur in chk:
            cv2.circle(frame, (int(cur[1][1]), int(cur[1][0])), int(cur[2]/2), (0,0,255), 3)
        cv2.imshow('Webcam', frame)
        cv2.waitKey(10)
    vid_in.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)