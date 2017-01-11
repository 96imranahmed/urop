import sys
import cv2

def main(argv):
    vid_in = cv2.VideoCapture(0)
    eye_cascade = cv2.CascadeClassifier('haar_eye.xml')
    ret = True
    while (ret == True):
        ret, frame = vid_in.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in eyes:
            cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0),2)
        cv2.imshow('Webcam', frame)
        cv2.waitKey(10)
    vid_in.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)