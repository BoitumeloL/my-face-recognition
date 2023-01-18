import dlib
import numpy as np
from numpy.linalg import norm
import cv2 as cv
from imutils import face_utils
from time import sleep

#load classifiers to detect a face

# initialize the threshold for the blink detection and the number of blinks
threshold = 0.2
eye_closed = False
blinks = 0

#use dlib face detector and facelandmark predictor models instead of classifiers
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

#instantiate video capture
video_capture = cv.VideoCapture(0)


#do calculations to get the middle of the line between the eyes before starting video capture
def mid_line_distance(p1 ,p2, p3, p4):
    """compute the euclidean distance between the midpoints of two sets of points"""
    p5 = np.array([int((p1[0] + p2[0])/2), int((p1[1] + p2[1])/2)])
    p6 = np.array([int((p3[0] + p4[0])/2), int((p3[1] + p4[1])/2)])

    return norm(p5 - p6)

def aspect_ratio(landmarks, eye_range):
    # Get the eye coordinates
    eye = np.array(
        [np.array([landmarks.part(i).x, landmarks.part(i).y]) 
         for i in eye_range]
        )
    # compute the euclidean distances
    B = norm(eye[0] - eye[3])
    A = mid_line_distance(eye[1], eye[2], eye[5], eye[4])
    # Use the euclidean distance to compute the aspect ratio
    ear = A / B
    return ear

#start video capture and check if camera is able to open
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #turn picture into grayscale picture
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #detect face in grayscale
    rects = detector(gray, 0)

    # loop over the face detections
    for (i,rect) in enumerate(rects):

        
        landmarks = predictor(gray, rect)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
        # Use the coordinates of each eye to compute the eye aspect ratio.
        left_aspect_ratio = aspect_ratio(landmarks, range(42, 48))
        right_aspect_ratio = aspect_ratio(landmarks, range(36, 42))
        ear = (left_aspect_ratio + right_aspect_ratio) / 2.0

        # if the eye aspect ratio is below the blink threshold, set the eye_closed flag to True.
        if ear < threshold:
            eye_closed = True
        # if the eye aspect ratio is above the blink threshold and 
        # the eye_closed flag is True, increment the number of blinks.
        elif ear >= threshold and eye_closed:
            blinks += 1
            eye_closed = False

        # draw the landmarks on the frame
        for n in range(36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv.circle(frame, (x, y), 4, (255, 0, 0), -1)

        # draw the eye aspect ratio and the number of blinks on the frame
        cv.putText(frame, "Blinks: {}".format(blinks), (10, 30),
            cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
       
    cv.imshow("Frame", frame)
    if cv.waitKey(1) == ord("q"):
        break
    #if cv.waitKey(1) == ord("q") and blinks == 4:
    #if cv.waitKey(1) == ord("q") and blinks == 4:
    #    cv.putText(frame, "This is the real user", (10, 30),
    #        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #cv.waitKey(120000)
        
        
video_capture.release()
cv.destroyAllWindows()



    