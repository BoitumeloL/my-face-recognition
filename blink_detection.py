import dlib
import cv2
import numpy as np
from numpy.linalg import norm


# initialize the threshold for the blink detection and the number of blinks
threshold = 0.2
eye_closed = False
blinks = 0

# initialize dlib's face detector and face landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# iniatialize the video capture object.
vs = cv2.VideoCapture(0)

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

while True:
    # grab the frame from the video capture object, 
    # resize it, and convert it to grayscale.
    _, frame = vs.read()
    frame = cv2.resize(frame, (600, 450))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

# loop over the face detections
    for rect in rects:
        landmarks = predictor(gray, rect)

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
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)

        # draw the eye aspect ratio and the number of blinks on the frame
        cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Eye Aspect Ratio: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()