import cv2
import numpy as np
from components import array_op
from collections import namedtuple

Point = namedtuple("Point", ['x', 'y'])
Face = namedtuple("Face", ['image', 'x', 'y', 'width', 'height'])
# get the first camera
cap = cv2.VideoCapture(0)

# get the first frame of the img to obtain the size of the frame
success, frame = cap.read()

face_color = (199, 199, 199)
eye_color = (100, 100, 100)

face_cascade = cv2.CascadeClassifier("face.xml")
left_eye_cascade = cv2.CascadeClassifier("left_eye.xml")
right_eye_cascade = cv2.CascadeClassifier("right_eye.xml")

print(1)

# get the source frame
success, frame = cap.read()

while success:

    # get new frame and the frame size
    success, frame = cap.read()
    size = frame.shape[:2]

    # to Gray
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.equalizeHist(image, image)  # 灰度图像进行直方图等距化

    # get the result
    face_rectangles = face_cascade.detectMultiScale(image, 1.1, 5)
    faces = list()
    filtered_eye_rectangles = list()

    for face in face_rectangles:
        fx, fy, fw, fh = face
        cut = Face(image[fy:fy + fh, fx:fx + fw], fx, fy, fw, fh)
        faces.append(cut)

    index = 0
    for now_face in faces:
        assert isinstance(now_face, Face)
        cv2.imshow("Face" + str(index), now_face.image)  # 显示图像
        index += 1

        left_eye_rectangles = left_eye_cascade.detectMultiScale(
            now_face.image, 1.17, 12)
        right_eye_rectangles = right_eye_cascade.detectMultiScale(
            now_face.image, 1.17, 12)

        eye_rectangles = array_op.mosaic_array(
            left_eye_rectangles, right_eye_rectangles)

        # Ensure eyes are always on the upper side of the horizontal middle line
        # of the face
        if len(eye_rectangles) > 0:
            for eye in eye_rectangles:
                x, y, w, h = eye
                mid_point = Point(int(x + (w / 2)), int(y + (h / 2)))

                hold_eye = False
                for face in face_rectangles:

                    fx, fy, fw, fh = face
                    if mid_point.y <= int(now_face.height / 2):
                        hold_eye = True

                if hold_eye:
                    eye = [x + now_face.x, y + now_face.y, w, h]
                    filtered_eye_rectangles.append(eye)

    # simplify operations to a function
    def draw_result_rectangles(input_image, color, rectangles):
        if len(rectangles) > 0:
            for rect in rectangles:
                x, y, w, h = rect
                cv2.rectangle(input_image, (x, y), (x + w, y + h), color)
        return input_image

    face_drawn = draw_result_rectangles(frame, face_color, face_rectangles)
    eye_drawn = draw_result_rectangles(
        face_drawn, eye_color, filtered_eye_rectangles)

    cv2.imshow("Face Detection", eye_drawn)  # 显示图像

    key = cv2.waitKey(10)
    c = chr(key & 255)
    if c in ['q', 'Q', chr(27)]:
        break

cv2.destroyAllWindows()
