import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from components import facepp_api
from collections import namedtuple
import win32api
import win32con
from components.thread_rebuild import ReThread


Point = namedtuple("Point", ['x', 'y'])
Face = namedtuple("Face", ['image', 'x', 'y', 'width', 'height'])
# get the first camera
cap = cv2.VideoCapture(0)

# get the first frame of the img to obtain the size of the frame
success, frame = cap.read()

face_color = (199, 199, 199)
eye_color = (100, 100, 100)
pupil_color = (180, 200, 0)

face_cascade = cv2.CascadeClassifier("face.xml")
eye_cascade = cv2.CascadeClassifier("eye.xml")

main_face = None

# get the source frame
success, frame = cap.read()
tick = 0
emotion = ""

public_thread = None


def detect_face_emotion(image, argument):

    emo = facepp_api.analyze_face(main_face.image, "emotion")

    if emo and isinstance(emo, dict):
        emo = emo["emotion"]
        max_rate = 0
        max_key = ""
        print(emo)
        for key in emo:
            value = int(emo[key])
            if value > max_rate:
                max_key = key
                max_rate = value

        return max_key


while success:

    # get new frame and the frame size
    success, frame = cap.read()
    size = frame.shape[:2]

    # to Gray
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # get the result
    face_rectangles = face_cascade.detectMultiScale(image, 1.22, 10)
    faces = list()
    filtered_eye_rectangles = list()

    for face in face_rectangles:
        fx, fy, fw, fh = face
        cut = Face(image[fy:fy + int(fh * 1), fx:fx + fw], fx, fy, fw, fh)
        faces.append(cut)

    index = 0
    for now_face in faces:

        # get every face and do comparison with face++ API
        assert isinstance(now_face, Face)
        nf = now_face
        now_face = Face(cv2.resize(nf.image, (256, 256)),
                        nf.x, nf.y, nf.width, nf.height)
        main_face = now_face
        cv2.imshow("Face" + str(index), now_face.image)  # 显示图像
        index += 1

        eye_rectangles = ""
        # eye_cascade.detectMultiScale(now_face.image, 1.07, 10)

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
    def draw_result_rectangles(input_image, color, rectangles, is_eye=False):
        if len(rectangles) > 0:
            for rect in rectangles:
                x, y, w, h = rect
                cv2.rectangle(input_image, (x, y), (x + w, y + h), color)
                if is_eye:
                    pupil_point = Point(int(x + (w / 2)), int(y + (h / 2)))
                    cv2.circle(input_image, pupil_point, 1, pupil_color, 5)
        return input_image

    face_drawn = draw_result_rectangles(frame, face_color, face_rectangles)
    eye_drawn = draw_result_rectangles(
        face_drawn, eye_color, filtered_eye_rectangles, True)

    if main_face and tick == 2:

        im = Image.fromarray(eye_drawn)
        public_thread = ReThread(
            detect_face_emotion, args=(
                main_face.image, "emotion",))
        public_thread.run()

    elif main_face:
        if public_thread:
            emotion = public_thread.get_result()
        im = Image.fromarray(eye_drawn)
        font = ImageFont.truetype('font.otf', 20)
        draw = ImageDraw.Draw(im)
        x, y = (main_face.x, main_face.y - 20)
        draw.text((x, y), emotion, font=font, fill=(119, 85, 0))
        eye_drawn = np.array(im)

    if tick == 5:
        tick = 0

    tick += 1
    cv2.imshow("Face", eye_drawn)  # 显示图像

    key = cv2.waitKey(10)
    c = chr(key & 255)
    if c in ['q', 'Q', chr(27)]:
        break
    if c in ['e', 'E', chr(69)]:
        if main_face:
            print(facepp_api.create_face(main_face.image))
        else:
            win32api.MessageBox(0, "没有找到可用人脸", "错误", win32con.MB_OK)
    if c in ['c', 'c', chr(70)]:
        if main_face:
            facepp_api.compare_face(main_face.image, facepp_api.model_token)
        else:
            win32api.MessageBox(0, "没有找到可用人脸", "错误", win32con.MB_OK)
    main_face = False

cv2.destroyAllWindows()
