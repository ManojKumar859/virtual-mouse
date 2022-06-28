import mediapipe as mp
import cv2
import time
import numpy as np
from collections import deque
from cv2 import VideoCapture
import mediapipe as mp
import cv2
import speech_recognition as sr
import pyttsx3
import numpy as np
import subprocess
from mediapipe.framework.formats import landmark_pb2
import time
from math import sqrt
import win32api
import pyautogui
import webbrowser
import mediapipe as mp
from math import hypot
import cv2
from cvzone.HandTrackingModule import HandDetector
import screen_brightness_control as sbc
import numpy as np
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
pinkytip_x=0
pinkytip_y=0
def SpeakText(command):

    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
click=0

video = cv2.VideoCapture(0)
SpeakText("welcome to virtual mouse using opencv the available functinalities are click,browser,notepad,volume and brightness adjustment")
numb=int(input("enter"))
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:

        SpeakText("you have enabled click functionality")
        while video.isOpened():
            _, frame = video.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = cv2.flip(image, 1)

            imageHeight, imageWidth, _ = image.shape

            results = hands.process(image)


            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                            )

            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    for point in mp_hands.HandLandmark:


                        normalizedLandmark = handLandmarks.landmark[point]
                        pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)

                        point=str(point)


                        if point=='HandLandmark.THUMB_TIP':
                            try:
                                thumbfingertip_x=pixelCoordinatesLandmark[0]
                                thumbfingertip_y=pixelCoordinatesLandmark[1]
                                #print("thumb",thumbfingertip_x)

                            except:
                                pass
                        if point=='HandLandmark.INDEX_FINGER_TIP':
                            try:
                                indexfingertip_x=pixelCoordinatesLandmark[0]
                                indexfingertip_y=pixelCoordinatesLandmark[1]
                                win32api.SetCursorPos((indexfingertip_x*4,indexfingertip_y*5))

                            except:
                                pass

                        elif point=="HandLandmark.RING_FINGER_TIP":
                            try:
                                rtip_x=pixelCoordinatesLandmark[0]
                                rtip_y=pixelCoordinatesLandmark[1]
                                #print("thumb",thumbfingertip_x)

                            except:
                                pass
                        elif point=="HandLandmark.PINKY_TIP":
                            try:
                                pinkytip_x=pixelCoordinatesLandmark[0]
                                pinkytip_y=pixelCoordinatesLandmark[1]
                                #print("thumb",thumbfingertip_x)

                            except:
                                pass
                        elif point=="HandLandmark.PINKY_DIP":
                            try:
                                pinkydip_x=pixelCoordinatesLandmark[0]
                                pinkydip_y=pixelCoordinatesLandmark[1]
                                #print("thumb",thumbfingertip_x)

                            except:
                                pass
                        elif point=="HandLandmark.MIDDLE_FINGER_TIP":
                            try:
                                mt_x=pixelCoordinatesLandmark[0]
                                mt_y=pixelCoordinatesLandmark[1]
                                #print("thumb",thumbfingertip_x)

                            except:
                                pass
                        elif point=="HandLandmark.INDEX_FINGER_DIP":
                            try:
                                id_x=pixelCoordinatesLandmark[0]
                                id_y=pixelCoordinatesLandmark[1]
                                #print("thumb",thumbfingertip_x)

                            except:
                                pass
                            try:
                                    #pyautogui.moveTo(indexfingertip_x,indexfingertip_y)
                                    Distance_x= sqrt((indexfingertip_x-thumbfingertip_x)**2 + (indexfingertip_x-thumbfingertip_x)**2)
                                    Distance_y= sqrt((indexfingertip_y-thumbfingertip_y)**2 + (indexfingertip_y-thumbfingertip_y)**2)
                                    if Distance_x<15 or Distance_x<-15:
                                        if Distance_y<15 or Distance_y<-15:
                                            click=click+1
                                            if click%5==0:
                                                print("single click")
                                                pyautogui.click()

                            except:
                                    pass
                            #if pinkytip_y!=0 and pinkytip_x!=0:
                            try:
                                Distance_x= sqrt((pinkytip_x-thumbfingertip_x)**2 + (pinkytip_x-thumbfingertip_x)**2)
                                Distance_y= sqrt((pinkytip_y-thumbfingertip_y)**2 + (pinkytip_y-thumbfingertip_y)**2)
                                if Distance_x<5 or Distance_x<-5:
                                    if Distance_y<5 or Distance_y<-5:

                                        mpHands = mp.solutions.hands
                                        hands = mpHands.Hands()
                                        mpDraw = mp.solutions.drawing_utils
                                        SpeakText("you have enabled brightness functionality")
                                        while True:
                                            success,img = video.read()
                                            imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                                            results = hands.process(imgRGB)

                                            lmList = []
                                            if results.multi_hand_landmarks:
                                                for handlandmark in results.multi_hand_landmarks:
                                                    for id,lm in enumerate(handlandmark.landmark):
                                                        h,w,_ = img.shape
                                                        cx,cy = int(lm.x*w),int(lm.y*h)
                                                        lmList.append([id,cx,cy])
                                                    mpDraw.draw_landmarks(img,handlandmark,mpHands.HAND_CONNECTIONS)

                                            if lmList != []:
                                                x1,y1 = lmList[4][1],lmList[4][2]
                                                x2,y2 = lmList[8][1],lmList[8][2]

                                                cv2.circle(img,(x1,y1),4,(255,0,0),cv2.FILLED)
                                                cv2.circle(img,(x2,y2),4,(255,0,0),cv2.FILLED)
                                                cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)

                                                length = hypot(x2-x1,y2-y1)

                                                bright = np.interp(length,[15,220],[0,100])
                                                print(bright,length)
                                                sbc.set_brightness(int(bright))

                                                # Hand range 15 - 220
                                                # Brightness range 0 - 100

                                            cv2.imshow('Image',img)
                                            if cv2.waitKey(1) & 0xff==ord('q'):
                                                break
                            except:
                                        pass
                            try:
                                Distance_x= sqrt((pinkydip_x-thumbfingertip_x)**2 + (pinkydip_x-thumbfingertip_x)**2)
                                Distance_y= sqrt((pinkydip_y-thumbfingertip_y)**2 + (pinkydip_y-thumbfingertip_y)**2)
                                if Distance_x<5 or Distance_x<-5:
                                    if Distance_y<5 or Distance_y<-5:
                                            SpeakText("you have enabled volume functionality")
                                            mpHands = mp.solutions.hands
                                            hands = mpHands.Hands()
                                            mpDraw = mp.solutions.drawing_utils
                                            devices = AudioUtilities.GetSpeakers()
                                            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                                            volume = cast(interface, POINTER(IAudioEndpointVolume))

                                            volMin,volMax = volume.GetVolumeRange()[:2]

                                            while True:
                                                success,img = video.read()
                                                imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                                                results = hands.process(imgRGB)

                                                lmList = []
                                                if results.multi_hand_landmarks:
                                                    for handlandmark in results.multi_hand_landmarks:
                                                        for id,lm in enumerate(handlandmark.landmark):
                                                            h,w,_ = img.shape
                                                            cx,cy = int(lm.x*w),int(lm.y*h)
                                                            lmList.append([id,cx,cy])
                                                        mpDraw.draw_landmarks(img,handlandmark,mpHands.HAND_CONNECTIONS)

                                                if lmList != []:
                                                    x1,y1 = lmList[4][1],lmList[4][2]
                                                    x2,y2 = lmList[8][1],lmList[8][2]

                                                    cv2.circle(img,(x1,y1),4,(255,0,0),cv2.FILLED)
                                                    cv2.circle(img,(x2,y2),4,(255,0,0),cv2.FILLED)
                                                    cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)

                                                    length = hypot(x2-x1,y2-y1)

                                                    vol = np.interp(length,[15,220],[volMin,volMax])
                                                    print(vol,length)
                                                    volume.SetMasterVolumeLevel(vol, None)

                                                    # Hand range 15 - 220
                                                    # Volume range -63.5 - 0.0

                                                cv2.imshow('Image',img)
                                                if cv2.waitKey(1) & 0xff==ord('q'):
                                                    break
                            except:
                                pass
                            try:
                                Distance_x= sqrt((rtip_x-thumbfingertip_x)**2 + (rtip_x-thumbfingertip_x)**2)
                                Distance_y= sqrt((rtip_y-thumbfingertip_y)**2 + (rtip_y-thumbfingertip_y)**2)
                                if Distance_x<5 or Distance_x<-5:
                                    if Distance_y<5 or Distance_y<-5:
                                        SpeakText("you have enabled browser functionality")
                                        webbrowser.open('http://www.google.com')
                            except:
                                pass
                            try:
                                Distance_x= sqrt((mt_x-thumbfingertip_x)**2 + (mt_x-thumbfingertip_x)**2)
                                Distance_y= sqrt((mt_y-thumbfingertip_y)**2 + (mt_y-thumbfingertip_y)**2)
                                if Distance_x<5 or Distance_x<-5:
                                    if Distance_y<5 or Distance_y<-5:
                                        SpeakText("you have enabled Notepad  functionality")
                                        subprocess.Popen(['notepad.exe'])
                            except:
                                pass
                            try:
                                Distance_x= sqrt((id_x-thumbfingertip_x)**2 + (id_x-thumbfingertip_x)**2)
                                Distance_y= sqrt((id_y-thumbfingertip_y)**2 + (id_y-thumbfingertip_y)**2)
                                if Distance_x<5 or Distance_x<-5:
                                    if Distance_y<5 or Distance_y<-5:
                                            SpeakText("virtual piant is enabled")
                                            count_frames = 0
                                            Brush_Size = 7
                                            b = [0, 0, 0, 0, 0]
                                            a = deque([[0, 0]], maxlen=5)
                                            color = (255, 255, 255)

                                            class HandsDetection():

                                                def __init__(self, num_hands = 1, detection_confidence = 0.7,
                                                            static_mode = False, tracking_confidence = 0.7,):
                                                    self.num_hands = num_hands
                                                    self.detection_confidence = detection_confidence
                                                    self.static_mode = static_mode
                                                    self.tracking_confidence = tracking_confidence

                                                    self.mpHands = mp.solutions.hands
                                                    self.mpDraw = mp.solutions.drawing_utils
                                                    self.hands = self.mpHands.Hands(max_num_hands = self.num_hands,
                                                                                    min_detection_confidence = self.detection_confidence,
                                                                                    static_image_mode = self.static_mode,
                                                                                    min_tracking_confidence = self.tracking_confidence)


                                                def DrawHandsLandmarks(self, frame):

                                                    for i in self.results.multi_hand_landmarks:
                                                        self.mpDraw.draw_landmarks(frame, i, self.mpHands.HAND_CONNECTIONS)

                                                    return frame


                                                def SelectColor(self, x, y): # , default = True
                                                    '''
                                                    To select color of Brush
                                                    '''

                                                    global color

                                                    if ((50 < x < 250) and (50 < y < 150)):
                                                        color = (255, 0, 0)
                                                        return color

                                                    elif ((350 < x < 550) and (50 < y < 150)):
                                                        color = (0, 255, 0)
                                                        return color

                                                    elif ((650 < x < 850) and (50 < y < 150)):
                                                        color = (0, 0, 255)
                                                        return color

                                                    elif ((950 < x < 1150) and (50 < y < 150)):
                                                        color = (0, 0, 0)
                                                        return color

                                                    return None


                                                def BrushSize(self, x, y):
                                                    '''
                                                    To determine the size of Brush
                                                    '''

                                                    global Brush_Size

                                                    if ((80 < x < 120) and (230 < y < 570)):
                                                        Brush_Size = int((y-70)/9.2)

                                                        if Brush_Size < 5:
                                                            Brush_Size = 5

                                                        elif Brush_Size > 45:
                                                            Brush_Size = 45

                                                    return None


                                                def DrawOnScreen(self, frame, canvas, x, y):
                                                    '''
                                                    To draw on screen
                                                    '''

                                                    global a, count_frames, color, Brush_Size

                                                    # To select color of Brush
                                                    self.SelectColor(x, y)

                                                    # To determine the size of Brush
                                                    self.BrushSize(x, y)

                                                    if ((x, y) != (0, 0) and (a[-1] == [0,0])):
                                                        a.append([x, y])
                                                        return frame, canvas

                                                    if ((x, y) != (0, 0) and (a[-1] != [0,0])):
                                                        cv2.circle(frame, (x, y), 15, (200, 255, 200), -1)
                                                        # cv2.rectangle(frame, (50, 50), (1150, 550), (0, 0, 200), 2)   # for RoI of drawable area
                                                        a.append([x, y])

                                                        if sum(b) == 1:

                                                            if color == (0, 0, 0):
                                                                x0, y0 = a[-2]
                                                                cv2.circle(canvas, (x, y), Brush_Size-3, color, -1)
                                                                cv2.line(canvas, (x , y), (x0, y0), color, Brush_Size+7)
                                                                cv2.putText(frame, str(Brush_Size), (75, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

                                                            else:
                                                                x0, y0 = a[-2]
                                                                cv2.circle(canvas, (x, y), Brush_Size-3, color, -1)
                                                                cv2.line(canvas, (x , y), (x0, y0), color, Brush_Size+7)
                                                                cv2.putText(frame, str(Brush_Size), (75, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)


                                                    return frame, canvas


                                                def DrawBoxes(self, frame):
                                                    '''
                                                    To draw diffrent req boxes for colors
                                                    '''

                                                    global color

                                                    cv2.rectangle(frame, (50, 50), (250, 150), (0, 0, 255), -1)
                                                    cv2.putText(frame, 'Blue', (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                                                    cv2.rectangle(frame, (350, 50), (550, 150), (0, 255, 0), -1)
                                                    cv2.putText(frame, 'Green', (370, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                                                    cv2.rectangle(frame, (650, 50), (850, 150), (255, 0, 0), -1)
                                                    cv2.putText(frame, 'Red', (670, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                                                    cv2.rectangle(frame, (950, 50), (1150, 150), (0, 0, 0), 4)
                                                    cv2.putText(frame, 'Eraser', (970, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                                                    pts = [(100, 250), (100, 550), (110, 550)]
                                                    cv2.line(frame, pts[0], pts[1], (255, 0, 255), 1)
                                                    cv2.line(frame, pts[1], pts[2], (255, 0, 255), 1)
                                                    cv2.line(frame, pts[1], pts[2], (255, 0, 255), 1)
                                                    cv2.fillPoly(frame, np.array([pts]), (255, 0, 255))
                                                    cv2.putText(frame, 'Brush Size', (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

                                                    if color == (0, 0, 255):
                                                        c = 'Red'
                                                        cv2.rectangle(frame, (645, 45), (855, 155), (255, 0, 255), 8)

                                                    elif color == (0, 255, 0):
                                                        c = 'Green'
                                                        cv2.rectangle(frame, (345, 45), (555, 155), (255, 0, 255), 8)

                                                    elif color == (255, 0, 0):
                                                        c = 'Blue'
                                                        cv2.rectangle(frame, (45, 45), (255, 155), (255, 0, 255), 8)

                                                    elif color == (255, 255, 255):
                                                        c = 'White'
                                                        # cv2.rectangle(frame, (950, 50), (1150, 170), (255, 0, 255)), 6)

                                                    elif color == (0, 0, 0):
                                                        c = 'Eraser'
                                                        cv2.rectangle(frame, (945, 45), (1155, 155), (255, 0, 255), 8)

                                                    if c == 'Eraser':
                                                        cv2.putText(frame, 'Eraser Selected', (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                                                    else:
                                                        cv2.putText(frame, 'Color Selected: ', (540, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                                                        cv2.putText(frame, c, (800, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color[::-1], 3)

                                                    return frame


                                                def Landmarks(self, frame, canvas, draw = True):

                                                    global a, b, color, Brush_Size

                                                    self.results = self.hands.process(frame)

                                                    frame = self.DrawBoxes(frame)

                                                    if cv2.waitKey(1) & 0xFF == ord('c'): # for clearing the screen
                                                        canvas = np.zeros_like(frame)
                                                        x, y = 0, 0
                                                        a = deque([[x, y], [x, y]])
                                                        color = (255, 255, 255)
                                                        Brush_Size = 7
                                                        cv2.rectangle(frame, (250, 150), (1100, 300), (0, 255, 0), -1)
                                                        cv2.putText(frame, 'Screen Cleared', (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 10)

                                                        return frame, canvas

                                                    if self.results.multi_hand_landmarks:

                                                        h, w = frame.shape[0], frame.shape[1]
                                                        x, y = int(self.results.multi_hand_landmarks[0].landmark[8].x*w), int(self.results.multi_hand_landmarks[0].landmark[8].y*h)

                                                        x2, x4 = self.results.multi_hand_landmarks[0].landmark[2].x*w, self.results.multi_hand_landmarks[0].landmark[4].x*w
                                                        if x4 > x2:
                                                            b[0] = 0
                                                        elif x4 < x2:
                                                            b[0] = 1

                                                        # Co-ordinates of Index finger:
                                                        y6, y8 = self.results.multi_hand_landmarks[0].landmark[6].y*h, self.results.multi_hand_landmarks[0].landmark[8].y*h
                                                        if y8 < y6:
                                                            b[1] = 1
                                                        elif y8 > y6:
                                                            b[1] = 0

                                                        # Co-ordinates of Middle finger:
                                                        y10, y12 = self.results.multi_hand_landmarks[0].landmark[10].y*h, self.results.multi_hand_landmarks[0].landmark[12].y*h
                                                        if y12 < y10:
                                                            b[2] = 1
                                                        elif y12 > y10:
                                                            b[2] = 0

                                                        # Co-ordinates of Ring finger:
                                                        y14, y16 = self.results.multi_hand_landmarks[0].landmark[14].y*h, self.results.multi_hand_landmarks[0].landmark[16].y*h
                                                        if y16 < y14:
                                                            b[3] = 1
                                                        elif y16 > y14:
                                                            b[3] = 0

                                                        # Co-ordinates of Pinky finger:
                                                        y18, y20 = self.results.multi_hand_landmarks[0].landmark[18].y*h, self.results.multi_hand_landmarks[0].landmark[20].y*h
                                                        if y20 < y18:
                                                            b[4] = 1
                                                        elif y20 > y18:
                                                            b[4] = 0

                                                        frame, canvas = self.DrawOnScreen(frame, canvas, x, y)

                                                        if draw:
                                                            frame = self.DrawHandsLandmarks(frame)

                                                    return frame, canvas



                                            def main():

                                                global count_frames

                                                s = 0
                                                t = 0
                                                canvas = np.zeros((720, 1280, 3), dtype = np.uint8)

                                                video = cv2.VideoCapture(0)
                                                video.set(3, 1280)
                                                video.set(4, 720)
                                                hands = HandsDetection(num_hands = 1, detection_confidence = 0.6,
                                                                    static_mode = False, tracking_confidence = 0.6,)

                                                while video.isOpened():
                                                    ret, frame = video.read()
                                                    if not ret:
                                                        break

                                                    frame, canvas = hands.Landmarks(cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB), canvas, draw = False)

                                                    s = time.time()
                                                    fps = int(1/(s-t))
                                                    t = s

                                                    cv2.putText(frame, 'FPS: ' + str(fps), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

                                                    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                                                    _, canvas_inv = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
                                                    canvas_inv = cv2.cvtColor(canvas_inv, cv2.COLOR_GRAY2BGR)
                                                    frame = cv2.bitwise_and(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), canvas_inv)
                                                    output = cv2.bitwise_or(frame, canvas)

                                                    cv2.imshow('output', output)


                                                    # result.write(output)

                                                    if cv2.waitKey(1) & 0xFF == ord('q'):
                                                        break

                                                    count_frames += 1

                                                # result.release()
                                                video.release()

                                                cv2.destroyAllWindows()
                                                print("Done processing video")
                                            main()
                            except:
                                pass
            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cv2.waitKey()
cv2.destroyAllWindows()

video.release()