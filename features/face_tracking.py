import cv2
from numpy.lib.type_check import imag
import pygame
import time
import numpy as np

from move import Controls

import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

MOUTH2NOSE_DISTANCE_THRESHOLD = [40, 50] #pixels

class FaceTracker(Controls):
    """ 
    Once the drone detect a face it will follow that face, face to face    
    """

    def __init__(self):
        super(FaceTracker, self).__init__()
        self.area_to_stabilize = (25000, 30000)
        self.height = 720
        self.width = 960

        self.upper_limit = int(self.height//3)
        self.lower_limit = self.upper_limit * 2
        
        self.rightmost_limit = int(self.width//3)
        self.leftmost_limit = self.rightmost_limit*2
        
    def run(self):
        self.start_up()

        frame_read = self.tello.get_frame_read()

        should_stop = False
        while not should_stop:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    elif event.key == pygame.K_t:
                        self.tello.takeoff()
                        self.send_rc_control = True

                    elif event.key == pygame.K_l:  # land
                        self.tello.land()
                        self.send_rc_control = False
            
            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])
            self.update()

            frame = frame_read.frame                

            text = "Battery: {}%".format(self.tello.get_battery())
            cv2.putText(frame, text, (5, 720 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            with mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5) as face_detection:

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                height, width, shape = image.shape
                results = face_detection.process(image)

                # Draw the face detection annotations on the image.
                # image.flags.writeable = True
                if results.detections:
                    for detection in results.detections:
                        mp_drawing.draw_detection(image, detection)

                        nose_x = mp_face_detection.get_key_point(
                            detection, mp_face_detection.FaceKeyPoint.NOSE_TIP).x * width
                        nose_y = mp_face_detection.get_key_point(
                            detection, mp_face_detection.FaceKeyPoint.NOSE_TIP).y * height

                        mouth_x = mp_face_detection.get_key_point(
                            detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER).x * width
                        mouth_y = mp_face_detection.get_key_point(
                            detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER).y * height
                        
                        output_text = []

                        #mouth to nose distance
                        m2n_distance = np.sqrt(np.square(mouth_x - nose_x) + np.square(mouth_y - nose_y))
                    
                        #Control up and down
                        upper_limit = int(height/3)
                        lower_limit = int(height/3)*2
                        cv2.line(frame, (0,upper_limit),(width, upper_limit), (255, 0, 0), 1, 1)
                        cv2.line(frame, (0,lower_limit),(width, lower_limit), (255, 0, 0), 1, 1)
                        
                        if(nose_y < upper_limit):
                            output_text.append('down')
                            self.up_down_velocity = -self.S
                        elif(nose_y > lower_limit):
                            output_text.append('up')
                            self.up_down_velocity = self.S
                        else:
                            output_text.append('stable_horizontal')
                            self.up_down_velocity = 0

                        #Control right and left
                        rightmost_limit = int(width/3)
                        leftmost_limit =  int(width/3) * 2
                        cv2.line(frame, (rightmost_limit, 0),(rightmost_limit, height), (255, 0, 0), 1, 1)
                        cv2.line(frame, (leftmost_limit, 0),(leftmost_limit, height), (255, 0, 0), 1, 1)

                        if(nose_x < rightmost_limit):
                            output_text.append('move_left')
                            self.left_right_velocity = self.S
                        elif(nose_x > leftmost_limit):
                            output_text.append('move_right')
                            self.left_right_velocity = -self.S
                        else:
                            output_text.append('stable_vertical')
                            self.left_right_velocity = 0
                        

                        #Control forward and backward
                        if(m2n_distance > MOUTH2NOSE_DISTANCE_THRESHOLD[1]):
                            output_text.append('move_backwards')
                            self.for_back_velocity = -self.S
                        elif(m2n_distance < MOUTH2NOSE_DISTANCE_THRESHOLD[0]):
                            output_text.append('move_forwards')
                            self.for_back_velocity = self.S
                        else:
                            output_text.append('stable_depth')
                            self.for_back_velocity = 0

                        # cv2.putText(img = frame, text = f'{m2n_distance}', 
                        #     org = (15,15), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        #     fontScale = 1,color = (0, 0, 255), thickness = 1)
                        cv2.putText(img = frame, text = '|'.join(output_text), 
                            org = (30,15), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale = 1,color = (0, 0, 255), thickness = 1)
                else:
                    print('stationary')

            self.update()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame = np.rot90(frame)
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / self.FPS)

        self.tello.streamoff()
        # Call it always before finishing. To deallocate resources.
        self.tello.end()

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)

def webcam_findFace():
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            height, width, shape = image.shape
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)

                    nose_x = mp_face_detection.get_key_point(
                        detection, mp_face_detection.FaceKeyPoint.NOSE_TIP).x * width
                    nose_y = mp_face_detection.get_key_point(
                        detection, mp_face_detection.FaceKeyPoint.NOSE_TIP).y * height

                    mouth_x = mp_face_detection.get_key_point(
                        detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER).x * width
                    mouth_y = mp_face_detection.get_key_point(
                        detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER).y * height

                    #mouth to nose distance
                    m2n_distance = np.sqrt(np.square(mouth_x - nose_x) + np.square(mouth_y - nose_y))

                    cv2.putText(img = image, text = f'{m2n_distance}', 
                        org = (30,30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 1,color = (0, 0, 255), thickness = 1)
                
                    #Control up and down
                    upper_limit = int(height/3)
                    lower_limit = int(height/3)*2
                    cv2.line(image, (0,upper_limit),(width, upper_limit), (255, 0, 0), 1, 1)
                    cv2.line(image, (0,lower_limit),(width, lower_limit), (255, 0, 0), 1, 1)
                    
                    if(nose_y < upper_limit):
                        print('down')
                    elif(nose_y > lower_limit):
                        print('up')

                    #Control right and left
                    rightmost_limit = int(width/3)
                    leftmost_limit =  int(width/3) * 2
                    cv2.line(image, (rightmost_limit, 0),(rightmost_limit, height), (255, 0, 0), 1, 1)
                    cv2.line(image, (leftmost_limit, 0),(leftmost_limit, height), (255, 0, 0), 1, 1)

                    if(nose_x < rightmost_limit):
                        print('move_left')
                    elif(nose_x > leftmost_limit):
                        print('move_right')

                    #Control forward and backward
                    if(m2n_distance > MOUTH2NOSE_DISTANCE_THRESHOLD[1]):
                        print('move backwards')
                    elif(m2n_distance < MOUTH2NOSE_DISTANCE_THRESHOLD[0]):
                        print('move forwards')
                    else:
                        print('stationary')
                
            cv2.imshow('MediaPipe Face Detection', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

'''
def findFace(img, mode = 'CascadeClassifier'):
    if mode == 'CascadeClassifier':
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)
    elif mode == 'BlazeFace':
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection
            results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                return img, [[0,0], [0]]
            annotated_image = img.copy()
            for detection in results.detections:
                print('Nose tip:')
                print(mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                mp_drawing.draw_detection(annotated_image, detection)
    else:
        faces = []

    face_list = []
    face_area_list = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w*h
        face_list.append([cx, cy])
        face_area_list.append(area)

    if len(face_list) != 0:
        i = face_area_list.index(max(face_area_list))
        return img, [face_list[i], face_area_list[i]]

    else:
        return img, [[0,0], [0]]
    '''

def main():
    controls = FaceTracker()
    controls.run()
    # webcam_findFace()

if __name__ == '__main__':
    main()

