import cv2
import pygame
import time
import numpy as np

from move import Controls

import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

MOUTH2NOSE_DISTANCE_THRESHOLD = [35, 45] #pixels

class FaceTracker(Controls):
    """ 
    Once the drone detect a face it will follow that face, face to face    
    """

    def __init__(self):
        super(FaceTracker, self).__init__()
        self.S = 25
        self.max_S = 50

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
                        should_stop = True
            
            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])
            frame = frame_read.frame                

            with mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.6) as face_detection:

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                frame.flags.writeable = False
                height, width, shape = frame.shape
                results = face_detection.process(frame)

                # Draw the face detection annotations on the image.
                # image.flags.writeable = True
                if results.detections:
                    for detection in results.detections:
                        mp_drawing.draw_detection(frame, detection)

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
                        upper_limit = int(height/2.5)
                        lower_limit = height - upper_limit
                        height_range = upper_limit # one third of the full height
                        cv2.line(frame, (0,upper_limit),(width, upper_limit), (255, 0, 0), 1, 1)
                        cv2.line(frame, (0,lower_limit),(width, lower_limit), (255, 0, 0), 1, 1)
                        
                        if(nose_y < upper_limit):
                            output_text.append('up')
                            absolute_distance = abs(nose_y - upper_limit)
                            self.up_down_velocity = self.get_pid_velocity(distance_range=height_range, curr_distance=absolute_distance)
                        elif(nose_y > lower_limit):
                            output_text.append('down')
                            absolute_distance = abs(nose_y - lower_limit)
                            self.up_down_velocity = -self.get_pid_velocity(distance_range=height_range, curr_distance=absolute_distance)
                        else:
                            output_text.append('stable_horizontal')
                            self.up_down_velocity = 0

                        #Control right and left
                        rightmost_limit = int(width/2.5)
                        leftmost_limit =  width - rightmost_limit
                        width_range = rightmost_limit
                        cv2.line(frame, (rightmost_limit, 0),(rightmost_limit, height), (255, 0, 0), 1, 1)
                        cv2.line(frame, (leftmost_limit, 0),(leftmost_limit, height), (255, 0, 0), 1, 1)

                        if(nose_x < rightmost_limit):
                            output_text.append('move_right')
                            absolute_distance = abs(nose_x - rightmost_limit)
                            self.left_right_velocity = self.get_pid_velocity(distance_range=width_range, curr_distance=absolute_distance)
                        elif(nose_x > leftmost_limit):
                            output_text.append('move_left')
                            absolute_distance = abs(nose_x - leftmost_limit)
                            self.left_right_velocity = -self.get_pid_velocity(distance_range=width_range, curr_distance=absolute_distance)
                        else:
                            output_text.append('stable_vertical')
                            self.left_right_velocity = 0
                        
                        #Control forward and backward
                        if(m2n_distance > MOUTH2NOSE_DISTANCE_THRESHOLD[1]):
                            output_text.append('move_backwards')
                            self.for_back_velocity = 0 #-self.S
                        elif(m2n_distance < MOUTH2NOSE_DISTANCE_THRESHOLD[0]):
                            output_text.append('move_forwards')
                            self.for_back_velocity = 0# self.S
                        else:
                            output_text.append('stable_depth')
                            self.for_back_velocity = 0

                        # cv2.putText(img = frame, text = f'{m2n_distance}', 
                        #     org = (15,15), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        #     fontScale = 1,color = (0, 0, 255), thickness = 1)
                        cv2.putText(img = frame, text = '|'.join(output_text), 
                            org = (30,15), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale = 1,color = (0, 0, 255), thickness = 1)
                        print('|'.join(output_text))
                else:
                    print('stationary')
                    self.stationary()
           
            self.show_battery(frame)
            self.update()
            self.preproc_frame(frame)

            #Down-sample to 10 fps processing
            # time.sleep(1 / self.FPS)
            time.sleep(1 / (self.FPS/3))

        self.tello.streamoff()
        self.tello.end()

    def get_pid_velocity(self, curr_distance, distance_range):   
        # The velocity would be a ratioed of the maximum velocity, 
        # with the same ratio between the distance to the border to the max distance
        
        return int(curr_distance/distance_range * self.max_S)

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

