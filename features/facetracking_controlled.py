from turtle import width
import cv2
import pygame
import time
import numpy as np

import torch

from move import Controls
from utils import draw_bbox, plot_landmarks

from facenet_pytorch import MTCNN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

CONF_THRESHOLD = 0.75
DISTANCE_THRESHOLD = (30,10)

class FaceTracker(Controls):
    """ 
    Once the drone detect a face it will follow that face, face to face    
    """

    def __init__(self):
        super(FaceTracker, self).__init__()
        self.S = 30
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
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            height, width, _ = frame.shape

            image_array = np.array(frame, dtype=np.float32)
            bounding_boxes, conf, landmarks = mtcnn.detect(frame, landmarks=True) 

            if bounding_boxes is not None:
                areas = []
                new_bboxes = []
                # new_conf = []
                # new_landmarks = []

                for index in range(len(bounding_boxes)):
                    if conf[index] > CONF_THRESHOLD:
                        x1, y1, x2, y2 = bounding_boxes[index]
                        areas.append(abs((x1 - x2)*(y2 - y1)))
                        new_bboxes.append(bounding_boxes[index])
                        # new_conf.append(conf[index])
                        # new_landmarks.append(landmarks[index])
                if(len(new_bboxes) == 0): break
                max_index = np.argmax(areas)
            
                bounding_boxes = [new_bboxes[max_index]]
                # conf = [new_conf[max_index]]
                # landmarks = [new_landmarks[max_index]]

                frame = draw_bbox(bounding_boxes, frame)

                # plot the facial landmarks
                # image_array = plot_landmarks(landmarks, image_array)
                
                center =((bounding_boxes[0][0] + bounding_boxes[0][2])//2, (bounding_boxes[0][1] + bounding_boxes[0][3])//2)
                cv2.circle(frame, 
                        (int(center[0]),int(center[1]) ),
                        2, (0, 0, 255), -1, cv2.LINE_AA)

                #Control up and down
                upper_limit = int(height/2.1)
                lower_limit = height - upper_limit
                height_range = upper_limit # one third of the full height
                cv2.line(frame, (0,upper_limit),(width, upper_limit), (255, 0, 0), 1, 1)
                cv2.line(frame, (0,lower_limit),(width, lower_limit), (255, 0, 0), 1, 1)
                
                nose_x, nose_y = center
                output_text = []
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
                face_height = abs(bounding_boxes[0][1] - bounding_boxes[0][3])
                if(face_height > DISTANCE_THRESHOLD[1]):
                    output_text.append('move_backwards')
                    self.for_back_velocity = 0 #-self.S
                elif(face_height < DISTANCE_THRESHOLD[0]):
                    output_text.append('move_forwards')
                    self.for_back_velocity = 0# self.S
                else:
                    output_text.append('stable_depth')
                    self.for_back_velocity = 0

                cv2.putText(img = frame, text = f'{face_height}', 
                    org = (15,15), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 1,color = (0, 0, 255), thickness = 1)
                cv2.putText(img = frame, text = '|'.join(output_text), 
                    org = (30,30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
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
        
        return int(curr_distance/distance_range * self.max_S) + 2

def webcam_findFace():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image_array = np.array(image, dtype=np.float32)
        bounding_boxes, conf, landmarks = mtcnn.detect(image, landmarks=True)        

        if bounding_boxes is not None:
            areas = []
            new_bboxes = []
            # new_conf = []
            # new_landmarks = []

            for index, bounding_box in enumerate(bounding_boxes):
                if conf[index] > CONF_THRESHOLD:
                    x1, y1, x2, y2 = bounding_boxes[index]
                    areas.append(abs((x1 - x2)*(y2 - y1)))
                    new_bboxes.append(bounding_boxes[index])
                    # new_conf.append(conf[index])
                    # new_landmarks.append(landmarks[index])
            if(len(new_bboxes) == 0): break
            max_index = np.argmax(areas)
            
            bounding_boxes = [new_bboxes[max_index]]
            # conf = [new_conf[max_index]]
            # landmarks = [new_landmarks[max_index]]

            image_array = draw_bbox(bounding_boxes, image_array)

            # plot the facial landmarks
            # image_array = plot_landmarks(landmarks, image_array)
            
            center =((bounding_boxes[0][0] + bounding_boxes[0][2])//2, (bounding_boxes[0][1] + bounding_boxes[0][3])//2)
            cv2.circle(image_array, 
                      (int(center[0]),int(center[1]) ),
                      2, (0, 0, 255), -1, cv2.LINE_AA)

        cv2.imshow('MediaPipe Face Detection', image_array/255.0)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()

def main():
    controls = FaceTracker()
    controls.run()
    # webcam_findFace()

if __name__ == '__main__':
    main()

