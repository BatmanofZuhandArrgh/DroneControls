import cv2
import pygame
import numpy as np
import time

from move import Controls

import torch
from facenet_pytorch import MTCNN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

CONF_THRESHOLD = 0.75

class FaceAttacker(Controls):
    """ 
    The drone will take off and reach the height of 1.5 meters, then spins until detecting the face, and kamikaze into it
    """

    def __init__(self):
        super(FaceAttacker, self).__init__()

    def run(self):
        self.start_up()

        frame_read = self.tello.get_frame_read()

        should_stop = False
        is_spinning = False
        is_tracking = False
        while not should_stop:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    elif event.key == pygame.K_t:
                        self.tello.takeoff()
                        self.tello.move_up(50)
                        self.send_rc_control = True
                        is_spinning = True

                    elif event.key == pygame.K_l:  # land
                        self.tello.land()
                        self.send_rc_control = False
                        should_stop = True
            
            if is_spinning:
                self.tello.rotate_clockwise(x = 24)

            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])

            frame = frame_read.frame                
            bounding_boxes, conf, landmarks = mtcnn.detect(frame, landmarks=True) 
            print(bounding_boxes, conf, landmarks)
            # if conf > CONF_THRESHOLD:
            #     self.tello.land()
            #     self.send_rc_control = False
            #     should_stop = True

            text = "Battery: {}%".format(self.tello.get_battery())
            cv2.putText(frame, text, (5, 720 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
  
                    # print('stationary')
                    # self.stationary()

            self.update()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame = np.rot90(frame)
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            # time.sleep(1 / self.FPS) #Since the model run on 30FPS, it is already 1/30 seconds between every update to the drone

        self.tello.streamoff()
        # Call it always before finishing. To deallocate resources.
        self.tello.end()

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)



def main():

    #Drone inference
    controls = FaceAttacker()
    controls.run()

if __name__ == '__main__':
    main()