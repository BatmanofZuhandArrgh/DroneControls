import cv2
import pygame
import numpy as np
import time

from move import Controls

class UserControls(Controls):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations (yaw)
            - W and S: Up and down.
    """

    def __init__(self):
        super(UserControls, self).__init__()
        
    def run(self):
        #Should start a loop to 
        self.start_up()

        frame_read = self.tello.get_frame_read()

        should_stop = False
        while not should_stop:

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.update()
                elif event.type == pygame.QUIT:
                    should_stop = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == pygame.KEYUP:
                    self.keyup(event.key)

            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])

            frame = frame_read.frame
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

            self.show_battery(frame)
            self.preproc_frame(frame)

            time.sleep(1 / self.FPS)

        # Call it always before finishing. To deallocate resources.
        self.tello.streamoff()
        self.tello.end()


def main():
    frontend = UserControls()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
