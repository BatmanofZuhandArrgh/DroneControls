from djitellopy import Tello
import pygame
import cv2
import numpy as np

class Controls(object):
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

        self.height = 720
        self.width = 960

        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.stationary()
        self.speed = 20

        self.send_rc_control = False

        # Speed of the drone
        self.S = 50
        # Frames per second of the pygame window display
        # A low number also results in input lag, as input information is processed once per frame.
        self.FPS = 30

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // self.FPS)

    def start_up(self):
        self.tello.connect()
        self.tello.set_speed(self.speed)

        # In case streaming is on. This happens when we quit this program without the escape key.
        self.tello.streamoff()
        self.tello.streamon()

    def stationary(self):
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0

    def run(self):
        #Should start a loop to 
        raise NotImplementedError()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity
            self.for_back_velocity = self.S
        elif key == pygame.K_DOWN:  # set backward velocity
            self.for_back_velocity = -self.S
        elif key == pygame.K_LEFT:  # set left velocity
            self.left_right_velocity = -self.S
        elif key == pygame.K_RIGHT:  # set right velocity
            self.left_right_velocity = self.S
        elif key == pygame.K_w:  # set up velocity
            self.up_down_velocity = self.S
        elif key == pygame.K_s:  # set down velocity
            self.up_down_velocity = -self.S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity
            self.yaw_velocity = -self.S
        elif key == pygame.K_d:  # set yaw clockwise velocity
            self.yaw_velocity = self.S
        elif key == pygame.K_f:
            self.tello.flip_forward()

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)
    
    def show_battery(self, frame):
        text = "Battery: {}%".format(self.tello.get_battery())
        cv2.putText(frame, text, (5, 720 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def show_velocity(self):
        # For debug
        print("left_right_velocity: ", self.left_right_velocity)
        print("up_down_velocity: ", self.up_down_velocity)
        print("for_back_velocity: ", self.for_back_velocity)
        print("yaw_velocity: ", self.yaw_velocity)

    def preproc_frame(self, frame): 
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = np.rot90(frame)
        frame = np.flipud(frame)

        frame = pygame.surfarray.make_surface(frame)
        self.screen.blit(frame, (0, 0))
        pygame.display.update()

def main():
    frontend = Controls()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
