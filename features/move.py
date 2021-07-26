from djitellopy import Tello
import pygame

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
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("Tello video stream")
        self.screen = pygame.display.set_mode([960, 720])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 10

        self.send_rc_control = False

        # Speed of the drone
        self.S = 60
        # Frames per second of the pygame window display
        # A low number also results in input lag, as input information is processed once per frame.
        self.FPS = 120

        # create update timer
        pygame.time.set_timer(pygame.USEREVENT + 1, 1000 // self.FPS)
    
    def start_up(self):
        self.tello.connect()
        self.tello.set_speed(self.speed)

        # In case streaming is on. This happens when we quit this program without the escape key.
        self.tello.streamoff()
        self.tello.streamon()

    def run(self):
        #Should start a loop to 
        raise NotImplementedError()

def main():
    frontend = Controls()

    # run frontend
    frontend.run()


if __name__ == '__main__':
    main()
