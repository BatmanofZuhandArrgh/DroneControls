from djitellopy import Tello
import time

tello = Tello()

tello.connect(False)
tello.takeoff()


# tello.move_left(100)
time.sleep(10)
tello.move_forward(100)
time.sleep(20)
tello.rotate_counter_clockwise(90)
time.sleep(20)
tello.land()