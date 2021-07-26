from djitellopy import Tello
import time

tello = Tello()

tello.connect()
tello.takeoff()
print('Finish Taking off')
time.sleep(5)
print('Sending move forward')
# tello.move_left(100)



tello.move_forward(100)
time.sleep(10)
tello.rotate_clockwise(90)
time.sleep(10)
tello.land()
