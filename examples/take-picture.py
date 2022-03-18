import cv2
from djitellopy import Tello

tello = Tello()
tello.connect()

print('---------Done Connecting--------------')
tello.streamon()

print('-----------------------')
frame_read = tello.get_frame_read()

tello.takeoff()
print('done taking off')
cv2.imwrite("output/picture.png", frame_read.frame)

tello.land()
