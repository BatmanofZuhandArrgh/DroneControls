import cv2
import pygame
import mediapipe as mp
import numpy as np
import time

from move import Controls
from handpose import (get_index_finger_direction, 
                    get_ringmid_direction, 
                    get_thumb_direction,
                    get_all_point_of_hands,
                    get_ringmid_direction,
                    handGesture_rockNroll,
                    handGesture_mockingJay)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class HandPoseControl(Controls):
    """ 
    Once the drone detect a face it will follow that face, face to face    
    """

    def __init__(self):
        super(HandPoseControl, self).__init__()

    def run(self):
        self.start_up()

        frame_read = self.tello.get_frame_read()

        should_stop = False
        order_acceptance_count = 0
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

            text = "Battery: {}%".format(self.tello.get_battery())
            cv2.putText(frame, text, (5, 720 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            order_queue = ''
            with mp_hands.Hands(
                max_num_hands = 1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image_height, image_width, _ = image.shape
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                print('--------------------------')
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    hand_dict = get_all_point_of_hands(mp_hands, hand_landmarks, image_width, image_height)
                    order = get_index_finger_direction(hand_dict)
                    
                    mockingJay = handGesture_mockingJay(hand_dict)
                    print('order:', order)

                    order_queue = order_queue + order
                    order_acceptance_count += 1

                    if order_acceptance_count % 2 == 0:     
                        if handGesture_rockNroll(hand_dict):
                            self.flip()

                        if handGesture_mockingJay(hand_dict):
                            self.tello.land()
                            self.send_rc_control = False
                            should_stop = True

                        #Only takes order once every 1/2 seconds (software and model running at 30FPS)
                        if('down' in order_queue):
                            self.up_down_velocity = -self.S
                        elif('up' in order_queue):
                            self.up_down_velocity = self.S
                        else:
                            self.up_down_velocity = 0

                        if('left' in order_queue):
                            self.left_right_velocity = self.S
                        elif('right' in order_queue):
                            self.left_right_velocity = -self.S
                        else:
                            self.left_right_velocity = 0
                            
                        # Control forward and backward
                        if('backwards' in order_queue):
                            self.for_back_velocity = -self.S
                        elif('forwards' in order_queue):
                            self.for_back_velocity = self.S
                        else:
                            self.for_back_velocity = 0

                        order_queue = ''
                    else:
                        print('stationary')
                        self.stationary()
                else:
                    print('stationary')
                    self.stationary()

            self.update()
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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

def image_handpose(image_files):
    # For static images:
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as hands:
        for idx, file in enumerate(image_files):
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread(file), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                print('hand_landmarks:', hand_landmarks)
                print(
                    f'Index finger tip coordinates: (',
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                    f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
                )
                mp_drawing.draw_landmarks(
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            print('sample_output/' + str(idx) + '.png')
            cv2.imwrite(
                'sample_output/' + str(idx) + '.png', cv2.flip(annotated_image, 1))

def webcam_handpose():
# For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        max_num_hands = 1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        
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
            image_height, image_width, _ = image.shape
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            print('--------------------------')
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
    
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                hand_dict = get_all_point_of_hands(mp_hands, hand_landmarks, image_width, image_height)
                # get_index_finger_direction(hand_dict)
                # get_thumb_direction(hand_dict)
                # get_ringmid_direction(hand_dict)
                # get_ringmid_direction(hand_dict)
                rockNroll = handGesture_rockNroll(hand_dict)
                mockingJay = handGesture_mockingJay(hand_dict)

                print('rockNroll', rockNroll )
                print('mockingJay', mockingJay)
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


def main():
    
    #Image inference
    # image_handpose(glob.glob(f'sample_input/*'))
    
    #Webcam inference
    webcam_handpose()

    #Drone inference
    # controls = HandPoseControl()
    # controls.run()

if __name__ == '__main__':
    main()