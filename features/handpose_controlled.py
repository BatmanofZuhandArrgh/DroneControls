import cv2
import glob as glob
import pygame
import mediapipe as mp
import numpy as np
import matplotlib.path as mpltPath
from sklearn.linear_model import LinearRegression
import time

from move import Controls

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_styles = mp.solutions.drawing_styles

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
            
            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])
            self.update()

            frame = frame_read.frame                

            text = "Battery: {}%".format(self.tello.get_battery())
            cv2.putText(frame, text, (5, 720 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            with mp_hands.Hands(
                max_num_hands = 1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
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
                    order += get_thumb_direction(hand_dict)
                    order += get_ringmid_direction(hand_dict)

                    if('down' in order):
                        self.up_down_velocity = -self.S
                    elif('up' in order):
                        self.up_down_velocity = self.S
                    else:
                        self.up_down_velocity = 0

                    if('left' in order):
                        self.left_right_velocity = self.S
                    elif('right' in order):
                        self.left_right_velocity = -self.S
                    else:
                        self.left_right_velocity = 0
                        
                    # Control forward and backward
                    if('backwards' in order):
                        self.for_back_velocity = -self.S
                    elif('forwards' in order):
                        self.for_back_velocity = self.S
                    else:
                        self.for_back_velocity = 0

                    cv2.putText(img = frame, text = '|'.join(order), 
                        org = (30,15), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 1,color = (0, 0, 255), thickness = 1)
                else:
                    print('stationary')
                    self.stationary()

            self.update()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame = np.rot90(frame)
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / self.FPS)

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
                    annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmark_style(),
                    drawing_styles.get_default_hand_connection_style())
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
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmark_style(),
                        drawing_styles.get_default_hand_connection_style())

                hand_dict = get_all_point_of_hands(mp_hands, hand_landmarks, image_width, image_height)
                get_index_finger_direction(hand_dict)
                get_thumb_direction(hand_dict)
                get_ringmid_direction(hand_dict)


            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

def get_all_point_of_hands(
    mp_hands, 
    hand_landmarks, 
    image_width, 
    image_height,
):
    hand_dict = {}
    
    hand_dict['thumb'] = {}
    hand_dict['thumb']['thumb_cmc'] = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height) #Renamed for easy parsing
    hand_dict['thumb']['thumb_mcp'] = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
    hand_dict['thumb']['thumb_ip']  = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
    hand_dict['thumb']['thumb_tip'] = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

    hand_dict['index_finger'] = {}
    hand_dict['index_finger']['index_finger_mcp'] = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
    hand_dict['index_finger']['index_finger_pip'] = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
    hand_dict['index_finger']['index_finger_dip'] = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
    hand_dict['index_finger']['index_finger_tip'] = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)

    hand_dict['wrist'] = {}
    hand_dict['wrist']['wrist_mcp'] = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)

    hand_dict['middle_finger'] = {}
    hand_dict['middle_finger']['middle_finger_mcp'] = (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
    hand_dict['middle_finger']['middle_finger_pip'] = (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
    hand_dict['middle_finger']['middle_finger_dip'] = (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
    hand_dict['middle_finger']['middle_finger_tip'] = (hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

    hand_dict['ring_finger'] = {}
    hand_dict['ring_finger']['ring_finger_mcp'] = (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
    hand_dict['ring_finger']['ring_finger_pip'] = (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
    hand_dict['ring_finger']['ring_finger_dip'] = (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
    hand_dict['ring_finger']['ring_finger_tip'] = (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)
    
    hand_dict['pinky_finger'] = {}
    hand_dict['pinky_finger']['pinky_finger_mcp'] = (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
    hand_dict['pinky_finger']['pinky_finger_pip'] = (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
    hand_dict['pinky_finger']['pinky_finger_dip'] = (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
    hand_dict['pinky_finger']['pinky_finger_tip'] = (hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)

    return hand_dict

def get_angle_fingerVSox(
    finger,
    hand_dict,
):
    finger_points = [(int(value[0]), int(value[1])) for value in hand_dict[finger].values()]
    x = [point[0] for point in finger_points]
    x = np.array(x).reshape((-1,1))
    y = [point[1] for point in finger_points] 

    
    reg = LinearRegression(fit_intercept=True).fit(x, np.array(y))
    slope = reg.coef_[0]
    angle = np.arctan(slope) * 180 / np.pi 
    return angle if angle > 0 else 180 + angle

def get_angle_fingerVSfinger(
    finger0,
    finger1,
    hand_dict,
):
    angle0 = get_angle_fingerVSox(finger0, hand_dict)
    angle1 = get_angle_fingerVSox(finger1, hand_dict)
    # print(angle0 - angle1)
    return abs(angle0 - angle1)

def get_thumb_direction(
    hand_dict,
    vertical_threshold = 2,
    horizontal_threshold = 0.8,
    ):
    thump_cmc = hand_dict['thumb']['thumb_cmc']
    thump_mcp = hand_dict['thumb']['thumb_mcp']
    thump_ip  = hand_dict['thumb']['thumb_ip']
    thump_tip = hand_dict['thumb']['thumb_tip']

    thump_points = [thump_cmc, thump_mcp, thump_ip, thump_tip]

    if are_points_inside_polygon([thump_tip, thump_ip], hand_dict, 'thumb'):
        return 0, ''

    x = [p[0] for p in thump_points]
    x = np.array(x).reshape((-1,1))
    y = [p[1] for p in thump_points]
    reg = LinearRegression(fit_intercept=True).fit(x, np.array(y))
    slope = reg.coef_[0]

    order = ''
    if slope > vertical_threshold or slope < -vertical_threshold:
        if thump_tip[1] > thump_cmc[1]:
            order += '|down'
        elif thump_tip[1] < thump_cmc[1]:
            order += '|up'
        else:
            order += '|stable_horizontal'
            
    if slope < horizontal_threshold and slope > -horizontal_threshold:
        if thump_tip[0] < thump_cmc[0]:
            order += '|left'
        elif thump_tip[1] < thump_cmc[1]:
            order += '|right'
        else:
            order += '|stable_vertical'

    # print('thumb', slope, order)
    return order

def get_index_finger_direction(
    hand_dict, 
    vertical_threshold = 2,
    horizontal_threshold = 0.8,
    ):

    # wrist = (hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
    index_mcp = hand_dict['index_finger']['index_finger_mcp']
    index_pip = hand_dict['index_finger']['index_finger_pip']
    index_dip = hand_dict['index_finger']['index_finger_dip']
    index_tip = hand_dict['index_finger']['index_finger_tip']

    if are_points_inside_polygon([index_tip, index_pip], hand_dict, 'index_finger'):
        return 0, ''

    index_points = [index_mcp, index_pip, index_dip, index_tip]
    x = [p[0] for p in index_points]
    x = np.array(x).reshape((-1,1))
    y = [p[1] for p in index_points]
    reg = LinearRegression(fit_intercept=True).fit(x, np.array(y))
    slope = reg.coef_[0]

    order = ''
    if slope > vertical_threshold or slope < -vertical_threshold:
        if index_tip[1] > index_mcp[1]:
            order += '|down'
        elif index_tip[1] < index_mcp[1]:
            order += '|up'
        else:
            order += '|stable_horizontal'
    
    if slope < horizontal_threshold and slope > -horizontal_threshold:
        if index_tip[0] < index_mcp[0]:
            order += '|left'
        elif index_tip[1] < index_mcp[1]:
            order += '|right'
        else:
            order += '|stable_vertical'

    # print('index', slope, order)
    return order

def get_extreme(hand_dict, ref_finger):
    hand_points = []
    for key in [x for x in hand_dict.keys() if x != ref_finger]:
        hand_points.extend([coords for coords in hand_dict[key].values()])

    hand_x = [point[0] for point in hand_points]
    hand_y = [point[1] for point in hand_points]

    left_most, right_most = np.argmin(hand_x), np.argmax(hand_x)
    top, bottom = np.argmin(hand_y), np.argmax(hand_y)

    left_most, right_most = hand_points[left_most], hand_points[right_most]
    top, bottom = hand_points[top], hand_points[bottom]

    return [left_most, top, right_most, bottom]

def are_points_inside_polygon(points_to_check, hand_dict, ref_finger):
    extreme_points = get_extreme(hand_dict, ref_finger)  
    polygon = [[int(point[0]), int(point[1])] for point in extreme_points]
    path = mpltPath.Path(polygon)
    
    is_inside = path.contains_points(points_to_check)
    
    return True in is_inside

def get_ringmid_direction(
    hand_dict, 
    lower_angle_threshold = 20,
    upper_angle_threshold = 30,
    ):
    diff_angle = get_angle_fingerVSfinger('middle_finger', 'ring_finger', hand_dict)
    if diff_angle > lower_angle_threshold and diff_angle < upper_angle_threshold:
        if hand_dict['middle_finger']['middle_finger_tip'][0] <= hand_dict['ring_finger']['ring_finger_tip'][0]:
            return '|backwards'
        elif hand_dict['middle_finger']['middle_finger_tip'][0] > hand_dict['ring_finger']['ring_finger_tip'][0]:
            return '|forwards'
    else:
        return '!stable_depth'
    
def main():
    
    #Image inference
    # image_handpose(glob.glob(f'sample_input/*'))
    
    #Webcam inference
    webcam_handpose()

    pass

if __name__ == '__main__':
    main()
    pass