import cv2
import glob as glob
import mediapipe as mp

from move import Controls

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_styles = mp.solutions.drawing_styles

class HandposeControls(Controls):
    """ 
    Once the drone detect a face it will follow that face, face to face    
    """

    def __init__(self):
        super(HandposeControls, self).__init__()
        self.area_to_stabilize = (25000, 30000)
        self.height = 720
        self.width = 960

        self.upper_limit = int(self.height//3)
        self.lower_limit = self.upper_limit * 2
        
        self.rightmost_limit = int(self.width//3)
        self.leftmost_limit = self.rightmost_limit*2
        
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
            
            if frame_read.stopped:
                break

            self.screen.fill([0, 0, 0])
            self.update()

            frame = frame_read.frame                

            text = "Battery: {}%".format(self.tello.get_battery())
            cv2.putText(frame, text, (5, 720 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame, [self.face_coords, self.area] = findFace(frame)
            
            #Move in left right up down
            if(self.face_coords == [0,0]):
                print('nothing`')
                continue
            else:
                if(self.face_coords[1] < self.upper_limit):
                    print('down')
                elif(self.face_coords[1] > self.lower_limit):
                    print('up')
                else:
                    print('center')

                if(self.face_coords[0] < self.rightmost_limit):
                    print('move_left')
                elif(self.face_coords[0] > self.leftmost_limit):
                    print('move_right')
                else:
                    print('center')
            
            #Move forward, backward
            if(self.area == 0):
                continue
            else:
                if(self.area > self.area_to_stabilize[-1]):
                    print('move backwards')
                elif(self.area < self.area_to_stabilize[0]):
                    print('move forwards')
                else:
                    print('stationary')
            

            #Draw grid
            cv2.line(frame, (0,self.lower_limit),(self.width, self.lower_limit), (255, 0, 0), 1, 1)
            cv2.line(frame, (0,self.upper_limit),(self.width, self.upper_limit), (255, 0, 0), 1, 1)
            
            cv2.line(frame, (self.rightmost_limit, 0),(self.rightmost_limit, self.height), (255, 0, 0), 1, 1)
            cv2.line(frame, (self.leftmost_limit, 0),(self.leftmost_limit, self.height), (255, 0, 0), 1, 1)

            cv2.putText(img = frame, text = f'{self.face_coords}_{self.area}', 
                        org = (30,30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale = 1,color = (0, 0, 255), thickness = 2)


            frame = np.rot90(frame)
            frame = np.flipud(frame)

            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            time.sleep(1 / self.FPS)

        self.tello.streamoff()
        # Call it always before finishing. To deallocate resources.
        self.tello.end()

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
            not self.tello.land()
            self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity,
                self.up_down_velocity, self.yaw_velocity)

def findFace(img, mode = 'CascadeClassifier'):
    if mode == 'CascadeClassifier':
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)
    else:
        faces = []

    face_list = []
    face_area_list = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w*h
        face_list.append([cx, cy])
        face_area_list.append(area)

    if len(face_list) != 0:
        i = face_area_list.index(max(face_area_list))
        return img, [face_list[i], face_area_list[i]]

    else:
        return img, [[0,0], [0]]

def webcam_findFace():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print('cannot open camera')
        exit()

    AREA_THRESHOLD = [25000, 30000]

    while True:
        _, img = cap.read()
        
        img, [face_coords, area] = findFace(img)
        cv2.putText(img = img, text = f'{face_coords}_{area}', 
                    org = (30,30), fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 1,color = (0, 0, 255), thickness = 2)
        
        cv2.circle(img, tuple(face_coords), 1, (255,0,0), 2,2)

        upper_limit = int(img.shape[0]/3)
        lower_limit = int(img.shape[0]/3)*2
        cv2.line(img, (0,int(img.shape[0]/3)),(img.shape[1], int(img.shape[0]/3)), (255, 0, 0), 1, 1)
        cv2.line(img, (0,int(img.shape[0]/3*2)),(img.shape[1], int(img.shape[0]/3*2)), (255, 0, 0), 1, 1)

        rightmost_limit = int(img.shape[1]/3)
        leftmost_limit =  int(img.shape[1]/3) * 2
        cv2.line(img, (int(img.shape[1]/3), 0),(int(img.shape[1]/3), img.shape[0]), (255, 0, 0), 1, 1)
        cv2.line(img, (int(img.shape[1]/3*2), 0),(int(img.shape[1]/3*2), img.shape[0]), (255, 0, 0), 1, 1)

        if(face_coords == [0,0]):
            continue
        else:
            if(face_coords[1] < upper_limit):
                print('down')
            elif(face_coords[1] > lower_limit):
                print('up')

            if(face_coords[0] < rightmost_limit):
                print('move_left')
            elif(face_coords[0] > leftmost_limit):
                print('move_right')

        if(area == 0):
            continue
        else:
            if(area > AREA_THRESHOLD[-1]):
                print('move backwards')
            elif(area < AREA_THRESHOLD[0]):
                print('move forwards')
            else:
                print('stationary')
            
        cv2.imshow("Output", img)

        press = cv2.waitKey(1)
        if press == ord('q'):
            print('Quitting')
            break


def image_handpose(image_files):
    # For static images:
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
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
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmark_style(),
                        drawing_styles.get_default_hand_connection_style())
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

def main():
    inputs = glob.glob(f'sample_input/*')
    image_handpose(inputs)
    pass

if __name__ == '__main__':
    main()
    pass