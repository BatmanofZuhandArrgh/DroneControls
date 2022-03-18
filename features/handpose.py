import matplotlib.path as mpltPath
import numpy as np

from sklearn.linear_model import LinearRegression

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

def areParellel(finger0, finger1, hand_dict, threshold_angle_diff = 10):
    angle0 = get_angle_fingerVSox(finger0,hand_dict)
    angle1 = get_angle_fingerVSox(finger1,hand_dict)
    
    if abs(angle0 - angle1) <= threshold_angle_diff:
        return True, angle0
    return False, 0

def areParallelandUpwards(finger0, finger1, hand_dict, threshold_angle_diff = 10):
    ifParallel, angle0 = areParellel(finger0, finger1, hand_dict, threshold_angle_diff)
    if ifParallel and abs(angle0 - 90) <= 2*threshold_angle_diff:
        return True
    return False

def handGesture_mockingJay(hand_dict):
    #If both pinky and index are not in the fist, check if they are parallel and upwards, return True
    if  are_points_inside_polygon([hand_dict['index_finger']['index_finger_tip']], hand_dict, 'index_finger') or \
        are_points_inside_polygon([hand_dict['middle_finger']['middle_finger_tip']], hand_dict, 'middle_finger') or \
        are_points_inside_polygon([hand_dict['ring_finger']['ring_finger_tip']], hand_dict, 'ring_finger') or \
        hand_dict['middle_finger']['middle_finger_tip'][1] > hand_dict['middle_finger']['middle_finger_pip'][1] or \
        hand_dict['ring_finger']['ring_finger_tip'][1] > hand_dict['ring_finger']['ring_finger_pip'][1] or \
        hand_dict['index_finger']['index_finger_tip'][1] > hand_dict['index_finger']['index_finger_pip'][1]:
        return False
    return areParallelandUpwards('ring_finger', 'index_finger', hand_dict) and areParallelandUpwards('middle_finger', 'index_finger', hand_dict)

def handGesture_rockNroll(hand_dict):
    #If both pinky and index are not in the fist, check if they are parallel and upwards, return True
    if are_points_inside_polygon([hand_dict['pinky_finger']['pinky_finger_tip']], hand_dict, 'pinky_finger') or \
        are_points_inside_polygon([hand_dict['index_finger']['index_finger_tip']], hand_dict, 'index_finger') or \
        hand_dict['pinky_finger']['pinky_finger_tip'][1] > hand_dict['pinky_finger']['pinky_finger_dip'][1]:
        return False
    return areParallelandUpwards('pinky_finger', 'index_finger', hand_dict)


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
        elif thump_tip[0] > thump_cmc[0]:
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

    index_mcp = hand_dict['index_finger']['index_finger_mcp']
    index_pip = hand_dict['index_finger']['index_finger_pip']
    index_dip = hand_dict['index_finger']['index_finger_dip']
    index_tip = hand_dict['index_finger']['index_finger_tip']

    if are_points_inside_polygon([index_tip, index_pip], hand_dict, 'index_finger'):
        return ''

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
        order += '|stable_vertical'
    
    if slope < horizontal_threshold and slope > -horizontal_threshold:
        if index_tip[0] < index_mcp[0]:
            order += '|left'
        elif index_tip[0] > index_mcp[0]:
            order += '|right'
    else:
        order += '|stable_horizontal'

    print('index', slope, order)
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
        return '|stable_depth'
    