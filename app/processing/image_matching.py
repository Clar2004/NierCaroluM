import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import random, sys, os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        return os.path.join(os.getcwd(), relative_path)

images = [
   "static/assets/image_matching/circle.jpg",
   "static/assets/image_matching/square.jpg"
]

prev_position = None  
smoothed_position = None  
drawing = False  
drawing_start_time = None  
countdown_time = 3  
last_thumbs_up_time = None  
game_started = False  
waiting_for_thumbs_up = True  
countdown_started = False

trigger_start_countdown = False 
trigger_end_countdown = False

alpha = 0.2 

def image_matching(drawn_image, target_image):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(drawn_image, None)
    kp2, des2 = orb.detectAndCompute(target_image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    similarity_percentage = len(matches) / len(kp1) * 100 if len(kp1) > 0 else 0
    return similarity_percentage

def is_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    # Check if thumb is up
    thumb_up = thumb_tip.y < thumb_ip.y < thumb_mcp.y < thumb_cmc.y
    
    # Check if other fingers are down
    index_down = index_tip.y > index_dip.y > index_pip.y > index_mcp.y
    middle_down = middle_tip.y > middle_dip.y > middle_pip.y > middle_mcp.y
    ring_down = ring_tip.y > ring_dip.y > ring_pip.y > ring_mcp.y
    pinky_down = pinky_tip.y > pinky_dip.y > pinky_pip.y > pinky_mcp.y
    
    thumbs_up = thumb_up and index_down and middle_down and ring_down and pinky_down
    return thumbs_up

def draw_text(frame, text, position=(50, 50), font_scale=1, color=(0, 255, 0), thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def is_pinch(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance < 0.05 

def draw_on_canvas(cursor_x, cursor_y, canvas, prev_position, smoothed_position, alpha=0.2):
    """Draw on the canvas with smoothing"""
    if prev_position is not None:
        smoothed_position = (
            int(alpha * cursor_x + (1 - alpha) * prev_position[0]),
            int(alpha * cursor_y + (1 - alpha) * prev_position[1]),
        )
        if 0 <= smoothed_position[0] < canvas.shape[1] and 0 <= smoothed_position[1] < canvas.shape[0]:
            cv2.line(canvas, prev_position, smoothed_position, (0, 0, 0), 5)  
        prev_position = smoothed_position  
    else:
        prev_position = (cursor_x, cursor_y)

    return prev_position, smoothed_position

def game_loop(cap):
    global prev_position, smoothed_position, drawing, drawing_start_time, countdown_time, last_thumbs_up_time, game_started, waiting_for_thumbs_up
    global trigger_end_countdown, trigger_start_countdown, countdown_started

    alpha = 0.2
    isInitialized = False
    canvas = None
    random_number = -1
    
    game_started = False
    drawing = False
    drawing_start_time = None
    last_thumbs_up_time = None
    waiting_for_thumbs_up = True
    trigger_start_countdown = False
    trigger_end_countdown = False
    prev_position = None
    smoothed_position = None
    countdown_started = False
    trigger_start_countdown = False 
    trigger_end_countdown = False

    pygame.mixer.init()
    pygame.mixer.music.load(get_resource_path("static/assets/sound/boss_bg_8bit.mp3"))
    pygame.mixer.music.play(loops=-1, start=0.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if not isInitialized and canvas is None:
            isInitialized = True
            canvas = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)
        
        last_thumbs_up_detection_time = 0 

        if results.multi_hand_landmarks:
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label

                if hand_label == 'Right':
                    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                    
                    # if prev_position and smoothed_position:
                        # print(f"Circle coordinates: (x: {x}, y: {y})")

                    if is_pinch(landmarks) and game_started:
                        if drawing:
                            print("Pinch detected! Stop drawing for now.")
                            drawing = False 
                            prev_position = None 
                            smoothed_position = None  
                    elif not is_pinch(landmarks) and game_started:
                        if not drawing:
                            prev_position = (x, y)
                            smoothed_position = (x, y)
                        drawing = True

                    if drawing and game_started:
                        prev_position, smoothed_position = draw_on_canvas(x, y, canvas, prev_position, smoothed_position, alpha)
                        
                        # print(f"Line from: {prev_position} to {smoothed_position}")

                    overlay = np.zeros_like(frame, dtype=np.uint8)
                    
                    # cv2.circle(overlay, (x, y), 10, (192, 192, 192), -1)

                    alpha = 0.5 
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    if is_thumbs_up(landmarks):
                        current_time = time.time()  

                        if current_time - last_thumbs_up_detection_time >= 1:  
                            if not game_started and waiting_for_thumbs_up:
                                trigger_start_countdown = True
                                waiting_for_thumbs_up = False
                                
                                random_number = random.randint(0, 1)
                                # print("Random number: ", random_number)
                                from app import match_start
                                match_start(random_number)
 
                                # print("Thumbs up detected! Starting countdown...")
                                canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255  
                                last_thumbs_up_detection_time = current_time
                                last_thumbs_up_time = time.time()  
                        else:
                            print("Cooldown active. Please wait for 3 seconds before detecting another thumbs-up.")
                        
        if trigger_start_countdown and last_thumbs_up_time is not None:
            # print("Starting game countdown")

            time_elapsed = time.time() - last_thumbs_up_time
            countdown_value = int(time_elapsed)
            
            from app import count_down_start
            count_down_start(countdown_value)

            if countdown_value > 3:
                print("Countdown ended! Start drawing...")
                game_started = True  
                drawing = True  
                
                trigger_start_countdown = False 
                last_thumbs_up_time = None
                drawing_start_time = time.time() 
                
        if game_started and drawing_start_time is not None:
            # print("Drawing time started!")
            time_elapsed = time.time() - drawing_start_time
            countdown_value = int(time_elapsed)
            
            from app import drawing_start
            drawing_start(countdown_value)
            
            if countdown_value > 20: 
                
                game_started = False
                drawing = False
                
                drawing_start_time = None  
                
                trigger_end_countdown = True
                last_thumbs_up_time = time.time()  
                print("Drawing time ended! Waiting for 3 seconds before resetting...")

        if trigger_end_countdown and last_thumbs_up_time is not None:
            # print("starting end cooldown")
            
            time_elapsed = time.time() - last_thumbs_up_time
            countdown_value = int(time_elapsed)
            
            from app import count_down_end
            count_down_end(countdown_value)
            if time_elapsed > 3:
                waiting_for_thumbs_up = True
                trigger_end_countdown = False
                last_thumbs_up_time = None
                print("Game reset! Waiting for thumbs up...")
                
                from app import send_accuracy, set_match_accuracy, reset_game
                target_image = cv2.imread(get_resource_path(images[random_number]))
                similarity_percentage = image_matching(canvas, target_image)
                set_match_accuracy(similarity_percentage)
                print(f"Accuracy: {similarity_percentage}")
                reset_game()
                send_accuracy()

        canvas_resized = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))
        frame_with_canvas = cv2.addWeighted(frame, 0, canvas_resized, 1, 0)

        _, jpeg_frame = cv2.imencode('.jpg', frame_with_canvas)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n')
