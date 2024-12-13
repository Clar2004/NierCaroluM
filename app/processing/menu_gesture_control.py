import cv2
import mediapipe as mp
from flask import Flask, Response, render_template
import os
import sys
import pygame
import time, numpy as np
from state import *

app = Flask(__name__)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        return os.path.join(os.getcwd(), relative_path)

# title_image_path = os.path.join(os.getcwd(), 'static', 'assets', 'images', 'Title_White.png')
# cursor_image_path = os.path.join(os.getcwd(), 'static', 'assets', 'cursors', 'CursorNier.png')
# background_video_path = os.path.join(os.getcwd(), 'static', 'assets', 'TitleBackground.mp4')

title_image = cv2.imread(get_resource_path('static/assets/images/Title_white.png'), cv2.IMREAD_UNCHANGED) 
cursor_image2 = cv2.imread(get_resource_path('static/assets/cursors/CursorNier.png'), cv2.IMREAD_UNCHANGED) 
resize_factor = 0.8 
new_width = int(cursor_image2.shape[1] * resize_factor)
new_height = int(cursor_image2.shape[0] * resize_factor)

cursor_image = cv2.resize(cursor_image2, (new_width, new_height))

cap_background = cv2.VideoCapture(get_resource_path('static/assets/TitleBackground.mp4'))

screen_width = 2860
screen_height = 1080

last_click_time = time.time()

def is_pinch(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance < 0.05

def is_one_finger(landmarks):
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    return index_tip < index_mcp and thumb_tip > index_tip and middle_tip > index_tip

def overlay_image_with_alpha(background, overlay, x, y):
    h, w = overlay.shape[:2]
    if x + w > background.shape[1] or y + h > background.shape[0]:
        return background

    for c in range(0, 3): 
        background[y:y+h, x:x+w, c] = background[y:y+h, x:x+w, c] * (1 - overlay[:, :, 3] / 255.0) + overlay[:, :, c] * (overlay[:, :, 3] / 255.0)
    return background

def resize_and_crop_background(frame_bg, screen_width, screen_height):
    """Resize and crop the background video to fill the whole screen."""
    bg_height, bg_width = frame_bg.shape[:2]
    aspect_ratio = bg_width / bg_height
    
    if aspect_ratio > 1:
        new_width = screen_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)

    frame_bg_resized = cv2.resize(frame_bg, (new_width, new_height))

    crop_x = (new_width - screen_width) // 2
    crop_y = (new_height - screen_height) // 2

    frame_bg_cropped = frame_bg_resized[crop_y:crop_y+screen_height, crop_x:crop_x+screen_width]

    return frame_bg_cropped

def detect_gestures_and_stream(cap_camera):
    global screen_height, screen_width, last_click_time  

    last_emitted = {'x': None, 'y': None, 'gesture': None}
    pygame.mixer.init()
    pygame.mixer.music.load(get_resource_path("static/assets/sound/TitleSound.mp3"))
    pygame.mixer.music.play(loops=-1, start=0.0)

    while True:
        ret_bg, frame_bg = cap_background.read()
        if not ret_bg:
            print("Error: Could not read background video frame")
            break

        frame_bg_resized_full = resize_and_crop_background(frame_bg, screen_width, screen_height)

        ret_cam, frame_cam = cap_camera.read()
        if not ret_cam:
            print("Error: Could not read camera frame")
            break

        frame_cam = cv2.resize(frame_cam, (screen_width, screen_height))

        frame_cam_rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_cam_rgb)

        if frame_bg_resized_full.shape[:2] != frame_cam.shape[:2]:
            frame_cam = cv2.resize(frame_cam, (frame_bg_resized_full.shape[1], frame_bg_resized_full.shape[0]))

        frame_combined = cv2.addWeighted(frame_bg_resized_full, 1, frame_cam, 0.0, 0)

        cursor_x, cursor_y = None, None 

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = hand_handedness.classification[0].label
                if hand_label == 'Left':
                    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    cursor_x = int((1 - index_finger.x) * screen_width)
                    cursor_y = int(index_finger.y * screen_height)

                    if is_pinch(hand_landmarks.landmark):
                        if (last_emitted['x'], last_emitted['y'], last_emitted['gesture']) != (cursor_x, cursor_y, 'pinch'):
                            last_emitted.update({'x': cursor_x, 'y': cursor_y, 'gesture': 'pinch'})

                    elif is_one_finger(hand_landmarks.landmark):
                        if (last_emitted['x'], last_emitted['y'], last_emitted['gesture']) != (cursor_x, cursor_y, 'click'):
                            last_emitted.update({'x': cursor_x, 'y': cursor_y, 'gesture': 'click'})


        title_image_width = int(screen_width // 3) 
        title_image_height = int(title_image.shape[0] * (title_image_width / title_image.shape[1]))  
        title_x = (screen_width - title_image_width) // 2  
        title_y = screen_height // 4 
        title_resized = cv2.resize(title_image, (title_image_width, title_image_height))
        frame_combined = overlay_image_with_alpha(frame_combined, title_resized, title_x, title_y)

        font_scale = 1.2 
        font = cv2.FONT_HERSHEY_SIMPLEX
        play_demo_text = "Play Demo"
        text_size = cv2.getTextSize(play_demo_text, font, font_scale, 2)[0] 
        text_x = (screen_width - text_size[0]) // 2 
        text_y = title_y + title_image_height + 100  

        cv2.putText(frame_combined, play_demo_text, (text_x, text_y), font, font_scale, (255, 255, 255), 2)

        if cursor_x is not None and cursor_y is not None:
            cursor_x = max(0, min(cursor_x, screen_width - cursor_image.shape[1]))
            cursor_y = max(0, min(cursor_y, screen_height - cursor_image.shape[0]))

            def is_cursor_hovering(cursor_x, cursor_y, text_x, text_y, text_width, text_height):
                return text_x <= cursor_x <= text_x + text_width and text_y - text_height <= cursor_y <= text_y + 30

            if is_cursor_hovering(cursor_x, cursor_y, text_x, text_y, text_size[0], text_size[1]):
                extra_width = 150  
                cv2.line(frame_combined, 
                        (text_x - extra_width // 2, text_y + 10),  
                        (text_x + text_size[0] + extra_width // 2, text_y + 10),  
                        (255, 255, 255), 2)  
            # if is_cursor_hovering(cursor_x, cursor_y, text_x, text_y, text_size[0], text_size[1]) and last_emitted['gesture'] == 'click':
            #     print("Play Demo clicked!")
            #     from app import play_demo
            #     play_demo()
            
            current_time = time.time()
            if is_cursor_hovering(cursor_x, cursor_y, text_x, text_y, text_size[0], text_size[1]) and last_emitted['gesture'] == 'click':
                if current_time - last_click_time >= 2: 
                    print("Play Demo clicked!")
                    from state import game_state
                    game_state.isPlayDemo = True
                    from app import play_demo
                    play_demo()
                    last_click_time = current_time 
                else:
                    print("Cooldown active. Please wait...")

        if cursor_x is not None and cursor_y is not None:
            cursor_x = max(0, min(cursor_x, screen_width - cursor_image.shape[1]))
            cursor_y = max(0, min(cursor_y, screen_height - cursor_image.shape[0]))

            cursor_resized = cv2.resize(cursor_image, (cursor_image.shape[1], cursor_image.shape[0]))
            frame_combined = overlay_image_with_alpha(frame_combined, cursor_resized, cursor_x, cursor_y)

        _, jpeg_frame = cv2.imencode('.jpg', frame_combined)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n\r\n')
