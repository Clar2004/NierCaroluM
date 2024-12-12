import cv2
import numpy as np
import mediapipe as mp
from threading import Lock
from collections import deque
import pygame

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

threshold_level = 128  
lock = Lock()
input_image = None
previous_x = None  

thumb_path = deque(maxlen=15)  
line_alpha = 1.0  

def initialize_threshold_image(image_path, resize_width=800, resize_height=600):
    global input_image
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    input_image = cv2.resize(input_image, (resize_width, resize_height))


def adjust_threshold(image, level):
    _, thresholded = cv2.threshold(image, level, 255, cv2.THRESH_BINARY)
    return thresholded


def generate_threshold_frames(camera, frame_width=800, frame_height=600):
    global threshold_level, input_image, previous_x, thumb_path, line_alpha
    
    pygame.mixer.init()
    pygame.mixer.music.load("static/assets/sound/boss_bg_8bit.mp3")
    pygame.mixer.music.play(loops=-1, start=0.0)

    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Could not read frame")
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            line_alpha = 1.0  
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                if hand_label == 'Right':
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                    thumb_x = int(thumb_tip.x * frame_width)
                    thumb_y = int(thumb_tip.y * frame_height)

                    if len(thumb_path) == 0 or np.linalg.norm(
                        np.array(thumb_path[-1]) - np.array((thumb_x, thumb_y))
                    ) > 10: 
                        thumb_path.append((thumb_x, thumb_y))

                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_x = int(index_tip.x * frame_width)

                    if previous_x is not None:
                        delta_x = index_x - previous_x
                        if delta_x > 15:  
                            with lock:
                                threshold_level = min(255, threshold_level + 5)
                                print(f"Threshold level increased to {threshold_level}")
                        elif delta_x < -15: 
                            with lock:
                                threshold_level = max(0, threshold_level - 5)
                                print(f"Threshold level decreased to {threshold_level}")

                    previous_x = index_x
        else:
            if len(thumb_path) > 0:
                line_alpha -= 0.1  
                if line_alpha <= 0:
                    thumb_path.clear()  
                    line_alpha = 0

        with lock:
            thresholded_image = adjust_threshold(input_image, threshold_level)

        threshold_display = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)
        line_overlay = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        for i in range(1, len(thumb_path)):
            cv2.line(
                line_overlay,
                thumb_path[i - 1],
                thumb_path[i],
                (255, 255, 255), 
                thickness=3,  
                lineType=cv2.LINE_AA,
            )

        combined_frame = cv2.addWeighted(threshold_display, 1.0, line_overlay, line_alpha, 0)

        cv2.putText(
            combined_frame,
            f"Threshold Level: {threshold_level}",
            (10, 25),  
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,  
            (0, 255, 0), 
            1,
        )
        
        _, buffer = cv2.imencode(".jpg", combined_frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
