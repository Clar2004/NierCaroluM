import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import wiener
import pygame
import sys, os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        return os.path.join(os.getcwd(), relative_path)

filter_type = "gaussian"  
filter_radius = 50  
base_image = None
blurred_image = None
circle_center = (320, 240)

def initialize_base_image(image_path, blur_strength=51):
    global base_image, blurred_image
    base_image = cv2.imread(image_path)
    if base_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    base_image = cv2.resize(base_image, (640, 480))
    blurred_image = create_blurred_image(base_image, blur_strength)

def create_blurred_image(image, blur_strength=51):
    if blur_strength % 2 == 0:
        blur_strength += 1
    return cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

def apply_circle_filter(image, original_image, center, radius, kernel_size):
    if kernel_size > 0 and kernel_size % 2 == 0:
        kernel_size += 1

    height, width = image.shape[:2]
    x_center = max(radius, min(center[0], width - radius))
    y_center = max(radius, min(center[1], height - radius))
    center = (x_center, y_center)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1) 

    if kernel_size > 0:
        filtered_image = cv2.GaussianBlur(original_image, (kernel_size, kernel_size), 0)
    else:
        filtered_image = original_image

    filtered_region = cv2.bitwise_and(filtered_image, filtered_image, mask=mask)
    unfiltered_region = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    updated_image = cv2.add(filtered_region, unfiltered_region)

    image[:, :] = updated_image

    visualization_frame = image.copy()
    cv2.circle(visualization_frame, center, radius, (255, 255, 255), 2) 

    return visualization_frame

def process_frame(camera):

    global filter_radius, circle_center, blurred_image, base_image
    kernel_size = 0  

    success, frame = camera.read()
    if not success:
        print("Failed to capture frame from camera.")
        return None

    frame = cv2.flip(frame, 1) 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_handedness:
        hands_data = []
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_handedness.classification[0].label 
            hands_data.append({
                "label": hand_label,
                "landmarks": hand_landmarks.landmark
            })

        for hand in hands_data:
            landmarks = hand["landmarks"]
            hand_label = hand["label"]

            if hand_label == "Left":
                fingers_up = 0
                if landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                    fingers_up += 1
                if landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
                    fingers_up += 1

                kernel_size = fingers_up * 10 

                if kernel_size > 0 and kernel_size % 2 == 0:
                    kernel_size += 1

                # print(f"Left hand detected: {fingers_up} fingers up. Kernel size: {kernel_size}")

            elif hand_label == "Right":
                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                pinch_distance = np.linalg.norm(
                    [thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]
                )
                filter_radius = max(10, min(int(pinch_distance * 200), min(base_image.shape[1], base_image.shape[0]) // 2))
                # print(f"Pinch radius set to {filter_radius}.")

                x_center = int((thumb_tip.x + index_tip.x) / 2 * base_image.shape[1])
                y_center = int((thumb_tip.y + index_tip.y) / 2 * base_image.shape[0])
                circle_center = (x_center, y_center)

    overlay_frame = apply_circle_filter(blurred_image, base_image, circle_center, filter_radius, kernel_size)
    return overlay_frame

def generate_frames(camera):
    pygame.mixer.init()
    pygame.mixer.music.load(get_resource_path("static/assets/sound/boss_bg_8bit.mp3"))
    pygame.mixer.music.play(loops=-1, start=0.0)
    
    initialize_base_image(get_resource_path("static/assets/images/Image_Filter_Asset.png"), blur_strength=51)
    
    while True:
        processed_frame = process_frame(camera)
        if processed_frame is None:
            print("No processed frame available.")
            break

        success, buffer = cv2.imencode(".jpg", processed_frame)
        if not success:
            print("Failed to encode frame.")
            continue

        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
