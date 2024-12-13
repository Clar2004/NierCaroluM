import cv2
import numpy as np
import mediapipe as mp
import time
import pygame, sys, os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)

thumb_path = []  
health_points = 3  
game_started = False
cheat_mode_activated = False  

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        return os.path.join(os.getcwd(), relative_path)

def initialize_maze_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    gray_float = np.float32(gray)
    corners = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)

    dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    dilated_corners = cv2.dilate(corners, np.ones((5, 5), np.uint8), iterations=1)

    image[edges != 0] = [0, 255, 0]  
    image[corners > 0.01 * corners.max()] = [0, 0, 255] 

    return image, dilated_edges, dilated_corners

def calculate_entry_points(width, height):
    return [
        (int(width * 0.1), int(height * 0.05)), 
        (int(width * 0.95), int(height * 0.16)),  
        (int(width * 0.05), int(height * 0.65)),  
        (int(width * 0.95), int(height * 0.77)),  
    ]

def calculate_cpu_zone(width, height):
    return (int(width * 0.5), int(height * 0.5)), int(min(width, height) * 0.08)

def are_all_fingers_extended(hand_landmarks, height):
    extended = []

    for finger_tip, knuckle in [
        (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
    ]:
        if hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[knuckle].y:
            extended.append(True)
        else:
            extended.append(False)

    return all(extended)

def generate_maze_interaction_frames(camera):
    from app import change_mini_game_three_state
    global thumb_path, health_points, game_started, cheat_mode_activated

    maze_image, edges, corners = initialize_maze_image(get_resource_path('static/assets/images/maze.jpg'))
    pygame.mixer.init()
    pygame.mixer.music.load(get_resource_path("static/assets/sound/boss_bg_8bit.mp3"))
    pygame.mixer.music.play(loops=-1, start=0.0)

    game_started = False
    thumb_path = [] 

    while True:
        success, frame = camera.read()
        if not success:
            break

        height, width, _ = frame.shape
        resized_maze_image = cv2.resize(maze_image, (width, height))
        resized_edges = cv2.resize(edges, (width, height))
        resized_corners = cv2.resize(corners, (width, height))

        entry_points = calculate_entry_points(width, height)
        cpu_center, cpu_radius = calculate_cpu_zone(width, height)

        if not game_started:
            for point in entry_points:
                cv2.circle(resized_maze_image, point, int(min(width, height) * 0.02), (255, 255, 255), -1)

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                if hand_label == 'Right':
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    tip_x, tip_y = int(index_tip.x * width), int(index_tip.y * height)
                    tip_x = max(0, min(tip_x, width - 1))
                    tip_y = max(0, min(tip_y, height - 1))
                    cv2.circle(resized_maze_image, (tip_x, tip_y), 10, (192, 192, 192), -1)

                    if not game_started:
                        for point in entry_points:
                            px, py = point 
                            distance = np.sqrt((tip_x - px) ** 2 + (tip_y - py) ** 2)
                            if distance <= int(min(width, height) * 0.02):
                                game_started = True
                                thumb_path = []
                                break

                    thumb_path.append((tip_x, tip_y))

                    for i in range(1, len(thumb_path)):
                        if i >= len(thumb_path):
                            break
                        
                        x1, y1 = thumb_path[i - 1]
                        x2, y2 = thumb_path[i]
                        x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
                        x2, y2 = max(0, min(x2, width - 1)), max(0, min(y2, height - 1))

                        if game_started:
                            if not cheat_mode_activated:
                                line_points = np.linspace((x1, y1), (x2, y2), num=20).astype(int)
                                collision_detected = False
                                for px, py in line_points:
                                    distance_to_cpu = np.sqrt((px - cpu_center[0]) ** 2 + (py - cpu_center[1]) ** 2)
                                    if distance_to_cpu <= cpu_radius:
                                        change_mini_game_three_state()
                                        print("Player reached the CPU")
                                        break

                                    if resized_edges[py, px] != 0 or resized_corners[py, px] > 0.01 * resized_corners.max():
                                        collision_detected = True
                                        break
                                
                                if collision_detected:
                                    thumb_path = thumb_path[:i] 
                                    break  

                        cv2.line(resized_maze_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', resized_maze_image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
