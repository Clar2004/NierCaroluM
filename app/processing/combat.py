import cv2
import numpy as np
import mediapipe as mp
from threading import Lock
import time
from ball import RedBall
import math
import os
import pygame
import sys

SHOT_COOLDOWN = 0.3
last_shot_time = 0

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        return os.path.join(os.getcwd(), relative_path)

player_image = cv2.imread(get_resource_path("static/assets/combat_assets/Player.png"), cv2.IMREAD_UNCHANGED)  
shot_image = cv2.imread(get_resource_path("static/assets/shot_animation/shot2_exp1.png"), cv2.IMREAD_UNCHANGED)  
shot2_exp2 = cv2.imread(get_resource_path("static/assets/shot_animation/shot2_exp2.png"), cv2.IMREAD_UNCHANGED)
shot2_exp3 = cv2.imread(get_resource_path("static/assets/shot_animation/shot2_exp3.png"), cv2.IMREAD_UNCHANGED)
shot2_exp4 = cv2.imread(get_resource_path("static/assets/shot_animation/shot2_exp4.png"), cv2.IMREAD_UNCHANGED)
shot2_exp5 = cv2.imread(get_resource_path("static/assets/shot_animation/shot2_exp5.png"), cv2.IMREAD_UNCHANGED)
laser_image = cv2.imread(get_resource_path("static/assets/boss_asset/red_ball.png"), cv2.IMREAD_UNCHANGED)

if laser_image is None:
    print("Error: Laser image failed to load.")

scale_factor_player = 1.3  
scale_factor_shot = 1.7 

player_image = cv2.resize(player_image, None, fx=scale_factor_player, fy=scale_factor_player, interpolation=cv2.INTER_LINEAR)
shot_image = cv2.resize(shot_image, None, fx=scale_factor_shot, fy=scale_factor_shot, interpolation=cv2.INTER_LINEAR)

bg_image = cv2.imread(get_resource_path("static/assets/images/combat_bg2.png"), cv2.IMREAD_UNCHANGED) 
bg_height, bg_width, _ = bg_image.shape

video_width = 1920
video_height = 1080

# video_width = 2560
# video_height = 1240

player_x = 100 
player_y = video_height // 2 
last_player_y = player_y

shots = []

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

isDead = False
isCheat = False
isCheatTriggered = False
isBossDead = False

BOSS_STATE_SHOOTING = "shooting"
BOSS_STATE_IDLE_INITIAL = "idle_initial"
BOSS_STATE_IDLE = "idle"
BOSS_STATE_IDLE2 = "idle2"
BOSS_STATE_IMAGE_CHANGE1 = "image_change1"
BOSS_STATE_IMAGE_CHANGE2 = "image_change2"
BOSS_STATE_IMAGE_CHANGEBACK2 = "image_change22"
BOSS_STATE_IMAGE_CHANGEBACK1 = "image_change11"
BOSS_STATE_SHOOTING_LASERS = "shooting laser"

boss_state = BOSS_STATE_IDLE_INITIAL
state_change_time = None

boss_image = cv2.imread(get_resource_path("static/assets/boss_asset/bos_1.png"), cv2.IMREAD_UNCHANGED)
if boss_image is None:
    print("Error loading boss image")
    
scale_factor_boss = 0.9
boss_image = cv2.resize(boss_image, None, fx=scale_factor_boss, fy=scale_factor_boss, interpolation=cv2.INTER_LINEAR)
print(boss_image.shape)

boss_width = boss_image.shape[1]
boss_height = boss_image.shape[0]
boss_x = video_width - boss_width - 50
boss_y = video_height // 4
boss_rotation_angle = 0

health_bar_width = 200
health_bar_height = 20
health_bar_x = 50 
health_bar_y = 50 

max_health = 1000
boss_health = 1000


red_ball_image2 = cv2.imread(get_resource_path('static/assets/boss_asset/red_ball.png'), cv2.IMREAD_UNCHANGED) 
scaling_factor_red_ball = 0.8
red_ball_image = cv2.resize(red_ball_image2, (0, 0), fx=scaling_factor_red_ball, fy=scaling_factor_red_ball)

ball_speed = 6
boss_speed = 7
ball2_speed = 6

BALL_SHOT_COOLDOWN = 1.5
last_ball_shot_time = 0  
last_laser_shot_time = 0

red_balls = [] 

last_shot_time = time.time() 
shooting_time = 10 
idle_time = 2  
image_change_duration = 2  
boss_image_index = 1  

player_health = 5  
health_image = cv2.imread(get_resource_path('static/assets/images/yorha-logo.png'), cv2.IMREAD_UNCHANGED) 

health_offset_x = 20
health_offset_y = 20

# health_offset_x = 20
# health_offset_y = 20

animation_folder = get_resource_path('static/assets/explosion')
animation_images = []
frame_delay = 500 

show_text_start_time = None

isGameTwoDone = False
isGameThreeDone = False

isPaused = False
isPausedActive = False

def is_peace_sign(hand_landmarks):
    index_finger_tip = hand_landmarks.landmark[8]
    middle_finger_tip = hand_landmarks.landmark[12]
    
    index_up = index_finger_tip.y < hand_landmarks.landmark[6].y
    middle_up = middle_finger_tip.y < hand_landmarks.landmark[10].y

    thumb_finger_tip = hand_landmarks.landmark[4]
    thumb_curl = thumb_finger_tip.y > hand_landmarks.landmark[3].y

    ring_finger_tip = hand_landmarks.landmark[16]
    ring_curl = ring_finger_tip.y > hand_landmarks.landmark[14].y

    pinky_tip = hand_landmarks.landmark[20]
    pinky_curl = pinky_tip.y > hand_landmarks.landmark[18].y

    return index_up and middle_up and thumb_curl and ring_curl and pinky_curl

def load_animation_images():
    global animation_images
    for filename in sorted(os.listdir(animation_folder)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(animation_folder, filename), cv2.IMREAD_UNCHANGED)
            animation_images.append(img)
            
def run_death_animation(frame_list, x, y, combined_frame):
    global player_image 
    
    player_height, player_width = player_image.shape[:2]

    for frame in frame_list:
        frame_resized = cv2.resize(frame, (player_width, player_height))

        if frame_resized.shape[2] == 4:
            alpha_channel = frame_resized[:, :, 3] / 255.0 

            for c in range(0, 3):
                combined_frame[y:y+player_height, x:x+player_width, c] = \
                    combined_frame[y:y+player_height, x:x+player_width, c] * (1 - alpha_channel) + \
                    frame_resized[:, :, c] * alpha_channel
        else:
            print("Warning: Frame does not have an alpha channel, skipping overlay.")
            
    return combined_frame

def check_collisions_and_update_health( combined_frame, hit_sound, isCheat):
    """Check for collisions with lasers or red balls and reduce health."""
    global player_health, lasers, red_balls, animation_images, player_x, player_y, isDead
    death_timer = time.time()

    for laser in lasers[:]:
        if is_collision_with_player(laser) and not isCheat: 
            lasers.remove(laser) 
            player_health -= 1 
            hit_sound.set_volume(0.6)
            hit_sound.play()
            # if player_health < 1:
            #     isDead = True
            #     pygame.mixer.music.stop()
            #     run_death_animation(animation_images, player_x, player_y, combined_frame)
                
            #     hit_sound.stop()
                
            #     from app import player_dead
            #     player_dead()
                
            #     death_timer = time.time()
            #     break
                
            break  

    for red_ball in red_balls[:]:
        if is_collision_with_player(red_ball) and not isCheat:
            red_balls.remove(red_ball)  
            player_health -= 1 
            hit_sound.set_volume(0.6)
            hit_sound.play()
            # if player_health < 1:
            #     isDead = True
            #     pygame.mixer.music.stop()
            #     run_death_animation(animation_images, player_x, player_y, combined_frame)
                
            #     hit_sound.stop()
                
            #     from app import player_dead
            #     player_dead()
                
            #     death_timer = time.time()

            break
    
    if is_collision_with_boss(combined_frame) and not isCheat:
        player_health = player_health - 5
        
    if player_health < 1:
        isDead = True
        pygame.mixer.music.stop()
        run_death_animation(animation_images, player_x, player_y, combined_frame)
        
        hit_sound.stop()
        
        from app import player_dead
        player_dead()
        
        death_timer = time.time()
            
    if isDead and time.time() - death_timer < 2:
        isCheat = True
    else:
        isCheat = False
        
    
    return combined_frame, isDead

def is_collision_with_boss(combined_frame):
    global player_x, player_y, player_image, boss_x, boss_y, boss_image

    player_width, player_height = player_image.shape[1], player_image.shape[0]
    boss_width, boss_height = boss_image.shape[1], boss_image.shape[0]

    margin = 50 

    player_left = player_x + margin
    player_right = player_x + player_width - margin
    player_top = player_y + margin
    player_bottom = player_y + player_height - margin

    boss_left = boss_x + margin
    boss_right = boss_x + boss_width - margin
    boss_top = boss_y + margin
    boss_bottom = boss_y + boss_height - margin
    
    #Gambar Collider Player dan Boss
    # cv2.rectangle(combined_frame, (boss_left, boss_top), (boss_right, boss_bottom), (0, 255, 0), 2)
    # cv2.rectangle(combined_frame, (player_left, player_top), (player_right, player_bottom), (255, 0, 0), 2)

    if (player_right > boss_left and player_left < boss_right and
        player_bottom > boss_top and player_top < boss_bottom):
        return True
    return False

def is_collision_with_player(projectile):
    global player_x, player_y, player_image
    
    collider_scale = 0.8
    
    projectile_radius = min(red_ball_image.shape[1],red_ball_image.shape[0]) // 2 * collider_scale

    dist = math.sqrt((projectile.x - player_x) ** 2 + (projectile.y - player_y) ** 2)

    if dist < projectile_radius + (player_image.shape[1] // 2):
        return True
    return False

def shoot_red_ball(boss_x, boss_y, boss_rotation_angle):
    """Launch multiple red balls from fixed positions around the boss."""
    global last_ball_shot_time
    
    current_time = time.time()
    if current_time - last_ball_shot_time >= BALL_SHOT_COOLDOWN: 
        
        num_balls = 3
        radius = 270  

        angle_increment = 2 * math.pi / num_balls

        spawn_positions = []
        for i in range(num_balls):
            angle = i * angle_increment  

            spawn_x_offset = radius * math.cos(angle) 
            spawn_y_offset = radius * math.sin(angle)  

            spawn_x = boss_x + spawn_x_offset * math.cos(math.radians(-boss_rotation_angle)) - spawn_y_offset * math.sin(math.radians(-boss_rotation_angle))
            spawn_y = boss_y + spawn_x_offset * math.sin(math.radians(-boss_rotation_angle)) + spawn_y_offset * math.cos(math.radians(-boss_rotation_angle))

            offset_x = 270 
            offset_y = 300  

            spawn_x += offset_x
            spawn_y += offset_y
            
            spawn_positions.append((spawn_x, spawn_y))

        for spawn_x, spawn_y in spawn_positions:
            new_ball = RedBall(spawn_x, spawn_y, boss_x, boss_y, ball_speed)
            red_balls.append(new_ball)

        last_ball_shot_time = current_time

def move_red_balls():
    global isPaused
    """Move all red balls in their respective directions."""
    global red_balls
    for ball in red_balls[:]:
        if not isPaused:
            ball.move()
        if ball.is_offscreen(video_width, video_height):
            red_balls.remove(ball)

def draw_red_balls(frame):
    """Draw all the red balls on the frame."""
    for ball in red_balls:
        x_start = int(ball.x)
        y_start = int(ball.y)

        ball_width = red_ball_image.shape[1]
        ball_height = red_ball_image.shape[0]

        x_end = x_start + ball_width
        y_end = y_start + ball_height

        x_end = min(x_end, frame.shape[1])
        y_end = min(y_end, frame.shape[0])

        if x_start < frame.shape[1] and y_start < frame.shape[0] and x_end > x_start and y_end > y_start:
            alpha_channel = red_ball_image[:, :, 3] / 255.0  
            red_ball_rgb = red_ball_image[:, :, :3] 

            valid_x_end = min(x_end, frame.shape[1])
            valid_y_end = min(y_end, frame.shape[0])
            valid_x_start = max(x_start, 0)
            valid_y_start = max(y_start, 0)

            if valid_x_end > valid_x_start and valid_y_end > valid_y_start:
                for c in range(3):  
                    frame[valid_y_start:valid_y_end, valid_x_start:valid_x_end, c] = \
                        (1 - alpha_channel[:valid_y_end - valid_y_start, :valid_x_end - valid_x_start]) * \
                        frame[valid_y_start:valid_y_end, valid_x_start:valid_x_end, c] + \
                        alpha_channel[:valid_y_end - valid_y_start, :valid_x_end - valid_x_start] * \
                        red_ball_rgb[:valid_y_end - valid_y_start, :valid_x_end - valid_x_start, c]

def shoot(shot_x, shot_y, direction):
    shots.append([shot_x, shot_y, direction]) 
    
def remove_offscreen_bullets():
    global shots
    threshold = 2 
    
    shots = [shot for shot in shots if (shot[0] > threshold and shot[0] < video_width - shot_image.shape[1] - threshold and 
                                        shot[1] > threshold and shot[1] < video_height - shot_image.shape[0] - threshold)]

def detect_hand_gesture(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    right_hand_position = None  
    left_hand_gesture = None  
    right_hand_gesture = None
    
    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = handedness.classification[0].label

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            index_finger_distance = math.sqrt(
                (index_finger_tip.x - wrist.x) ** 2 + (index_finger_tip.y - wrist.y) ** 2
            )


            if hand_label == 'Right': 
                right_hand_position = (int(index_finger_tip.x * video_width), int(index_finger_tip.y * video_height))

            elif hand_label == 'Left': 
                if (abs(thumb_tip.x - index_finger_tip.x) < 0.05 and
                    abs(index_finger_tip.x - middle_finger_tip.x) < 0.05 and
                    abs(middle_finger_tip.x - ring_finger_tip.x) < 0.05 and
                    abs(ring_finger_tip.x - pinky_finger_tip.x) < 0.05):
                    left_hand_gesture = "fist"
                    
                elif (abs(thumb_tip.x - index_finger_tip.x) > 0.1 and
                    abs(index_finger_tip.x - middle_finger_tip.x) > 0.1 and
                    abs(middle_finger_tip.x - ring_finger_tip.x) > 0.1 and
                    abs(ring_finger_tip.x - pinky_finger_tip.x) > 0.1 and
                    abs(thumb_tip.y - wrist.y) > 0.2 and
                    abs(index_finger_tip.y - wrist.y) > 0.2 and
                    abs(middle_finger_tip.y - wrist.y) > 0.2 and
                    abs(ring_finger_tip.y - wrist.y) > 0.2 and
                    abs(pinky_finger_tip.y - wrist.y) > 0.2):
                    right_hand_gesture = "open"  
                
                elif is_peace_sign(hand_landmarks): 
                    right_hand_gesture = "peace"

    return right_hand_position, left_hand_gesture, right_hand_gesture

def shoot(shot_x, shot_y, direction, shoot_sound):
    global last_shot_time
    
    current_time = time.time()  
    
    if current_time - last_shot_time < 0.5:
        return 
    
    shoot_sound.play() 

    if direction == "right":
        shot_x += player_image.shape[1]  
    else:
        shot_x -= shot_image.shape[1]
        
    shots.append([shot_x, shot_y, direction])
    last_shot_time = current_time

    
boss_height, boss_width, _ = boss_image.shape

def move_boss(isAlreadyCenter):
    global boss_x, boss_y, boss_state, boss_rotation_angle, state_change_time, isBossDead, isPaused, isPausedActive
    global isCheat
    
    if not isBossDead:
        if boss_state == BOSS_STATE_IDLE:
            pass
        
        elif boss_state == BOSS_STATE_SHOOTING:
            boss_rotation_angle += 1  
            if boss_rotation_angle >= 360:  
                boss_rotation_angle = 0
            
        elif boss_state == BOSS_STATE_SHOOTING_LASERS:
            boss_rotation_angle -= 1  
            if boss_rotation_angle < 0:  
                boss_rotation_angle = 359
        
        elif boss_state == BOSS_STATE_IDLE_INITIAL:
            boss_rotation_angle += 1  
            if boss_rotation_angle >= 360:  
                boss_rotation_angle = 0
    
    rotated_boss_image = rotate_boss_image(boss_image, boss_rotation_angle)
    screen_center_x = video_width // 2
    
    if boss_x > screen_center_x and not isPaused:
        boss_x -= boss_speed
        
    if isBossDead and not isPaused:
        if boss_y + boss_height < video_height:
            boss_y += 5
        else:
            boss_y = video_height - boss_height
            pygame.mixer.music.stop()
            from app import player_dead
            player_dead()
            
            isCheat = True
    
    if not isAlreadyCenter:
        if boss_x <= screen_center_x:
            boss_state = BOSS_STATE_SHOOTING 
            print("State changed to shooting")
            state_change_time = time.time()
            boss_x = screen_center_x 
            isAlreadyCenter = True 
        
    return rotated_boss_image, isAlreadyCenter

def shoot_boss_ball():
    global shots

    ball_x = boss_x + boss_width
    ball_y = boss_y + boss_height // 2  

    angle_rad = np.radians(boss_rotation_angle) 
    direction_x = np.cos(angle_rad)  
    direction_y = np.sin(angle_rad)  

    ball_speed_x = ball_speed * direction_x
    ball_speed_y = ball_speed * direction_y

    shots.append([ball_x, ball_y, ball_speed_x, ball_speed_y])

def preprocess_boss_image(image):
    gray_boss = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    edges = cv2.Canny(gray_boss, 50, 150)
    return edges

def get_boss_contours(edges):
    """
    Find the contours in the edge-detected image.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
        
def show_animation(frame_list, x, y, combined_frame):
    for frame in frame_list:
        h, w, _ = frame.shape

        for c in range(0, 3):  
            combined_frame[y:y+h, x:x+w, c] = \
                combined_frame[y:y+h, x:x+w, c] * (1 - frame[:, :, 3] / 255.0) + \
                frame[:, :, c] * (frame[:, :, 3] / 255.0)
        
def check_boss_shot_collision_with_edges(shots, boss_x, boss_y, boss_image, shot_image, combined_frame, hit_sound):
    edges = preprocess_boss_image(boss_image)
    contours = get_boss_contours(edges)
    for shot in shots[:]:
        shot_x, shot_y, direction = shot
        
        if shot_x + shot_image.shape[1] > boss_x and \
           shot_x < boss_x + boss_image.shape[1] and \
           shot_y + shot_image.shape[0] > boss_y and \
           shot_y < boss_y + boss_image.shape[0]:

            for contour in contours:
                for point in contour:
                    contour_x = boss_x + point[0][0]
                    contour_y = boss_y + point[0][1]

                    if (shot_x < contour_x < shot_x + shot_image.shape[1] and
                        shot_y < contour_y < shot_y + shot_image.shape[0]):
                        reduce_boss_health(10, hit_sound)
                        show_animation([shot2_exp2, shot2_exp3, shot2_exp4, shot2_exp5], shot_x, shot_y, combined_frame)
                        shots.remove(shot)
                        return

def check_boss_shot_collision(shots, boss_x, boss_y, boss_width, boss_height, boss_image, shot_image, combined_frame, hit_sound):
    check_boss_shot_collision_with_edges(shots, boss_x, boss_y, boss_image, shot_image, combined_frame, hit_sound)
    
def rotate_boss_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)  
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1) 
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))  
    return rotated_image

def reduce_boss_health(amount, hit_sound):
    global boss_health, isBossDead
    
    hit_sound.set_volume(0.6)
    hit_sound.play()
    boss_health = max(0, boss_health - amount)
    if (boss_health <= 0) :
        hit_sound.set_volume(0.6)
        hit_sound.play()
        isBossDead = True
    
def draw_health_bar(frame, health, max_health, health_bar_y, health_bar_width, health_bar_height):
    health_bar_width = 1000  
    health_bar_x = (frame.shape[1] - health_bar_width) // 2  
    health_bar_y += 40  
    health_bar_current_width = int(health / max_health * health_bar_width)

    cv2.rectangle(frame, (health_bar_x, health_bar_y), 
                  (health_bar_x + health_bar_width, health_bar_y + health_bar_height), 
                  (0, 0, 0), -1)  

    cv2.rectangle(frame, (health_bar_x, health_bar_y), 
                  (health_bar_x + health_bar_current_width, health_bar_y + health_bar_height), 
                  (255, 255, 255), -1)

    cv2.rectangle(frame, (health_bar_x, health_bar_y), 
                  (health_bar_x + health_bar_width, health_bar_y + health_bar_height), 
                  (255, 255, 255), 2) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "CaroluM: The Trainee Slayer"
    text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = health_bar_y - 20  
    cv2.putText(frame, text, (text_x, text_y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

def draw_health(frame):
    global health_image, player_health, health_offset_x, health_offset_y

    if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
        print("Invalid frame!")
        return frame

    if health_image is None:
        print("Error: health_image is not loaded properly!")
        return frame

    has_alpha = health_image.shape[2] == 4  

    new_width = int(health_image.shape[1] * 0.03)  
    new_height = int(health_image.shape[0] * 0.02)  
    resized_health_image = cv2.resize(health_image, (new_width, new_height))

    if frame.shape[2] != 4:
        print("Converting frame to BGRA.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    y_offset = frame.shape[0] - resized_health_image.shape[0] - health_offset_y

    if y_offset < 0:
        y_offset = frame.shape[0] - resized_health_image.shape[0]
        print("Warning: y_offset was negative. Adjusted to bottom of the frame.")

    for i in range(player_health):
        x_offset = health_offset_x + i * resized_health_image.shape[1]
        if x_offset + resized_health_image.shape[1] <= frame.shape[1] and y_offset + resized_health_image.shape[0] <= frame.shape[0]:
            roi = frame[y_offset:y_offset + resized_health_image.shape[0], x_offset:x_offset + resized_health_image.shape[1]]

            if has_alpha:
                alpha_mask = resized_health_image[:, :, 3] / 255.0 
                alpha_mask = np.dstack([alpha_mask, alpha_mask, alpha_mask]) 

                for c in range(3):
                    roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask[:, :, c]) + resized_health_image[:, :, c] * alpha_mask[:, :, c]
            else:
                roi[:, :, :3] = resized_health_image[:, :, :3]
                
            frame[y_offset:y_offset + resized_health_image.shape[0], x_offset:x_offset + resized_health_image.shape[1]] = roi

    return frame
    
## Laser from boss ##
lasers = []
laser_rotation_speed = 0.1

def shoot_lasers(boss_x, boss_y, boss_rotation_angle):
    """Launch multiple red balls from fixed positions around the boss."""
    global last_ball_shot_time
    
    current_time = time.time()
    if current_time - last_ball_shot_time >= BALL_SHOT_COOLDOWN: 
        
        num_balls = 5  
        radius = 270  
        angle_increment = 2 * math.pi / num_balls
        spawn_positions = []
        for i in range(num_balls):
            angle = i * angle_increment 
            
            spawn_x_offset = radius * math.cos(angle) 
            spawn_y_offset = radius * math.sin(angle) 

            spawn_x = boss_x + spawn_x_offset * math.cos(math.radians(boss_rotation_angle)) - spawn_y_offset * math.sin(math.radians(boss_rotation_angle))
            spawn_y = boss_y + spawn_x_offset * math.sin(math.radians(boss_rotation_angle)) + spawn_y_offset * math.cos(math.radians(boss_rotation_angle))

            offset_x = 270  
            offset_y = 300  

            spawn_x += offset_x
            spawn_y += offset_y
            
            spawn_positions.append((spawn_x, spawn_y))

        for spawn_x, spawn_y in spawn_positions:
            new_ball = RedBall(spawn_x, spawn_y, boss_x, boss_y, ball2_speed)
            red_balls.append(new_ball)

        last_ball_shot_time = current_time

def move_lasers():
    """Move all red balls in their respective directions."""
    global red_balls, isPaused
    for ball in red_balls[:]:
        if not isPaused:
            ball.move()
        if ball.is_offscreen(video_width, video_height):
            red_balls.remove(ball)

def draw_lasers(frame):
    """Draw all the red balls on the frame."""
    for ball in red_balls:
        x_start = int(ball.x)
        y_start = int(ball.y)

        ball_width = red_ball_image.shape[1]
        ball_height = red_ball_image.shape[0]

        x_end = x_start + ball_width
        y_end = y_start + ball_height

        x_end = min(x_end, frame.shape[1])
        y_end = min(y_end, frame.shape[0])

        if x_start < frame.shape[1] and y_start < frame.shape[0] and x_end > x_start and y_end > y_start:
            alpha_channel = red_ball_image[:, :, 3] / 255.0 
            red_ball_rgb = red_ball_image[:, :, :3]  

            valid_x_end = min(x_end, frame.shape[1])
            valid_y_end = min(y_end, frame.shape[0])
            valid_x_start = max(x_start, 0)
            valid_y_start = max(y_start, 0)

            if valid_x_end > valid_x_start and valid_y_end > valid_y_start:
                for c in range(3):  
                    frame[valid_y_start:valid_y_end, valid_x_start:valid_x_end, c] = \
                        (1 - alpha_channel[:valid_y_end - valid_y_start, :valid_x_end - valid_x_start]) * \
                        frame[valid_y_start:valid_y_end, valid_x_start:valid_x_end, c] + \
                        alpha_channel[:valid_y_end - valid_y_start, :valid_x_end - valid_x_start] * \
                        red_ball_rgb[:valid_y_end - valid_y_start, :valid_x_end - valid_x_start, c]

def update_boss_state(boss_state, state_change_time, current_time, combined_frame, boss_change_sound):
    global boss_image, boss_image_index, laser_image, lasers, boss_rotation_angle, isBossDead

    if not isBossDead:
        if boss_state == BOSS_STATE_SHOOTING:
            if current_time - state_change_time >= shooting_time:
                move_red_balls()
                draw_red_balls(combined_frame)
                boss_state = BOSS_STATE_IDLE
                state_change_time = current_time
                print("State changed to idle")
            else:
                shoot_red_ball(boss_x, boss_y, boss_rotation_angle)
                move_red_balls()
                draw_red_balls(combined_frame)

        elif boss_state == BOSS_STATE_IDLE:
            if current_time - state_change_time >= idle_time:
                boss_state = BOSS_STATE_IMAGE_CHANGE1
                state_change_time = current_time  
                print("State changed to image change to index 2")
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
            else:
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
        
        elif boss_state == BOSS_STATE_IMAGE_CHANGE1:
            if current_time - state_change_time >= image_change_duration:
                boss_state = BOSS_STATE_IMAGE_CHANGE2
                state_change_time = current_time 
                print("State changed to image change to index 3")
                
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
            else:
                boss_image = cv2.imread(get_resource_path('static/assets/boss_asset/bos_2.png'), cv2.IMREAD_UNCHANGED)
                scale_factor_boss = 0.9
                boss_image = cv2.resize(boss_image, None, fx=scale_factor_boss, fy=scale_factor_boss, interpolation=cv2.INTER_LINEAR)
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
                boss_change_sound.set_volume(0.5)
                boss_change_sound.play()
                
        elif boss_state == BOSS_STATE_IMAGE_CHANGE2:
            if current_time - state_change_time >= image_change_duration:
                state_change_time = current_time  
                boss_state = BOSS_STATE_SHOOTING_LASERS
                print("State changed to shooting lasers")
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
            else:
                boss_image = cv2.imread(get_resource_path('static/assets/boss_asset/bos_3.png'), cv2.IMREAD_UNCHANGED)
                scale_factor_boss = 0.9
                boss_image = cv2.resize(boss_image, None, fx=scale_factor_boss, fy=scale_factor_boss, interpolation=cv2.INTER_LINEAR)
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
                
        elif boss_state == BOSS_STATE_SHOOTING_LASERS:
            if current_time - state_change_time >= shooting_time:
                boss_state = BOSS_STATE_IDLE2
                state_change_time = current_time
                print("State changed to idle")
                move_lasers()
                draw_lasers(combined_frame)
                move_red_balls()
                draw_red_balls(combined_frame)
            else:
                shoot_lasers(boss_x, boss_y, boss_rotation_angle)
                move_lasers()
                draw_lasers(combined_frame)
                move_red_balls()
                draw_red_balls(combined_frame)
        
        elif boss_state == BOSS_STATE_IDLE2:
            if current_time - state_change_time >= idle_time:
                boss_state = BOSS_STATE_IMAGE_CHANGEBACK2
                state_change_time = current_time  
                print("State changed to image change to index 2")
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
            else:
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
        
        elif boss_state == BOSS_STATE_IMAGE_CHANGEBACK2:
            if current_time - state_change_time >= image_change_duration:
                boss_state = BOSS_STATE_IMAGE_CHANGEBACK1
                state_change_time = current_time 
                print("State changed to image change to index 3")
                
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
            else:
                boss_image = cv2.imread(get_resource_path('static/assets/boss_asset/bos_2.png'), cv2.IMREAD_UNCHANGED)
                scale_factor_boss = 0.9
                boss_image = cv2.resize(boss_image, None, fx=scale_factor_boss, fy=scale_factor_boss, interpolation=cv2.INTER_LINEAR)
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
                
        elif boss_state == BOSS_STATE_IMAGE_CHANGEBACK1:
            if current_time - state_change_time >= image_change_duration:
                state_change_time = current_time  
                boss_state = BOSS_STATE_SHOOTING
                print("State changed to shooting lasers")
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
            else:
                boss_image = cv2.imread(get_resource_path('static/assets/boss_asset/bos_1.png'), cv2.IMREAD_UNCHANGED)
                scale_factor_boss = 0.9
                boss_image = cv2.resize(boss_image, None, fx=scale_factor_boss, fy=scale_factor_boss, interpolation=cv2.INTER_LINEAR)
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
    else:
        move_red_balls()
        draw_red_balls(combined_frame)
        move_lasers()
        draw_lasers(combined_frame)
    
    return boss_state, state_change_time

def show_text(frame, isDone):
    global show_text_start_time
    
    text = "Cheat Activated"
    duration = 3 
    
    if show_text_start_time is None:
        show_text_start_time = time.time()
        print("Start time initialized:", show_text_start_time)

    current_time = time.time()

    if current_time - show_text_start_time < duration and not isDone:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        text_x = frame.shape[1] - text_size[0] - 10 
        text_y = frame.shape[0] - 10  

        print(f"Text position: ({text_x}, {text_y})")

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness)
    else:
        show_text_start_time = None
        isDone = True
    
    return frame, isDone

def is_five_fingers_extended(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_base = landmarks[mp_hands.HandLandmark.THUMB_MCP]

    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_base = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_base = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_base = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]

    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_base = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    thumb_extended = thumb_tip.y < thumb_base.y
    index_extended = index_tip.y < index_base.y
    middle_extended = middle_tip.y < middle_base.y
    ring_extended = ring_tip.y < ring_base.y
    pinky_extended = pinky_tip.y < pinky_base.y

    return thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended

def scroll_background(camera, isReset):
    global player_x, player_y, shots, player_image, boss_x, boss_y, boss_health, boss_rotation_angle, boss_image, boss_state, state_change_time, boss_image_index
    global state_change_time, isDead, last_player_y, isCheat, isCheatTriggered
    global max_health, isGameTwoDone, isGameThreeDone, lasers, red_balls, boss_width, player_health, isBossDead, isPaused, isPausedActive
    
    isAlreadyCenter = False
    gesture_start_time = None
    last_open_hand_time = 0
    cooldown_period = 2 
    
    if isReset:
        player_x = 100  
        player_y = video_height // 2 
        last_player_y = player_y
        player_health = 5
        
        boss_x = video_width - boss_width - 50 
        boss_y = video_height // 4  
        boss_rotation_angle = 0
        boss_state = BOSS_STATE_IDLE_INITIAL
        lasers = []
        red_balls = []
        boss_health = max_health
        
        isCheat = False
        isCheatTriggered = False
        isDead = False
        isBossDead = False
        isPaused = False
        isPausedActive = False
    
    x_offset = 0
    sensitivity = 10  
    last_player_x = player_x 
    last_player_y
    last_facing_direction = "right"  
    
    print("Initial state: ", boss_state)
    load_animation_images()
    
    isDone = False
    
    pygame.mixer.init()
    
    pygame.mixer.music.load(get_resource_path("static/assets/sound/boss_bg.mp3"))
    pygame.mixer.music.play(loops=-1, start=0.0)
    
    shoot_sound = pygame.mixer.Sound(get_resource_path("static/assets/sound/shoot_sound.mp3"))
    boss_change_sound = pygame.mixer.Sound(get_resource_path("static/assets/sound/boss_change_sound.mp3"))
    hit_sound = pygame.mixer.Sound(get_resource_path("static/assets/sound/dead_sound.mp3"))

    while True:
        from state import game_state
        from app import change_game_one_state,  change_game_two_state
        from app import change_game_three_state, change_game_four_state
        
        if boss_health <= max_health*0.8 and game_state.is_game_one_done == False:
            change_game_one_state()
            print("Redirect to Game 1")
        if boss_health <= max_health*0.6 and game_state.is_game_two_done == False:
            change_game_two_state()
            print("Redirect to Game 2")
        if boss_health <= max_health*0.4 and game_state.is_game_three_done == False:
            change_game_three_state()
            print("Redirect to Game 3")
        if boss_health <= max_health*0.2 and game_state.is_game_four_done == False:
            change_game_four_state()
            print("Redirect to Game 4")
        
        x_offset -= 10  
        if x_offset <= -bg_width:
            x_offset = 0  

        if not isPaused:
            shifted_bg = np.roll(bg_image, x_offset, axis=1)

        resized_bg = cv2.resize(shifted_bg, (video_width, video_height))

        ret, frame = camera.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (video_width, video_height))

        right_hand_position, left_hand_gesture, right_hand_gesture = detect_hand_gesture(frame)

        if right_hand_position:
            adjusted_x = video_width - (right_hand_position[0] + player_image.shape[1] // 2)
            player_x += (adjusted_x - player_x) // sensitivity  

            adjusted_y = right_hand_position[1] - player_image.shape[0] // 2
            player_y = adjusted_y

        player_x = max(0, min(player_x, video_width - player_image.shape[1]))
        player_y = max(0, min(player_y, video_height - player_image.shape[0]))

        player_image_flipped = player_image  

        if not isDead:
            if player_x < last_player_x:
                last_facing_direction = "left"
                player_image_flipped = cv2.flip(player_image, 1)
            elif player_x > last_player_x:
                last_facing_direction = "right"
                player_image_flipped = player_image

            last_player_x = player_x

            if right_hand_position:
                adjusted_x = video_width - (right_hand_position[0] + player_image.shape[1] // 2)
                player_x += (adjusted_x - player_x) // sensitivity 

                adjusted_y = right_hand_position[1] - player_image.shape[0] // 2
                player_y = adjusted_y
                last_player_y = player_y

            if left_hand_gesture == "fist":
                shoot(player_x, player_y, last_facing_direction, shoot_sound)
                
            if right_hand_gesture == "peace" and isCheatTriggered == False:
                print("Activate cheat")
                isCheat = True
                isCheatTriggered = True
                isDone = False

        else:
            player_image_flipped = player_image  
            last_facing_direction = last_facing_direction  
            
            player_x = last_player_x
            player_y = last_player_y
            
        current_time_pause = time.time()

        if right_hand_gesture == "open":
            if gesture_start_time is None:
                gesture_start_time = current_time
            elif current_time_pause - gesture_start_time >= 1:
                if current_time_pause - last_open_hand_time >= cooldown_period:
                    print("Palm hand detected for 2 seconds")
                    
                    if isPaused:
                        isPaused = False
                    else:
                        isPaused = True
                    
                    if isPaused:
                        print("Pause is active")
                        isDead = not isDead
                        isBossDead = not isBossDead
                        pygame.mixer.music.pause()
                    elif not isPaused:
                        print("Pause deactivated")
                        isDead = not isDead
                        isBossDead = not isBossDead
                        pygame.mixer.music.unpause()

                    last_open_hand_time = current_time_pause
        else:
            gesture_start_time = None

        combined_frame = resized_bg.copy()

        player_height, player_width = player_image_flipped.shape[:2]

        player_x = max(0, min(player_x, video_width - player_width))
        player_y = max(0, min(player_y, video_height - player_height))

        if player_y + player_height <= video_height and player_x + player_width <= video_width:
            for c in range(0, 3):
                combined_frame[player_y:player_y + player_height, player_x:player_x + player_width, c] = \
                    combined_frame[player_y:player_y + player_height, player_x:player_x + player_width, c] * (1 - player_image_flipped[:, :, 3] / 255.0) + \
                    player_image_flipped[:, :, c] * (player_image_flipped[:, :, 3] / 255.0)
        else:
            print("Error: player image exceeds background bounds")

        if 'boss_width' not in locals() or 'boss_height' not in locals():
            boss_width, boss_height = boss_image.shape[1], boss_image.shape[0]

        if boss_x + boss_width > video_width:
            boss_x = video_width - boss_width  

        if boss_y + boss_height > video_height:
            boss_y = video_height - boss_height  

        if boss_width > video_width or boss_height > video_height:
            scale_factor = min(video_width / boss_width, video_height / boss_height)
            new_width = int(boss_width * scale_factor)
            new_height = int(boss_height * scale_factor)
            boss_image_resized = cv2.resize(boss_image, (new_width, new_height))
            boss_width = new_width
            boss_height = new_height
        else:
            boss_image_resized = boss_image

        boss_x = int(boss_x)
        boss_y = int(boss_y)
        boss_width = int(boss_width)
        boss_height = int(boss_height)
        
        check_boss_shot_collision(shots, boss_x, boss_y, boss_width, boss_height, boss_image, shot_image, combined_frame, hit_sound)

        rotated_boss_image, isAlreadyCenter = move_boss(isAlreadyCenter)
        
        current_time = time.time()
        boss_state, state_change_time = update_boss_state(boss_state, state_change_time, current_time, combined_frame, boss_change_sound)

        rotated_boss_image_height = int(rotated_boss_image.shape[0])
        rotated_boss_image_width = int(rotated_boss_image.shape[1])
        
        boss_x = int(boss_x)
        boss_y = int(boss_y)

        if boss_y + rotated_boss_image_height <= video_height and boss_x + rotated_boss_image_width <= video_width:
            for c in range(0, 3):
                combined_frame[boss_y:boss_y + rotated_boss_image_height, boss_x:boss_x + rotated_boss_image_width, c] = \
                    combined_frame[boss_y:boss_y + rotated_boss_image_height, boss_x:boss_x + rotated_boss_image_width, c] * (1 - rotated_boss_image[:, :, 3] / 255.0) + \
                    rotated_boss_image[:, :, c] * (rotated_boss_image[:, :, 3] / 255.0)
        else:
            print(f"Boss image out of screen: boss_x={boss_x}, boss_y={boss_y}, boss_width={boss_width}, boss_height={boss_height}")

        for shot in shots:
            shot_x, shot_y, direction = shot
            if direction == "right":
                shot_x += 20
            else:
                shot_x -= 20

            shot_x = max(0, min(shot_x, video_width - shot_image.shape[1]))
            shot_y = max(0, min(shot_y, video_height - shot_image.shape[0]))

            shot[0] = shot_x

            shot_height, shot_width, _ = shot_image.shape
            for c in range(0, 3):
                combined_frame[shot_y:shot_y + shot_height, shot_x:shot_x + shot_width, c] = \
                    combined_frame[shot_y:shot_y + shot_height, shot_x:shot_x + shot_width, c] * (1 - shot_image[:, :, 3] / 255.0) + \
                    shot_image[:, :, c] * (shot_image[:, :, 3] / 255.0)

        remove_offscreen_bullets()
        
        draw_health_bar(combined_frame, boss_health, max_health, health_bar_y, health_bar_width, health_bar_height)
        
        if not isDone and isCheatTriggered == True:
            global show_text_start_time
            
            text = "Cheat Activated"
            duration = 3 
            
            if show_text_start_time is None:
                show_text_start_time = time.time()
                print("Start time initialized:", show_text_start_time)

            current_time = time.time()

            if current_time - show_text_start_time < duration:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (255, 255, 255)
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                
                text_x = combined_frame.shape[1] - text_size[0] - 70 # harusnya 220
                text_y = combined_frame.shape[0] - 70
                
                rect_color = (130, 130, 130)  
                rect_thickness = -1 
                cv2.rectangle(combined_frame, (text_x - 5, text_y - text_size[1] - 5), 
                            (text_x + text_size[0] + 5, text_y + 5), rect_color, rect_thickness)

                cv2.putText(combined_frame, text, (text_x, text_y), font, font_scale, font_color, thickness)
            else:
                show_text_start_time = None
                isDone = True

        combined_frame, isDead = check_collisions_and_update_health(combined_frame, hit_sound, isCheat)
        combined_frame = draw_health(combined_frame)
        
        if isPaused:
            overlay = combined_frame.copy()
            alpha = 0.7
            cv2.rectangle(overlay, (0, 0), (video_width, video_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, combined_frame, 1 - alpha, 0, combined_frame)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 5
            font_color = (255, 255, 255)
            thickness = 10
            text = "Game Paused"
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (video_width - text_size[0]) // 2
            text_y = (video_height + text_size[1]) // 2
            cv2.putText(combined_frame, text, (text_x, text_y), font, font_scale, font_color, thickness)

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', combined_frame)[1].tobytes() + b'\r\n')