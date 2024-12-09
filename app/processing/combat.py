import cv2
import numpy as np
import mediapipe as mp
from threading import Lock
import time
from ball import RedBall
import math
import os
import pygame

SHOT_COOLDOWN = 0.3
last_shot_time = 0
# socketio_lock = Lock()

# Load player and shot images
player_image = cv2.imread("static/assets/combat_assets/Player.png", cv2.IMREAD_UNCHANGED)  # Assuming it's a PNG with transparency
shot_image = cv2.imread("static/assets/shot_animation/shot2_exp1.png", cv2.IMREAD_UNCHANGED)  # Assuming it's a PNG with transparency
shot2_exp2 = cv2.imread("static/assets/shot_animation/shot2_exp2.png", cv2.IMREAD_UNCHANGED)
shot2_exp3 = cv2.imread("static/assets/shot_animation/shot2_exp3.png", cv2.IMREAD_UNCHANGED)
shot2_exp4 = cv2.imread("static/assets/shot_animation/shot2_exp4.png", cv2.IMREAD_UNCHANGED)
shot2_exp5 = cv2.imread("static/assets/shot_animation/shot2_exp5.png", cv2.IMREAD_UNCHANGED)
laser_image = cv2.imread("static/assets/boss_asset/red_ball.png", cv2.IMREAD_UNCHANGED)

if laser_image is None:
    print("Error: Laser image failed to load.")
    exit()

scale_factor_player = 1.5  # Increase or decrease this to change the spaceship size
scale_factor_shot = 2.0  # Increase or decrease this to change the bullet size

player_image = cv2.resize(player_image, None, fx=scale_factor_player, fy=scale_factor_player, interpolation=cv2.INTER_LINEAR)
shot_image = cv2.resize(shot_image, None, fx=scale_factor_shot, fy=scale_factor_shot, interpolation=cv2.INTER_LINEAR)

# Background image and size
bg_image = cv2.imread("static/assets/images/combat_bg2.png", cv2.IMREAD_UNCHANGED)  # Assuming it's a PNG with transparency
bg_height, bg_width, _ = bg_image.shape
video_width = 2560
video_height = 1240

# Player initial position (middle left)
player_x = 100  # x position at the left side of the screen
player_y = video_height // 2  # y position at the center
last_player_y = player_y

# Store the shots (each shot has x, y coordinates and direction)
shots = []

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

isDead = False
isCheat = False
isCheatTriggered = False
isBossDead = False

##boss section ##

# Define boss states
BOSS_STATE_SHOOTING = "shooting"
BOSS_STATE_IDLE_INITIAL = "idle_initial"
BOSS_STATE_IDLE = "idle"
BOSS_STATE_IDLE2 = "idle2"
BOSS_STATE_IMAGE_CHANGE1 = "image_change1"
BOSS_STATE_IMAGE_CHANGE2 = "image_change2"
BOSS_STATE_IMAGE_CHANGEBACK2 = "image_change22"
BOSS_STATE_IMAGE_CHANGEBACK1 = "image_change11"
BOSS_STATE_SHOOTING_LASERS = "shooting laser"

# Initialize boss state and state change time
boss_state = BOSS_STATE_IDLE_INITIAL
state_change_time = None

# Load boss image
boss_image = cv2.imread("static/assets/boss_asset/bos_1.png", cv2.IMREAD_UNCHANGED)
if boss_image is None:
    print("Error loading boss image")
    exit()
scale_factor_boss = 1.0  # Change this value to scale the boss image
boss_image = cv2.resize(boss_image, None, fx=scale_factor_boss, fy=scale_factor_boss, interpolation=cv2.INTER_LINEAR)
print(boss_image.shape)

# Boss position (right side of the screen)
boss_width = boss_image.shape[1]
boss_height = boss_image.shape[0]
boss_x = video_width - boss_width - 50  # 50 pixels from the right edge
boss_y = video_height // 4  # Fixed vertical position (you can adjust this based on your needs)
boss_rotation_angle = 0

# Define the health bar
health_bar_width = 200
health_bar_height = 20
health_bar_x = 50  # X-position of the health bar
health_bar_y = 50  # Y-position of the health bar

# Maximum health value for the boss
max_health = 500
boss_health = 500

#load the circle image
red_ball_image = cv2.imread('static/assets/boss_asset/red_ball.png', cv2.IMREAD_UNCHANGED) 
# Define Ball Speed
ball_speed = 10  # Speed of the red ball
boss_speed = 10
ball2_speed = 10

# Red Ball launch cooldown for the boss (interval between shots)
BALL_SHOT_COOLDOWN = 1.5  # Boss shot cooldown
last_ball_shot_time = 0  # Time of the last shot
last_laser_shot_time = 0

# Define red ball parameters
red_balls = []  # List to store red balls

last_shot_time = time.time()  # Track last shot time
shooting_time = 10  # Shooting duration in seconds
idle_time = 2  # Idle duration in seconds
image_change_duration = 2  # Duration for each image change in seconds
boss_image_index = 1  # Keeps track of the image cycle

# Global variable for health count and image
player_health = 5  # Starting health
health_image = cv2.imread('static/assets/images/yorha-logo.png', cv2.IMREAD_UNCHANGED)  # The image representing a health point
health_offset_x = 20  # Offset for drawing the health at the bottom-left corner
health_offset_y = 20  # Position from the bottom of the screen (adjust as needed)

animation_folder = 'static/assets/explosion'  # Replace with your actual folder path
animation_images = []
frame_delay = 300  # Delay between frames in milliseconds (adjust as needed)

show_text_start_time = None

isGameTwoDone = False
isGameThreeDone = False

isPaused = False
isPausedActive = False

def load_animation_images():
    global animation_images
    for filename in sorted(os.listdir(animation_folder)):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Assuming the images are in .png or .jpg format
            img = cv2.imread(os.path.join(animation_folder, filename), cv2.IMREAD_UNCHANGED)
            animation_images.append(img)
            
def run_death_animation(frame_list, x, y, combined_frame):
    global player_image  # Ensure this is the image for the player (which can be resized accordingly)
    
    # Get the player image dimensions to adjust the animation size
    player_height, player_width = player_image.shape[:2]

    for frame in frame_list:
        # Resize the frame to match the player size or the size of the area to overlay it on
        frame_resized = cv2.resize(frame, (player_width, player_height))

        # Ensure the frame has an alpha channel for transparency (RGBA)
        if frame_resized.shape[2] == 4:  # Check if the frame has an alpha channel (RGBA)
            alpha_channel = frame_resized[:, :, 3] / 255.0  # Normalize alpha to [0, 1]

            # Overlay the resized frame on the combined_frame at the player position
            for c in range(0, 3):  # Loop through each color channel (BGR)
                combined_frame[y:y+player_height, x:x+player_width, c] = \
                    combined_frame[y:y+player_height, x:x+player_width, c] * (1 - alpha_channel) + \
                    frame_resized[:, :, c] * alpha_channel
        else:
            print("Warning: Frame does not have an alpha channel, skipping overlay.")
            
    return combined_frame

def check_collisions_and_update_health( combined_frame, hit_sound):
    """Check for collisions with lasers or red balls and reduce health."""
    global player_health, lasers, red_balls, animation_images, player_x, player_y, isDead, isCheat

    # Check for collisions with lasers
    for laser in lasers[:]:
        if is_collision_with_player(laser) and not isCheat:  # Check if laser hits the player
            lasers.remove(laser)  # Remove the laser from the game
            player_health -= 1  # Decrease health
            hit_sound.set_volume(0.6)
            hit_sound.play()
            if player_health < 1:
                isDead = True
                pygame.mixer.music.stop()
                run_death_animation(animation_images, player_x, player_y, combined_frame)
                
                pygame.time.delay(1000)
                hit_sound.stop()
                
                from app import player_dead
                player_dead()
            break  # Break to prevent multiple health reductions for one collision

    # Check for collisions with red balls
    for red_ball in red_balls[:]:
        if is_collision_with_player(red_ball) and not isCheat:  # Check if red ball hits the player
            red_balls.remove(red_ball)  # Remove the red ball
            player_health -= 1  # Decrease health
            hit_sound.set_volume(0.6)
            hit_sound.play()
            if player_health < 1:
                pygame.mixer.music.stop()
                run_death_animation(animation_images, player_x, player_y, combined_frame)
                
                pygame.time.delay(1000)
                hit_sound.stop()
                
                from app import player_dead
                player_dead()

            break  # Break to prevent multiple health reductions for one collision
    
    if is_collision_with_boss() and not isCheat:
        # If player collides with the boss, directly set health to 0
        player_health = player_health - 5
        
        if player_health < 1:
            isDead = True
            pygame.mixer.music.stop()
            run_death_animation(animation_images, player_x, player_y, combined_frame)
            
            hit_sound.stop()
    
            from app import player_dead
            player_dead()
    
    return combined_frame, isDead

def is_collision_with_boss():
    global player_x, player_y, player_image, boss_x, boss_y, boss_image
    
    # Use both width and height to compute a more accurate "radius"
    player_radius = (player_image.shape[1] + player_image.shape[0]) / 4 - 20
    boss_radius = (boss_image.shape[1] + boss_image.shape[0]) / 4 - 20

    # Calculate distance between the centers of the player and the boss
    dist = math.sqrt((boss_x - player_x) ** 2 + (boss_y - player_y) ** 2)
    
    if dist < (player_radius + boss_radius):
        return True
    return False

def is_collision_with_player(projectile):
    global player_x, player_y, player_image
    
    player_radius = player_image.shape[1] // 2
    dist = math.sqrt((projectile.x - player_x) ** 2 + (projectile.y - player_y) ** 2)
    if dist < player_radius:
        return True
    return False

def shoot_red_ball(boss_x, boss_y, boss_rotation_angle):
    """Launch multiple red balls from fixed positions around the boss."""
    global last_ball_shot_time
    
    current_time = time.time()
    if current_time - last_ball_shot_time >= BALL_SHOT_COOLDOWN:  # Check if enough time has passed since the last shot
        
        num_balls = 3  # Number of balls to spawn
        radius = 270  # Radius for the circular spawn pattern (distance from the boss)
        
        # Fixed angle increment for even spacing (120 degrees for three balls)
        angle_increment = 2 * math.pi / num_balls
        
        # Create spawn positions based on a fixed circular pattern
        spawn_positions = []
        for i in range(num_balls):
            angle = i * angle_increment  # Angle for each ball (evenly spaced)
            
            # Calculate the x and y offsets based on the angle and radius
            spawn_x_offset = radius * math.cos(angle)  # X offset based on cosine
            spawn_y_offset = radius * math.sin(angle)  # Y offset based on sine
            
            # Calculate the spawn positions based on the boss's position and rotation angle
            spawn_x = boss_x + spawn_x_offset * math.cos(math.radians(-boss_rotation_angle)) - spawn_y_offset * math.sin(math.radians(-boss_rotation_angle))
            spawn_y = boss_y + spawn_x_offset * math.sin(math.radians(-boss_rotation_angle)) + spawn_y_offset * math.cos(math.radians(-boss_rotation_angle))
            
            # Apply the downward and rightward offset
            offset_x = 270  # Shift to the right (adjust as needed)
            offset_y = 300  # Shift downward (adjust as needed)
            
            # Apply the offset
            spawn_x += offset_x
            spawn_y += offset_y
            
            spawn_positions.append((spawn_x, spawn_y))
            
        # Now spawn the red balls with the boss's rotation angle
        for spawn_x, spawn_y in spawn_positions:
            # Use the boss's rotation angle to direct the shot
            new_ball = RedBall(spawn_x, spawn_y, boss_x, boss_y, ball_speed)
            red_balls.append(new_ball)
        
        # Update the last shot time
        last_ball_shot_time = current_time

# Function to move red balls based on their direction
def move_red_balls():
    global isPaused
    """Move all red balls in their respective directions."""
    global red_balls
    for ball in red_balls[:]:
        if not isPaused:
            ball.move()
        if ball.is_offscreen(video_width, video_height):
            red_balls.remove(ball)

# Function to draw red balls on the frame
def draw_red_balls(frame):
    """Draw all the red balls on the frame."""
    for ball in red_balls:
        x_start = int(ball.x)
        y_start = int(ball.y)
        
        # Get the red ball image dimensions
        ball_width = red_ball_image.shape[1]
        ball_height = red_ball_image.shape[0]
        
        # Calculate the end positions based on the ball's position and image size
        x_end = x_start + ball_width
        y_end = y_start + ball_height
        
        # Ensure the end positions do not exceed the frame bounds
        x_end = min(x_end, frame.shape[1])
        y_end = min(y_end, frame.shape[0])

        # Ensure the slice has non-zero size
        if x_start < frame.shape[1] and y_start < frame.shape[0] and x_end > x_start and y_end > y_start:
            # Apply alpha blending for transparent background (RGBA image handling)
            alpha_channel = red_ball_image[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
            red_ball_rgb = red_ball_image[:, :, :3]  # RGB channels
            
            # Handle the region size to ensure no empty slice
            valid_x_end = min(x_end, frame.shape[1])
            valid_y_end = min(y_end, frame.shape[0])
            valid_x_start = max(x_start, 0)
            valid_y_start = max(y_start, 0)
            
            # Ensure the region dimensions are non-zero
            if valid_x_end > valid_x_start and valid_y_end > valid_y_start:
                for c in range(3):  # Iterate over the RGB channels
                    # Apply the alpha blending to the corresponding region of the frame
                    frame[valid_y_start:valid_y_end, valid_x_start:valid_x_end, c] = \
                        (1 - alpha_channel[:valid_y_end - valid_y_start, :valid_x_end - valid_x_start]) * \
                        frame[valid_y_start:valid_y_end, valid_x_start:valid_x_end, c] + \
                        alpha_channel[:valid_y_end - valid_y_start, :valid_x_end - valid_x_start] * \
                        red_ball_rgb[:valid_y_end - valid_y_start, :valid_x_end - valid_x_start, c]
                    
# Function to simulate the shooting
def shoot(shot_x, shot_y, direction):
    shots.append([shot_x, shot_y, direction])  # Store the direction to handle movement
    
def remove_offscreen_bullets():
    global shots
    threshold = 2  # Define the threshold for "near the border"
    
    # Remove bullets that are near the edges
    shots = [shot for shot in shots if (shot[0] > threshold and shot[0] < video_width - shot_image.shape[1] - threshold and 
                                        shot[1] > threshold and shot[1] < video_height - shot_image.shape[0] - threshold)]

def detect_hand_gesture(frame):
    # Convert the image to RGB (MediaPipe uses RGB, not BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    right_hand_position = None  # Right hand position
    left_hand_gesture = None  # Gesture of the left hand
    right_hand_gesture = None
    
    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            # Determine if it's the left or right hand
            hand_label = handedness.classification[0].label
            
            # Initialize finger landmarks for both hands
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Calculate the distance between the wrist and the index finger tip
            index_finger_distance = math.sqrt(
                (index_finger_tip.x - wrist.x) ** 2 + (index_finger_tip.y - wrist.y) ** 2
            )

            # Right hand (pointing with index finger)
            if hand_label == 'Right':  # This is the right hand
                # Check if the index finger is extended (pointing gesture)
                right_hand_position = (int(index_finger_tip.x * video_width), int(index_finger_tip.y * video_height))

            # Left hand (fist gesture)
            elif hand_label == 'Left':  # This is the left hand
                # Detect fist gesture (simple check)
                if (abs(thumb_tip.x - index_finger_tip.x) < 0.05 and
                    abs(index_finger_tip.x - middle_finger_tip.x) < 0.05 and
                    abs(middle_finger_tip.x - ring_finger_tip.x) < 0.05 and
                    abs(ring_finger_tip.x - pinky_finger_tip.x) < 0.05):
                    left_hand_gesture = "fist"
                
                elif (index_finger_tip.y < middle_finger_tip.y and   # Index finger is above middle finger
                    index_finger_tip.y < ring_finger_tip.y and     # Index finger is above ring finger
                    index_finger_tip.y < pinky_finger_tip.y and    # Index finger is above pinky finger
                    middle_finger_tip.y < ring_finger_tip.y and    # Middle finger is above ring finger
                    middle_finger_tip.y < pinky_finger_tip.y and   # Middle finger is above pinky finger
                    abs(index_finger_tip.x - middle_finger_tip.x) > 0.1):  # Fingers should be apart (distance threshold)
                    right_hand_gesture = "cheat" #(peace sign)
                    
                elif (abs(thumb_tip.x - index_finger_tip.x) > 0.1 and
                    abs(index_finger_tip.x - middle_finger_tip.x) > 0.1 and
                    abs(middle_finger_tip.x - ring_finger_tip.x) > 0.1 and
                    abs(ring_finger_tip.x - pinky_finger_tip.x) > 0.1 and
                    abs(thumb_tip.y - wrist.y) > 0.2 and
                    abs(index_finger_tip.y - wrist.y) > 0.2 and
                    abs(middle_finger_tip.y - wrist.y) > 0.2 and
                    abs(ring_finger_tip.y - wrist.y) > 0.2 and
                    abs(pinky_finger_tip.y - wrist.y) > 0.2):
                    right_hand_gesture = "open"  # Detect open hand gesture

    return right_hand_position, left_hand_gesture, right_hand_gesture

# Function to create and fire a shot with a cooldown mechanism
def shoot(shot_x, shot_y, direction, shoot_sound):
    global last_shot_time
    
    current_time = time.time()  # Get the current time in seconds
    
    if current_time - last_shot_time < 0.5:
        return 
    
    shoot_sound.play()  # Play the shooting sound

    # If cooldown is over, fire the shot
    if direction == "right":
        shot_x += player_image.shape[1]  # Adjust shot_x to be in front of the spaceship when facing right
    else:
        shot_x -= shot_image.shape[1]  # Adjust shot_x to be in front of the spaceship when facing left
    
    # Add the shot to the list with its direction
    shots.append([shot_x, shot_y, direction])

    # Update the time of the last shot
    last_shot_time = current_time

    
boss_height, boss_width, _ = boss_image.shape

# Modify the move_boss function to stop when the boss reaches the middle of the screen
def move_boss(isAlreadyCenter):
    global boss_x, boss_y, boss_state, boss_rotation_angle, state_change_time, isBossDead, isPaused, isPausedActive
    
    if not isBossDead:
        # Handle different boss states
        if boss_state == BOSS_STATE_IDLE:
            pass
        
        elif boss_state == BOSS_STATE_SHOOTING:
            # In shooting state, rotate clockwise and move rightwards
            boss_rotation_angle += 1  # Increase the angle for clockwise rotation
            if boss_rotation_angle >= 360:  # Reset to 0 degrees after full rotation
                boss_rotation_angle = 0
            
        elif boss_state == BOSS_STATE_SHOOTING_LASERS:
        # In shooting lasers state, rotate anticlockwise and move leftwards
            boss_rotation_angle -= 1  # Decrease the angle for anticlockwise rotation
            if boss_rotation_angle < 0:  # Ensure the angle stays between 0-360 degrees
                boss_rotation_angle = 359
        
        elif boss_state == BOSS_STATE_IDLE_INITIAL:
            # In shooting state, rotate clockwise and move rightwards
            boss_rotation_angle += 1  # Increase the angle for clockwise rotation
            if boss_rotation_angle >= 360:  # Reset to 0 degrees after full rotation
                boss_rotation_angle = 0
    
    # Rotate the boss image with the updated angle
    rotated_boss_image = rotate_boss_image(boss_image, boss_rotation_angle)
    
    # Determine the middle of the screen (horizontal center)
    screen_center_x = video_width // 2
    
    # Move boss leftwards until it reaches the center of the screen
    if boss_x > screen_center_x and not isPaused:
        boss_x -= boss_speed  # Move boss leftwards
        
    if isBossDead and not isPaused:
        if boss_y + boss_height < video_height:
            boss_y += 5  # Move the boss down
        else:
            boss_y = video_height - boss_height
            pygame.mixer.music.stop()
            from app import player_dead
            player_dead()
    
    # Optionally, change state or behavior after reaching the middle
    if not isAlreadyCenter:
        if boss_x <= screen_center_x:
            boss_state = BOSS_STATE_SHOOTING  # Change state when boss reaches the middle
            print("State changed to shooting")
            state_change_time = time.time()
            boss_x = screen_center_x  # Keep the boss at the center position
            isAlreadyCenter = True  # Set the flag to True after reaching the center
        
    return rotated_boss_image, isAlreadyCenter

def shoot_boss_ball():
    global shots
    
    # Get the position of the boss's right edge (where the ball will be fired from)
    ball_x = boss_x + boss_width  # Right edge of the boss
    ball_y = boss_y + boss_height // 2  # Middle of the boss height (adjust if needed)
    
    # Get the direction of the ball based on the boss's rotation angle
    angle_rad = np.radians(boss_rotation_angle)  # Convert angle to radians
    direction_x = np.cos(angle_rad)  # Ball's x direction
    direction_y = np.sin(angle_rad)  # Ball's y direction
    
    # Adjust the ball speed
    ball_speed_x = ball_speed * direction_x
    ball_speed_y = ball_speed * direction_y
    
    # Store the shot with the starting position and direction
    shots.append([ball_x, ball_y, ball_speed_x, ball_speed_y])  # [x, y, direction_x, direction_y]

def preprocess_boss_image(image):
    """
    Convert the boss image to grayscale and apply edge detection to extract the boss's outline.
    """
    # Convert the image to grayscale
    gray_boss = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    # Apply Canny edge detection
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
        # Get the height and width of the frame and the target position
        h, w, _ = frame.shape
        
        # Overlay the animation frame at the given (x, y) position in the combined_frame
        for c in range(0, 3):  # Loop through each channel (BGR)
            combined_frame[y:y+h, x:x+w, c] = \
                combined_frame[y:y+h, x:x+w, c] * (1 - frame[:, :, 3] / 255.0) + \
                frame[:, :, c] * (frame[:, :, 3] / 255.0)
        
def check_boss_shot_collision_with_edges(shots, boss_x, boss_y, boss_image, shot_image, combined_frame, hit_sound):
    """
    Check if any shot collides with the boss using detailed edge-based collision detection.
    """
    # Preprocess the boss image to get edges
    edges = preprocess_boss_image(boss_image)
    
    # Get the contours of the boss
    contours = get_boss_contours(edges)
    
    # Iterate over all shots
    for shot in shots[:]:
        shot_x, shot_y, direction = shot
        
        # Check if the shot is within the bounding box of the boss (for optimization)
        if shot_x + shot_image.shape[1] > boss_x and \
           shot_x < boss_x + boss_image.shape[1] and \
           shot_y + shot_image.shape[0] > boss_y and \
           shot_y < boss_y + boss_image.shape[0]:
            
            # Iterate over the contours of the boss to check for actual collision
            for contour in contours:
                # Check if the shot intersects with any of the boss's contours
                for point in contour:
                    # Transform the contour point into the boss's position on the screen
                    contour_x = boss_x + point[0][0]
                    contour_y = boss_y + point[0][1]

                    # Check if the shot pixel collides with the boss's contour
                    if (shot_x < contour_x < shot_x + shot_image.shape[1] and
                        shot_y < contour_y < shot_y + shot_image.shape[0]):
                        # Handle the collision (e.g., show animation or reduce boss health)
                        reduce_boss_health(10, hit_sound)

                        # Play the animation for the shot hit
                        show_animation([shot2_exp2, shot2_exp3, shot2_exp4, shot2_exp5], shot_x, shot_y, combined_frame)

                        # Remove shot after collision
                        shots.remove(shot)

                        # Optional: change boss state on hit
                        boss_state = "damaged"

                        # Return after handling the first collision (to avoid multiple hits)
                        return

def check_boss_shot_collision(shots, boss_x, boss_y, boss_width, boss_height, boss_image, shot_image, combined_frame, hit_sound):
    """
    Check for both edge-based and fallback simple AABB collision detection.
    """
    # First, attempt the edge-based collision detection
    check_boss_shot_collision_with_edges(shots, boss_x, boss_y, boss_image, shot_image, combined_frame, hit_sound)
    
def rotate_boss_image(image, angle):
    """ Rotate the boss image by a given angle (clockwise). """
    center = (image.shape[1] // 2, image.shape[0] // 2)  # Get the center of the image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)  # Get the rotation matrix
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))  # Apply rotation
    return rotated_image

def reduce_boss_health(amount, hit_sound):
    """
    Reduces the boss's health when a shot hits.
    Ensures health does not drop below 0.
    """
    global boss_health, isBossDead
    
    hit_sound.set_volume(0.6)
    hit_sound.play()
    boss_health = max(0, boss_health - amount)
    if (boss_health <= 0) :
        hit_sound.set_volume(0.6)
        hit_sound.play()
        isBossDead = True
    
def draw_health_bar(frame, health, max_health, health_bar_y, health_bar_width, health_bar_height):
    """
    Draw the health bar on the screen based on the current health.
    
    Parameters:
    - frame: The image/frame on which the health bar is drawn.
    - health: The current health value.
    - max_health: The maximum health value for scaling.
    - health_bar_y: The top-left corner y-coordinate of the health bar.
    - health_bar_width, health_bar_height: The width and height of the health bar.
    """
    # Increase the width of the health bar (adjust as needed)
    health_bar_width = 1000  # Adjust the width as per your requirement
    
    # Calculate the x-coordinate to center the health bar
    health_bar_x = (frame.shape[1] - health_bar_width) // 2  # Centers the health bar horizontally
    
    # Move the health bar lower by adjusting the y position
    health_bar_y += 40  # Adjust the value here to move it lower or higher
    
    # Calculate the current width based on health
    health_bar_current_width = int(health / max_health * health_bar_width)

    # Draw the health bar background (empty health bar)
    cv2.rectangle(frame, (health_bar_x, health_bar_y), 
                  (health_bar_x + health_bar_width, health_bar_y + health_bar_height), 
                  (0, 0, 0), -1)  # black background

    # Draw the health bar foreground (filled health bar)
    cv2.rectangle(frame, (health_bar_x, health_bar_y), 
                  (health_bar_x + health_bar_current_width, health_bar_y + health_bar_height), 
                  (255, 255, 255), -1)  # white foreground
    
    # Draw white border around the health bar
    cv2.rectangle(frame, (health_bar_x, health_bar_y), 
                  (health_bar_x + health_bar_width, health_bar_y + health_bar_height), 
                  (255, 255, 255), 2)  # white border with thickness of 2
    
    # Add text on top of the health bar
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "CaroluM: The Trainee Slayer"
    text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2  # center the text horizontally
    text_y = health_bar_y - 20  # position the text just above the health bar
    cv2.putText(frame, text, (text_x, text_y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

def draw_health(frame):
    """Draw the health images at the bottom-left corner of the screen."""
    global health_image, player_health, health_offset_x, health_offset_y
    
    # Check if the frame is valid
    if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
        print("Invalid frame!")
        return frame
    
    # Ensure health_image is not None and is properly loaded
    if health_image is None:
        print("Error: health_image is not loaded properly!")
        return frame

    # Check if the health image has 4 channels (alpha channel)
    has_alpha = health_image.shape[2] == 4  # Check for alpha channel (BGRA or RGBA)
    
    # Resize the health image to a smaller size (e.g., 10% of original size)
    new_width = int(health_image.shape[1] * 0.04)  # 10% of original width
    new_height = int(health_image.shape[0] * 0.03)  # 10% of original height
    resized_health_image = cv2.resize(health_image, (new_width, new_height))

    # Convert the frame to BGRA if it has no alpha channel (convert BGR to BGRA)
    if frame.shape[2] != 4:
        print("Converting frame to BGRA.")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    # Calculate y_offset to place health bar at the bottom-left corner
    y_offset = frame.shape[0] - resized_health_image.shape[0] - health_offset_y
    
    # If y_offset is still negative, adjust it to a reasonable value
    if y_offset < 0:
        y_offset = frame.shape[0] - resized_health_image.shape[0]  # Set to bottom of the frame
        print("Warning: y_offset was negative. Adjusted to bottom of the frame.")
    
    # Loop to place each resized health image
    for i in range(player_health):
        x_offset = health_offset_x + i * resized_health_image.shape[1]
        
        # Ensure the health image fits within the frame
        if x_offset + resized_health_image.shape[1] <= frame.shape[1] and y_offset + resized_health_image.shape[0] <= frame.shape[0]:
            # Get the region of interest (ROI) in the frame where the health image will go
            roi = frame[y_offset:y_offset + resized_health_image.shape[0], x_offset:x_offset + resized_health_image.shape[1]]

            if has_alpha:
                # Create a mask of the health image where the alpha channel is not 0
                alpha_mask = resized_health_image[:, :, 3] / 255.0  # Normalize alpha channel to range [0, 1]
                alpha_mask = np.dstack([alpha_mask, alpha_mask, alpha_mask])  # Convert to 3 channels (RGB)

                # Blend the health image with the frame using the alpha mask
                for c in range(3):  # Iterate over RGB channels
                    roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask[:, :, c]) + resized_health_image[:, :, c] * alpha_mask[:, :, c]
            else:
                # If no alpha channel, we assume full opacity, so no blending required
                roi[:, :, :3] = resized_health_image[:, :, :3]  # Only copy the RGB channels

            # Place the blended image back into the frame
            frame[y_offset:y_offset + resized_health_image.shape[0], x_offset:x_offset + resized_health_image.shape[1]] = roi

    return frame
    
## Laser from boss ##
lasers = []
laser_rotation_speed = 0.1

def shoot_lasers(boss_x, boss_y, boss_rotation_angle):
    """Launch multiple red balls from fixed positions around the boss."""
    global last_ball_shot_time
    
    current_time = time.time()
    if current_time - last_ball_shot_time >= BALL_SHOT_COOLDOWN:  # Check if enough time has passed since the last shot
        
        num_balls = 5  # Number of balls to spawn
        radius = 270  # Radius for the circular spawn pattern (distance from the boss)
        
        # Fixed angle increment for even spacing (120 degrees for three balls)
        angle_increment = 2 * math.pi / num_balls
        
        # Create spawn positions based on a fixed circular pattern
        spawn_positions = []
        for i in range(num_balls):
            angle = i * angle_increment  # Angle for each ball (evenly spaced)
            
            # Calculate the x and y offsets based on the angle and radius
            spawn_x_offset = radius * math.cos(angle)  # X offset based on cosine
            spawn_y_offset = radius * math.sin(angle)  # Y offset based on sine
            
            # Calculate the spawn positions based on the boss's position and rotation angle
            spawn_x = boss_x + spawn_x_offset * math.cos(math.radians(boss_rotation_angle)) - spawn_y_offset * math.sin(math.radians(boss_rotation_angle))
            spawn_y = boss_y + spawn_x_offset * math.sin(math.radians(boss_rotation_angle)) + spawn_y_offset * math.cos(math.radians(boss_rotation_angle))
            
            # Apply the downward and rightward offset
            offset_x = 270  # Shift to the right (adjust as needed)
            offset_y = 300  # Shift downward (adjust as needed)
            
            # Apply the offset
            spawn_x += offset_x
            spawn_y += offset_y
            
            spawn_positions.append((spawn_x, spawn_y))
            
        # Now spawn the red balls with the boss's rotation angle
        for spawn_x, spawn_y in spawn_positions:
            # Use the boss's rotation angle to direct the shot
            new_ball = RedBall(spawn_x, spawn_y, boss_x, boss_y, ball2_speed)
            red_balls.append(new_ball)
        
        # Update the last shot time
        last_ball_shot_time = current_time

# Function to move lasers around the boss in anticlockwise direction
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
        
        # Get the red ball image dimensions
        ball_width = red_ball_image.shape[1]
        ball_height = red_ball_image.shape[0]
        
        # Calculate the end positions based on the ball's position and image size
        x_end = x_start + ball_width
        y_end = y_start + ball_height
        
        # Ensure the end positions do not exceed the frame bounds
        x_end = min(x_end, frame.shape[1])
        y_end = min(y_end, frame.shape[0])

        # Ensure the slice has non-zero size
        if x_start < frame.shape[1] and y_start < frame.shape[0] and x_end > x_start and y_end > y_start:
            # Apply alpha blending for transparent background (RGBA image handling)
            alpha_channel = red_ball_image[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
            red_ball_rgb = red_ball_image[:, :, :3]  # RGB channels
            
            # Handle the region size to ensure no empty slice
            valid_x_end = min(x_end, frame.shape[1])
            valid_y_end = min(y_end, frame.shape[0])
            valid_x_start = max(x_start, 0)
            valid_y_start = max(y_start, 0)
            
            # Ensure the region dimensions are non-zero
            if valid_x_end > valid_x_start and valid_y_end > valid_y_start:
                for c in range(3):  # Iterate over the RGB channels
                    # Apply the alpha blending to the corresponding region of the frame
                    frame[valid_y_start:valid_y_end, valid_x_start:valid_x_end, c] = \
                        (1 - alpha_channel[:valid_y_end - valid_y_start, :valid_x_end - valid_x_start]) * \
                        frame[valid_y_start:valid_y_end, valid_x_start:valid_x_end, c] + \
                        alpha_channel[:valid_y_end - valid_y_start, :valid_x_end - valid_x_start] * \
                        red_ball_rgb[:valid_y_end - valid_y_start, :valid_x_end - valid_x_start, c]

def update_boss_state(boss_state, state_change_time, current_time, combined_frame, boss_change_sound):
    global boss_image, boss_image_index, laser_image, lasers, boss_rotation_angle, isBossDead

    if not isBossDead:
        # Handle shooting state
        if boss_state == BOSS_STATE_SHOOTING:
            if current_time - state_change_time >= shooting_time:
                move_red_balls()
                draw_red_balls(combined_frame)
                # Transition to idle state after shooting time is over
                boss_state = BOSS_STATE_IDLE
                state_change_time = current_time
                print("State changed to idle")
            else:
                # Keep shooting red balls during the shooting state
                shoot_red_ball(boss_x, boss_y, boss_rotation_angle)
                move_red_balls()
                draw_red_balls(combined_frame)

        # Handle idle state
        elif boss_state == BOSS_STATE_IDLE:
            if current_time - state_change_time >= idle_time:
                boss_state = BOSS_STATE_IMAGE_CHANGE1
                state_change_time = current_time  # Reset the timer for image change state
                print("State changed to image change to index 2")
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
            else:
                # Let the red balls that already deployed continue to move
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
        
        # Handle image change state
        elif boss_state == BOSS_STATE_IMAGE_CHANGE1:
            if current_time - state_change_time >= image_change_duration:
                boss_state = BOSS_STATE_IMAGE_CHANGE2
                state_change_time = current_time  # Reset the timer for image change
                print("State changed to image change to index 3")
                
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
            else:
                boss_image = cv2.imread('static/assets/boss_asset/bos_2.png', cv2.IMREAD_UNCHANGED)
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
                boss_change_sound.set_volume(0.5)
                boss_change_sound.play()
                
        elif boss_state == BOSS_STATE_IMAGE_CHANGE2:
            if current_time - state_change_time >= image_change_duration:
                state_change_time = current_time  # Reset the timer for image change
                boss_state = BOSS_STATE_SHOOTING_LASERS
                print("State changed to shooting lasers")
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
            else:
                boss_image = cv2.imread('static/assets/boss_asset/bos_3.png', cv2.IMREAD_UNCHANGED)
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
                
        elif boss_state == BOSS_STATE_SHOOTING_LASERS:
            if current_time - state_change_time >= shooting_time:
                # Transition to idle state after shooting time is over
                boss_state = BOSS_STATE_IDLE2
                state_change_time = current_time
                print("State changed to idle")
                move_lasers()
                draw_lasers(combined_frame)
                move_red_balls()
                draw_red_balls(combined_frame)
            else:
                # Keep shooting laser during the state
                shoot_lasers(boss_x, boss_y, boss_rotation_angle)
                move_lasers()
                draw_lasers(combined_frame)
                move_red_balls()
                draw_red_balls(combined_frame)
        
        elif boss_state == BOSS_STATE_IDLE2:
            if current_time - state_change_time >= idle_time:
                boss_state = BOSS_STATE_IMAGE_CHANGEBACK2
                state_change_time = current_time  # Reset the timer for image change state
                print("State changed to image change to index 2")
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
            else:
                # Let the red balls that already deployed continue to move
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
        
        elif boss_state == BOSS_STATE_IMAGE_CHANGEBACK2:
            if current_time - state_change_time >= image_change_duration:
                boss_state = BOSS_STATE_IMAGE_CHANGEBACK1
                state_change_time = current_time  # Reset the timer for image change
                print("State changed to image change to index 3")
                
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
            else:
                boss_image = cv2.imread('static/assets/boss_asset/bos_2.png', cv2.IMREAD_UNCHANGED)
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
                
        elif boss_state == BOSS_STATE_IMAGE_CHANGEBACK1:
            if current_time - state_change_time >= image_change_duration:
                state_change_time = current_time  # Reset the timer for image change
                boss_state = BOSS_STATE_SHOOTING
                print("State changed to shooting lasers")
                move_red_balls()
                draw_red_balls(combined_frame)
                move_lasers()
                draw_lasers(combined_frame)
            else:
                boss_image = cv2.imread('static/assets/boss_asset/bos_1.png', cv2.IMREAD_UNCHANGED)
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
    duration = 3  # Seconds
    
    if show_text_start_time is None:
        show_text_start_time = time.time()
        print("Start time initialized:", show_text_start_time)

    # Get the current time
    current_time = time.time()

    # Check if the duration has passed
    if current_time - show_text_start_time < duration and not isDone:
        # Put text on the bottom-right corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 255, 255)  # White text
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Positioning the text at the bottom-right corner
        text_x = frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
        text_y = frame.shape[0] - 10  # 10 pixels from the bottom

        # Debug: Print text position to verify if it's visible on the screen
        print(f"Text position: ({text_x}, {text_y})")
        
        # Display the text on the frame
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness)
    else:
        # Reset the start time after duration
        show_text_start_time = None
        isDone = True
    
    return frame, isDone

def is_five_fingers_extended(landmarks):
    # Thumb tip and base (MCP joint)
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_base = landmarks[mp_hands.HandLandmark.THUMB_MCP]

    # Index tip and base (MCP joint)
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_base = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    # Middle tip and base (MCP joint)
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_base = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    # Ring tip and base (MCP joint)
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_base = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]

    # Pinky tip and base (MCP joint)
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_base = landmarks[mp_hands.HandLandmark.PINKY_MCP]

    # Check if each finger's tip is higher than its base (extended)
    thumb_extended = thumb_tip.y < thumb_base.y
    index_extended = index_tip.y < index_base.y
    middle_extended = middle_tip.y < middle_base.y
    ring_extended = ring_tip.y < ring_base.y
    pinky_extended = pinky_tip.y < pinky_base.y

    # Return True if all fingers are extended
    return thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended

# Function to scroll the background and move the player
def scroll_background(camera, isReset):
    global player_x, player_y, shots, player_image, boss_x, boss_y, boss_health, boss_rotation_angle, boss_image, boss_state, state_change_time, boss_image_index
    global state_change_time, isDead, last_player_y, isCheat, isCheatTriggered
    global max_health, isGameTwoDone, isGameThreeDone, lasers, red_balls, boss_width, player_health, isBossDead, isPaused, isPausedActive
    
    isAlreadyCenter = False
    gesture_start_time = None
    last_open_hand_time = 0
    cooldown_period = 2 
    
    if isReset:
        # Player initial position (middle left)
        player_x = 100  # x position at the left side of the screen
        player_y = video_height // 2  # y position at the center
        last_player_y = player_y
        player_health = 5
        
        boss_x = video_width - boss_width - 50  # 50 pixels from the right edge
        boss_y = video_height // 4  # Fixed vertical position (you can adjust this based on your needs)
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
    sensitivity = 10  # Adjust sensitivity here
    last_player_x = player_x  # Track the last x position for flip decision
    last_player_y
    last_facing_direction = "right"  # Track the player's last facing direction (default is right)
    
    print("Initial state: ", boss_state)
    load_animation_images()
    
    isDone = False
    
    pygame.mixer.init()
    
    pygame.mixer.music.load("static/assets/sound/boss_bg.mp3")
    pygame.mixer.music.play(loops=-1, start=0.0)
    
    shoot_sound = pygame.mixer.Sound("static/assets/sound/shoot_sound.mp3")
    boss_change_sound = pygame.mixer.Sound("static/assets/sound/boss_change_sound.mp3")
    hit_sound = pygame.mixer.Sound("static/assets/sound/dead_sound.mp3")

    while True:
        from app import change_game_one_state, is_game_one_done, change_game_two_state, is_game_two_done
        from app import change_game_three_state, is_game_three_done, is_game_four_done, change_game_four_state
        
        if boss_health <= max_health*0.95 and is_game_four_done == False:
            change_game_four_state()
            print("Redirect to Game 4")
        
        # if boss_health <= max_health*0.8 and is_game_one_done == False:
        #     change_game_one_state()
        #     print("Redirect to Game 1")
        # elif boss_health <= max_health*0.6 and is_game_two_done == False:
        #     change_game_two_state()
        #     print("Redirect to Game 2")
        # elif boss_health <= max_health*0.4 and is_game_three_done == False:
        #     change_game_three_state()
        #     print("Redirect to Game 3")
        # elif boss_health <= max_health*0.2 and is_game_four_done == False:
        #     change_game_four_state()
        #     print("Redirect to Game 4")
        
        # Move the image by changing the x_offset
        x_offset -= 10  # Adjust speed here
        if x_offset <= -bg_width:
            x_offset = 0  # Reset the offset when it moves off-screen

        # Create a copy of the background and shift it horizontally
        if not isPaused:
            shifted_bg = np.roll(bg_image, x_offset, axis=1)

        # Resize background to fit the video dimensions
        resized_bg = cv2.resize(shifted_bg, (video_width, video_height))

        # Capture a frame from the webcam
        ret, frame = camera.read()
        if not ret:
            break
        
        # frame = cv2.flip(frame, 1)
        
        # Resize the webcam feed to fit the video dimensions
        resized_frame = cv2.resize(frame, (video_width, video_height))

        # Detect hand gestures
        right_hand_position, left_hand_gesture, right_hand_gesture = detect_hand_gesture(frame)
        
        # Move the player image based on the right hand's index finger position
        if right_hand_position:
            adjusted_x = video_width - (right_hand_position[0] + player_image.shape[1] // 2)
            player_x += (adjusted_x - player_x) // sensitivity  # Smooth player movement

            adjusted_y = right_hand_position[1] - player_image.shape[0] // 2
            player_y = adjusted_y

        # Ensure player doesn't go out of bounds
        player_x = max(0, min(player_x, video_width - player_image.shape[1]))
        player_y = max(0, min(player_y, video_height - player_image.shape[0]))

        # Ensure player_image_flipped is initialized
        player_image_flipped = player_image  # Default to the normal player image

        # Only allow movement and actions when the player is not dead
        if not isDead:
            # Horizontal movement: flip based on direction
            if player_x < last_player_x:
                last_facing_direction = "left"
                player_image_flipped = cv2.flip(player_image, 1)
            elif player_x > last_player_x:
                last_facing_direction = "right"
                player_image_flipped = player_image

            last_player_x = player_x

            # Vertical movement: update player_y
            if right_hand_position:
                adjusted_x = video_width - (right_hand_position[0] + player_image.shape[1] // 2)
                player_x += (adjusted_x - player_x) // sensitivity  # Smooth player movement

                adjusted_y = right_hand_position[1] - player_image.shape[0] // 2
                player_y = adjusted_y
                last_player_y = player_y

            # Create the shot if the left hand is detected as a fist
            if left_hand_gesture == "fist":
                shoot(player_x, player_y, last_facing_direction, shoot_sound)
                
            if right_hand_gesture == "cheat" and isCheatTriggered == False:
                print("Activate cheat")
                isCheat = True
                isCheatTriggered = True
                isDone = False
        
        # if right_hand_gesture == "open":
        #     print("Palm hand detected")
            
        #     if isPaused:
        #         isPaused = False
        #     else:
        #         isPaused = True
            
        #     if isPaused:
        #         print("Pause is active")
        #         isDead = not isDead
        #         isBossDead = not isBossDead
        #         pygame.mixer.music.pause()
        #     elif not isPaused:
        #         print("Pause deactivated")
        #         isDead = not isDead
        #         isBossDead = not isBossDead
        #         pygame.mixer.music.unpause()

        # If the player is dead, prevent any further movement or actions
        else:
            # Keep the player in the current position, do not change player_x or player_y
            player_image_flipped = player_image  # Keep the image static
            last_facing_direction = last_facing_direction  # Maintain the previous facing direction
            
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
                        
                        # if not isCheatTriggered:
                        #     isCheat = True
                            
                        pygame.mixer.music.pause()
                    elif not isPaused:
                        print("Pause deactivated")
                        isDead = not isDead
                        isBossDead = not isBossDead
                        
                        # if not isCheatTriggered:
                        #     isCheat = False
                        
                        pygame.mixer.music.unpause()

                    last_open_hand_time = current_time_pause
        else:
            gesture_start_time = None

        # Draw the background and player image
        combined_frame = resized_bg.copy()

        # Ensure player image fits in the frame (handling index bounds)
        player_height, player_width = player_image_flipped.shape[:2]

        # Clamp player position to ensure it stays within the background bounds
        player_x = max(0, min(player_x, video_width - player_width))
        player_y = max(0, min(player_y, video_height - player_height))

        # Ensure the player image doesn't exceed the background size
        if player_y + player_height <= video_height and player_x + player_width <= video_width:
            # Proceed only if player image fits within the frame
            for c in range(0, 3):
                # Blend player image with the background using the alpha channel
                combined_frame[player_y:player_y + player_height, player_x:player_x + player_width, c] = \
                    combined_frame[player_y:player_y + player_height, player_x:player_x + player_width, c] * (1 - player_image_flipped[:, :, 3] / 255.0) + \
                    player_image_flipped[:, :, c] * (player_image_flipped[:, :, 3] / 255.0)
        else:
            print("Error: player image exceeds background bounds")

        # First, ensure we have a valid boss_width and boss_height before using them.
        if 'boss_width' not in locals() or 'boss_height' not in locals():
            boss_width, boss_height = boss_image.shape[1], boss_image.shape[0]

        # Adjust the x-position if the image exceeds the screen width
        if boss_x + boss_width > video_width:
            boss_x = video_width - boss_width  # Adjust the x position

        # Adjust the y-position if the image exceeds the screen height
        if boss_y + boss_height > video_height:
            boss_y = video_height - boss_height  # Adjust the y position

        # Resize if necessary (only resize if the boss image is larger than the screen)
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
                
        # Ensure the boss image is within the frame boundaries
        rotated_boss_image_height = int(rotated_boss_image.shape[0])
        rotated_boss_image_width = int(rotated_boss_image.shape[1])
        
        boss_x = int(boss_x)
        boss_y = int(boss_y)

        # Overlay the boss image (right side of the screen)
        if boss_y + rotated_boss_image_height <= video_height and boss_x + rotated_boss_image_width <= video_width:
            for c in range(0, 3):
                combined_frame[boss_y:boss_y + rotated_boss_image_height, boss_x:boss_x + rotated_boss_image_width, c] = \
                    combined_frame[boss_y:boss_y + rotated_boss_image_height, boss_x:boss_x + rotated_boss_image_width, c] * (1 - rotated_boss_image[:, :, 3] / 255.0) + \
                    rotated_boss_image[:, :, c] * (rotated_boss_image[:, :, 3] / 255.0)
        else:
            print(f"Boss image out of screen: boss_x={boss_x}, boss_y={boss_y}, boss_width={boss_width}, boss_height={boss_height}")

        # Move and draw the shots
        for shot in shots:
            shot_x, shot_y, direction = shot
            if direction == "right":
                shot_x += 20
            else:
                shot_x -= 20

            # Ensure shot stays within bounds
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
            duration = 3  # Seconds
            
            if show_text_start_time is None:
                show_text_start_time = time.time()
                print("Start time initialized:", show_text_start_time)

            # Get the current time
            current_time = time.time()

            # Check if the duration has passed
            if current_time - show_text_start_time < duration:
                # Put text on the bottom-right corner
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_color = (255, 255, 255)
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                
                # Positioning the text at the bottom-right corner
                text_x = combined_frame.shape[1] - text_size[0] - 20
                text_y = combined_frame.shape[0] - 70
                
                rect_color = (130, 130, 130)  # grey rectangle
                rect_thickness = -1  # Filled rectangle
                cv2.rectangle(combined_frame, (text_x - 5, text_y - text_size[1] - 5), 
                            (text_x + text_size[0] + 5, text_y + 5), rect_color, rect_thickness)

                # Display the text on the frame
                cv2.putText(combined_frame, text, (text_x, text_y), font, font_scale, font_color, thickness)
            else:
                # Reset the start time after duration
                show_text_start_time = None
                isDone = True
        
        #player health
        combined_frame, isDead = check_collisions_and_update_health(combined_frame, hit_sound)
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

        # Return the combined frame for displaying in Flask template
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', combined_frame)[1].tobytes() + b'\r\n')