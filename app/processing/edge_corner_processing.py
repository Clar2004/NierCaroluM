import cv2
import numpy as np
import mediapipe as mp
import time
import pygame

# Initialize MediaPipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

thumb_path = []  # Track drawn line
health_points = 3  # Initialize health points
game_started = False  # To track if the game has started
cheat_mode_activated = False  # Cheat mode flag

five_finger_start_time = None  # Start time when five fingers are detected
five_finger_time_threshold = 2.0  # Minimum duration (seconds) to hold the gesture for cheat activation
is_cheat_triggered = False

# Load and process maze image
def initialize_maze_image(image_path):
    # Load the maze image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection (Canny)
    edges = cv2.Canny(gray, 100, 200)

    # Corner detection (Harris Corner)
    gray_float = np.float32(gray)
    corners = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)

    # Dilate edges and corners to reduce sensitivity (increase margin for errors)
    dilated_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)  # Dilation by a kernel size of 5x5
    dilated_corners = cv2.dilate(corners, np.ones((5, 5), np.uint8), iterations=1)

    # Draw edges and corners on original image
    image[edges != 0] = [0, 255, 0]  # Green edges
    image[corners > 0.01 * corners.max()] = [0, 0, 255]  # Red corners

    return image, dilated_edges, dilated_corners

# Entry points of the maze (adjusted coordinates to match the actual maze image)
entry_points = [
    (125, 25),   # Left entry
    (1230, 115),  # Top entry
    (50, 470),   # Bottom-left entry
    (1230, 560)  # Right entry
]

# CPU detection zone
cpu_center = (700, 350)  # Approximate center coordinates of the CPU
cpu_radius = 50  # Radius around the CPU to detect touch

# Function to check if all five fingers are extended
def are_all_fingers_extended(hand_landmarks, height):
    extended = []

    # Check if each finger tip is above its respective knuckle
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

def update_health_(new_health):
    from app import update_health
    update_health(new_health)
    
# To activate cheat mode
def activate_cheat_mode(message="Cheat activated!"):
    from app import show_cheat
    
    show_cheat()

# Generate maze frames combined with interaction logic
def generate_maze_interaction_frames(camera):
    from app import change_mini_game_three_state
    global thumb_path, health_points, game_started, cheat_mode_activated, five_finger_start_time
    global is_cheat_triggered

    maze_image, edges, corners = initialize_maze_image('static/assets/images/maze.jpg')
    
    last_health_emit = health_points
    cheat_mode_activated = False
    is_cheat_triggered = False
    
    # Initialize the mixer for pygame
    pygame.mixer.init()

    # Load the music file
    pygame.mixer.music.load("static/assets/sound/boss_bg_8bit.mp3")
    # Play the music in a loop (-1 means infinite loop)
    pygame.mixer.music.play(loops=-1, start=0.0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Resize the maze image to match the frame dimensions
        height, width, _ = frame.shape
        resized_maze_image = cv2.resize(maze_image, (width, height))
        resized_edges = cv2.resize(edges, (width, height))
        resized_corners = cv2.resize(corners, (width, height))

        # Draw the entry circles on the maze image if the game hasn't started yet
        if not game_started:
            for point in entry_points:
                cv2.circle(resized_maze_image, point, 20, (255, 255, 255), -1)  # Draw white circles

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe for gesture detection
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                if hand_label == 'Right':
                    # Get index finger tip position
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    tip_x, tip_y = int(index_tip.x * width), int(index_tip.y * height)

                    # Add boundary checks for tip_x and tip_y
                    tip_x = max(0, min(tip_x, width - 1))
                    tip_y = max(0, min(tip_y, height - 1))

                    # Check for five fingers extended
                    if are_all_fingers_extended(hand_landmarks, height):
                        if five_finger_start_time is None:
                            five_finger_start_time = time.time()
                        else:
                            elapsed_time = time.time() - five_finger_start_time
                            if elapsed_time >= five_finger_time_threshold and is_cheat_triggered == False:
                                cheat_mode_activated = True
                                is_cheat_triggered = True
                                print("Cheat mode activated: Edge and corner detection disabled.")
                                
                                cheat_text = "Cheat Activated"
                                
                                activate_cheat_mode()
        
                    else:
                        five_finger_start_time = None

                    # If the game hasn't started yet, check if user touches any entry point
                    if not game_started:
                        for point in entry_points:
                            px, py = point 
                            distance = np.sqrt((tip_x - px) ** 2 + (tip_y - py) ** 2)
                            if distance <= 20:  # If the finger touches the entry circle
                                game_started = True
                                thumb_path = []  # Clear any existing path
                                print("Game started: Player entered the maze.")
                                break

                    # Track the finger path always (whether game started or not)
                    thumb_path.append((tip_x, tip_y))

                    # Draw the path on the maze image (white line)
                    for i in range(1, len(thumb_path)):
                        x1, y1 = thumb_path[i - 1]
                        x2, y2 = thumb_path[i]
                        # Ensure the coordinates stay within bounds
                        x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
                        x2, y2 = max(0, min(x2, width - 1)), max(0, min(y2, height - 1))
                        cv2.line(resized_maze_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

                    # Apply game rules if the game has started
                    if game_started and health_points > 0:
                        # Check if the player reaches the CPU
                        distance_to_cpu = np.sqrt((tip_x - cpu_center[0]) ** 2 + (tip_y - cpu_center[1]) ** 2)
                        if distance_to_cpu <= cpu_radius:
                            print("Player reached the CPU!")
                            change_mini_game_three_state()

                        # Check collision with dilated edges or corners (only if cheat mode is not activated)
                        if not cheat_mode_activated:
                            if resized_edges[tip_y, tip_x] != 0 or resized_corners[tip_y, tip_x] > 0.01 * resized_corners.max():
                                health_points -= 1
                                print(f"Health updated: {health_points} health points remaining.")
                                thumb_path = []  # Clear the path

                                # Emit health update only if health changes
                                if last_health_emit != health_points:
                                        last_health_emit = health_points
                                        update_health_(health_points)

                                if health_points == 0:
                                    print("Game Over: Line cleared after 3 collisions.")
                                    # Reset the game
                                    health_points = 3
                                    game_started = False
                                    cheat_mode_activated = False
                                    thumb_path = []  # Clear the path
                                    print("Game reset: Health restored to 3 and entry points redrawn.")
                                    update_health_(health_points)

            # Encode and yield the frame
            ret, buffer = cv2.imencode('.jpg', resized_maze_image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')