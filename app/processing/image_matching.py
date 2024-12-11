import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

images = [
   "static/assets/image_matching/circle.jpg",
   "static/assets/image_matching/rhombus.jpg",
   "static/assets/image_matching/square.jpg"
]

prev_position = None  # Store previous position for line drawing
smoothed_position = None  # For applying smoothing
drawing = False  # Track whether drawing is enabled
drawing_start_time = None  # Track when drawing started
countdown_time = 3  # Countdown duration in seconds
last_thumbs_up_time = None  # Time when thumbs up was last detected
game_started = False  # Track whether the game has started
waiting_for_thumbs_up = True  # Track if we are waiting for thumbs up gesture
countdown_started = False

trigger_start_countdown = False 
trigger_end_countdown = False

alpha = 0.2  # Smoothing factor

def image_matching(drawn_image, target_image):
    """Match the drawn image with the target image using ORB"""
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(drawn_image, None)
    kp2, des2 = orb.detectAndCompute(target_image, None)

    # Use Brute-Force Matcher to compare the descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    # Calculate similarity percentage
    similarity_percentage = len(matches) / len(kp1) * 100 if len(kp1) > 0 else 0
    return similarity_percentage

# Function to detect thumbs up gesture
def is_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    # Check if thumb is up (thumb tip above the thumb MCP and index finger is down)
    thumbs_up = thumb_tip.y < thumb_mcp.y and index_tip.y > thumb_tip.y
    return thumbs_up

def draw_text(frame, text, position=(50, 50), font_scale=1, color=(0, 255, 0), thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# Function to detect pinch gesture
def is_pinch(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance < 0.05  # If thumb and index finger are close, return True for pinch

def draw_on_canvas(cursor_x, cursor_y, canvas, prev_position, smoothed_position, alpha=0.2):
    """Draw on the canvas with smoothing"""
    if prev_position is not None:
        smoothed_position = (
            int(alpha * cursor_x + (1 - alpha) * prev_position[0]),
            int(alpha * cursor_y + (1 - alpha) * prev_position[1]),
        )
        # Draw the line between previous and smoothed position
        if 0 <= smoothed_position[0] < canvas.shape[1] and 0 <= smoothed_position[1] < canvas.shape[0]:
            cv2.line(canvas, prev_position, smoothed_position, (0, 0, 0), 5)  # Black line
        prev_position = smoothed_position  # Update previous position
    else:
        prev_position = (cursor_x, cursor_y)  # Set initial position

    return prev_position, smoothed_position

def game_loop(cap):
    global prev_position, smoothed_position, drawing, drawing_start_time, countdown_time, last_thumbs_up_time, game_started, waiting_for_thumbs_up
    global trigger_end_countdown, trigger_start_countdown, countdown_started

    alpha = 0.2  # Smoothing factor
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

    
    # Initialize the mixer for pygame
    pygame.mixer.init()

    # Load the music file
    pygame.mixer.music.load("static/assets/sound/boss_bg_8bit.mp3")
    pygame.mixer.music.play(loops=-1, start=0.0)

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        if not isInitialized and canvas is None:
            isInitialized = True
            canvas = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255  # White background

        # Convert to RGB (MediaPipe requires RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand landmarks
        results = hands.process(rgb_frame)
        
        last_thumbs_up_detection_time = 0 

        # If hands are detected
        if results.multi_hand_landmarks:
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Check which hand it is (left or right)
                hand_label = handedness.classification[0].label
                
                # Process only the right hand
                if hand_label == 'Right':
                    # Get the right hand landmarks
                    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x, y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                    
                    # if prev_position and smoothed_position:
                        # print(f"Circle coordinates: (x: {x}, y: {y})")

                    # Check for pinch gesture
                    if is_pinch(landmarks) and game_started:
                        if drawing:
                            print("Pinch detected! Stop drawing for now.")
                            drawing = False  # Stop drawing immediately, but don't reset the game
                            prev_position = None  # Reset previous position
                            smoothed_position = None  # Reset smoothed position
                    elif not is_pinch(landmarks) and game_started:
                        if not drawing:
                            prev_position = (x, y)
                            smoothed_position = (x, y)
                        drawing = True

                    if drawing and game_started:
                        prev_position, smoothed_position = draw_on_canvas(x, y, canvas, prev_position, smoothed_position, alpha)
                        
                        # print(f"Line from: {prev_position} to {smoothed_position}")
                        
                    # Create a transparent overlay
                    overlay = np.zeros_like(frame, dtype=np.uint8)
                    
                    # Draw the circle on the overlay
                    # cv2.circle(overlay, (x, y), 10, (192, 192, 192), -1)

                    # Blend the overlay with the frame
                    alpha = 0.5  # Transparency factor
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    # Check if the thumbs up gesture is detected for the right hand
                    if is_thumbs_up(landmarks):
                        current_time = time.time()  # Get the current time

                        # Check if enough time has passed since the last thumbs-up (2-second cooldown)
                        if current_time - last_thumbs_up_detection_time >= 2:  # If 2 seconds have passed
                            if not game_started and waiting_for_thumbs_up:
                                # Random number between 0 to 2
                                random_number = random.randint(0, 2)
                                
                                print("Random number: ", random_number)
                                
                                from app import match_start
                                match_start(random_number)

                                # Start the countdown for 3 seconds after thumbs-up detected
                                trigger_start_countdown = True
                                last_thumbs_up_time = time.time()  # Start the drawing timer
                                waiting_for_thumbs_up = False  # No longer waiting for thumbs-up
                                
                                # Reset canvas and start the game countdown
                                print("Thumbs up detected! Starting countdown...")
                                canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255  # Reset canvas

                                # Update the time of the last thumbs-up detection
                                last_thumbs_up_detection_time = current_time  # Record the time when thumbs-up was detected

                        else:
                            # If the cooldown is still active, print that it's in cooldown
                            print("Cooldown active. Please wait for 2 seconds before detecting another thumbs-up.")
                        
        # Handle countdown logic (decrease countdown time as seconds pass)
        if trigger_start_countdown and last_thumbs_up_time is not None:
            # print("Starting game countdown")
                        
            # Time elapsed since thumbs up was detected
            time_elapsed = time.time() - last_thumbs_up_time
            countdown_value = int(time_elapsed)
            
            from app import count_down_start
            count_down_start(countdown_value)

            if countdown_value > 3:
                print("Countdown ended! Start drawing...")
                game_started = True  # Enable drawing after countdown
                drawing = True  # Start drawing
                
                trigger_start_countdown = False  # Stop countdown trigger
                last_thumbs_up_time = None
                drawing_start_time = time.time()  # Initialize the drawing timer at game start

        # Handle drawing timer and 30-second rule
        elif game_started and drawing_start_time is not None:
            
            time_elapsed = time.time() - drawing_start_time
            countdown_value = int(time_elapsed)
            
            from app import drawing_start
            drawing_start(countdown_value)
            
            if countdown_value > 20:  # Disable drawing after 10 seconds for testing
                game_started = False
                drawing = False
                
                drawing_start_time = None  # Reset drawing timer
                trigger_end_countdown = True
                last_thumbs_up_time = time.time()  # Start the countdown for game reset
                print("Drawing time ended! Waiting for 3 seconds before resetting...")

        # Reset game after 3 seconds of idle time (if drawing is done)
        elif trigger_end_countdown and last_thumbs_up_time is not None:
            # print("starting end cooldown")
            
            time_elapsed = time.time() - last_thumbs_up_time
            countdown_value = int(time_elapsed)
            
            from app import count_down_end
            count_down_end(countdown_value)
            if time_elapsed > 4:
                waiting_for_thumbs_up = True
                trigger_end_countdown = False
                last_thumbs_up_time = None
                # print("Game reset! Waiting for thumbs up...")
                
                from app import send_accuracy, set_match_accuracy
                target_image = cv2.imread(images[random_number])
                similarity_percentage = image_matching(canvas, target_image)
                set_match_accuracy(similarity_percentage)
                print(f"Accuracy: {similarity_percentage}")
                send_accuracy()

        # Blend the frame with the canvas and draw text
        canvas_resized = cv2.resize(canvas, (frame.shape[1], frame.shape[0]))
        frame_with_canvas = cv2.addWeighted(frame, 0, canvas_resized, 1, 0)

        # Encode the frame to JPEG
        _, jpeg_frame = cv2.imencode('.jpg', frame_with_canvas)

        # Yield the frame to the client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n')
