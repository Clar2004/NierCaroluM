import cv2
import mediapipe as mp
import numpy as np
import random
import time
from threading import Lock

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load images for the user to draw
images = [
    "assets/.jpg", 
    "image2.jpg", 
    "image3.jpg"
    ]  # Use your image file paths here

# Initialize drawing state
drawing_started = False
drawn_image = np.zeros((500, 500, 3), dtype=np.uint8)  # Canvas for drawing

# Gesture state tracking
last_position = None
is_drawing = False

# Initialize the video capture
screen_width, screen_height = 640, 480  # Default resolution for webcam
lock = Lock()

# Function to check for fist gesture
def is_fist(hand_landmarks):
    """Detect if the hand is a fist based on finger positions"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    return index_tip > thumb_tip  # Fist check, adjust as per needs

# Function to check if the index finger is being used for drawing
def is_index_finger_up(hand_landmarks):
    """Detect if the index finger is up based on its position"""
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    return index_tip.y < index_mcp.y  # Index finger up check

# Drawing function
def draw_on_board(frame, cursor_x, cursor_y):
    """Draw on the canvas (drawing_board)"""
    if drawing_started:
        cv2.circle(drawn_image, (cursor_x, cursor_y), 5, (255, 0, 0), -1)

# Image Matching using ORB (Oriented FAST and Rotated BRIEF)
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

# Countdown timer
def countdown_timer():
    """Start the countdown for 30 seconds"""
    global drawing_started
    for i in range(3, 0, -1):
        print(f"Game starting in {i}...")
        time.sleep(1)
    drawing_started = True
    print("Start drawing!")

# Main Game Loop
def game_loop(camera):
    with lock:
        global drawing_started, drawn_image, last_position, is_drawing

        # Select a random image from the list
        target_image = cv2.imread(random.choice(images), cv2.IMREAD_GRAYSCALE)
        target_image = cv2.resize(target_image, (500, 500))  # Resize for consistency

        # Initialize timer and drawing state
        start_time = time.time()
        drawing_started = False

        # Start countdown
        countdown_timer()

        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture image")
                break

            # Flip and convert to RGB for MediaPipe
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Display countdown and drawing instructions
            elapsed_time = time.time() - start_time
            time_remaining = max(0, 30 - int(elapsed_time))
            
            if drawing_started:
                cv2.putText(frame, "Start drawing!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif time_remaining > 0:
                cv2.putText(frame, f"Game starting in {time_remaining}...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Time's up! Press 'q' to exit.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    cursor_x = int(index_finger.x * screen_width)
                    cursor_y = int(index_finger.y * screen_height)

                    # If fist is detected, stop drawing
                    if is_fist(hand_landmarks):
                        drawing_started = False

                    # If index finger is up and drawing is enabled, start drawing
                    elif is_index_finger_up(hand_landmarks) and drawing_started:
                        draw_on_board(frame, cursor_x, cursor_y)

            # Display the canvas and webcam feed together
            combined_frame = np.hstack((frame, cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB)))
            cv2.imshow("Drawing Board - Press 'q' to quit", combined_frame)

            # Check if 30 seconds have passed
            if elapsed_time >= 30:
                drawing_started = False
                print("Time's up!")
                break

        # After 30 seconds, perform image matching
        similarity = image_matching(drawn_image, target_image)
        print(f"Image similarity: {similarity:.2f}%")

        if similarity > 70:
            print("You did a great job!")
        else:
            print("Try again! The game will restart with a new image.")
            game_loop(camera)  # Restart the game

        # Preparing frame for streaming (if necessary)
        ret, buffer = cv2.imencode('.jpg', drawn_image)  # Use drawn_image for final output
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')