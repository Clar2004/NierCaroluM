import cv2
import numpy as np
import mediapipe as mp
from threading import Lock
from collections import deque

# MediaPipe initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Globals
threshold_level = 128  # Default threshold level
lock = Lock()
input_image = None
previous_x = None  # Track previous x position of the index finger

# Store recent thumb positions for the line effect
thumb_path = deque(maxlen=15)  # Keeps up to 15 recent positions (shorter line)
line_alpha = 1.0  # Alpha value for fade-out animation

def initialize_threshold_image(image_path, resize_width=800, resize_height=600):
    """
    Load and resize the image for thresholding.
    """
    global input_image
    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    input_image = cv2.resize(input_image, (resize_width, resize_height))


def adjust_threshold(image, level):
    """
    Apply a threshold to the image.
    """
    _, thresholded = cv2.threshold(image, level, 255, cv2.THRESH_BINARY)
    return thresholded


def generate_threshold_frames(camera, frame_width=800, frame_height=600):
    """
    Generator for streaming thresholded frames to the frontend.
    """
    global threshold_level, input_image, previous_x, thumb_path, line_alpha

    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Could not read frame")
            break

        # Resize the frame
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Flip and convert frame for MediaPipe
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        # If hand is detected, update the thumb path
        if results.multi_hand_landmarks:
            line_alpha = 1.0  # Reset alpha when hand is detected
            for hand_landmarks in results.multi_hand_landmarks:
                # Get thumb tip position
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Convert to screen coordinates
                thumb_x = int(thumb_tip.x * frame_width)
                thumb_y = int(thumb_tip.y * frame_height)

                # Add the current thumb position to the path, but reduce sensitivity
                if len(thumb_path) == 0 or np.linalg.norm(
                    np.array(thumb_path[-1]) - np.array((thumb_x, thumb_y))
                ) > 10:  # Only add if the distance is >10 pixels
                    thumb_path.append((thumb_x, thumb_y))

                # Get index finger tip position for swipe gesture
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x = int(index_tip.x * frame_width)

                # Detect horizontal swipe
                if previous_x is not None:
                    delta_x = index_x - previous_x
                    if delta_x > 15:  # Swipe right
                        with lock:
                            threshold_level = min(255, threshold_level + 5)
                            print(f"Threshold level increased to {threshold_level}")
                    elif delta_x < -15:  # Swipe left
                        with lock:
                            threshold_level = max(0, threshold_level - 5)
                            print(f"Threshold level decreased to {threshold_level}")

                # Update previous x position
                previous_x = index_x
        else:
            # If no hands are detected, gradually fade out the line
            if len(thumb_path) > 0:
                line_alpha -= 0.1  # Gradually decrease alpha
                if line_alpha <= 0:
                    thumb_path.clear()  # Clear the path once fully faded
                    line_alpha = 0  # Ensure alpha doesn't go negative

        # Apply the threshold
        with lock:
            thresholded_image = adjust_threshold(input_image, threshold_level)

        # Convert thresholded image to BGR for display
        threshold_display = cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR)

        # Create a blank canvas for the line effect
        line_overlay = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        # Draw the line effect with consistent width
        for i in range(1, len(thumb_path)):
            cv2.line(
                line_overlay,
                thumb_path[i - 1],
                thumb_path[i],
                (255, 255, 255),  # White color
                thickness=3,  # Consistent width
                lineType=cv2.LINE_AA,
            )

        # Blend the line effect onto the threshold display with alpha
        combined_frame = cv2.addWeighted(threshold_display, 1.0, line_overlay, line_alpha, 0)

        # Add the threshold level text
        cv2.putText(
            combined_frame,
            f"Threshold Level: {threshold_level}",
            (10, 25),  # Position
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,  # Font size (smaller)
            (0, 255, 0),  # Color (Green)
            1,  # Thickness
        )

        # Encode and yield the frame
        _, buffer = cv2.imencode(".jpg", combined_frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
