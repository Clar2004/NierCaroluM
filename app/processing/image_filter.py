import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import wiener
import pygame

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Global variables
filter_type = "gaussian"  # Default filter type
filter_radius = 50  # Default filter radius
base_image = None
blurred_image = None
circle_center = (320, 240)

def initialize_base_image(image_path, blur_strength=51):
    """
    Load and preprocess the base image.
    """
    global base_image, blurred_image
    base_image = cv2.imread(image_path)
    if base_image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    base_image = cv2.resize(base_image, (640, 480))
    blurred_image = create_blurred_image(base_image, blur_strength)
    # print(f"Base image and blurred image initialized with blur strength {blur_strength}")

def create_blurred_image(image, blur_strength=51):
    """
    Create a heavily blurred version of the image.
    """
    if blur_strength % 2 == 0:
        blur_strength += 1
    return cv2.GaussianBlur(image, (blur_strength, blur_strength), 0)

def apply_circle_filter(image, original_image, center, radius, kernel_size):
    """
    Apply a Gaussian filter to a circular region of the image permanently, and add a visual-only white border.

    Args:
        image (np.ndarray): The current working image (updated after applying the filter).
        original_image (np.ndarray): The original (unblurred) image.
        center (tuple): The center of the circle (x, y).
        radius (int): The radius of the circle.
        kernel_size (int): The kernel size for Gaussian blur.

    Returns:
        np.ndarray: The updated image with the filter applied permanently and a white border for visualization.
    """
    # print(f"Applying Gaussian filter with kernel size {kernel_size} at center {center}, radius {radius}")

    # Ensure kernel size is odd and greater than 0
    if kernel_size > 0 and kernel_size % 2 == 0:
        kernel_size += 1

    # Ensure the center and radius are within bounds
    height, width = image.shape[:2]
    x_center = max(radius, min(center[0], width - radius))
    y_center = max(radius, min(center[1], height - radius))
    center = (x_center, y_center)

    # Create a mask for the circular region
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)  # Draw a white-filled circle on the mask

    if kernel_size > 0:
        # Apply Gaussian blur to the original image
        filtered_image = cv2.GaussianBlur(original_image, (kernel_size, kernel_size), 0)
    else:
        # Restore the original region (no blur)
        filtered_image = original_image

    # Combine the filtered region with the current image using the mask
    filtered_region = cv2.bitwise_and(filtered_image, filtered_image, mask=mask)
    unfiltered_region = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    updated_image = cv2.add(filtered_region, unfiltered_region)

    # Update the image permanently
    image[:, :] = updated_image

    # Add a white border to the circle for visualization
    visualization_frame = image.copy()
    cv2.circle(visualization_frame, center, radius, (255, 255, 255), 2)  # White border with thickness of 2 pixels

    return visualization_frame

def process_frame(camera):
    """
    Process the camera frame, detect gestures, and apply filters dynamically.
    """
    global filter_radius, circle_center, blurred_image, base_image
    kernel_size = 0  # Default kernel size (no blur)

    success, frame = camera.read()
    if not success:
        print("Failed to capture frame from camera.")
        return None

    frame = cv2.flip(frame, 1)  # Flip for a mirrored view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand gestures with MediaPipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_handedness:
        # Store handedness and landmarks as a dictionary for easy access
        hands_data = []
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_handedness.classification[0].label  # "Left" or "Right"
            hands_data.append({
                "label": hand_label,
                "landmarks": hand_landmarks.landmark
            })

        for hand in hands_data:
            landmarks = hand["landmarks"]
            hand_label = hand["label"]
            # print(f"Detected {hand_label} hand.")

            if hand_label == "Left":
                # Detect raised fingers
                fingers_up = 0
                if landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                    fingers_up += 1
                if landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
                    fingers_up += 1

                # Map the number of raised fingers to kernel size
                kernel_size = fingers_up * 10  # 0 for no blur, 10 for light blur, 20 for stronger blur

                # Ensure kernel size is valid
                if kernel_size > 0 and kernel_size % 2 == 0:
                    kernel_size += 1

                # print(f"Left hand detected: {fingers_up} fingers up. Kernel size: {kernel_size}")

            elif hand_label == "Right":
                # Control pinch radius and center with right hand
                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Calculate pinch distance
                pinch_distance = np.linalg.norm(
                    [thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]
                )
                filter_radius = max(10, min(int(pinch_distance * 200), min(base_image.shape[1], base_image.shape[0]) // 2))
                # print(f"Pinch radius set to {filter_radius}.")

                # Calculate dynamic circle center
                x_center = int((thumb_tip.x + index_tip.x) / 2 * base_image.shape[1])
                y_center = int((thumb_tip.y + index_tip.y) / 2 * base_image.shape[0])
                circle_center = (x_center, y_center)

    # else:
        # Retain last known circle position if no hand is detected
        # print("No hand detected. Retaining last circle position.")

    # Apply the circular filter to the blurred image (permanent changes)
    overlay_frame = apply_circle_filter(blurred_image, base_image, circle_center, filter_radius, kernel_size)
    return overlay_frame

def generate_frames(camera):
    """
    Generate video frames for the Flask route.
    """
    
    # Initialize the mixer for pygame
    pygame.mixer.init()

    # Load the music file
    pygame.mixer.music.load("static/assets/sound/boss_bg_8bit.mp3")
    pygame.mixer.music.play(loops=-1, start=0.0)
    
    while True:
        processed_frame = process_frame(camera)
        if processed_frame is None:
            print("No processed frame available.")
            break

        # Encode the frame as JPEG
        success, buffer = cv2.imencode(".jpg", processed_frame)
        if not success:
            print("Failed to encode frame.")
            continue

        frame = buffer.tobytes()
        # print("Generated a new frame.")
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
