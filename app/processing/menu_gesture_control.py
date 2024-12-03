import cv2
import mediapipe as mp
from threading import Lock
import pygame
from flask_socketio import SocketIO

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Gesture Detection Functions
def is_pinch(landmarks):
    """Check if thumb and index finger are close (pinch gesture)."""
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].x
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
    return abs(thumb_tip - index_tip) < 0.01  # Adjust based on testing

def is_one_finger(landmarks):
    """Check if only index finger is up (one-finger gesture)."""
    index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP].y
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
    return index_tip < index_mcp and thumb_tip > index_tip and middle_tip > index_tip

# Gesture Detection Function
def detect_gestures(cap):
    print("Gesture detection thread started")
    if not cap.isOpened():
        print("Error: Could not open video capture")
        # socketio.emit('error', {'message': 'Camera initialization failed'})
        return

    # Get screen resolution dynamically
    screen_width, screen_height = 2560, 1440
    last_emitted = {'x': None, 'y': None, 'gesture': None}
    socketio_lock = Lock()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    cursor_x = int(index_finger.x * screen_width)
                    cursor_y = int(index_finger.y * screen_height)

                    # Pinch gesture detection
                    if is_pinch(hand_landmarks.landmark):
                        if (last_emitted['x'], last_emitted['y'], last_emitted['gesture']) != (cursor_x, cursor_y, 'pinch'):
                            with socketio_lock:
                                # socketio.emit('move_cursor', {'x': cursor_x, 'y': cursor_y})
                                last_emitted.update({'x': cursor_x, 'y': cursor_y, 'gesture': 'pinch'})
                                # socketio.sleep(0.01)

                    # One finger gesture detection
                    elif is_one_finger(hand_landmarks.landmark):
                        if (last_emitted['x'], last_emitted['y'], last_emitted['gesture']) != (cursor_x, cursor_y, 'click'):
                            with socketio_lock:
                                # socketio.emit('click', {'x': cursor_x, 'y': cursor_y})
                                last_emitted.update({'x': cursor_x, 'y': cursor_y, 'gesture': 'click'})
                                # socketio.sleep(0.01)

            # Quit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # cap.release()
        cv2.destroyAllWindows()
        print("Camera and resources released.")
