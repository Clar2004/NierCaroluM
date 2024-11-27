from gevent import monkey
monkey.patch_all()

from flask import Flask, render_template, request, redirect, url_for, Response
from flask_socketio import SocketIO
import os
import threading
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'processing'))
from processing import menu_gesture_control, image_filter
from processing.image_filter import initialize_base_image, generate_frames
from flask_cors import CORS
import cv2
from processing.threshold_processing import (
    initialize_threshold_image,
    generate_threshold_frames,
)
from threading import Lock
from processing.edge_corner_processing import generate_maze_interaction_frames

lock = Lock()

app = Flask(__name__)
app.config['DEBUG'] = True 
app.config['SECRET_KEY'] = 'your_secret_key'

CORS(app)
socketio = SocketIO(app, cors_allowed_origins=["http://127.0.0.1:5000"], async_mode="gevent")

# Start Gesture Detection in a Separate Thread
def start_gesture_detection():
    menu_gesture_control.detect_gestures(socketio)

camera = cv2.VideoCapture(0)

@app.route('/release_camera', methods=['POST'])
def release_camera():
    global camera
    with lock:
        if camera.isOpened():
            camera.release()
            print("Camera released.")
    return "Camera released", 200

@app.route('/initialize_camera', methods=['POST'])
def initialize_camera():
    global camera
    with lock:
        if not camera.isOpened():
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("Failed to initialize camera.")
                return "Failed to initialize camera", 500
            print("Camera initialized.")
    return "Camera initialized", 200

@socketio.on("connect")
def handle_connect():
    print('Client connected')
    socketio.start_background_task(start_gesture_detection)

@app.route('/')
def menu():
    return render_template('menu.html')

@app.route('/play')
def play_demo():
    return "Game Demo Started!"

@app.route('/exit')
def exit_game():
    # shutdown_server()
    return "Goodbye!"
 
def shutdown_server():
    global camera
    if camera.isOpened():
        camera.release()  # Release the camera resource
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        print("Server shutdown function not available.")
        os._exit(0)  # Force exit if shutdown function is not available
    func()

## Image Filter Routes ##
initialize_base_image("static/assets/images/Image_Filter_Asset.png", blur_strength=51) 

@app.route('/image_filter')
def image_filter_page():
    return render_template('image_filter.html')

@app.route('/video_feed')
def video_feed():
    global camera

    # Check if the camera is opened
    if not camera.isOpened():
        camera.open(0)
        if not camera.isOpened():
            print("Error: Camera not opened or unavailable.")
            return Response(
                "Camera not available. Please ensure the camera is connected and not used by another application.",
                status=503,  # 503 Service Unavailable
            )

    # Generate and stream frames
    try:
        return Response(
            generate_frames(camera),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
    except Exception as e:
        print(f"Error during frame generation: {e}")
        return Response(
            "An error occurred while streaming frames.",
            status=500,  # Internal Server Error
        )


##  Threshold Processing Routes ##
image_path = "static/assets/images/Threshold.png"
initialize_threshold_image(image_path)

# Route for Threshold Puzzle
@app.route("/threshold")
def threshold_page():
    return render_template("threshold.html")


# Route for streaming thresholded video feed
@app.route("/threshold_feed")
def threshold_feed():
    global camera

    # Check if the camera is opened
    if not camera.isOpened():
        camera.open(0)
        if not camera.isOpened():
            print("Error: Camera not opened or unavailable.")
            return Response(
                "Camera not available. Please ensure the camera is connected and not used by another application.",
                status=503,  # 503 Service Unavailable
            )

    # Generate and stream frames
    try:
        return Response(
            generate_threshold_frames(camera),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as e:
        print(f"Error during frame generation: {e}")
        return Response(
            "An error occurred while streaming frames.",
            status=500,  # Internal Server Error
        )

## Edge corner processing routes ##
@app.route("/edge_corner")
def edge_corner_page():
    return render_template("edge_corner.html")

@app.route('/maze_game_feed')
def edge_corner_feed():
    # global camera

    # # Check if the camera is opened
    # if not camera.isOpened():
    #     camera.open(0)
    #     if not camera.isOpened():
    #         print("Error: Camera not opened or unavailable.")
    #         return Response(
    #             "Camera not available. Please ensure the camera is connected and not used by another application.",
    #             status=503,  # 503 Service Unavailable
    #         )

    # Generate and stream frames
    try:
        return Response(
            generate_maze_interaction_frames(socketio),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
    except Exception as e:
        print(f"Error during frame generation: {e}")
        return Response(
            "An error occurred while streaming frames.",
            status=500,  # Internal Server Error
        )

if __name__ == '__main__':
    try:
        socketio.run(app, debug=True, host='127.0.0.1', port=5000)
    finally:
        if camera.isOpened():
            camera.release()
