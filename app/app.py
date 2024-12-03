from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
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
from processing.combat import scroll_background
import processing.state as state
import time
from processing.image_matching import game_loop

app = Flask(__name__)
app.config['DEBUG'] = True 
app.config['SECRET_KEY'] = 'your_secret_key'

CORS(app)

# Start Gesture Detection in a Separate Thread
def start_gesture_detection():
    menu_gesture_control.detect_gestures(camera)

camera = cv2.VideoCapture(0)

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
            generate_maze_interaction_frames(camera),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
    except Exception as e:
        print(f"Error during frame generation: {e}")
        return Response(
            "An error occurred while streaming frames.",
            status=500,  # Internal Server Error
        )
        
## Combat routes ##
isReset = False
@app.route('/combat')
def combat_page():
    global isReset
    reset_game = request.args.get('reset', 'false') == 'true'
    
    if reset_game:
        isReset = True
        
    return render_template('combat.html')

@app.route('/combat_feed')
def combat_feed():
    global isReset
    # Check if the camera is opened
    if not camera.isOpened():
        camera.open(0)
        if not camera.isOpened():
            print("Error: Camera not opened or unavailable.")
            return Response(
                "Camera not available. Please ensure the camera is connected and not used by another application.",
                status=503,  # 503 Service Unavailable
            )

    # Generate and stream frames with the scrolling background
    if isReset:
        isReset = False
        try:
            return Response(
                scroll_background(camera, isReset=True),
                mimetype='multipart/x-mixed-replace; boundary=frame',
            )
        except Exception as e:
            print(f"Error during frame generation: {e}")
            return Response(
                "An error occurred while streaming frames.",
                status=500,  # Internal Server Error
            )
    else:
        try:
            return Response(
                scroll_background(camera, isReset=False),
                mimetype='multipart/x-mixed-replace; boundary=frame',
            )
        except Exception as e:
            print(f"Error during frame generation: {e}")
            return Response(
                "An error occurred while streaming frames.",
                status=500,  # Internal Server Error
            )

is_game_one_done = False
is_mini_game_one_done = False
is_game_two_done = False
is_mini_game_two_done = False
is_game_three_done = False
is_mini_game_three_done = False
is_cheat = False
is_update = False
current_health = 3
isDead = False

def player_dead():
    global isDead
    isDead = True
    print("Player is dead!")

def show_cheat():
    global is_cheat
    is_cheat = True
    print("Cheat activated!")
    
def update_health(hp):
    global current_health, is_update
    current_health = hp
    is_update = True
    
def change_game_one_state():
    global is_game_one_done
    is_game_one_done = True  
    print("Game one is done!")

def change_game_two_state():
    global is_game_two_done
    is_game_two_done = True  
    print("Game two is done!")
    
def change_game_three_state():
    global is_game_three_done
    is_game_three_done = True  
    print("Game three is done!") 
    
@app.route('/game_one', methods=['POST'])
def change_mini_game_one_state():
    data = request.get_json() 
    if data :
        
        message = data.get('message', 'No message provided')
        print(message)
        if message != "NAR25-1 Semangat Jangan Merasa Aman":
            return jsonify({"message": "Invalid message"}), 400
        global is_mini_game_one_done
        is_mini_game_one_done = True #minigamenya uda kelar
            
        print("Mini game one is done!")
        return jsonify({"message": "Mini game one is done!"}),200
    else :
        return jsonify({"message": "No message provided"}), 400
    
@app.route('/game_two', methods=['POST'])
def change_mini_game_two_state():
    data = request.get_json() 
    if data :
        
        message = data.get('message', 'No message provided')
        print(message)
        if message != "For The Glory Of Mankind":
            return jsonify({"message": "Invalid message"}), 400
        global is_mini_game_two_done
        is_mini_game_two_done = True #minigamenya uda kelar
            
        print("Mini game one is done!")
        return jsonify({"message": "Mini game two is done!"}),200
    else :
        return jsonify({"message": "No message provided"}), 400
    
def change_mini_game_three_state():
    global is_mini_game_three_done
    is_mini_game_three_done = True

# SSE route
@app.route('/sse_game_status')
def sse_game_status():
    def event_stream():
        global is_game_one_done, is_mini_game_one_done, is_game_two_done, is_mini_game_two_done, isDead
        while True:
            # Memeriksa apakah game sudah selesai
            if is_game_one_done and not is_mini_game_one_done:
                yield f"data: game_one\n\n"
                break  # Menghentikan stream jika game selesai
            elif is_game_two_done and not is_mini_game_two_done:
                yield f"data: game_two\n\n"
                break
            elif is_game_three_done and not is_mini_game_three_done:
                yield f"data: game_three\n\n"
                break
            elif isDead:
                isDead = False
                yield f"data: dead\n\n"
                break
            time.sleep(1)

    return Response(event_stream(), content_type='text/event-stream')

@app.route('/sse_mini_game_one')
def sse_mini_game_one():
    def event_stream():
        global is_game_one_done, is_mini_game_one_done
        while True:
            # Memeriksa apakah game sudah selesai
            if is_game_one_done and is_mini_game_one_done:
                yield f"data: redirect\n\n"
                break  # Menghentikan stream jika game selesai
            time.sleep(1)

    return Response(event_stream(), content_type='text/event-stream')

@app.route('/sse_mini_game_two')
def sse_mini_game_two():
    def event_stream():
        global is_game_two_done, is_mini_game_two_done
        while True:
            # Memeriksa apakah game sudah selesai
            if is_game_two_done and is_mini_game_two_done:
                yield f"data: redirect\n\n"
                break  # Menghentikan stream jika game selesai
            time.sleep(1)

    return Response(event_stream(), content_type='text/event-stream')

@app.route('/sse_mini_game_three')
def sse_mini_game_three():
    def event_stream():
        global is_cheat, current_health, is_update, is_game_three_done, is_mini_game_three_done
        while True:
            if is_game_three_done and is_mini_game_three_done:
                yield f"data: redirect\n\n"
                break

            elif is_cheat:
                is_cheat = False
                yield f"data: cheat\n\n"            
            elif is_update and current_health == 0:
                is_update = False
                yield f"data: zero\n\n"
            elif is_update and current_health == 1:
                is_update = False
                yield f"data: one\n\n"
            elif is_update and current_health == 2:
                is_update = False
                yield f"data: two\n\n"
            elif is_update and current_health == 3:
                is_update = False
                yield f"data: three\n\n"
                
            time.sleep(1)
                
    return Response(event_stream(), content_type='text/event-stream')

## Image matching routes ##
@app.route('/image_match')
def image_match_page():
    return render_template('image_matching.html')

@app.route('/match_video_feed')
def match_video_feed():

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
            game_loop(camera),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
    except Exception as e:
        print(f"Error during frame generation: {e}")
        return Response(
            "An error occurred while streaming frames.",
            status=500,  # Internal Server Error
        )

if __name__ == '__main__':
    app.run()
