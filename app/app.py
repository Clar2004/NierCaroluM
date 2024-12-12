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
from processing.edge_corner_processing import generate_maze_interaction_frames
from processing.combat import scroll_background
import time
from processing.image_matching import game_loop
from processing.menu_gesture_control import detect_gestures_and_stream
from processing.state import *

app = Flask(__name__)
app.config['DEBUG'] = True 
app.config['SECRET_KEY'] = 'your_secret_key'

CORS(app)

camera = cv2.VideoCapture(1)

@app.route('/')
def menu():
    return render_template('menu.html')

@app.route('/menu_video_feed')
def menu_feed():
    
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
            detect_gestures_and_stream(camera),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
    except Exception as e:
        print(f"Error during frame generation: {e}")
        return Response(
            "An error occurred while streaming frames.",
            status=500,  # Internal Server Error
        )

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
@app.route('/combat')
def combat_page():
    
    from processing.state import game_state
    
    reset_game = request.args.get('reset', 'false') == 'true'
    
    if reset_game:
        game_state.isReset = True
        game_state.is_game_one_done = False
        game_state.is_mini_game_one_done = False
        game_state.is_game_two_done = False
        game_state.is_mini_game_two_done = False
        game_state.is_game_three_done = False
        game_state.is_mini_game_three_done = False
        game_state.is_game_four_done = False
        game_state.is_mini_game_four_done = False
        
    return render_template('combat.html')

@app.route('/combat_feed')
def combat_feed():
    from processing.state import game_state
    
    print("Current reset state:", game_state.isReset)
    
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
    if game_state.isReset:
        game_state.isReset = False
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
            
# isReset = False [v]
# is_game_one_done = False [v]
# is_mini_game_one_done = False
# is_game_two_done = False
# is_mini_game_two_done = False
# is_game_three_done = False
# is_mini_game_three_done = False
# is_game_four_done = False
# is_mini_game_four_done = False
# is_cheat = False
# is_update = False
# current_health = 3
# isDead = False

def player_dead():
    from processing.state import game_state
    game_state.is_dead = True
    game_state.isPlayDemo = False
    print("Player is dead!")

def show_cheat():
    from processing.state import game_state
    game_state.is_cheat = True
    print("Cheat activated!")
    
def update_health(hp):
    from processing.state import game_state
    game_state.current_health = hp
    game_state.is_update = True
    
def change_game_one_state():
    from processing.state import game_state
    game_state.is_game_one_done = True  
    print("Game one is done!")

def change_game_two_state():
    from processing.state import game_state
    game_state.is_game_two_done = True  
    print("Game two is done!")
    
def change_game_three_state():
    from processing.state import game_state
    game_state.is_game_three_done = True  
    print("Game three is done!") 

def change_game_four_state():
    from processing.state import game_state
    game_state.is_game_four_done = True  
    print("Game four is done!")
    
@app.route('/game_one', methods=['POST'])
def change_mini_game_one_state():
    data = request.get_json() 
    if data :
        
        message = data.get('message', 'No message provided')
        print(message)
        if message != "NAR25-1 Semangat Jangan Merasa Aman":
            return jsonify({"message": "Invalid message"}), 400
        from processing.state import game_state
        game_state.is_mini_game_one_done = True #minigamenya uda kelar
            
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
        from processing.state import game_state
        game_state.is_mini_game_two_done = True #minigamenya uda kelar
            
        print("Mini game one is done!")
        return jsonify({"message": "Mini game two is done!"}),200
    else :
        return jsonify({"message": "No message provided"}), 400
    
@app.route('/game_four', methods=['POST'])
def change_mini_game_four_state():
    from processing.state import game_state
    data = request.get_json() 
    if data :
        
        message = data.get('message', 'No message provided')
        print(message)
        if message != "Success":
            return jsonify({"message": "Invalid message"}), 400
        game_state.is_mini_game_four_done = True
        game_state.is_game_four_done = True
            
        print("Mini game four is done!")
        return jsonify({"message": "Mini game two is done!"}),200
    else :
        return jsonify({"message": "No message provided"}), 400
    
def change_mini_game_three_state():
    from processing.state import game_state
    game_state.is_mini_game_three_done = True

# SSE route
@app.route('/sse_game_status')
def sse_game_status():
    def event_stream():
        from processing.state import game_state
        while True:
            if game_state.is_game_one_done and not game_state.is_mini_game_one_done:
                yield f"data: game_one\n\n"
                break 
            elif game_state.is_game_two_done and not game_state.is_mini_game_two_done:
                yield f"data: game_two\n\n"
                break
            elif game_state.is_game_three_done and not game_state.is_mini_game_three_done:
                yield f"data: game_three\n\n"
                break
            elif game_state.is_game_four_done and not game_state.is_mini_game_four_done:
                yield f"data: game_four\n\n"
                break
            elif game_state.is_dead:
                game_state.is_dead = False
                yield f"data: dead\n\n"
                break
            
            time.sleep(1)

    return Response(event_stream(), content_type='text/event-stream')

@app.route('/sse_mini_game_one')
def sse_mini_game_one():
    def event_stream():
        from processing.state import game_state
        while True:
            # Memeriksa apakah game sudah selesai
            if game_state.is_game_one_done and game_state.is_mini_game_one_done:
                yield f"data: redirect\n\n"
                break  # Menghentikan stream jika game selesai
            time.sleep(1)

    return Response(event_stream(), content_type='text/event-stream')

@app.route('/sse_mini_game_two')
def sse_mini_game_two():
    def event_stream():
        from processing.state import game_state
        while True:
            # Memeriksa apakah game sudah selesai
            if game_state.is_game_two_done and game_state.is_mini_game_two_done:
                yield f"data: redirect\n\n"
                break  # Menghentikan stream jika game selesai
            time.sleep(1)

    return Response(event_stream(), content_type='text/event-stream')

@app.route('/sse_mini_game_three')
def sse_mini_game_three():
    def event_stream():
        from processing.state import game_state
        while True:
            if game_state.is_game_three_done and game_state.is_mini_game_three_done:
                yield f"data: redirect\n\n"
                break

            elif game_state.is_cheat:
                game_state.is_cheat = False
                yield f"data: cheat\n\n"            
            elif game_state.is_update and game_state.current_health == 0:
                game_state.is_update = False
                yield f"data: zero\n\n"
            elif game_state.is_update and game_state.current_health == 1:
                game_state.is_update = False
                yield f"data: one\n\n"
            elif game_state.is_update and game_state.current_health == 2:
                game_state.is_update = False
                yield f"data: two\n\n"
            elif game_state.is_update and game_state.current_health == 3:
                game_state.is_update = False
                yield f"data: three\n\n"
                
            time.sleep(1)
                
    return Response(event_stream(), content_type='text/event-stream')

def play_demo():
    from processing.state import game_state
    game_state.isPlayDemo = True
    print(f"Current game state: {game_state.isPlayDemo}")
    
@app.route('/sse_menu')    
def sse_menu():
    def event_stream():        
        from processing.state import game_state
        while game_state.isPlayDemo == False:
            yield f"data: dasdas\n\n"
            time.sleep(1)
        game_state.isPlayDemo == False
        yield f"data: redirect\n\n"
        sys.stdout.flush()
                
    return Response(event_stream(), content_type='text/event-stream')


## Image matching routes ##
# isGameStart = False [v]
# isCountDownStart = False [v]
# isDrawingStart = False [v]
# isCountDownEnd = False [v]
# targetImageIndex = None [v]
# isSendAccuracy = False [v]
# match_accuracy = None [v]
# current_seconds = 0 [v]

def match_start(num):
    from processing.state import game_state
    game_state.targetImageIndex = num
    game_state.isGameStart = True

def count_down_start(seconds):
    from processing.state import game_state
    game_state.isCountDownStart = True
    game_state.current_seconds = seconds

def drawing_start(seconds):
    from processing.state import game_state
    game_state.isDrawingStart = True
    game_state.current_seconds = seconds

def count_down_end(seconds):
    from processing.state import game_state
    game_state.isCountDownEnd = True
    game_state.current_seconds = seconds
    
def send_accuracy():
    from processing.state import game_state
    game_state.isSendAccuracy = True
    
def set_match_accuracy(accuracy):
    from processing.state import game_state
    game_state.match_accuracy = accuracy

@app.route('/image_match')
def image_match_page():
    return render_template('image_matching.html')

@app.route('/match_video_feed')
def matches_video_feed():

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
        
@app.route('/sse_mini_game_four')
def sse_mini_game_four():
    def event_stream():
        from processing.state import game_state
        
        while True:
            if game_state.is_mini_game_four_done:
                print("is mini game four true", game_state.is_mini_game_four_done)
            
            # Memeriksa apakah game sudah selesai
            if game_state.is_game_four_done and game_state.is_mini_game_four_done:
                yield f"data: redirect\n\n"
                break
            
            yield "data: heartbeat\n\n" 
            time.sleep(1)
            
    return Response(event_stream(), content_type='text/event-stream')

@app.route('/sse_mini_game_four_accuracy')
def sse_mini_game_four_accuracy():
    def event_stream():
        from processing.state import game_state
                
        while True:
            if game_state.isSendAccuracy:
                game_state.isSendAccuracy = False
                print("Sending accuracy:", game_state.match_accuracy)
                yield f"data: {{\"event\": \"accuracy\", \"accuracy\": {game_state.match_accuracy}}}\n\n"
            
            elif game_state.isGameStart:
                game_state.isGameStart = False
                print("Sending image index:", game_state.targetImageIndex)
                yield f"data:{{\"event\": \"game_start\", \"image_index\": {game_state.targetImageIndex}}}\n\n"
            
            elif game_state.isCountDownStart:
                print("Countdown start time", game_state.current_seconds)
                if game_state.current_seconds <= 3 :
                    yield f"data:{{\"event\": \"countdown_start\", \"time\": {3 - game_state.current_seconds}}}\n\n"
                elif game_state.current_seconds > 3:
                    game_state.isCountDownStart = False
                    game_state.current_seconds = 0
            
            elif game_state.isDrawingStart:
                print("Drawing time", game_state.current_seconds)
                if game_state.current_seconds <= 20:
                    yield f"data:{{\"event\": \"drawing_start\", \"time\": {20 - game_state.current_seconds}}}\n\n"
                else:
                    game_state.isDrawingStart = False
                    game_state.current_seconds = 0
            elif game_state.isCountDownEnd:
                print("Countdown end", game_state.current_seconds)
                if game_state.current_seconds <= 3:
                    yield f"data:{{\"event\": \"countdown_end\"}}\n\n"
                elif game_state.current_seconds > 3:
                    game_state.isCountDownEnd = False
                    game_state.current_seconds = 0
                    
            time.sleep(0.01)
            
    return Response(event_stream(), content_type='text/event-stream')

def run_flask():
    serve(app, host="127.0.0.1", port=5000)
        
import webview
from waitress import serve

if __name__ == "__main__":
    from threading import Thread
    
    def run_flask():
        app.run(debug=False, port=5000, use_reloader=False)

    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    window = webview.create_window('BPCV', 'http://127.0.0.1:5000', fullscreen=True)

    webview.start()

    flask_thread.join()
