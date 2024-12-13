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
import numpy as np

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        return os.path.join(os.getcwd(), relative_path)

app = Flask(__name__)
app.config['DEBUG'] = True 
app.config['SECRET_KEY'] = 'your_secret_key'

CORS(app)

camera = cv2.VideoCapture(1)

# def generate_black_frame_with_text():
#     # Create a black frame
#     frame_size = (480, 640, 3)
#     black_frame = np.zeros(frame_size, dtype=np.uint8)

#     # Add text message using OpenCV
#     text = "Camera not detected. Please connect a camera."
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1
#     font_thickness = 2
#     text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
#     text_x = (frame_size[1] - text_size[0]) // 2
#     text_y = (frame_size[0] + text_size[1]) // 2
#     cv2.putText(black_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

#     # Encode the black frame to JPEG
#     _, buffer = cv2.imencode('.jpg', black_frame)
#     return buffer.tobytes()

@app.route('/')
def menu():
    return render_template('menu.html')

@app.route('/menu_video_feed')
def menu_feed():
    # def generate_black_frame_with_text():
    #     # Create a black frame
    #     frame_size = (480, 640, 3)
    #     black_frame = np.zeros(frame_size, dtype=np.uint8)

    #     # Add text message using OpenCV
    #     text = "Camera not detected. Please connect a camera."
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 1
    #     font_thickness = 2
    #     text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    #     text_x = (frame_size[1] - text_size[0]) // 2
    #     text_y = (frame_size[0] + text_size[1]) // 2
    #     cv2.putText(black_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    #     # Encode the black frame to JPEG
    #     _, buffer = cv2.imencode('.jpg', black_frame)
    #     return buffer.tobytes()
    
    if not camera.isOpened():
        camera.open(0)
        if not camera.isOpened():
            print("Error: Camera not opened or unavailable.")
            return Response(
                "Camera not available. Please ensure the camera is connected and not used by another application.",
                status=503,  
            )
        # while True:
        #     frame = generate_black_frame_with_text()
        #     yield (b'--frame\r\n'
        #             b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    # else:
    #     # Camera is available; stream real footage
    #     while True:
    #         ret, frame = camera.read()
    #         if not ret:
    #             break

    #         # Encode the frame to JPEG
    #         _, buffer = cv2.imencode('.jpg', frame)
    #         frame = buffer.tobytes()
    #         yield (b'--frame\r\n'
    #                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    try:
        return Response(
            detect_gestures_and_stream(camera),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
    except Exception as e:
        print(f"Error during frame generation: {e}")
        return Response(
            "An error occurred while streaming frames.",
            status=500, 
        )

initialize_base_image(get_resource_path("static/assets/images/Image_Filter_Asset.png"), blur_strength=51)

@app.route('/image_filter')
def image_filter_page():
    return render_template('image_filter.html')

@app.route('/video_feed')
def video_feed():
    
    # def generate_black_frame_with_text():
    #     frame_size = (480, 640, 3)
    #     black_frame = np.zeros(frame_size, dtype=np.uint8)

    #     text = "Camera not detected. Please connect a camera."
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 1
    #     font_thickness = 2
    #     text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    #     text_x = (frame_size[1] - text_size[0]) // 2
    #     text_y = (frame_size[0] + text_size[1]) // 2
    #     cv2.putText(black_frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    #     _, buffer = cv2.imencode('.jpg', black_frame)
    #     return buffer.tobytes()
    
    if not camera.isOpened():
        camera.open(0)
        if not camera.isOpened():
            print("Error: Camera not opened or unavailable.")
            return Response(
                "Camera not available. Please ensure the camera is connected and not used by another application.",
                status=503, 
            )
    #     while True:
    #         frame = generate_black_frame_with_text()
    #         yield (b'--frame\r\n'
    #                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    # else:
    #     while True:
    #         ret, frame = camera.read()
    #         if not ret:
    #             break

    #         _, buffer = cv2.imencode('.jpg', frame)
    #         frame = buffer.tobytes()
    #         yield (b'--frame\r\n'
    #                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    try:
        return Response(
            generate_frames(camera),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
    except Exception as e:
        print(f"Error during frame generation: {e}")
        return Response(
            "An error occurred while streaming frames.",
            status=500,  
        )

##  Threshold Processing Routes ##
image_path = get_resource_path("static/assets/images/Threshold.png")
initialize_threshold_image(image_path)

@app.route("/threshold")
def threshold_page():
    return render_template("threshold.html")


@app.route("/threshold_feed")
def threshold_feed():

    if not camera.isOpened():
        camera.open(0)
        if not camera.isOpened():
            print("Error: Camera not opened or unavailable.")
            return Response(
                "Camera not available. Please ensure the camera is connected and not used by another application.",
                status=503, 
            )
        # while True:
        #     frame = generate_black_frame_with_text()
        #     yield (b'--frame\r\n'
        #             b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    try:
        return Response(
            generate_threshold_frames(camera),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    except Exception as e:
        print(f"Error during frame generation: {e}")
        return Response(
            "An error occurred while streaming frames.",
            status=500, 
        )

## Edge corner processing routes ##
@app.route("/edge_corner")
def edge_corner_page():
    return render_template("edge_corner.html")

@app.route('/maze_game_feed')
def edge_corner_feed():
    if not camera.isOpened():
        camera.open(0)
        if not camera.isOpened():
            print("Error: Camera not opened or unavailable.")
            return Response(
                "Camera not available. Please ensure the camera is connected and not used by another application.",
                status=503,
            )
        # while True:
        #     frame = generate_black_frame_with_text()
        #     yield (b'--frame\r\n'
        #             b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    try:
        return Response(
            generate_maze_interaction_frames(camera),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
    except Exception as e:
        print(f"Error during frame generation: {e}")
        return Response(
            "An error occurred while streaming frames.",
            status=500,  
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

    if not camera.isOpened():
        camera.open(0)
        if not camera.isOpened():
            print("Error: Camera not opened or unavailable.")
            return Response(
                "Camera not available. Please ensure the camera is connected and not used by another application.",
                status=503,  
            )
        # while True:
        #     frame = generate_black_frame_with_text()
        #     yield (b'--frame\r\n'
        #             b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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
                status=500, 
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
                status=500, 
            )

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
        game_state.is_mini_game_one_done = True 
            
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
        game_state.is_mini_game_two_done = True 
            
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
            if game_state.is_game_one_done and game_state.is_mini_game_one_done:
                yield f"data: redirect\n\n"
                break  
            time.sleep(1)

    return Response(event_stream(), content_type='text/event-stream')

@app.route('/sse_mini_game_two')
def sse_mini_game_two():
    def event_stream():
        from processing.state import game_state
        while True:
            if game_state.is_game_two_done and game_state.is_mini_game_two_done:
                yield f"data: redirect\n\n"
                break  
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
def reset_game():
    from processing.state import game_state
    game_state.isGameStart = False
    game_state.isStartTriggered = False

def match_start(num):
    from processing.state import game_state
    game_state.targetImageIndex = num
    game_state.isGameStart = True
    game_state.isStartTriggered = True
    print("Game start with image index:", num)

def count_down_start(seconds):
    from processing.state import game_state
    game_state.isCountDownStart = True
    game_state.current_seconds_2 = seconds

def drawing_start(seconds):
    from processing.state import game_state
    game_state.isDrawingStart = True
    game_state.current_seconds = seconds

def count_down_end(seconds):
    from processing.state import game_state
    game_state.isCountDownEnd = True
    game_state.current_seconds_3 = seconds
    
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

    if not camera.isOpened():
        camera.open(0)
        if not camera.isOpened():
            print("Error: Camera not opened or unavailable.")
            return Response(
                "Camera not available. Please ensure the camera is connected and not used by another application.",
                status=503, 
            )
        # while True:
        #     frame = generate_black_frame_with_text()
        #     yield (b'--frame\r\n'
        #             b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    try:
        return Response(
            game_loop(camera),
            mimetype='multipart/x-mixed-replace; boundary=frame',
        )
    except Exception as e:
        print(f"Error during frame generation: {e}")
        return Response(
            "An error occurred while streaming frames.",
            status=500, 
        )
        
@app.route('/sse_mini_game_four')
def sse_mini_game_four():
    def event_stream():
        from processing.state import game_state
        
        while True: 
            if game_state.is_game_four_done and game_state.is_mini_game_four_done:
                yield f"data: redirect\n\n"
                break
            
            time.sleep(1)
            
    return Response(event_stream(), content_type='text/event-stream')

def reset_game_four():
    from processing.state import game_state
    game_state.isCountDownStart = False
    game_state.isDrawingStart = False
    game_state.isCountDownEnd = False
    game_state.isSendAccuracy = False
    game_state.match_accuracy = None
    game_state.current_seconds = 0
    game_state.current_seconds_2 = 0
    game_state.current_seconds_3 = 0
    game_state.isStartTriggered = False

@app.route('/sse_mini_game_four_accuracy')
def sse_mini_game_four_accuracy():
    def event_stream():
        # from processing.state import game_state
        last_update_time = time.time()
        last_update_time2 = time.time()
                
        while True:
            from processing.state import game_state
            current_time = time.time()
            current_time2 = time.time()
            
            # print(game_state.isCountDownEnd)
            
            if current_time2 - last_update_time2 > 1:  # Send every 15 seconds
                yield "data: {}\n\n"
                last_update_time2 = current_time2
                
            if game_state.isGameStart == False:
                yield f"data: {{\"event\": \"wait\"}}\n\n"
            
            if game_state.isSendAccuracy:
                game_state.isCountDownEnd = False
                game_state.isSendAccuracy = False
                game_state.isGameStart = False
                print("Sending accuracy:", game_state.match_accuracy)
                yield f"data: {{\"event\": \"accuracy\", \"accuracy\": {game_state.match_accuracy}}}\n\n"
                
            if game_state.isGameStart and game_state.isStartTriggered and (current_time - last_update_time > 0.05):
                game_state.isStartTriggered = False
                last_update_time = current_time
                print("Sending image index:", game_state.targetImageIndex)
                yield f"data:{{\"event\": \"game_start\", \"image_index\": {game_state.targetImageIndex}}}\n\n"
            
            if game_state.isCountDownStart:
                print("Countdown start time", game_state.current_seconds)
                if game_state.current_seconds_2 > 3:
                    game_state.isCountDownStart = False
                    game_state.current_seconds_2 = 0
                else:
                    yield f"data:{{\"event\": \"countdown_start\", \"time\": {3 - game_state.current_seconds_2}}}\n\n"
            
            if game_state.isDrawingStart and game_state.isGameStart:
                print("Drawing time", game_state.current_seconds)
                if game_state.current_seconds <= 20:
                    yield f"data:{{\"event\": \"drawing_start\", \"time\": {20 - game_state.current_seconds}}}\n\n"
                else:
                    game_state.isDrawingStart = False
                    game_state.current_seconds = 0
            if game_state.isCountDownEnd:
                print("Countdown end", game_state.current_seconds_3)
                if game_state.current_seconds_3 <= 3:
                    yield f"data:{{\"event\": \"countdown_end\"}}\n\n"
                elif game_state.current_seconds_3 > 3:
                    game_state.isCountDownEnd = False
                    game_state.current_seconds_3 = 0
            
            time.sleep(0.05) 
            
    return Response(event_stream(), content_type='text/event-stream')

shutdown_event = threading.Event()
import signal
import os

def stop_flask():
    """Stop Flask server and terminate the program."""
    print("Closing window and stopping Flask...")
    shutdown_event.set()
    os.kill(os.getpid(), signal.SIGTERM) 
    os._exit(0)

def run_flask():
    serve(app, host="127.0.0.1", port=5000)
        
import webview
from waitress import serve

if __name__ == "__main__":
    # serve(app, host="127.0.0.1", port=5000, threads=8)
    
    from threading import Thread
    
    def run_flask():
        app.run(debug=False, port=5000, use_reloader=False)

    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    window = webview.create_window('NieR: CaroluM', 'http://127.0.0.1:5000', fullscreen=True)

    try:
        webview.start()
    except Exception as e:
        print("Error starting webview:", e)
    finally:
        stop_flask()

    flask_thread.join()
