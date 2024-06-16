from threading import Thread

from flask import Flask, request, render_template, jsonify  
from predict import predict_posture, stop_predict, get_posture_model 

flag = False  # Flag to keep track of whether the posture prediction is running or not

app = Flask(__name__, static_url_path='/static') 

@app.route('/', methods=['GET']) 
def index():
    return render_template('index.html') 

@app.route('/start', methods=['GET'])  
def start():
    predict_posture()  # Call the predict_posture function to start the prediction

@app.route('/stop', methods=['GET']) 
def stop():
    stop_predict()  # Call the stop_predict function to stop the prediction

@app.route('/get_posture', methods=['GET'])
def get_posture():
    global flag  
    if not flag:  # If the flag is False (posture prediction is not running)
        t = Thread(target=get_posture_from_model)  # Create a new thread to run the get_posture_from_model function
        t.daemon = True  # Set the thread as a daemon thread (will exit when the main program exits)
        t.start()  # Start the thread
        flag = True  # Set the flag to True (posture prediction is running)
    
    return get_posture_from_model()  # Call the get_posture_from_model function and return its result

def get_posture_from_model():
    prediction, confidence = get_posture_model()  # Call the get_posture_model function to get the predicted posture and confidence
    print(confidence)  # Print the confidence
    print(jsonify((prediction,confidence)))  # Print the prediction and confidence as JSON
    prediction = "Correct Posture" if prediction == "correct_pose" else "Incorrect Posture"  # Map the prediction to human-readable form
    return jsonify({"posture": prediction, "confidence": confidence})  # Return the prediction and confidence as JSON

if __name__ == '__main__':
    app.run(debug=True, port=1234)  
