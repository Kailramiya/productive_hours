import cv2
import dlib
import numpy as np
import time
import threading
import sounddevice as sd
from flask import Flask, Response
from flask_cors import CORS

# Global flag for noisy environment
noisy_environment = False

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Utility function to format time in hh:mm:ss format
def format_time(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

# Noise monitoring function
def check_ambient_noise():
    global noisy_environment
    while True:
        audio_sample = sd.rec(int(1 * 44100), samplerate=44100, channels=1, dtype='float64')
        sd.wait()
        rms = np.sqrt(np.mean(np.square(audio_sample)))
        decibel_level = 20 * np.log10(rms) if rms > 0 else 0
        noisy_environment = decibel_level > -45
        time.sleep(1)

# Streaming function to capture and process frames
def generate_frames():
    # Set up face and object detection
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(r"C:\Users\amank\OneDrive\Desktop\programming\requirements\dlib\shape_predictor_68_face_landmarks.dat")
    net = cv2.dnn.readNet(r"C:\Users\amank\OneDrive\Desktop\programming\requirements\yolo\yolov3.weights", r"C:\Users\amank\OneDrive\Desktop\programming\requirements\yolo\yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    with open(r"C:\Users\amank\OneDrive\Desktop\programming\requirements\yolo\coco.names", 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    book_class_id = 73  # Change this if necessary for the book class ID
    cap = cv2.VideoCapture(0)

    focused_time = distraction_time = 0
    is_focused = False

    # Start noise monitoring in a separate thread
    threading.Thread(target=check_ambient_noise, daemon=True).start()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = detector(gray)
        face_detected = len(faces) > 0

        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Object detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        book_detected = False
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == book_class_id:
                    center_x, center_y = int(obj[0] * width), int(obj[1] * height)
                    w, h = int(obj[2] * width), int(obj[3] * height)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    book_detected = True
                    break

        # Determine focus state
        if book_detected and face_detected:
            if not is_focused:
                is_focused = True
        else:
            if is_focused:
                is_focused = False

        # Update focus and distraction time
        if is_focused:
            focused_time += 1
        else:
            distraction_time += 1

        # Display status text
        cv2.putText(frame, "Focused" if is_focused else "Not Focused", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if noisy_environment:
            cv2.putText(frame, "Noisy Environment", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Stream frame as part of MJPEG stream
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
