from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime

app = Flask(__name__)

# Load models
model = tf.keras.models.load_model('facemodel.keras')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def generate_frames(camera_active):
    if not camera_active:
        # Create attractive placeholder image
       ''' placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder[:] = (41, 90, 168)  # Dark blue background
        
        # Add text and decoration
        text = "Camera Stopped"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
        text_x = (placeholder.shape[1] - text_size[0]) // 2
        text_y = (placeholder.shape[0] + text_size[1]) // 2
        
        cv2.putText(placeholder, text, (text_x, text_y), 
                   font, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Add camera icon
        cv2.circle(placeholder, (320, 180), 80, (255, 255, 255), 3)
        cv2.circle(placeholder, (320, 180), 30, (255, 255, 255), -1)
        
        ret, buffer = cv2.imencode('.jpg', placeholder)
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return'''

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while camera_active:
        success, frame = camera.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=(0, -1))
            
            preds = model.predict(face_roi)
            emotion = emotion_labels[np.argmax(preds)]
            cv2.putText(frame, emotion, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    camera.release()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed/<int:camera_active>')
def video_feed(camera_active):
    return Response(generate_frames(camera_active),
                  mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)