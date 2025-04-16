import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import time

# Function to detect faces and emotions in real-time using MediaPipe
def detect_emotion_realtime():
    # Load the pre-trained emotion detection model
    model = load_model('tests_l/newerfacemodel.keras')
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    # Define emotion labels
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Start webcam capture
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # For FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Calculate FPS
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
            
        # Convert BGR to RGB for MediaPipe
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect faces
        results = face_detection.process(rgb_frame)
        
        # Convert to grayscale for emotion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # If faces are detected
        if results.detections:
            for detection in results.detections:
                # Get bounding box coordinates
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Ensure coordinates are within frame boundaries
                x = max(0, x)
                y = max(0, y)
                w = min(iw - x, w)
                h = min(ih - y, h)
                
                # Extract face ROI
                face_roi = gray[y:y+h, x:x+w]
                
                if face_roi.size == 0:
                    continue
                
                try:
                    # Resize to 48x48 (model input size)
                    resized_face = cv2.resize(face_roi, (48, 48))
                    
                    # Normalize pixel values
                    normalized_face = resized_face / 255.0
                    
                    # Reshape for model input
                    input_face = normalized_face.reshape(1, 48, 48, 1)
                    
                    # Predict emotion
                    emotion_predictions = model.predict(input_face, verbose=0)
                    emotion_index = np.argmax(emotion_predictions)
                    emotion = emotion_labels[emotion_index]
                    confidence = emotion_predictions[0][emotion_index] * 100
                    
                    # Draw rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display emotion text
                    text = f"{emotion}: {confidence:.1f}%"
                    y_offset = y - 10 if y - 10 > 10 else y + h + 30
                    cv2.putText(frame, text, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
        
        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotion_realtime()