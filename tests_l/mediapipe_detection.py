import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque, Counter

def detect_emotion_realtime():
    # Load your pre-trained FER CNN model
    model = load_model('newerfacemodel.keras')

    # Set up MediaPipe face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    # Emotion labels (match your model's training order)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    recent_emotions = deque(maxlen=3)  # Rolling window to smooth emotion display
    recent_confidences = deque(maxlen=3)  # Rolling window for confidence values

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))  # Resize to boost performance

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        ih, iw, _ = frame.shape

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Add padding around face ROI
                pad_x = int(0.1 * w)
                pad_y = int(0.1 * h)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(iw, x + w + pad_x)
                y2 = min(ih, y + h + pad_y)

                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue

                try:
                    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    resized_face = cv2.resize(gray_face, (48, 48))
                    normalized_face = resized_face / 255.0
                    input_face = normalized_face.reshape(1, 48, 48, 1)

                    emotion_predictions = model.predict(input_face, verbose=0)
                    emotion_index = np.argmax(emotion_predictions)
                    emotion = emotion_labels[emotion_index]
                    confidence = emotion_predictions[0][emotion_index] * 100

                    # Add prediction and confidence to rolling window
                    recent_emotions.append(emotion)
                    recent_confidences.append(confidence)

                    # Get most common emotion and its average confidence in recent frames
                    common_emotion = Counter(recent_emotions).most_common(1)[0][0]
                    avg_confidence = np.mean([conf for emo, conf in zip(recent_emotions, recent_confidences) if emo == common_emotion])

                    # Draw bounding box and emotion label with confidence
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    y_offset = y1 - 10 if y1 - 10 > 10 else y2 + 30
                    cv2.putText(frame, f"{common_emotion}: {avg_confidence:.1f}%", (x1, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                except Exception as e:
                    print(f"Error processing face: {e}")

        # Display frame
        cv2.imshow('Real-time Emotion Detection', frame)

        # Exit on ESC key or window close
        if cv2.waitKey(1) & 0xFF == 27 or cv2.getWindowProperty('Real-time Emotion Detection', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_emotion_realtime()