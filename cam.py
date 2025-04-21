import cv2
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Webcam not accessible")
else:
    print("Webcam is working")