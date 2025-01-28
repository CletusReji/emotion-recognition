import cv2
faceclass = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

def detectbox(vid):
    grayimage = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = faceclass.detectMultiScale(grayimage, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x+w, y+h), (0, 255, 0), 4)
    return faces

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    faces = detectbox(frame)
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) == 13 or cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
        break
cam.release()
cv2.destroyAllWindows()