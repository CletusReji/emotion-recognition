#gui with tkinter and testing
 
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading

# Face detection model
faceclass = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cam = None
running = False

# Function to detect faces
def detectbox(vid):
    grayimage = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = faceclass.detectMultiScale(grayimage, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return vid

# Function to start webcam
def start_webcam():
    global cam, running
    if not running:
        running = True
        cam = cv2.VideoCapture(0)
        show_frame()

# Function to stop webcam
def stop_webcam():
    global cam, running
    running = False
    if cam is not None:
        cam.release()
        cam = None
    webcam_label.config(image='')


# Function to continuously update video feed
def show_frame():
    global cam, running
    if running and cam is not None:
        ret, frame = cam.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = detectbox(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            webcam_label.imgtk = imgtk
            webcam_label.configure(image=imgtk)
        if running:
            root.after(10, show_frame)  # Asynchronous update

# Main Tkinter window
root = tk.Tk()
root.title("Face Detection App")
root.geometry("800x600")
root.configure(bg="#1e1e1e")

# Styling buttons and frames
btn_frame = tk.Frame(root, bg="#1e1e1e")
btn_frame.pack(pady=10)

btn_style1 = {"font": ("Arial", 14, "bold"), "fg": "white", "bg": "lawngreen", "width": 20, "bd": 0, "relief": "flat"}
btn_style2 = {"font": ("Arial", 14, "bold"), "fg": "white", "bg": "red", "width": 20, "bd": 0, "relief": "flat"}

btn_start = tk.Button(btn_frame, text="Start Webcam", command=start_webcam, **btn_style1)
btn_start.pack(side=tk.LEFT, padx=10, pady=5)

btn_stop = tk.Button(btn_frame, text="Stop Webcam", command=stop_webcam, **btn_style2)
btn_stop.pack(side=tk.RIGHT, padx=10, pady=5)

# Frame for video feed
video_frame = tk.Frame(root, bg="#1e1e1e", bd=2, relief="solid")
video_frame.pack(pady=20)

webcam_label = tk.Label(video_frame, bg="#1e1e1e")
webcam_label.pack()

root.mainloop()
