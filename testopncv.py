import cv2

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height which is 640x480 by default
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()
    #read() produces 2 values. first is boolean if frame get was successful. second is frame itself
    
    # Write the frame to the output file
    frame=cv2.flip(frame, 1)
    #flipcode 0 for x axis, 1 for y axis, -1 for both axes
    
    #frame=frame[:, 80:560]
    #slicing the matrix in y axis and x axis for making a square cut

    #out.write(frame)

    # Display the captured frame
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == 13:
        break

# Release the capture and writer objects
cam.release()
#out.release()
cv2.destroyAllWindows()