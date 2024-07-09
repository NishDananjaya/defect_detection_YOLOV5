import torch  # Deep learning library
import cv2  # Computer vision library
import numpy as np  # Numerical computation library
from PIL import Image  # Image processing library
from ultralytics import YOLO

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp2/weights/best.pt')  # Load the YOLOv5 model with custom weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize a counter for the number of frames captured
capture_count = 1

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

# Define the display window size
window_width = 800
window_height = 600

while True:
    ret, frame = cap.read()  # Read a frame from the video capture
    if ret:  # If a frame was successfully read
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB
        results = model(frame_rgb)  # Run the model on the frame
        print(results)

        # Render the model's results on the frame
        annotated_img = results.render()[0]
        # Convert the image to RGB
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # If the model made any detections in the frame
        if results.pred[0].shape[0] != 0:
            # Define the path where the frame will be saved
            save_path = f"captured images/captured_frame_auto_{capture_count}.jpg"
            # Save the image
            cv2.imwrite(save_path, annotated_img_rgb)
            # Increment the capture count
            capture_count += 1

        # Resize the frame to fit the display window
        annotated_img_resized = cv2.resize(annotated_img_rgb, (window_width, window_height))

        # Display the frame in a window named 'Video Feed'
        cv2.imshow('Video Feed', annotated_img_resized)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
