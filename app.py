import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image


st.title("Real-time Object Detection with YOLOv5")

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to draw the bounding box while selecting ROI
def draw_roi_box(event, x, y, flags, param):
    global roi_start, roi_end, selecting_roi, frame_with_box

    if event == cv2.EVENT_LBUTTONDOWN:
        selecting_roi = True
        roi_start = (x, y)
        roi_end = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting_roi:
            roi_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        selecting_roi = False
        roi_end = (x, y)
        cv2.rectangle(frame_with_box, roi_start, roi_end, (0, 255, 0), 2)

selecting_roi = False
roi_start = (0, 0)
roi_end = (0, 0)
frame_with_box = None

# Set the mouse callback function for ROI selection
cv2.namedWindow("ROI Selector")
cv2.setMouseCallback("ROI Selector", draw_roi_box)

st.text("Select ROI by clicking and dragging on the 'ROI Selector' window.")

# Display the camera input widget
img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # Convert the file-like object to a PIL image
    img = Image.open(img_file_buffer)
    img_array = np.array(img)  # Convert PIL Image to numpy array

    st.text("Press 'q' to exit the video stream")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break

        frame_with_box = frame.copy()

        if not selecting_roi:
            cv2.rectangle(frame_with_box, roi_start, roi_end, (0, 255, 0), 2)

        cv2.imshow("ROI Selector", frame_with_box)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key
            break

    cv2.destroyWindow("ROI Selector")

    # Crop the selected ROI
    roi = frame[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for consistent processing
        frame = cv2.resize(frame, (1020, 500))

        # Perform object detection on the selected ROI
        results = model(roi)

        # Draw bounding boxes and labels
        for index, row in results.pandas().xyxy[0].iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(roi, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display the resulting frame with detections
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
