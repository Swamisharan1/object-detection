import torch
import cv2
import numpy as np
import streamlit as st

drawing = False
roi_selected = False
point1 = (0, 0)
point2 = (0, 0)
canvas = None

# Standardize the size of the video frames
STANDARD_WIDTH = 1020
STANDARD_HEIGHT = 500

def mouse_draw_rect(event, x, y, flags, params):
    global drawing, point1, point2, canvas, roi_selected

    frame = params[0]

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        point1 = (x, y)
        point2 = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            canvas = frame.copy()
            point2 = (x, y)
            cv2.rectangle(canvas, point1, point2, (255, 0, 0), 2)  # Blue for ROI selection
            cv2.imshow('Video', canvas)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        point2 = (x, y)
        roi_selected = True
        canvas = frame.copy()
        cv2.rectangle(canvas, point1, point2, (255, 0, 0), 2)  # Blue for ROI selection
        cv2.imshow('Video', canvas)

# Function to initialize camera and start capturing video
def start_camera():
    cap = cv2.VideoCapture(0)  # Access the default camera (0)

    if not cap.isOpened():
        st.error("Error: Unable to access the camera.")
        return None

    return cap

# Function to release camera and close windows
def stop_camera(cap):
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

# Main function to run the Streamlit application
def main():
    st.title("Camera Object Detection with YOLOv5")

    cap = start_camera()
    if cap is None:
        return

    cv2.namedWindow('Video')

    ret, frame = cap.read()
    if not ret:
        st.error("Error: Failed to capture video frame.")
        stop_camera(cap)
        return

    frame = cv2.resize(frame, (STANDARD_WIDTH, STANDARD_HEIGHT))
    canvas = frame.copy()

    cv2.setMouseCallback('Video', mouse_draw_rect, [frame])

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    while True:
        if not roi_selected:
            if drawing:
                cv2.rectangle(canvas, point1, point2, (255, 0, 0), 2)  # Keep ROI rectangle visible in blue
            cv2.imshow('Video', canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (STANDARD_WIDTH, STANDARD_HEIGHT))
        roi = frame[point1[1]:point2[1], point1[0]:point2[0]]
        results = model(roi)

        for index, row in results.pandas().xyxy[0].iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])
            label = row['name']
            confidence = row['confidence']
            label_text = f'{label} {confidence:.2f}'

            # Draw bounding box and label
            cv2.rectangle(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for detection boxes
            cv2.putText(roi, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        frame[point1[1]:point2[1], point1[0]:point2[0]] = roi
        cv2.rectangle(frame, point1, point2, (255, 0, 0), 2)  # Draw ROI rectangle on the frame in blue
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    stop_camera(cap)

if __name__ == "__main__":
    main()
