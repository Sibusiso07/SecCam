import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

# Read the first frame to initialize the background
ret, frame = cap.read()
if not ret:
    print("Error: Could not read the frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

background_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

frame_duration = 1 / 17  # Duration for each frame in seconds

# Try to create the CSRT tracker
try:
    tracker = cv2.TrackerCSRT_create()
except AttributeError:
    tracker = cv2.TrackerCSRT.create()

initBB = None  # Bounding box for the tracked object
motion_threshold = 500  # Minimum contour area to be considered as motion

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Compute the absolute difference between the current frame and background
    frame_delta = cv2.absdiff(background_gray, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours on thresholded image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Update the tracker if initialized
    if initBB is not None:
        success, box = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        else:
            initBB = None  # If tracking fails, reset the bounding box

    # Initialize tracker if motion is detected and no object is currently being tracked
    if initBB is None:
        for contour in contours:
            if cv2.contourArea(contour) < motion_threshold:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            initBB = (x, y, w, h)
            tracker.init(frame, initBB)
            break  # Initialize tracker for the first detected object only

    cv2.imshow('desktop_cam', frame)

    # Update the background if no object is being tracked
    if initBB is None:
        background_gray = gray.copy()

    # Calculate the time taken for processing and displaying the frame
    elapsed_time = time.time() - start_time
    # Sleep for the remaining time to achieve 17 fps
    time.sleep(max(0, frame_duration - elapsed_time))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
