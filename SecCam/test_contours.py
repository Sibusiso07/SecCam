import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

# Read the first frame to initialize the background
ret, background = cap.read()
if not ret:
    print("Error: Could not read the frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

frame_duration = 1 / 17  # Duration for each frame in seconds

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

    # Draw contours on the frame
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('desktop_cam', frame)

    # Update the background
    background_gray = gray.copy()

    # Calculate the time taken for processing and displaying the frame
    elapsed_time = time.time() - start_time
    # Sleep for the remaining time to achieve 17 fps
    time.sleep(max(0, frame_duration - elapsed_time))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
