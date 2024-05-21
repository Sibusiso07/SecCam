import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
last_mean = 0
frame_duration = 1 / 17  # Duration for each frame in seconds

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('desktop_cam', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = np.abs(np.mean(gray) - last_mean)

    if result > 1:
        print("Motion detected!")
    else:
        print("0")

    last_mean = np.mean(gray)

    # Calculate the time taken for processing and displaying the frame
    elapsed_time = time.time() - start_time
    # Sleep for the remaining time to achieve 17 fps
    time.sleep(max(0, frame_duration - elapsed_time))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
