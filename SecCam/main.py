import cv2
import time
import os
import numpy as np


def record_camera(camera_ip, username, password, output_dir, duration):
    # Ensure output directory exists, create if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # URL of the IP camera feed
    camera_url = f"rtsp://{username}:{password}@{camera_ip}/"

    # Open the IP camera
    cap = cv2.VideoCapture(camera_url)

    last_mean = 0

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to connect to IP camera")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    motion_out = None
    start_time = time.time()
    motion_start_time = start_time
    motion_last_detected = start_time
    segment_count = 0
    motion_segment_count = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print("Error: Unable to capture frame")
            break

        # Get current time
        current_time = time.time()

        # Create new VideoWriter object for every minute
        if current_time - start_time >= duration * 60 or out is None:
            start_time = current_time
            segment_count += 1
            if out is not None:
                out.release()  # Release previous VideoWriter object
            output_file = f"{output_dir}/output_{segment_count}.avi"
            out = cv2.VideoWriter(output_file, fourcc, 20.0, (720, 480))

        # Resize frame
        smaller_frame = cv2.resize(frame, (720, 480))

        # Motion detection
        gray = cv2.cvtColor(smaller_frame, cv2.COLOR_BGR2GRAY)
        result = np.abs(np.mean(gray) - last_mean)
        if result > 0.2:
            motion_last_detected = current_time
            if current_time - motion_start_time >= duration * 60 or motion_out is None:
                motion_start_time = current_time
                motion_segment_count += 1
                if motion_out is not None:
                    motion_out.release()  # Release previous VideoWriter object
                motion_output_file = f"{output_dir}/motion_output_{motion_segment_count}.avi"
                motion_out = cv2.VideoWriter(motion_output_file, fourcc, 20.0, (720, 480))

        # Check if no motion detected for 30 seconds
        if current_time - motion_last_detected >= 30:
            if motion_out is not None:
                motion_out.release()
                motion_out = None

        last_mean = np.mean(gray)

        # Write the frame to the output files
        out.write(smaller_frame)
        if motion_out is not None:
            motion_out.write(smaller_frame)

        # Display the frame (optional)
        cv2.imshow('Recording', smaller_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when recording is done
    cap.release()
    if out is not None:
        out.release()
    if motion_out is not None:
        motion_out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Replace these with your camera's details
    ip_address = '#'
    username = '#'
    password = '#'

    # Specify the output directory
    output_directory = 'recordings'

    # Specify the duration of each video segment (in minutes)
    duration = 1  # Record for every minute

    record_camera(ip_address, username, password, output_directory, duration)
