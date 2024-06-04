# import cv2
# import time
# import os
# import numpy as np
# from dotenv import find_dotenv, load_dotenv
#
# # Find .env file
# path = find_dotenv()
# # Load data stored in the .env file
# load_dotenv(path)
#
#
# def record_camera(cam1_ip, cam2_ip, username, password, output_dir, duration):
#     # Ensure output directory exists, create if it doesn't
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     # URL of the IP camera feed
#     # camera_url = f"rtsp://{username}:{password}@{cam1_ip}/"
#     # camera_url = f"rtsp://{username}:{password}@{cam2_ip}/"
#
#     # Open the IP camera
#     cap1 = cv2.VideoCapture(f"rtsp://{username}:{password}@{cam1_ip}/")
#     cap2 = cv2.VideoCapture(f"rtsp://{username}:{password}@{cam2_ip}/")
#
#     last_mean = 0
#
#     # Check if the camera opened successfully
#     if not cap1.isOpened() or not cap2.isOpened():
#         print("Error: Unable to connect to IP camera")
#         return
#
#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = None
#     motion_out = None
#     start_time = time.time()
#     motion_start_time = start_time
#     motion_last_detected = start_time
#     segment_count = 0
#     motion_segment_count = 0
#
#     while True:
#         # Capture frame-by-frame
#         ret1, frame1 = cap1.read()
#         ret2, frame2 = cap2.read()
#
#         # Check if the frame was captured successfully
#         if not ret1 or not ret2:
#             print("Error: Unable to capture frame")
#             break
#
#         # Get current time
#         current_time = time.time()
#
#         # Create new VideoWriter object for every minute
#         if current_time - start_time >= duration * 60 or out is None:
#             start_time = current_time
#             segment_count += 1
#             if out is not None:
#                 out.release()  # Release previous VideoWriter object
#             output_file = f"{output_dir}/output_{segment_count}.avi"
#             out = cv2.VideoWriter(output_file, fourcc, 20.0, (720, 480))
#
#         # Resize frame
#         cap1_frame = cv2.resize(frame1, (720, 480))
#         cap2_frame = cv2.resize(frame2, (720, 480))
#
#         # Concatenate the videos from the cameras
#         combine_frame = cv2.hconcat([cap1_frame, cap2_frame])
#
#         # Motion detection
#         gray = cv2.cvtColor(combine_frame, cv2.COLOR_BGR2GRAY)
#         result = np.abs(np.mean(gray) - last_mean)
#         if result > 0.5:
#             motion_last_detected = current_time
#             if current_time - motion_start_time >= duration * 60 or motion_out is None:
#                 motion_start_time = current_time
#                 motion_segment_count += 1
#                 if motion_out is not None:
#                     motion_out.release()  # Release previous VideoWriter object
#                 motion_output_file = f"{output_dir}/motion_output_{motion_segment_count}.avi"
#                 motion_out = cv2.VideoWriter(motion_output_file, fourcc, 20.0, (720, 480))
#
#         # Check if no motion detected for 30 seconds
#         if current_time - motion_last_detected >= 30:
#             if motion_out is not None:
#                 motion_out.release()
#                 motion_out = None
#
#         last_mean = np.mean(gray)
#
#         # Write the frame to the output files
#         # out.write(combine_frame)
#         # if motion_out is not None:
#         #     motion_out.write(combine_frame)
#
#         # Display the frame (optional)
#         cv2.imshow('Recording', combine_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Release everything when recording is done
#     cap1.release()
#     cap2.release()
#     if out is not None:
#         out.release()
#     if motion_out is not None:
#         motion_out.release()
#     cv2.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     # Replace these with your camera's details
#     cam1 = os.getenv('camera1_ip')
#     cam2 = os.getenv('camera2_ip')
#     u_name = os.getenv('cam_user')
#     p_word = os.getenv('cam_pass')
#
#     # Specify the output directory
#     output_directory = 'recordings'
#
#     # Specify the duration of each video segment (in minutes)
#     output_duration = 1  # Record for every minute
#
#     record_camera(cam1, cam2, u_name, p_word, output_directory, output_duration)


import cv2
import time
import os
import numpy as np
from threading import Thread, Lock
from dotenv import find_dotenv, load_dotenv

# Find .env file
path = find_dotenv()
# Load data stored in the .env file
load_dotenv(path)


class CameraCapture:
    def __init__(self, cam_ip, username, password):
        self.cap = cv2.VideoCapture(f"rtsp://{username}:{password}@{cam_ip}/")
        self.frame = None
        self.lock = Lock()
        self.running = True

    def start(self):
        Thread(target=self.update, args=()).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
        return frame

    def stop(self):
        self.running = False
        self.cap.release()


def record_camera(cam1_ip, cam2_ip, username, password, output_dir, duration):
    # Ensure output directory exists, create if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cam1 = CameraCapture(cam1_ip, username, password)
    cam2 = CameraCapture(cam2_ip, username, password)

    cam1.start()
    cam2.start()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    motion_out = None
    start_time = time.time()
    motion_start_time = start_time
    motion_last_detected = start_time
    segment_count = 0
    motion_segment_count = 0
    last_mean = 0

    while True:
        frame1 = cam1.read()
        frame2 = cam2.read()

        if frame1 is None or frame2 is None:
            continue

        # Get current time
        current_time = time.time()

        # Create new VideoWriter object for every minute
        if current_time - start_time >= duration * 60 or out is None:
            start_time = current_time
            segment_count += 1
            if out is not None:
                out.release()  # Release previous VideoWriter object
            output_file = f"{output_dir}/output_{segment_count}.avi"
            out = cv2.VideoWriter(output_file, fourcc, 20.0, (1440, 480))

        # Resize frame
        cap1_frame = cv2.resize(frame1, (720, 480))
        cap2_frame = cv2.resize(frame2, (720, 480))

        # Concatenate the videos from the cameras
        combine_frame = cv2.hconcat([cap1_frame, cap2_frame])

        # Motion detection
        gray = cv2.cvtColor(combine_frame, cv2.COLOR_BGR2GRAY)
        result = np.abs(np.mean(gray) - last_mean)
        if result > 0.5:
            motion_last_detected = current_time
            if current_time - motion_start_time >= duration * 60 or motion_out is None:
                motion_start_time = current_time
                motion_segment_count += 1
                if motion_out is not None:
                    motion_out.release()  # Release previous VideoWriter object
                motion_output_file = f"{output_dir}/motion_output_{motion_segment_count}.avi"
                motion_out = cv2.VideoWriter(motion_output_file, fourcc, 20.0, (1440, 480))

        # Check if no motion detected for 30 seconds
        if current_time - motion_last_detected >= 30:
            if motion_out is not None:
                motion_out.release()
                motion_out = None

        last_mean = np.mean(gray)

        # Write the frame to the output files
        out.write(combine_frame)
        if motion_out is not None:
            motion_out.write(combine_frame)

        # Display the frame (optional)
        cv2.imshow('Recording', combine_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when recording is done
    cam1.stop()
    cam2.stop()
    if out is not None:
        out.release()
    if motion_out is not None:
        motion_out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Replace these with your camera's details
    cam1 = os.getenv('camera1_ip')
    cam2 = os.getenv('camera2_ip')
    u_name = os.getenv('cam_user')
    p_word = os.getenv('cam_pass')

    # Specify the output directory
    output_directory = 'recordings'

    # Specify the duration of each video segment (in minutes)
    output_duration = 1  # Record for every minute

    record_camera(cam1, cam2, u_name, p_word, output_directory, output_duration)
