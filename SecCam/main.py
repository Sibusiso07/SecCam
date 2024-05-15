import cv2
import time
import os


def record_camera(camera_ip, output_dir, timer):
    # Ensure output directory exists, create if it doesn't
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # URL of the IP camera feed
    camera_url = f"rtsp://{ip_address}/"

    # Open the IP camera
    cap = cv2.VideoCapture(camera_url)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Unable to connect to IP camera")
        return

        # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    start_time = time.time()
    output_file = None
    segment_count = 0

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
        if current_time - start_time >= duration * 60 or out is None:  # Check if out is None
            start_time = current_time
            segment_count += 1
            if out is not None:
                out.release()  # Release previous VideoWriter object
            output_file = f"{output_directory}/output_{segment_count}.avi"
            out = cv2.VideoWriter(output_file, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        # Write the frame to the output file
        out.write(frame)

        # Display the frame (optional)
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything when recording is done
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Replace 'your_ip_address' with the IP address of your camera
    ip_address = 'camera_ip_address'

    # Specify the output directory
    output_directory = 'recordings'

    # Specify the duration of each video segment (in minutes)
    duration = 1  # Record for every minute

    record_camera(ip_address, output_directory, duration)
