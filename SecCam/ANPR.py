import cv2
import numpy as np
import easyocr
import imutils
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread('Cars/RedMini.jpg')
if img is None:
    print("Error: Image not found or unable to load.")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    plt.title('Grayscale Image')
    plt.show()

    # Noise Reduction
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    # Edge Detection
    edged = cv2.Canny(bfilter, 30, 200)
    plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
    plt.title('Edge Detection')
    plt.show()

    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    # Find the location of the license plate
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        # Create mask and extract the license plate
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_img = cv2.bitwise_and(img, img, mask=mask)
        plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
        plt.title('License Plate Isolated')
        plt.show()

        # Image isolation
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.title('Cropped License Plate')
        plt.show()

        # OCR recognition
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        print(result)

        # Rendering the results
        # text = result[0][-2]
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1,
        #                   color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        # res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
        # plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

    else:
        print("License plate location could not be determined.")



# ANPR on a video or camera
# import cv2
# import numpy as np
# import easyocr
# import imutils
# from matplotlib import pyplot as plt
#
# # Initialize video capture (0 for webcam, or provide a video file path)
# video_path = 'path/to/your/video.mp4'  # Replace with your video file path or 0 for webcam
# cap = cv2.VideoCapture(video_path)
#
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()
#
# reader = easyocr.Reader(['en'])
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Noise Reduction
#     bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
#
#     # Edge Detection
#     edged = cv2.Canny(bfilter, 30, 200)
#
#     # Find contours
#     keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours = imutils.grab_contours(keypoints)
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
#
#     # Find the location of the license plate
#     location = None
#     for contour in contours:
#         approx = cv2.approxPolyDP(contour, 10, True)
#         if len(approx) == 4:
#             location = approx
#             break
#
#     if location is not None:
#         # Create mask and extract the license plate
#         mask = np.zeros(gray.shape, np.uint8)
#         new_image = cv2.drawContours(mask, [location], 0, 255, -1)
#         new_img = cv2.bitwise_and(frame, frame, mask=mask)
#
#         # Image isolation
#         (x, y) = np.where(mask == 255)
#         (x1, y1) = (np.min(x), np.min(y))
#         (x2, y2) = (np.max(x), np.max(y))
#         cropped_image = gray[x1:x2 + 1, y1:y2 + 1]
#
#         # OCR recognition
#         result = reader.readtext(cropped_image)
#         print(result)
#
#         # Rendering the results on the original frame
#         if result:
#             text = result[0][-2]
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             cv2.putText(frame, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1,
#                         color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
#             cv2.rectangle(frame, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
#
#     # Display the frame with detected license plate
#     cv2.imshow('Frame', frame)
#
#     # Break the loop if 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture object and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

