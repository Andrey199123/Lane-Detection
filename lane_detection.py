#Andrey Vasilyev 2/23/25
import cv2 as cv
import numpy as np

# Set the dimensions for the ROI mask (now smaller)
mask_height = 300
mask_width = 300

def processImage(image):
    # Convert the image to grayscale for processing.
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur (kernel size: 15x15, sigma: 1) to smooth the image.
    blurred = cv.GaussianBlur(gray, (15, 15), 1)

    # Apply a median blur (kernel size: 5) to further reduce noise.
    median_blur = cv.medianBlur(blurred, 5)

    # Use Canny edge detection with thresholds 100 and 20.
    canny_image = cv.Canny(median_blur, 100, 20)

    # Create a blank mask (same size as the grayscale image).
    mask = np.zeros_like(gray)

    # Calculate coordinates to center the ROI in the image.
    height, width = mask.shape
    x1 = (width - mask_width) // 2
    y1 = (height - mask_height) // 2
    x2 = x1 + mask_width
    y2 = y1 + mask_height

    # Set the ROI region in the mask to white (255).
    mask[y1:y2, x1:x2] = 255

    # Apply the ROI mask to the Canny edge image to focus processing on the ROI.
    roi_edges = cv.bitwise_and(canny_image, canny_image, mask=mask)

    # Detect contours in the masked edge image.
    contours, _ = cv.findContours(roi_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # If contours are detected, draw them and compute a centerline.
    if contours:
        # Draw all detected contours on the original image in green with thickness 5.
        cv.drawContours(image, contours, -1, (0, 255, 0), 5)

        # Determine the minimum contour length among all contours.
        min_length = min(len(cnt) for cnt in contours)

        # For each contour, take only the first 'min_length' points and calculate the average coordinates.
        midpoint_x_arr = np.mean([contour[:, 0, :][:min_length][:, 0] for contour in contours], axis=0).astype(int)
        midpoint_y_arr = np.mean([contour[:, 0, :][:min_length][:, 1] for contour in contours], axis=0).astype(int)

        # Draw the computed centerline in blue by connecting successive averaged points.
        for i in range(len(midpoint_x_arr) - 1):
            cv.line(image,
                    (midpoint_x_arr[i], midpoint_y_arr[i]),
                    (midpoint_x_arr[i + 1], midpoint_y_arr[i + 1]),
                    (255, 0, 0), 5)

    # Outline the ROI on the original image with a yellow rectangle (BGR color: (0, 255, 255)).
    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)

def displayImage(image):
    cv.imshow("Detected Lines", image)
    cv.waitKey(1)

if __name__ == "__main__":
    try:
        # Open the default webcam.
        video = cv.VideoCapture(0)

        # Continuously process frames until the video stream ends or an error occurs.
        while True:
            ret, frame = video.read()
            if not ret:
                break

            processImage(frame)
            displayImage(frame)

        # Release the video capture and close all OpenCV windows.
        video.release()
        cv.destroyAllWindows()

    except Exception as e:
        print("Exiting program due to error:", e)
