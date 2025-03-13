"""
Lane Detection System - Version 3 | Andrey Vasilyev 3/12/25.

1. IMPORT required modules: os, logging, math, time, numpy, pickle, cv2, Flask, etc.
2. CONFIGURE the Flask app, secret key, and logging.
3. INITIALIZE global variables:
   - VIDEO_PATH: Path to the video file.
   - cap_raw, cap_overlay: OpenCV VideoCapture objects for raw and overlay video feeds.
   - frame_count_overlay, playback_active, last_raw_frame, last_overlay_frame.
   - Lane detection state variables: lane_frame_count, prev_right_poly, prev_left_poly,
     prev_left_avg, alpha, turned_right, jumped, lane_last_result.
   - Default parameters: default_contrast, default_l_h, default_l_s, default_l_v,
     default_u_h, default_u_s, default_u_v, default_left_offset.

--------------------------------------------
FUNCTIONS:
--------------------------------------------
process_lane_detection(frame):
    Input: frame (a video frame from cap_overlay)
    Steps:
      a. Update lane_frame_count and handle jump condition if frame count > 2400.
      b. Set smoothing factor and adjust contrast based on lane_frame_count.
      c. Process only every third frame; if not, return lane_last_result or the original frame.
      d. Resize the frame to 640x480 and adjust contrast.
      e. Convert frame to a blue-scale version if lane_frame_count < 8700.
      f. Set up perspective transformation:
         - Define points (tl, tr, bl, br) and compute transformation matrices (pts1, pts2).
         - Apply warpPerspective to obtain a transformed frame.
      g. Apply HSV thresholding:
         - Use yellow thresholds if lane_frame_count > 8700, else use default thresholds.
      h. Remove noise
      i. Enhance edges by performing Canny edge detection on the blue channel.
      j. LEFT lane detection via sliding window:
         - Compute a histogram on the lower half of the mask.
         - Identify a base point (left_base) and use sliding windows to collect lane points.
         - Fit a polynomial (left_curve_sliding) if enough points are detected.
      k. Perform sliding window search for both left lane and right lane, use smoothing and previous frames if the lanes aren't perfectly detected.
      l. Draw the detected lane curves (left and right) on the transformed frame.
      m. Warp the overlay back to the original perspective and blend with the original frame.
      n. Compute and draw the center line and a compass rose indicating the vehicle direction.
      o. Save the processed frame as lane_last_result and return it.

save_to_pkl(data, filename):
    - Save the given data to a pickle file.

load_from_pkl(filename):
    - Load and return data from the specified pickle file; return an empty list if EOFError occurs.

--------------------------------------------
FLASK ENDPOINTS:
--------------------------------------------
/ (login_register):
    - GET: Render the login page.
    - POST: Process login or registration submissions (supports both JSON and form data).
      * Validate credentials or create a new user; set global user_logged_in and return appropriate
        JSON responses or flash messages.

/home:
    - GET: Render the main interface (index.html) if the user is logged in; otherwise, redirect to login.

/logs/<filename>:
    - GET: Read and return the contents of the specified log file as plain text.

/toggle_play:
    - POST: Toggle the global playback_active variable (pausing/resuming video feeds).
    - Return a JSON response with the current status ("Playing" or "Paused").

--------------------------------------------
VIDEO FEED GENERATORS:
--------------------------------------------
generate_raw_feed():
    Loop indefinitely:
      - If playback_active is True:
          * Read the next frame from cap_raw.
          * Update last_raw_frame.
        Else:
          * Use last_raw_frame to freeze the video.
      - Encode the frame to JPEG.
      - Yield the frame as part of an MJPEG multipart response.

generate_overlay_feed():
    Loop indefinitely:
      - If playback_active is True:
          * Read the next frame from cap_overlay.
          * Process the frame through process_lane_detection().
          * Update last_overlay_frame and increment frame_count_overlay.
        Else:
          * Use last_overlay_frame to freeze the video.
      - Encode the processed frame to JPEG.
      - Sleep briefly to control FPS.
      - Yield the frame as part of an MJPEG multipart response.

--------------------------------------------
MAIN:
--------------------------------------------
- Run the Flask app on host 0.0.0.0, port 10000 with debug mode enabled.
========================================================================================
"""

import os
import logging
import math
import time
import numpy as np
import pickle
import cv2

from flask import Flask, jsonify, request, render_template, flash, redirect, url_for, Response
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = 'some_secret_key'

logging.basicConfig(filename='api_output.log', level=logging.DEBUG)
log = logging.getLogger("werkzeug")
log.disabled = True

# ------------------------------------------------------
#                  GLOBALS & INITIALIZATION
# ------------------------------------------------------
VIDEO_PATH = "/Users/andreyvasilyev/Documents/PWP/Lane-Detection-With-GUI/My Movie 2.mov"

cap_raw = cv2.VideoCapture(VIDEO_PATH)
cap_overlay = cv2.VideoCapture(VIDEO_PATH)

# FRAME STARTING LOCATION
frame_count_overlay = 0
cap_overlay.set(cv2.CAP_PROP_POS_FRAMES, frame_count_overlay)

playback_active = True
last_raw_frame = None
last_overlay_frame = None

# ------------------------------------------------------
#      Lane Detection Global State & Defaults
# ------------------------------------------------------
lane_frame_count = 0
prev_right_poly = None
prev_left_poly = None
prev_left_avg = None
alpha = 0.1
turned_right = False
jumped = True
lane_last_result = None

# HSV VALUES
default_contrast = 95  # Contrast
default_l_h = 0  # Lower Hue
default_l_s = 0  # Lower Saturation
default_l_v = 200  # Lower Value
default_u_h = 200  # Upper Hue
default_u_s = 150  # Upper Saturation
default_u_v = 255  # Upper Value
default_left_offset = 540  # Left Offset


# ------------------------------------------------------
#            LANE DETECTION PROCESSING FUNCTION
# ------------------------------------------------------
def process_lane_detection(frame):
    """
    The majority of this code comes from lane detection v2.

    Process a video frame to detect lane markings and overlay lane detection results.
      1. Updates a global frame counter
      2. Adjusts a smoothing factor (alpha) and contrast based on the current frame count.
      3. Processes every third frame (returning the last processed frame if not due for processing).
      4. Resizes the input frame to 640x480 and applies contrast adjustment.
      5. Converts the frame to a blue-scale version for frames below a specified count.
      6. Sets up and applies a perspective transformation (bird’s-eye view) using predefined points.
      7. Performs HSV thresholding:
            - Uses yellow thresholds if the frame count exceeds 8700.
            - Otherwise, applies default HSV thresholds.
      8. Removes noise.
      9. Enhances lane line visibility using Canny edge detection on the blue channel and combines the result with the mask.
     10. Detects the left lane using a sliding window approach:
            - Computes a histogram on the lower half of the mask.
            - Uses sliding windows to locate lane points and fits a polynomial if sufficient points are found.
     11. Draws the detected lane curves on the transformed frame.
     12. Warps the lane overlay back to the original perspective and blends it with the original frame.
     13. Computes and draws the center line between lanes.
     14. Draws a compass rose in the top-left corner of the frame.

    Global Variables Used:
      - lane_frame_count: Counts the processed frames.
      - jumped: Flag to simulate a jump condition in the frame count.
      - prev_right_poly, prev_left_poly: Previous polynomial coefficients for right and left lanes used for smoothing.
      - prev_left_avg: Previously averaged polynomial coefficients for the left lane.
      - alpha: Smoothing factor for polynomial coefficient updates.
      - turned_right: Boolean flag indicating if the vehicle has turned right.
      - lane_last_result: Stores the last processed frame.

    Parameters:
      frame (numpy.ndarray): The current video frame to be processed.

    Returns:
      numpy.ndarray: The processed frame with lane detection overlays including lane curves, center line, and a compass rose.

    """
    global lane_frame_count, jumped, prev_right_poly, prev_left_poly, prev_left_avg
    global alpha, turned_right, lane_last_result, arrow_angle

    # --- Update frame count and simulate jump condition (if applicable) ---
    print("Lane Frame Count:", lane_frame_count)

    if lane_frame_count > 2400 and not jumped:
        lane_frame_count = 2550
        jumped = True
        # (In the original code a new frame was read here; we simply continue using the provided frame.)

    # --- Set smoothing factor and contrast based on frame count ---
    if 2000 < lane_frame_count < 2600:
        alpha = 0.2
    elif 1200 < lane_frame_count < 1500:
        alpha = 0.1
    else:
        alpha = 0.05

    contrast = default_contrast / 100.0

    if 1200 < lane_frame_count < 1600:
        contrast = 0.85

    if lane_frame_count > 2400:
        turned_right = False
        contrast = 0.95

    lane_frame_count += 1

    # --- Process only every third frame ---
    if lane_frame_count % 3 != 0:

        if lane_last_result is not None:

            return lane_last_result

        else:

            return frame

    # --- Resize and adjust contrast ---
    frame = cv2.resize(frame, (640, 480))
    frame = cv2.addWeighted(frame, contrast, np.zeros(frame.shape, frame.dtype), 0, 0)

    # --- Convert frame to Blue-scale ---
    if lane_frame_count < 8700:
        blue_channel = frame[:, :, 0]
        frame = cv2.merge([blue_channel, blue_channel, blue_channel])

    # --- Perspective Transformation Setup ---
    mask_h = 480
    mask_w = 640
    tl = (int(mask_w * 0.4), int(mask_h * 0.5) + 50)
    tr = (int(mask_w * 0.6), int(mask_h * 0.5) + 50)
    bl = (0, mask_h)
    br = (int(mask_w * 0.97), mask_h)
    # (In the original code, circles were drawn for debugging.)
    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, mask_h], [mask_w, 0], [mask_w, mask_h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (mask_w, mask_h))

    # --- HSV Thresholding ---
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)

    if lane_frame_count > 8700:

        lower_yellow = np.array([20, 80, 80])
        upper_yellow = np.array([30, 255, 255])
        mask = cv2.inRange(hsv_transformed_frame, lower_yellow, upper_yellow)

    else:

        l_h = default_l_h
        l_s = default_l_s
        l_v = 180 if not turned_right else default_l_v
        u_h = default_u_h
        u_s = default_u_s
        u_v = default_u_v
        lower = np.array([l_h, l_s, l_v])
        upper = np.array([u_h, u_s, u_v])
        mask = cv2.inRange(hsv_transformed_frame, lower, upper)

    # --- Remove Small Shadows using Morphology ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # --- Blue Edge Detection to Enhance White Lines ---
    blue_channel = transformed_frame[:, :, 0]
    edges_blue = cv2.Canny(blue_channel, 50, 150)
    mask = cv2.bitwise_or(mask, edges_blue)

    # --- LEFT sliding window detection ---
    msk = mask.copy()
    histogram_left = np.sum(mask[mask_h // 2:, :], axis=0)
    midpoint = int(histogram_left.shape[0] / 2)
    left_base = int(np.argmax(histogram_left[:midpoint]))
    lx_points = []
    y_val = 472  # bottom y-value of ROI
    left_y_lower_bound = 200

    while y_val > left_y_lower_bound:

        win_left = max(left_base - 50, 0)
        win_right = min(left_base + 50, mask_w)
        left_window = mask[y_val - 40:y_val, win_left:win_right]
        contours, _ = cv2.findContours(left_window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            M = cv2.moments(contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                detected_x = win_left + cx
                lx_points.append((detected_x, y_val - 20))
                left_base = detected_x
        cv2.rectangle(msk, (max(left_base - 50, 0), y_val - 40),
                      (min(left_base + 50, mask_w), y_val), (255, 255, 255), 2)
        y_val -= 40

    if len(lx_points) >= 2:

        lx_array = np.array(lx_points)
        coeffs_left_sliding = np.polyfit(lx_array[:, 1], lx_array[:, 0], 2)
        y_samples = np.linspace(0, mask_h, mask_h)
        left_x_samples_sliding = (coeffs_left_sliding[0] * y_samples ** 2 + coeffs_left_sliding[1] * y_samples + coeffs_left_sliding[2])
        left_curve_sliding = np.vstack((left_x_samples_sliding, y_samples)).T.astype(np.int32)

    else:

        left_curve_sliding = None

    # --- Depending on frame count, choose right lane detection & left lane refinement ---

    if lane_frame_count > 11000:

        # LEFT lane detection via alternative sliding window search
        histogram = np.sum(mask[mask_h // 2:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        left_base = int(np.argmax(histogram[:midpoint]))
        lx_points = []
        y_val = 472
        left_y_lower_bound = 200
        msk = mask.copy()

        while y_val > left_y_lower_bound:

            win_left = max(left_base - 50, 0)
            win_right = min(left_base + 50, mask_w)
            left_window = mask[y_val - 40:y_val, win_left:win_right]
            contours, _ = cv2.findContours(left_window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:

                M = cv2.moments(contour)

                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    detected_x = win_left + cx
                    lx_points.append((detected_x, y_val - 20))
                    left_base = detected_x
            cv2.rectangle(msk, (max(left_base - 50, 0), y_val),
                          (min(left_base + 50, mask_w), y_val - 40), (255, 255, 255), 2)
            y_val -= 40

        if len(lx_points) < 2:

            if prev_left_poly is not None:

                coeffs_left = prev_left_poly
                y_samples = np.linspace(0, mask_h, mask_h)
                left_x_samples = coeffs_left[0] * y_samples ** 2 + coeffs_left[1] * y_samples + coeffs_left[2]
                left_curve_pts = np.vstack((left_x_samples, y_samples)).T.astype(np.int32)

            else:

                y_samples = np.linspace(0, mask_h, mask_h)
                x_samples = np.linspace(0, mask_w, mask_h)
                left_curve_pts = np.vstack((x_samples, y_samples)).T.astype(np.int32)
                coeffs_left = np.polyfit(left_curve_pts[:, 1], left_curve_pts[:, 0], 2)

        else:
            lx_array = np.array(lx_points)
            coeffs_left_new = np.polyfit(lx_array[:, 1], lx_array[:, 0], 2)

            if prev_left_poly is not None:

                coeffs_left = alpha * coeffs_left_new + (1 - alpha) * prev_left_poly

            else:

                coeffs_left = coeffs_left_new
            prev_left_poly = coeffs_left
            y_samples = np.linspace(0, mask_h, mask_h)
            left_x_samples = coeffs_left[0] * y_samples ** 2 + coeffs_left[1] * y_samples + coeffs_left[2]
            left_curve_pts = np.vstack((left_x_samples, y_samples)).T.astype(np.int32)

        bottom_idx = np.argmin(np.abs(left_curve_pts[:, 1] - 472))
        L = left_curve_pts[bottom_idx, 0]
        left_offset = default_left_offset

        if 1200 < lane_frame_count < 1600:
            left_offset = 450

        symmetry_line = L + left_offset / 2.0
        a, b, c = coeffs_left
        coeffs_right_new = np.array([-a, -b, 2 * symmetry_line - c])

        if prev_right_poly is not None:

            coeffs_right = alpha * coeffs_right_new + (1 - alpha) * prev_right_poly

        else:

            coeffs_right = coeffs_right_new

        prev_right_poly = coeffs_right
        y_samples = np.linspace(0, mask_h, mask_h)
        right_x_samples = coeffs_right[0] * y_samples ** 2 + coeffs_right[1] * y_samples + coeffs_right[2]
        right_curve_pts = np.vstack((right_x_samples, y_samples)).T.astype(np.int32)
        left_curve_avg = left_curve_pts  # from the alternative left lane search

    else:

        histogram = np.sum(mask[mask_h // 2:, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        right_base = int(np.argmax(histogram[midpoint:])) + midpoint
        rx_points = []
        y_val = 472
        right_y_lower_bound = 200
        msk = mask.copy()

        while y_val > right_y_lower_bound:

            win_left = max(right_base - 50, 0)
            win_right = min(right_base + 50, mask_w)
            right_window = mask[y_val - 40:y_val, win_left:win_right]
            contours, _ = cv2.findContours(right_window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    detected_x = win_left + cx
                    rx_points.append((detected_x, y_val - 20))
                    right_base = detected_x
            cv2.rectangle(msk, (max(right_base - 50, 0), y_val),
                          (min(right_base + 50, mask_w), y_val - 40), (255, 255, 255), 2)
            y_val -= 40

        if len(rx_points) < 2:

            print("No right lane points detected, using previous detection")

            if prev_right_poly is not None:

                coeffs_right = prev_right_poly

            else:

                y_samples = np.linspace(0, mask_h, mask_h)
                x_samples = np.linspace(midpoint, mask_w, mask_h)
                right_curve_pts = np.vstack((x_samples, y_samples)).T.astype(np.int32)
                coeffs_right = np.polyfit(right_curve_pts[:, 1], right_curve_pts[:, 0], 2)
            y_samples = np.linspace(0, mask_h, mask_h)
            right_x_samples = coeffs_right[0] * y_samples ** 2 + coeffs_right[1] * y_samples + coeffs_right[2]
            right_curve_pts = np.vstack((right_x_samples, y_samples)).T.astype(np.int32)

        else:

            rx_array = np.array(rx_points)
            coeffs_right_new = np.polyfit(rx_array[:, 1], rx_array[:, 0], 2)

            if prev_right_poly is not None:

                coeffs_right = alpha * coeffs_right_new + (1 - alpha) * prev_right_poly

            else:

                coeffs_right = coeffs_right_new
            prev_right_poly = coeffs_right
            y_samples = np.linspace(0, mask_h, mask_h)
            right_x_samples = coeffs_right[0] * y_samples ** 2 + coeffs_right[1] * y_samples + coeffs_right[2]
            right_curve_pts = np.vstack((right_x_samples, y_samples)).T.astype(np.int32)

        bottom_idx = np.argmin(np.abs(right_curve_pts[:, 1] - 472))
        R = right_curve_pts[bottom_idx, 0]
        left_offset = default_left_offset
        symmetry_line = R - left_offset / 2.0
        coeffs_left_mirror = np.array([-coeffs_right[0], -coeffs_right[1], 2 * symmetry_line - coeffs_right[2]])
        left_x_samples_mirror = (coeffs_left_mirror[0] * y_samples ** 2 + coeffs_left_mirror[1] * y_samples + coeffs_left_mirror[2])
        left_curve_mirror = np.vstack((left_x_samples_mirror, y_samples)).T.astype(np.int32)

        # --- Average the two left lane estimates (sliding and mirrored) ---
        if left_curve_sliding is not None and left_curve_mirror is not None:
            coeffs_left_avg = (np.polyfit(left_curve_sliding[:, 1], left_curve_sliding[:, 0],
                                          2) + coeffs_left_mirror) / 2.0
        elif left_curve_sliding is not None:
            coeffs_left_avg = np.polyfit(left_curve_sliding[:, 1], left_curve_sliding[:, 0], 2)
        elif left_curve_mirror is not None:
            coeffs_left_avg = coeffs_left_mirror
        else:
            coeffs_left_avg = None

        if coeffs_left_avg is not None:
            if prev_left_avg is not None:
                coeffs_left_avg = alpha * coeffs_left_avg + (1 - alpha) * prev_left_avg
            prev_left_avg = coeffs_left_avg
            left_x_samples_avg = coeffs_left_avg[0] * y_samples ** 2 + coeffs_left_avg[1] * y_samples + coeffs_left_avg[
                2]
            left_curve_avg = np.vstack((left_x_samples_avg, y_samples)).T.astype(np.int32)

        else:
            left_curve_avg = None

    # --- Determine turning mode and adjust lane curves if needed ---
    turning_mode = None
    if 1800 < lane_frame_count < 2450:
        turning_mode = "right"

    if 10500 < lane_frame_count < 10900:
        turning_mode = "left"
        arrow_angle = -math.pi / 4  # compass point left

    if turning_mode == "right":
        left_lane_x = 90
        lane_width = 540
        arrow_angle = math.pi / 4  # compass point right
        right_lane_x = left_lane_x + lane_width
        y_samples = np.linspace(0, mask_h, mask_h)
        left_curve_avg = np.vstack((np.full_like(y_samples, left_lane_x), y_samples)).T.astype(np.int32)
        right_curve_pts = np.vstack((np.full_like(y_samples, right_lane_x), y_samples)).T.astype(np.int32)

    # --- Draw the curved lanes ---
    curve_overlay = transformed_frame.copy()
    cv2.polylines(curve_overlay, [right_curve_pts.reshape((-1, 1, 2))],
                  isClosed=False, color=(0, 255, 0), thickness=5)
    if left_curve_avg is not None:
        cv2.polylines(curve_overlay, [left_curve_avg.reshape((-1, 1, 2))],
                      isClosed=False, color=(0, 255, 0), thickness=5)

    # --- Warp the lane overlay back to the original perspective ---
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    original_perspective = cv2.warpPerspective(curve_overlay, inv_matrix, (mask_w, mask_h))
    result = cv2.addWeighted(frame, 0.3, original_perspective, 0.7, 0)

    # --- Compute and draw the center line ---
    if left_curve_avg is not None:
        center_line = ((left_curve_avg + right_curve_pts) // 2)
        center_line[:, 0] = np.clip(center_line[:, 0], 0, mask_w - 1)
        center_line[:, 1] = np.clip(center_line[:, 1], 0, mask_h - 1)
        center_poly = np.polyfit(center_line[:, 1], center_line[:, 0], 2)
        y_samples = np.linspace(0, mask_h, mask_h)
        center_x_samples = center_poly[0] * y_samples ** 2 + center_poly[1] * y_samples + center_poly[2]
        center_curve_pts = np.vstack((center_x_samples, y_samples)).T.astype(np.int32)
        center_curve_pts_warped = cv2.perspectiveTransform(center_curve_pts.reshape(-1, 1, 2).astype(np.float32), inv_matrix)
        center_curve_pts_warped = center_curve_pts_warped.astype(np.int32)
        cv2.polylines(result, [center_curve_pts_warped.reshape((-1, 1, 2))], isClosed=False, color=(0, 255, 255), thickness=5)

    # --- Draw a Compass Rose in the top-left corner ---
    if turning_mode is None and left_curve_avg is not None:
        center_line_for_arrow = ((left_curve_avg + right_curve_pts) // 2)
        pt_bottom = center_line_for_arrow[-1]
        pt_above = center_line_for_arrow[-min(11, len(center_line_for_arrow))]
        dx = pt_bottom[0] - pt_above[0]
        dy = pt_bottom[1] - pt_above[1]
        arrow_angle = math.atan2(dx, dy)

    compass_center = (550, 60)  # Location of compass is in the top right

    radius = 40
    tip = (int(compass_center[0] + radius * math.sin(arrow_angle)),
           int(compass_center[1] - radius * math.cos(arrow_angle)))
    cv2.circle(result, compass_center, radius, (0, 255, 0), 2)
    cv2.arrowedLine(result, compass_center, tip, (0, 255, 0), 2, tipLength=0.3)

    lane_last_result = result.copy()
    return result


# ------------------------------------------------------
#      Flask Login & Registration Logic
# ------------------------------------------------------
users_file = 'users.pkl'
if not os.path.exists(users_file):
    open(users_file, 'wb').close()


def save_to_pkl(data, filename):
    """Save user data to pkl files."""
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_from_pkl(filename):
    """Load user data from pkl files."""
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except EOFError:
        return []


user_logged_in = False


@app.route('/', methods=['GET', 'POST'])
def login_register():
    """API Code for user login."""
    global user_logged_in
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            form_type = data.get('form_type')

            if form_type == 'login':
                username_or_email = data.get('username_or_email')
                password = data.get('password')
                users = load_from_pkl(users_file)
                user = next((u for u in users if u['username'] == username_or_email or u['email'] == username_or_email),
                            None)
                if user and check_password_hash(user['password'], password):
                    user_logged_in = True
                    return jsonify(success=True, redirect_url=url_for('home'))
                else:
                    return jsonify(success=False, message="Invalid username/email or password")

            elif form_type == 'register':
                name = data.get('name')
                username = data.get('username')
                email = data.get('email')
                password = data.get('password')
                users = load_from_pkl(users_file)

                if any(u['username'] == username for u in users):
                    return jsonify(success=False, message="Username already exists")
                elif any(u['email'] == email for u in users):
                    return jsonify(success=False, message="Email already exists")
                else:
                    hashed_password = generate_password_hash(password)
                    users.append({'name': name, 'username': username, 'email': email, 'password': hashed_password})
                    save_to_pkl(users, users_file)
                    return jsonify(success=True, message="Registration successful! Please log in.")

        else:
            form_type = request.form['form_type']
            if form_type == 'login':
                username_or_email = request.form['username_or_email']
                password = request.form['password']
                users = load_from_pkl(users_file)
                user = next((u for u in users if u['username'] == username_or_email or u['email'] == username_or_email),
                            None)
                if user and check_password_hash(user['password'], password):
                    flash('Login successful')
                    user_logged_in = True
                    return redirect('/home')
                else:
                    flash('Invalid username/email or password')

            elif form_type == 'register':
                name = request.form['name']
                username = request.form['username']
                email = request.form['email']
                password = request.form['password']
                users = load_from_pkl(users_file)

                if any(u['username'] == username for u in users):
                    flash('Username already exists')
                elif any(u['email'] == email for u in users):
                    flash('Email already exists')
                else:
                    hashed_password = generate_password_hash(password)
                    users.append({'name': name, 'username': username, 'email': email, 'password': hashed_password})
                    save_to_pkl(users, users_file)
                    flash('Registration successful. Please login.')
                    return redirect('/')

    return render_template('login.html')


@app.route('/home')
def home():
    """Check if the user is logged in and returns the webpage."""
    global user_logged_in
    if user_logged_in:
        return render_template('index.html')
    else:
        return redirect('/')


@app.route('/logs/<filename>', methods=['GET'])
def logs(filename):
    """Return the contents of the specified log file as plain text.

    If the log file does not exist, return a JSON error message.
    """
    log_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(log_path):
        with open(log_path, 'r') as file:
            return file.read(), 200, {'Content-Type': 'text/plain'}
    else:
        return jsonify({'error': 'Log file not found'}), 404


# ------------------------------------------------------
#         PLAY / PAUSE ENDPOINT
# ------------------------------------------------------
@app.route('/toggle_play', methods=['POST'])
def toggle_play():
    """Pause/Play button in the top right of the screen."""
    global playback_active
    playback_active = not playback_active
    status = "Playing" if playback_active else "Paused"
    return jsonify({"status": status})


# ------------------------------------------------------
#   RAW VIDEO FEED (Only)
# ------------------------------------------------------
def generate_raw_feed():
    """Display the unchanged video feed in the top left of the screen."""
    global playback_active, cap_raw, last_raw_frame
    while True:
        if playback_active:
            success, frame = cap_raw.read()
            if not success:
                # Loop back to start if we reach the end of the video
                cap_raw.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            last_raw_frame = frame
        else:
            # When paused, use the last captured frame
            frame = last_raw_frame
            if frame is None:
                continue

        # Encode frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Display the RAW video feed."""
    return Response(generate_raw_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ------------------------------------------------------
#   OVERLAY FEED
# ------------------------------------------------------
def generate_overlay_feed():
    """Play the video at 60 FPS to match the speed of the RAW video, also call the process_lane_detection() function from here."""
    global playback_active, cap_overlay, last_overlay_frame, frame_count_overlay

    desired_fps = 60
    frame_delay = 1.0 / desired_fps

    while True:
        if playback_active:
            success, frame = cap_overlay.read()
            if not success:
                frame_count_overlay = 0
                cap_overlay.set(cv2.CAP_PROP_POS_FRAMES, frame_count_overlay)
                continue

            # Process the frame (for instance, with your lane detection algorithm)
            output = process_lane_detection(frame)
            last_overlay_frame = output
            frame_count_overlay += 1
        else:
            output = last_overlay_frame
            if output is None:
                continue

        ret, buffer = cv2.imencode('.jpg', output)
        if not ret:
            continue

        time.sleep(frame_delay)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


@app.route('/overlay_feed')
def overlay_feed():
    """Overlay feed now shows the lane detection results in the bottom right grid."""
    return Response(generate_overlay_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
