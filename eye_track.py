import numpy as np
import cv2
import dlib
import collections

# Constants
DELAY_SECONDS = 0.2
ZOOM_FACTOR = 2
CENTER_THRESHOLD = 1.0
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1
TEXT_COLOR = (255, 255, 255)
TEXT_THICKNESS = 2
TEXT_POSITION = (10, 50)
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 500

# Face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Capture video from mac webcam
cap = cv2.VideoCapture(0)
FPS = int(cap.get(cv2.CAP_PROP_FPS))
BUFFER_SIZE = int(FPS * DELAY_SECONDS)

# Create deques for frames and text for delay
frame_buffer = collections.deque(maxlen=BUFFER_SIZE)
text_buffer = collections.deque(maxlen=BUFFER_SIZE)

# Functions
def detect_faces_and_landmarks(gray_frame):
    """
    Uses the detector to find landmarks.
    """
    faces = detector(gray_frame)
    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(gray_frame, face)
        return landmarks
    return None


def process_eye_region(landmarks):
    """
    Processes the eye region to return the thresholded (black & white) eye image,
    and relative x-position of the iris.
    """
    # Collect the points (landmarks) that outline the right eye (36-41).
    right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])

    # Create a mask with the same dimensions as the frame
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [right_eye_points], 255)

    # Extract and crop the eye region
    eye_region = cv2.bitwise_and(gray, gray, mask=mask)
    x, y, w, h = cv2.boundingRect(right_eye_points)
    eye_region_cropped = eye_region[y:y + h, x:x + w]

    # Threshold based on median intensity
    median_intensity = np.median(eye_region_cropped[eye_region_cropped > 0])
    threshold_value = max(0, int(median_intensity))
    _, thresholded_eye = cv2.threshold(eye_region_cropped, threshold_value, 255, cv2.THRESH_BINARY)

    # Collect dark pixels in the thresholded eye region
    dark_pixels = np.column_stack(np.where(thresholded_eye == 0))

    # Find the "center" of the dark pixels
    if dark_pixels.size > 0:
        mean_dark_x = np.mean(dark_pixels[:, 1])  # x-coordinates are in the second column
        center_x = w / 2
        relative_x_position = mean_dark_x - center_x
        return thresholded_eye, relative_x_position
    return thresholded_eye, None


def estimate_gaze(relative_x_position):
    """
    Estimates the location of gaze based on the relative x-position of the iris.
    """
    if relative_x_position is None:
        return "Unknown"
    elif abs(relative_x_position) <= CENTER_THRESHOLD:
        return "Center"
    elif relative_x_position > CENTER_THRESHOLD:
        return "Left"
    else:
        return "Right"


def draw_lines_on_eye(eye_region, mean_dark_x, center_x, h):
    """
    Draws lines on eye to visualize estimated iris location.
    """
    # Draw a vertical line at the center of the eye region
    cv2.line(eye_region, (int(center_x), 0), (int(center_x), h), (0, 255, 0), 1)

    # Draw a vertical line at the mean x-position of the dark pixels
    if mean_dark_x is not None:
        cv2.line(eye_region, (int(mean_dark_x), 0), (int(mean_dark_x), h), (255, 0, 0), 1)


def update_buffers(zoomed_eye, location_text):
    """
    Updates buffers with delayed frames and related text.
    """
    frame_buffer.append(zoomed_eye)
    text_buffer.append(location_text)


def display_delayed_output():
    """
    Displays the delayed video and text output.
    """
    if len(frame_buffer) == BUFFER_SIZE:
        delayed_zoomed_eye = frame_buffer.popleft()
        delayed_text = text_buffer.popleft()

        # Display delayed video
        cv2.imshow('Delayed Zoomed Eye', delayed_zoomed_eye)

        # Display delayed text
        text_size = cv2.getTextSize(delayed_text, TEXT_FONT, TEXT_SCALE, TEXT_THICKNESS)[0]
        text_window = np.zeros((100, text_size[0] + 20, 3), dtype=np.uint8)
        cv2.putText(text_window, delayed_text, TEXT_POSITION, TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
        cv2.imshow('Direction', text_window)

        # Get the size of the displayed image, and put the windows in the center of the screen
        window_width = delayed_zoomed_eye.shape[1]
        window_height = delayed_zoomed_eye.shape[0]
        window_x = (SCREEN_WIDTH - window_width) // 2
        window_y = (SCREEN_HEIGHT - window_height) // 2
        cv2.moveWindow('Delayed Zoomed Eye', window_x, window_y)
        cv2.moveWindow('Direction', window_x, window_y - 100)


# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and landmarks
    landmarks = detect_faces_and_landmarks(gray)
    if landmarks is None:
        continue

    # Process the eye region and calculate relative x-position
    thresholded_eye, relative_x_position = process_eye_region(landmarks)

    # Estimate gaze direction (Left, Right, Center)
    gaze_text = estimate_gaze(relative_x_position)

    # Zoom in on the thresholded eye region
    zoomed_thresholded_eye = cv2.resize(thresholded_eye, (0, 0), fx=ZOOM_FACTOR, fy=ZOOM_FACTOR)

    # Update buffers with the zoomed frame and gaze text
    update_buffers(zoomed_thresholded_eye, gaze_text)

    # Display delayed output
    display_delayed_output()

    # Break the loop when ESC is pressed
    if cv2.waitKey(1) == 27:
        break

# When finished
cap.release()
cv2.destroyAllWindows()
