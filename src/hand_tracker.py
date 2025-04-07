# ===== Import Required Libraries =====
# MediaPipe Hand Tracking task modules
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
# MediaPipe Drawing modules for visualization
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

# OpenCV import for video capture and display
import cv2
# NumPy import for numerical operations
import numpy as np

# ===== Constants for Visualization =====
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# ===== Hand Tracking with MediaPipe =====

# Define the model path for hand tracking
model_path = 'src/model/hand_landmarker.task'

# Create a HandLandmarker object 
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


def draw_landmarks(rgb_frame, detection_result):
    """
    Draws hand landmarks and handedness on the given RGB frame.

    Args:
        rgb_frame (numpy.ndarray): The RGB frame to draw on.
        detection_result (HandLandmarkerResult): The result of hand landmark detection.

    Returns:
        numpy.ndarray: The annotated frame with landmarks and handedness text.
    """
    hand_landmarks_list = detection_result.hand_landmarks    # List of detected hands with landmarks
    handedness_list = detection_result.handedness            # List of detected hands with handedness (left/right)
    
    # Initialize a new image to draw the landmarks on, based on the original RGB frame
    annotated_frame = np.copy(rgb_frame)
    
    # Loop through each detected hand and visualize the landmarks
    for idx in range(len(hand_landmarks_list)):
        # Get the landmarks and handedness for the current hand
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Create a new normalized landmark list for the current hand
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()    

        # Extend the landmark list with the detected landmarks within the hand
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x, 
                y=landmark.y, 
                z=landmark.z
            ) for landmark in hand_landmarks
        ])
        
        # Draw the landmarks on the annotated frame using the MediaPipe drawing module
        mp_drawing.draw_landmarks(
            annotated_frame,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Get the top left corner of the bounding box for the detected hand
        height, width = annotated_frame.shape                           # Get the height and width of the frame
        x_coordinates = [landmark.x for landmark in hand_landmarks]     # Get the x-coordinates of the landmarks
        y_coordinates = [landmark.y for landmark in hand_landmarks]     # Get the y-coordinates of the landmarks
        text_x = int(min(x_coordinates) * width)                        # Calculate the x-coordinate for the text   
        text_y = int(min(y_coordinates) * height) - MARGIN              # Calculate the y-coordinate for the text

        # Draw the handedness text on the annotated frame
        cv2.putText(
            annotated_frame,                # Image to draw on
            handedness[0].category_name,    # Text to display (left/right hand)
            (text_x, text_y),               # Position of the text
            cv2.FONT_HERSHEY_DUPLEX,        # Font type
            FONT_SIZE,                      # Font size
            HANDEDNESS_TEXT_COLOR,          # Text color
            FONT_THICKNESS,                 # Font thickness
            cv2.LINE_AA                     # Line type
        )

    # Return the annotated frame with landmarks and handedness text
    return annotated_frame


def process_result(result, output_frame, timestamp_ms):
    # Draw the landmarks on the output frame
    annotated_frame = draw_landmarks(output_frame.numpy_view(), result)
    # Convert the RGB annotated frame to BGR format for OpenCV display
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    # Display the annotated frame in a window
    cv2.imshow('Hand Tracking', annotated_frame_bgr)

    
# Create a hand landmarker instance with the live stream mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=process_result
    )

# Initialize the webcam video capture
cap = cv2.VideoCapture(0)

# Initialize the HandLandmarker with the specified options
with HandLandmarker.create_from_options(options) as landmarker:
  # Loop to continuously capture frames from the webcam
  # and process them with the hand landmarker
  while cap.isOpened():
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Failed to capture frame from webcam.")
        break
    
    # Convert the BGR frame to RGB format for processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame to MediaPipe Image format
    mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Get the current timestamp in milliseconds
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

    # Send the frame data to perform hand landmark detection
    # This will call the callback function print_result when the result is ready
    landmarker.detect_async(mp_frame, timestamp_ms)

    # Check for 'q' or 'ESC' key press to exit the loop
    # or if the X button is clicked on the window
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

# Release the webcam and close all OpenCV windows
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()


