# ========== Hand Tracking with MediaPipe Tasks (IMAGE MODE) ==========
# ===== Import Required Libraries =====
# MediaPipe Hand Tracking task modules
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
# MediaPipe Drawing modules for visualization
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
# OpenCV import for video capture and display
import cv2
# NumPy import for numerical operations
import numpy as np
# Time import for timestamping
import time

# ===== Constants for Visualization =====
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green


# ===== Hand Tracking with MediaPipe =====

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
    
    # Flip the frame horizontally for a mirror effect
    annotated_frame = cv2.flip(annotated_frame, 1)
    # Get the height and width of the frame
    height, width = annotated_frame.shape[:2]                       

    # Loop through each detected hand and visualize the handedness text
    for idx in range(len(hand_landmarks_list)):
        # Get the landmarks and handedness for the current hand
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        
        # Get the top left corner of the bounding box for the detected hand
        x_coordinates = [landmark.x for landmark in hand_landmarks]     # Get the x-coordinates of the landmarks
        y_coordinates = [landmark.y for landmark in hand_landmarks]     # Get the y-coordinates of the landmarks
        text_x = int(max(x_coordinates) * width)                        # Calculate the x-coordinate for the text   
        text_y = int(min(y_coordinates) * height) - MARGIN              # Calculate the y-coordinate for the text
        
        # Invert the x-coordinate for the mirror effect
        text_x = width - text_x - MARGIN

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


def run_hand_tracker():
    """
    Main function to run the hand tracking application using MediaPipe Tasks.
    
    This function captures video from the webcam, processes each frame to detect hands,
    and draws the detected landmarks and handedness on the frame.

    The application runs in a loop until the user presses 'q' or 'ESC' to exit.

    MediaPipe Tasks is the latest high-level API for hand tracking.
    """   
    # Define the model path for hand tracking
    model_path = 'model/hand_landmarker.task'

    # Create a HandLandmarker object 
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a hand landmarker instance with the live stream mode:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),  # Path to the model file
        running_mode=VisionRunningMode.IMAGE,                   # Set the running mode to image
        min_hand_detection_confidence=0.3,                      # Minimum confidence for hand detection
        min_hand_presence_confidence=0.3,                       # Minimum confidence for hand presence
        min_tracking_confidence=0.3,                            # Minimum confidence for hand tracking
        num_hands=2                                             # Number of hands to detect (1 or 2)
       )                                                        # no callback function for image mode

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

            # Process the frame with the hand landmarker
            # Detect hands and landmarks in the frame
            result = landmarker.detect(mp_frame)

            if result is not None:
                # Draw the landmarks and handedness on the frame
                annotated_frame = draw_landmarks(rgb_frame, result)
                # Convert the annotated frame back to BGR format for OpenCV display
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                # Display the annotated frame in a window
                cv2.imshow("Hand Tracking", annotated_frame)
            else:
                # If no hands are detected, display the original frame
                cv2.imshow("Hand Tracking", frame)

            # Check for 'q' or 'ESC' key press to exit the loop
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

    # Release the webcam and close all OpenCV windows
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()



# Main function to run the hand tracking on webcam
if __name__ == "__main__":
    run_hand_tracker()
