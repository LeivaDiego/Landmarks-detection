# ========== Hand Tracking with MediaPipe Solution (LEGACY API) ==========
# ===== Import Required Libraries =====
# Argparse for command line argument parsing
import argparse
# OpenCV imports
import cv2
# Mediapipe mp solutions imports for hand tracking
import mediapipe.python.solutions.hands as mp_hands
# MediaPipe Drawing modules for visualization
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

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

    Raises:
        AttributeError: If the detection result is malformed or missing required attributes.
        IndexError: If the detection result does not contain expected data.
        TypeError: If the input frame is not a valid image.
        Exception: For any other exceptions that may occur during processing.
    """
    try:
        # Get the height and width of the frame for accurate text placement
        height, width = rgb_frame.shape[:2]

        # Draw hand landmarks and connections on the frame
        for hand_landmarks, handedness in zip(detection_result.multi_hand_landmarks, detection_result.multi_handedness):
            # Draw landmarks and connections for each hand
            mp_drawing.draw_landmarks(
                image=rgb_frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
            )
        
            # Get the top left corner of the bounding box for the detected hand
            x_coordinates = [landmark.x for landmark in hand_landmarks.landmark]    # Get the x-coordinates of the landmarks
            y_coordinates = [landmark.y for landmark in hand_landmarks.landmark]    # Get the y-coordinates of the landmarks
            text_x = int(min(x_coordinates) * width)                                # Calculate the x-coordinate for the text 
            text_y = int(min(y_coordinates) * height) - MARGIN                      # Calculate the y-coordinate for the text

            # Get the handedness label (left/right) and draw it on the frame
            label = handedness.classification[0].label  # 'Left' or 'Right'
            cv2.putText(
                rgb_frame,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE,
                HANDEDNESS_TEXT_COLOR,
                FONT_THICKNESS,
                cv2.LINE_AA,
            )

    # Handle exceptions related to the detection result
    except (AttributeError, IndexError, TypeError) as e:
        # Handle exceptions related to the detection result
        print(f"ERROR: Failed to draw landmarks due to malformed data - {e}")

    except Exception as e:
        # Handle any other exceptions
        print(f"EXCEPTION: {e}")

    return rgb_frame


def run_hand_tracker(max_hands=2):
    """
    Run hand tracking on the webcam feed using MediaPipe Hands solution.
    
    This function captures video from the webcam, processes each frame to detect hands,
    and draws landmarks and handedness labels on the frame.

    The application runs in a loop until the user presses 'q' or 'ESC' to exit.

    MediaPipe Hands solution is a legacy API for hand tracking.

    Args:
        max_hands (int): Maximum number of hands to detect. Default is 2.
    
    Returns:
        None

    Raises:
        RuntimeError: If the webcam is not accessible or any other runtime error occurs.
        KeyboardInterrupt: If the user interrupts the application using Ctrl+C.
        Exception: If any other exception occurs during the execution.
    """
    print(f"STARTED: Hand tracking application running (max hands: {max_hands}).")
    print("INFO: Press 'q' or 'ESC' to exit.")

    # Initialize Video Capture object to capture video from the webcam
    cap = cv2.VideoCapture(index=0)
    try:
        # Check if the webcam is opened successfully
        if not cap.isOpened():
            raise RuntimeError("ERROR: Unable to access the webcam. Make sure it's connected and not in use by another app.")
        
        # Configure Mediapipe Hands module
        with mp_hands.Hands(
            model_complexity=1,
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        ) as hands:
            
            # Initialize Mediapipe Drawing module
            while cap.isOpened():
                # Capture frame-by-frame from the webcam
                success, frame = cap.read()
                if not success:
                    print("WARNING: Empty camera frame, ignored...")
                    continue
                
                frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a mirror effect

                # Check the frame for hands
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB
                results = hands.process(frame_rgb)  # Process the frame and detect hands    

                # Draw the hand annotations on the image
                if results.multi_hand_landmarks and results.multi_handedness:
                    frame = draw_landmarks(frame, results)


                # Display the frame with hand landmarks and handedness labels
                cv2.imshow("Hand Tracking", frame)

                # Check for 'q' or 'ESC' key press to exit the loop
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    print("INFO: Exiting the application...")
                    break

    # Handle exceptions during the hand tracking process
    except KeyboardInterrupt:
        # Handle keyboard interrupt (Ctrl+C) gracefully
        print("INFO: Hand tracking application interrupted by user.")
    except RuntimeError as e:
        # Handle runtime errors (e.g., webcam not accessible)
        print(f"RUNTIME-ERROR: {e}")
    except Exception as e:
        # Handle any other exceptions
        print(f"EXCEPTION: {e}")

    # Ensure that the VideoCapture object is released and all OpenCV windows are closed
    finally:
        # Release the VideoCapture object and close all OpenCV windows
        cap.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()

        print("TERMINATED: Hand tracking application closed.")



def parse_args():
    """
    Parse command line arguments for the hand tracking application.

    Returns:
        Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Hand Tracking with MediaPipe")
    parser.add_argument(
        "--hands",
        type=int,
        default=2,
        help="Maximum number of hands to detect (1 to 10) [default: 2]"
    )
    return parser.parse_args()


# Main function to run the hand tracking on webcam
if __name__ == "__main__":
    
    try:
        # Parse command line arguments
        args = parse_args()
    
        # Validate the number of hands argument
        if 1 <= args.hands <= 10:
            # Run hand tracking with the specified number of hands
            run_hand_tracker(max_hands=args.hands)
        
        else:
            # If invalid, raise an error
            raise ValueError("Number of hands must be between 1 and 10.")
            
    except ValueError as e:
        print(f"VALUE-ERROR: {e}")