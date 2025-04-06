# OpenCV imports
import cv2

# Mediapipe mp solutions imports for hand tracking (Legacy API)
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

def run_hand_tracking_on_webcam():
    """
    Run hand tracking on webcam feed.
    This function captures video from the webcam and uses Mediapipe to detect and draw hand landmarks.
    It displays the video feed with hand landmarks drawn on it.
    The function will exit when the 'q' key is pressed.

    Args:
        None

    Returns:
        None
    """
    # Initialize Video Capture object to capture video from the webcam
    cap = cv2.VideoCapture(index=0)

    # Configure Mediapipe Hands module
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        # Initialize Mediapipe Drawing module
        while cap.isOpened():
            # Capture frame-by-frame from the webcam
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame...")
                continue

            # Check the frame for hands
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB
            results = hands.process(frame_rgb)  # Process the frame and detect hands    

            # Draw the hand annotations on the image
            if results.multi_hand_landmarks:
                # Draw hand landmarks and connections on the frame
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks and connections for each hand
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )

            # Flip the frame horizontally for a later selfie-view display
            # and convert the BGR frame to RGB for display
            cv2.imshow("Hand Tracking", cv2.flip(frame, 1))

            # Wait for 1 ms and check if 'q' key is pressed to exit the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    # Release the VideoCapture object and close all OpenCV windows
    cap.release()

# Main function to run the hand tracking on webcam
if __name__ == "__main__":
    run_hand_tracking_on_webcam()