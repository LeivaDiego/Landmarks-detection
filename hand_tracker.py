# ========== Hand Tracking with MediaPipe Tasks (IMAGE MODE) ==========
# ===== Import Required Libraries =====
# MediaPipe Hand Tracking task modules
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
# MediaPipe Drawing modules for visualization
from mediapipe.framework.formats import landmark_pb2
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
# OpenCV import for video capture and display
import cv2
# NumPy import for numerical operations
import numpy as np
# Argparse for command line argument parsing
import argparse

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
		IndexError: If the detection result contains no hands.
		TypeError: If the detection result is not of the expected type.
		Exception: If any other unexpected error occurs.
	"""
	try:
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

	# Exception handling for various errors
	except (AttributeError, IndexError, TypeError) as e:
		# Handle the case where the detection result is malformed or missing required attributes
		print(f"ERROR: Failed to draw landmarks due to malformed data - {e}")
	except Exception as e:
		# Handle any other exceptions gracefully
		print(f"ERROR: An unexpected error occurred - {e}")

	# Return the annotated frame with landmarks and handedness text
	return annotated_frame


def init_hand_landmarker(model_path, max_hands=2):
	"""
	Initializes the HandLandmarker object with the specified model path.

	Args:
		model_path (str): Path to the hand landmarker model file.

	Returns:
		HandLandmarker: Initialized HandLandmarker object.

	Raises:
		FileNotFoundError: If the model file is not found.
		Exception: If the hand landmarker initialization fails.
	"""
	try:
		# Create a HandLandmarker object 
		BaseOptions = mp.tasks.BaseOptions
		HandLandmarker = mp.tasks.vision.HandLandmarker
		HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
		VisionRunningMode = mp.tasks.vision.RunningMode

		# Create a hand landmarker instance with the live stream mode:
		options = HandLandmarkerOptions(
		base_options=BaseOptions(model_asset_path=model_path),  # Path to the model file
		running_mode=VisionRunningMode.VIDEO,                   # Set the running mode to image
		min_hand_detection_confidence=0.5,                      # Minimum confidence for hand detection
		min_hand_presence_confidence=0.5,                       # Minimum confidence for hand presence
		min_tracking_confidence=0.5,                            # Minimum confidence for hand tracking
		num_hands=max_hands                                     # Number of hands to detect (1 or 2)
		)

		# Create a HandLandmarker instance with the specified options
		hand_landmarker = HandLandmarker.create_from_options(options)
	
	# Handle exceptions during the initialization process
	except FileNotFoundError as e:
		# Handle the case where the model file is not found
		print(f"ERROR: Model file not found - {e}")
		
	except Exception as e:
		# Handle any exceptions that occur during initialization
		print(f"ERROR: Failed to initialize HandLandmarker - {e}")
		
	# Return the initialized hand landmarker
	return hand_landmarker


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



def run_hand_tracker(max_hands=2):
	"""
	Main function to run the hand tracking application using MediaPipe Tasks.

	This function captures video from the webcam, processes each frame to detect hands,
	and draws the detected landmarks and handedness on the frame.

	The application runs in a loop until the user presses 'q' or 'ESC' to exit.

	MediaPipe Tasks is the latest high-level API for hand tracking.

	Args:
		max_hands (int): Maximum number of hands to detect (1 to 10). Default is 2.

	Raises:
		RuntimeError: If the webcam is not accessible or any other runtime error occurs.
		KeyboardInterrupt: If the user interrupts the application using Ctrl+C.
		Exception: If any other unexpected error occurs during hand tracking.
	"""   
	print(f"STARTED: Hand tracking application running (max hands: {max_hands}).")
	print("INFO: Press 'q' or 'ESC' to exit.")

	try:
		# Initialize the HandLandmarker with the specified model path and max hands
		hand_landmarker = init_hand_landmarker(model_path='models/hand_landmarker.task', max_hands=max_hands)

		# Initialize the webcam video capture
		cap = cv2.VideoCapture(0)

		# Check if the webcam is opened successfully
		if not cap.isOpened():
			raise RuntimeError("ERROR: Unable to access the webcam. Make sure it's connected and not in use by another app.")
		

		# Loop to continuously capture frames from the webcam
		# and process them with the hand landmarker
		while cap.isOpened():
			# Capture a frame from the webcam
			ret, frame = cap.read()

			# Check if the frame was captured successfully
			if not ret:
				print("WARNING: Empty camera frame, ignored...")
				continue

			# Convert the BGR frame to RGB format for processing
			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# Process the RGB frame to MediaPipe Image format
			mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

			
			# Get the current timestamp in milliseconds
			frame_timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
			# Process the frame with the hand landmarker and
			# detect hands and landmarks in the frame
			result = hand_landmarker.detect_for_video(mp_frame, frame_timestamp)

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

	# Handle exceptions during the pose tracking process
	except KeyboardInterrupt:
		# Handle keyboard interrupt gracefully
		print("INFO: Hand tracking application interrupted by user.")
	except RuntimeError as e:
		# Handle runtime errors gracefully
		print(f"ERROR: Runtime error occurred - {e}")
	except Exception as e:
		# Handle any other exceptions gracefully
		print(f"ERROR: An unexpected error occurred - {e}")

	# Ensure that the video capture and writer objects are released properly
	finally:
		cap.release()		# Release capture device (webcam)
		cv2.destroyAllWindows()		# Close all OpenCV windows
		print("TERMINATED: Hand tracking application closed.")




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
		# Handle value errors gracefully
		print(f"VALUE-ERROR: {e}")
