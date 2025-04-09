# ========== Pose Tracking with MediaPipe Tasks ==========
# ===== Import Required Libraries =====
# MediaPipe Pose Tracking task modules
import mediapipe as mp
import mediapipe.python.solutions.pose as mp_pose
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
# OS module for file path operations
import os
# TQDM progress bar for progress indication
from tqdm import tqdm


# ===== Pose Tracking with MediaPipe =====

def draw_landmarks(rgb_image, detection_result):
	"""
	Draws pose landmarks on the given RGB frame.

	Args:
		rgb_image (numpy.ndarray): The RGB frame to draw on.
		detection_result (PoseLandmarkerResult): The result of pose landmark detection.

	Returns:
		numpy.ndarray: The annotated frame with pose landmarks.

	Raises:
		AttributeError: If the detection result is malformed or missing required attributes.
		IndexError: If the detection result does not contain expected data.
		TypeError: If the input frame is not a valid image.
		Exception: For any other exceptions that may occur during processing.
	"""
	try:
		pose_landmarks_list = detection_result.pose_landmarks     # List of detected poses with landmarks
		
		# Initialize a new image to draw the landmarks on, based on the original RGB frame
		annotated_frame = np.copy(rgb_image)

		# Loop through the detected poses to visualize
		for idx in range(len(pose_landmarks_list)):
			# Get the landmarks for the current pose
			pose_landmarks = pose_landmarks_list[idx]

			# Create a new normalized landmark list for the current pose
			pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

			# Extend the landmark list with the detected landmarks within the pose
			pose_landmarks_proto.landmark.extend([
			landmark_pb2.NormalizedLandmark(
				x=landmark.x, 
				y=landmark.y, 
				z=landmark.z) for landmark in pose_landmarks
			])

			# Draw the pose landmarks on the annotated frame using MediaPipe drawing module
			mp_drawing.draw_landmarks(
			annotated_frame,
			pose_landmarks_proto,
			mp_pose.POSE_CONNECTIONS,
			mp_drawing_styles.get_default_pose_landmarks_style())
	
	except (AttributeError, IndexError, TypeError) as e:
		# Handle the case where the detection result is malformed or missing required attributes
		print(f"ERROR: Failed to draw landmarks due to malformed data - {e}")
	except Exception as e:
		# Handle any other exceptions gracefully
		print(f"ERROR: An unexpected error occurred - {e}")

	# Return the annotated frame with landmarks
	return annotated_frame
	

def draw_skeleton(frame_shape, detection_result):
	"""
	Draws the pose skeleton on a black background.

	Args:
		frame_shape (tuple): The shape of the frame (height, width, channels).
		detection_result (PoseLandmarkerResult): The result of pose landmark detection.

	Returns:
		numpy.ndarray: The black frame with pose landmarks drawn on it.

	Raises:
		AttributeError: If the detection result is malformed or missing required attributes.
		IndexError: If the detection result does not contain expected data.
		TypeError: If the input frame shape is not valid.
		Exception: For any other exceptions that may occur during processing.
	"""
	# Initialize a black frame with the same shape as the input frame
	black_frame = np.zeros(frame_shape, dtype=np.uint8)

	try:
		pose_landmarks_list = detection_result.pose_landmarks	# List of detected poses with landmarks
		
		# Loop through the detected poses to visualize
		for idx in range(len(pose_landmarks_list)):
			# Get the landmarks for the current pose
			pose_landmarks = pose_landmarks_list[idx]

			# Create a new normalized landmark list for the current pose
			pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

			# Extend the landmark list with the detected landmarks within the pose
			pose_landmarks_proto.landmark.extend([
			landmark_pb2.NormalizedLandmark(
				x=landmark.x, 
				y=landmark.y, 
				z=landmark.z) for landmark in pose_landmarks
			])

			# Draw the pose landmarks on the annotated frame using MediaPipe drawing module
			mp_drawing.draw_landmarks(
			black_frame,
			pose_landmarks_proto,
			mp_pose.POSE_CONNECTIONS,
			mp_drawing_styles.get_default_pose_landmarks_style())
	
	except (AttributeError, IndexError, TypeError) as e:
		# Handle the case where the detection result is malformed or missing required attributes
		print(f"ERROR: Failed to draw landmarks due to malformed data - {e}")
	except Exception as e:
		# Handle any other exceptions gracefully
		print(f"ERROR: An unexpected error occurred - {e}")
	
	# Return the black frame with landmarks
	return black_frame


def init_pose_landmarker(model_path='models/pose_landmarker_heavy.task'):
	"""
	Initializes the pose landmarker with the specified model path.

	Args:
		model_path (str): The path to the pose landmarker model.

	Returns:
		PoseLandmarker: The initialized pose landmarker object.

	Raises:
		Exception: If the pose landmarker initialization fails.
	"""
	try:
		# Initialize the PoseLandmarker base objects
		BaseOptions = mp.tasks.BaseOptions
		PoseLandmarker = mp.tasks.vision.PoseLandmarker
		PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
		VisionRunningMode = mp.tasks.vision.RunningMode

		# Define the options for the pose landmarker
		options = PoseLandmarkerOptions(
			base_options=BaseOptions(model_asset_path=model_path),		# Path to the model
			running_mode=VisionRunningMode.VIDEO,						# Running mode for video
			num_poses=2,												# Number of poses to detect
			min_pose_detection_confidence=0.5,							# Minimum confidence for pose detection
			min_pose_presence_confidence=0.5,							# Minimum confidence for pose presence
			min_tracking_confidence=0.5,								# Minimum confidence for tracking
			output_segmentation_masks=False								# Whether to output segmentation masks
			)
		
		# Create the pose landmarker with the defined options
		pose_landmarker = PoseLandmarker.create_from_options(options)
	
	# Handle exceptions during the initialization process
	except FileNotFoundError as e:
		# Handle the case where the model file is not found
		print(f"ERROR: Model file not found - {e}")
		
	except Exception as e:
		# Handle any exceptions that occur during initialization
		print(f"ERROR: Failed to initialize pose landmarker - {e}")
	
	return pose_landmarker


def run_pose_tracker(input_path, output_path='output/pose_tracking_output.mp4'):
	"""
	Run pose tracking on the input video file using MediaPipe Pose Landmarker solution.

	This function captures video from the specified input path, processes each frame to detect pose landmarks,
	and saves the annotated video with landmarks drawn on it.

	The application runs in a loop until all frames are processed or the user interrupts the process.

	Args:
		input_path (str): The path to the input video file.
		output_path (str): The path fpr the output video file

	Returns:
		None
	"""
	print(f"STARTED: Pose tracking application running with input: {input_path}.")

	try:
		# Initialize the pose landmarker with the model path
		pose_landmarker = init_pose_landmarker()
		
		# Initialize the VideoCapture object to capture video from the input path
		cap = cv2.VideoCapture(input_path)
		fps = cap.get(cv2.CAP_PROP_FPS)		# Get the frames per second of the video
		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))	# Get the width of the video frame
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))	# Get the height of the video frame
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))	# Get the total frames of the video 

		# Initialize the video writer to save the annotated video
		out_writer = cv2.VideoWriter(
			filename=output_path, 						# Output video file path
			fourcc=cv2.VideoWriter_fourcc(*'MP4V'),		# Codec for the output video
			fps=fps,									# Frames per second
			frameSize=(width * 2, height)				# Size of the video frame (width * 2 for side-by-side display)
			)
		
		frame_idx = 0	# Initialize the frame index for video processing

		# Initialize progress bar
		progress_bar = tqdm(total=total_frames, desc="Porcessing frames", unit="frame")

		# Loop through each frame in the video
		while cap.isOpened():
			# Read the current frame from the video
			success, frame = cap.read()
			
			# Break the loop if the frame is not successfully read
			if not success:
				break

			# Convert the frame to RGB format for pose detection
			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# Convert the RGB frame to a MediaPipe Image object
			mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

			# Compute the frame timestamp in milliseconds
			frame_timestamp = int(frame_idx * 1000 / fps)

			# Detect the pose landmarks in the current frame
			detection_result = pose_landmarker.detect_for_video(mp_image, frame_timestamp)

			# Check if the detection result is valid
			if detection_result is None:
				# If no pose landmarks are detected
				tqdm.write(f"WARNING: Frame {frame_idx} - No pose detected.")
				# Write the original frame to the output video if no pose is detected
				out_writer.write(frame)
			else:
				# If pose landmarks are detected
				# Draw the pose landmarks on the original frame 
				annotated_frame = draw_landmarks(rgb_frame, detection_result)
				# Convert the annotated frame back to BGR format for OpenCV
				annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

				# Draw the pose skeleton on a black background
				black_frame = draw_skeleton(frame.shape, detection_result)
				# Convert the black frame to BGR format for OpenCV
				black_frame_bgr = cv2.cvtColor(black_frame, cv2.COLOR_RGB2BGR)

				# Combine the annotated frame and the black frame for visualization
				combined_frame = np.hstack((annotated_frame_bgr, black_frame_bgr))

				# Write the combined frame to the output video
				out_writer.write(combined_frame)

			# Update progress bar
			progress_bar.update(1)
			# Increment the frame index
			frame_idx += 1

	# Handle exceptions during the pose tracking process
	except KeyboardInterrupt:
		# Handle keyboard interrupt gracefully
		print("INFO: Exiting the application...")
	except RuntimeError as e:
		# Handle runtime errors gracefully
		print(f"ERROR: Runtime error occurred - {e}")
	except Exception as e:
		# Handle any other exceptions gracefully
		print(f"ERROR: An unexpected error occurred - {e}")
	
	# Ensure that the video capture and writer objects are released properly
	finally:
		cap.release()			
		out_writer.release() 	# Release video Writer
		cv2.destroyAllWindows()	# Close all OpenCV windows
		progress_bar.close()	# Close the progress bar
		# Display a message indicating the completion of pose tracking
		print(f"SUCCESS: Pose tracking completed. Output saved to {output_path}")
		print(f"TERMINATED: Pose tracking application closed.")
	

def parse_args():
	"""
	Parse command line arguments for the pose tracking application.

	Returns:
		Namespace: Parsed command line arguments.
	"""
	parser = argparse.ArgumentParser(description="Pose Tracking with MediaPipe")
	parser.add_argument(
		"--input",
		type=str,
		default='input/dancing-test.mp4',
		help="Path to the input video file."
	)
	parser.add_argument(
		"--output",
		type=str,
		default='output/pose_tracking_output.mp4',
		help="Path to the output video file."
	)
	return parser.parse_args()


if __name__ == "__main__":
	try:
		# Parse command line arguments
		args = parse_args()

		# === INPUT VALIDATION ===
		if not args.input:
			raise ValueError("ERROR: Input video path cannot be empty.")

		if '\\' in args.input:
			args.input = args.input.replace('\\', '/')

		if os.path.isdir(args.input):
			raise ValueError("ERROR: Input video path cannot be a directory.")

		if not os.path.isfile(args.input):
			raise ValueError("ERROR: Input video path is not a valid file.")

		if not args.input.lower().endswith('.mp4'):
			raise ValueError("ERROR: Input video must be an MP4 file.")

		# === OUTPUT VALIDATION ===
		if not args.output:
			raise ValueError("ERROR: Output video path cannot be empty.")

		if '\\' in args.output:
			args.output = args.output.replace('\\', '/')

		if os.path.isdir(args.output):
			raise ValueError("ERROR: Output path cannot be a directory.")

		if not args.output.lower().endswith('.mp4'):
			raise ValueError("ERROR: Output path must end with '.mp4'.")

		output_dir = os.path.dirname(args.output)
		if not output_dir:
			raise ValueError("ERROR: Output path must include a directory (e.g., output/video.mp4).")

		# Create output directory if it doesn't exist
		if not os.path.exists(output_dir):
			print(f"INFO: Output directory '{output_dir}' does not exist. Creating it...")
			os.makedirs(output_dir)

		# Run pose tracking
		run_pose_tracker(args.input, args.output)

	except ValueError as e:
		print(f"VALUE-ERROR: {e}")