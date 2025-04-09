# ========== Pose Tracking with MediaPipe Tasks ==========
# ===== Import Required Libraries =====
# MediaPipe Pose Tracking task modules
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
# MediaPipe Drawing modules for visualization
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
# OpenCV import for video capture and display
import cv2
# NumPy import for numerical operations
import numpy as np


# ===== Pose Tracking with MediaPipe =====

def draw_landmarks(rgb_image, detection_result):
	"""
	Draws pose landmarks on the given RGB frame.

	Args:
		rgb_image (numpy.ndarray): The RGB frame to draw on.
		detection_result (PoseLandmarkerResult): The result of pose landmark detection.

	Returns:
		numpy.ndarray: The annotated frame with pose landmarks.
	"""
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
		
		# Return the annotated frame with landmarks
		return annotated_frame
	

def init_pose_landmarker(model_path='models/pose_landmarker_heavy.task'):
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

	return pose_landmarker


def run_pose_tracker(input_path):
	# Initialize the pose landmarker with the model path
	pose_landmarker = init_pose_landmarker('models/pose_landmarker_heavy.task')
	
	# Initialize the VideoCapture object to capture video from the input path
	cap = cv2.VideoCapture(input_path)
	fps = cap.get(cv2.CAP_PROP_FPS)		# Get the frames per second of the video
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))	# Get the width of the video frame
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))	# Get the height of the video frame

	# Initialize the video writer to save the annotated video
	output_path = 'output/pose_tracking_output.mp4'
	out_writer = cv2.VideoWriter(
		filename=output_path, 						# Output video file path
		fourcc=cv2.VideoWriter_fourcc(*'mp4v'),		# Codec for the output video
		fps=fps,									# Frames per second
		frameSize=(width, height)					# Size of the video frame
		)
	
	frame_idx = 0	# Initialize the frame index for video processing

	# Loop through each frame in the video
	while cap.isOpened():
		# Read the current frame from the video
		success, frame = cap.read()
		
		# Break the loop if the frame is not successfully read
		if not success:
			break

		frame_idx += 1	# Increment the frame index

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
			print(f"WARNING: Frame {frame_idx} - No pose detected.")
			# Write the original frame to the output video if no pose is detected
			out_writer.write(frame)
		else:
			# If pose landmarks are detected, draw them on the frame
			# Draw the detected pose landmarks on the RGB frame
			annotated_frame = draw_landmarks(rgb_frame, detection_result)
			# Convert the annotated frame back to BGR format for OpenCV
			annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
			# Write the annotated frame to the output video
			out_writer.write(annotated_frame_bgr)	

		# Increment the frame index
		frame_idx += 1

	# after processing all frames, release the video capture and writer objects
	cap.release()
	out_writer.release()
	# Display a message indicating the completion of pose tracking
	print(f"SUCCESS: Pose tracking completed. Output saved to {output_path}")
	

if __name__ == "__main__":
	# Run the pose tracker on the input video file
	run_pose_tracker(input_path='input/dancing-test.mp4')