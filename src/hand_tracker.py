# MediaPipe Hand Tracking task modules
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# OpenCV imports for video capture and display
import cv2


# Define the model path for hand tracking
model_path = 'src/model/hand_landmarker.task'

# Create a HandLandmarker object 
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Define a callback function to handle the results
# This function will be called when the hand landmarks are detected
def print_result(result, output_image, timestamp_ms):
    print('Hand landmarks result:', result)


# Create a hand landmarker instance with the live stream mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result
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

    # Show the frame in a window
    cv2.imshow('Hand Tracking', frame)

    # Check for 'q' or 'ESC' key press to exit the loop
    # or if the X button is clicked on the window
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

# Release the webcam and close all OpenCV windows
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()


