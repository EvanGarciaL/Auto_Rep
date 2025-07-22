import os
import mediapipe as mp
from dotenv import find_dotenv, load_dotenv
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

load_dotenv(find_dotenv())
model_path = os.getenv("PATH_TO_MODEL")


BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarkerResult = vision.PoseLandmarkerResult
VisionRunningMode = vision.RunningMode

# Create a pose landmarker instance with the live stream mode:
def print_result(result, output_image, timestamp_ms):
    print('pose landmarker result: {}'.format(result))

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with PoseLandmarker.create_from_options(options) as landmarker:
  landmarker.detect_async()