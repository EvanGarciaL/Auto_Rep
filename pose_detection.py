import os
import mediapipe as mp
from dotenv import find_dotenv, load_dotenv
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils, pose_connections
from collections import deque

#Load model paths from .env
load_dotenv(find_dotenv())
model_path = os.getenv("PATH_TO_MODEL_FULL")


# Configurations, Objects and Constants for dealing with MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarkerResult = vision.PoseLandmarkerResult
# Turn frozenset to list
PoseLandmarkerConnections = list(pose_connections.POSE_CONNECTIONS) 
VisionRunningMode = vision.RunningMode


class PoseLandmarkLiveRunner:
  def __init__(self) -> None:
    self.options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=self._callback_function)
    self.landmarks = None
    self.results = deque(maxlen=1) 
    # Deque with a maxlen of one, keeping only the most recent landmarks and removing the rest
    # Deque is a list-like container similar to stacks and queues
    # Can append and pop from either end (FIFO and LIFO)
    # Maxlen sets a maximum length, automatically discarding items from the opposite end.

  def _callback_function(self, result, output_image , timestamp_ms) -> None:
    self.results.append((timestamp_ms, result.pose_landmarks))


  def draw_landmarks(self,frame,landmarks) -> None:
      if self.results:
        timestamp, pose_landmarks = self.results[-1]
        try:
          for landmarks in pose_landmarks:

            proto_list = landmark_pb2.NormalizedLandmarkList() #type:ignore 

            proto_list.landmark.extend([
            landmark_pb2.NormalizedLandmark( #type:ignore
              x=lm.x,
              y=lm.y, 
              z=lm.z, 
              visibility=lm.visibility,
              presence=lm.presence) for lm in landmarks 
            ])

            drawing_utils.draw_landmarks(
              image=frame,
              landmark_list= proto_list,
              connections= PoseLandmarkerConnections
              )
        except Exception as e:
          print(e, "\nFrame failed to load")
