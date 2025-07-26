import os
import mediapipe as mp
import cv2 as cv
import numpy as np
from dotenv import find_dotenv, load_dotenv
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils, pose_connections
from camera_capture import live_capture


load_dotenv(find_dotenv())
model_path = os.getenv("PATH_TO_MODEL_LIGHT")


# Constants for 
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
PoseLandmarkerResult = vision.PoseLandmarkerResult
PoseLandmarkerConnections = pose_connections.POSE_CONNECTIONS
VisionRunningMode = vision.RunningMode

# 
class pose_stream:
  def __init__(self) -> None:
    self.options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=self.draw_landmark_live)
    self.image = None

  def draw_landmark_live(self, result, output_image , timestamp_ms) -> None:
    landmark_list = result.pose_landmarks

    if result.pose_landmarks:
      try:
        self.image = np.copy(output_image.numpy_view())
        for idx in range(len(landmark_list)):
          pose_landmarks = landmark_list[idx]

          pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList() # type: ignore
          pose_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark( # pyright: ignore[reportAttributeAccessIssue]
            x=landmark.x,
            y=landmark.y, 
            z=landmark.z, 
            visibility=landmark.visibility,
            presence=landmark.presence) for landmark in pose_landmarks
          ])
          drawing_utils.draw_landmarks(
            image= self.image,
            landmark_list= pose_landmarks_proto,
            connections= list(PoseLandmarkerConnections)
            )
      except Exception as e:
        print(e)
  

ps = pose_stream()

with PoseLandmarker.create_from_options(ps.options) as landmarker:
  cam = live_capture()
  for mp_image, mp_timestamp, mp_frame in cam.stream():
    landmarker.detect_async(mp_image,int(mp_timestamp))

    d_image = mp_frame
    if ps.image is not None:
      d_image = ps.image

    cv.imshow('frame', d_image)
    if cv.waitKey(1) == ord('q'):
      break