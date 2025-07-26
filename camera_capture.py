import numpy as np
import cv2 as cv
import mediapipe as mp

class live_capture:
  def __init__(self,width: int = 1280, height: int = 720) -> None:
    self.cap = cv.VideoCapture(0)
    self.width = width
    self.height = height
    self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
    self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
    self.mp_image = None
  
  def stream(self):
    if not self.cap.isOpened():
      print("Cannot open camera")
      raise Exception("Shit wrong")

    while True:
      ret, frame = self.cap.read()

      if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


      self.mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(frame))
      yield self.mp_image, self.cap.get(cv.CAP_PROP_POS_MSEC), frame
      # cv.imshow('frame', frame)
      # if cv.waitKey(1) == ord('q'):
      #   break

    self.cap.release()
    cv.destroyAllWindows

if __name__ == "__main__":
  cam = live_capture()
  for mp_image, mp_timestamp, mp_frame in cam.stream():
    cv.imshow('frame', mp_frame)
    print(mp_image,mp_timestamp)
    if cv.waitKey(1) == ord('q'):
      break