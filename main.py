import pose_detection as pd
from camera_capture import CameraCapture
import cv2 as cv

if __name__ == "__main__":
    LandmarkRunner = pd.PoseLandmarkLiveRunner()
    Camera = CameraCapture()

    with pd.PoseLandmarker.create_from_options(LandmarkRunner.options) as landmarker:
        for mp_image, mp_timestamp, mp_frame in Camera.stream():
            landmarker.detect_async(mp_image,int(mp_timestamp))

            if LandmarkRunner.results:
                LandmarkRunner.draw_landmarks(mp_frame, LandmarkRunner.results[-1][1])

            cv.imshow("Auto Rep",mp_frame)

            if cv.waitKey(1) == ord('q'):
                exit()

            
    