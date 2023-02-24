import sys

from utils.pose_utils.pose import Pose, Pushup, Plank, Squat, Jumpingjack, Pullup, Curl, Abdominal
from utils.video_reader_utils import VideoReader

class Exercise():
  """ Toplevel class for exercises """
  def __init__(self, filename: str, exercise: str, out: str) -> None:
    self.video_reader = VideoReader(filename)
    if exercise == "predict":
      exercise = "pose"
    self.exercise = exercise.lower().capitalize()
    self.out = out

  def estimate_exercise(self):
    """ Run estimator """
    pose_estimator = getattr(sys.modules[__name__], self.exercise)
    pose_estimator = pose_estimator(self.video_reader, self.out)
    pose_estimator.estimate() if self.exercise == "Pose" else pose_estimator.measure()
