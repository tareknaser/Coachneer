import sys

from utils.pose_utils.pose import Pose, BicepCurl, Squat
from utils.video_reader_utils import VideoReaderUtils


class Exercise:
    """Top-level class for exercises."""
    def __init__(self, filename, exercise: str, OUT_PATH: str) -> None:
        self.video_in = VideoReaderUtils(filename)
        self.exercise = exercise.lower().title().replace('_', '')
        self.video_out = OUT_PATH

    def estimate_exercise(self):
        """Run estimator."""
        exercise_estimator = getattr(sys.modules[__name__], self.exercise)
        exercise_estimator = exercise_estimator(self.video_in, self.video_out)
        exercise_estimator.measure()
