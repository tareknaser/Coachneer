import argparse

from utils.exercise_utils import Exercise


VIDEOS_OUT_PATH = 'data/videos_out'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-v', "--video", 
                        required=False, 
                        default=0,
                        help="Path to video source file", 
                        type=str)

    parser.add_argument('-e', "--exercise", 
                        required=False, 
                        default="bicep_curl",
                        help="Type of exercise in video source",
                        type=str, 
                        choices=['bicep_curl', 'squat', 'plank', 'abdominal'])

    parser.add_argument("-f", "--filename", 
                        required=False,
                        default=f'excercise-out',
                        help="Name for video output file (without extension)",
                        type=str)

    args = parser.parse_args()

    exercise = Exercise(args.video, args.exercise, args.filename)
    exercise.estimate_exercise()

if __name__ == '__main__':
    main()