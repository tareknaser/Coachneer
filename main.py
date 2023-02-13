import argparse, os

from utils.exercise_utils import Exercise

out_path = 'data/videos/out/'

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('-v', "--video", 
                      required= False, 
                      default= int(0),
                      help= "Path to video source file", 
                      type= str)

  parser.add_argument('-e', "--exercise", 
                      required= False, 
                      default= "predict",
                      help= "Type of exercise in video source",
                      type= str, 
                      choices= ['predict', 'pushup', 'plank', 'squat', 'jumpingjack', 'pullup', 'curl', 'abdominal'])

  parser.add_argument("-o", "--output", 
                      required= False,
                      default= 'output'+ str(len(os.listdir(out_path)) +1),
                      help= "Path to video output file (without extension)",
                      type= str)

  # parser.add_argument("-b", "--black", 
  #                     type= str, 
  #                     default= False,
  #                     help="set black background")

  args = parser.parse_args()
  video = args.video
  exercise = args.exercise
  out = args.output
  pose = Exercise(video, exercise, out)
  
  pose.estimate_exercise()
