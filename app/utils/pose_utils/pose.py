import os
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from pathlib import Path
from utils.operation_utils import OperationUtils
from utils.drawing_utils import DrawingUtils
from utils.pose_utils.const import POSE_CONNECTIONS, NORM_POSE_CONNECTIONS


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence= 0.5, min_tracking_confidence= 0.5)

# Initialize paths
VIDEOS_OUT_PATH = 'data/videos_out/'
MODEL_PATH = 'data/models'

def get_output_filename(filename) -> str:
  """ Generate an output filename. """
  output_filename = Path(VIDEOS_OUT_PATH) / f'{filename}{len(os.listdir(VIDEOS_OUT_PATH)) +1}.mp4'
  return str(output_filename)


class Pose:
  """ Base: Pose Class """
  def __init__(self, video_in, video_out) -> None:
    self.video_out = video_out

    self.video_reader = video_in
    self.operation_utils = OperationUtils()
    
    self.curl_count = self.squat_count = 0
    self.keypoints = self.norm_keypoints = None

    self.width = int(self.video_reader.get_frame_width())
    self.height = int(self.video_reader.get_frame_height())
    self.video_fps = int(self.video_reader.get_video_fps())
    self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    self.drawing_utils = DrawingUtils(self.width, self.height)
                                    
  def get_norm_keypoints(self, results) -> dict:
    ''' Get normalized keypoints '''
    norm_keypoints = {}

    for pose, landmark in zip(NORM_POSE_CONNECTIONS, results.pose_landmarks.landmark):
      landmark = {
        'x': landmark.x, 
        'y': landmark.y, 
        'z': landmark.z, 
        'v': landmark.visibility
      }
      if landmark:
        norm_keypoints[pose] = landmark
    return norm_keypoints
  
  def get_norm_point(self, str_point: str) -> tuple:
    """ Get point from normalized keypoints """
    keys = ('x', 'y')
    return tuple(self.norm_keypoints[str_point][i] for i in keys)
      # if self.is_point_in_norm_keypoints(str_point)\
      # else None # f'N/A'
  
  def get_norm_point_depth(self, str_point: str) -> int:
    """Get the depth (Z-axis) of a specific point in the pose"""
    return int(self.norm_keypoints[str_point]['z'] * self.width) # if int(norm_keypoints[str_point]['z'] * self.width) >= 0 else 1

  def get_keypoints(self, results) -> dict:
    """ Get keypoints """
    keypoints = {}
    
    for pose, landmark in zip(POSE_CONNECTIONS, results.pose_landmarks.landmark):
      landmark_px = _normalized_to_pixel_coordinates(landmark.x, 
                                                     landmark.y, 
                                                     self.width, 
                                                     self.height)
      if landmark_px:
        keypoints[pose] = landmark_px
    return keypoints

  def is_point_in_keypoints(self, str_point: str) -> bool:
    """ Check if point is in keypoints """
    return str_point in self.keypoints

  def get_point(self, str_point: str) -> tuple:
    """ Get point from keypoints """
    return self.keypoints[str_point]\
      if self.is_point_in_keypoints(str_point)\
        else None # f'N/A'

  def get_available_point(self, points: list) -> tuple:
    """
    Get highest priority keypoint from points list.
    i.e. first index is 1st priority, second index is 2nd priority, and so on.
    """
    available_point = None

    for point in points:
      if self.is_point_in_keypoints(point)\
        and available_point is None:
          available_point = self.get_point(point)
          break
    return available_point

  def check_posture_side(self, right_str_point: str, left_str_point: str) -> str:
    """Check which side of the body the posture leans to"""
    return 'RIGHT' if self.get_norm_point_depth(right_str_point) < self.get_norm_point_depth(left_str_point)\
      else 'LEFT'

  ### EDIT load_model() TO ADAPT TO THE {EXCERCISE} ###
  def load_model(self, path: str, side: str) -> None:
    try:
      model_name = f'mdl_norm-bicep-classifier-{side.lower()}.pkl'
      with open(f'{path}/{model_name}', 'rb') as f:
        self.model =  pickle.load(f)
    except Exception as e:
      print ( f'Exception in load_model() -> {e}' )
      
  def show_text(self, image, text: str, pos: tuple, color: str = 'red'):
    ''' Shows stage text in the video '''
    image = self.drawing_utils.draw_text(
      image,
      text= text,
      pos= pos,
      txt_color= color,
      background= True,
    )
    return image


class BicepCurl(Pose):
  """ Subclass for Bicep Curl """
  def __init__(self, video_in, video_out):
    super().__init__(video_in, video_out)
    self.video_out = video_out
    self.video_reader = video_in
    self.curl_count = 0
    self.curl_started = False
    self.curl_stage = 'UP'
    self.curl_state = 'UNKNOWN'	
  
  def _draw(self, image, side, color):    
    if side == 'RIGHT':
      right_shoulder = self.get_point('right_shoulder')
      right_elbow = self.get_point('right_elbow')
      right_wrist = self.get_point('right_wrist')
      right_hip = self.get_point('right_hip')
      
      self.drawing_utils.draw_circle(image, right_shoulder, 7, 'yellow', -1)
      self.drawing_utils.draw_circle(image, right_elbow, 7, 'yellow', -1)
      self.drawing_utils.draw_circle(image, right_wrist, 7, 'yellow', -1)
      
      # Join landmarks.
      self.drawing_utils.draw_line(image, right_shoulder, right_elbow, color, 4)
      ### EDIT: to draw a dotted line ###
      # self.drawing_utils.drawDottedLine(image, right_shoulder, (right_shoulder[0], right_shoulder[1] - 100), color, 4)
      self.drawing_utils.draw_line(image, right_elbow, right_wrist, color, 4)
      self.drawing_utils.draw_line(image, right_elbow, right_hip, color, 4)
      return image
    
    elif side == 'LEFT':
      left_shoulder = self.get_point('left_shoulder')
      left_elbow = self.get_point('left_elbow')
      left_wrist = self.get_point('left_wrist')
      left_hip = self.get_point('left_hip')
  
      self.drawing_utils.draw_circle(image, left_shoulder, 7, 'yellow', -1)
      self.drawing_utils.draw_circle(image, left_elbow, 7, 'yellow', -1)
      self.drawing_utils.draw_circle(image, left_wrist, 7, 'yellow', -1)
      
      # Join landmarks.
      self.drawing_utils.draw_line(image, left_shoulder, left_elbow, color, 4)
      # self.drawing_utils.drawDottedLine(image, left_shoulder, (left_shoulder[0], left_shoulder[1] - 100), color, 4)
      self.drawing_utils.draw_line(image, left_elbow, left_wrist, color, 4)
      self.drawing_utils.draw_line(image, left_elbow, left_hip, color, 4)
      return image

    return image

  def _draw_text(self, image, side, state_class_index= None, state_prob= None):
    side_txt, side_pos = f'SIDE: {str(side[0])}', (10, 80)
    image = self.drawing_utils.draw_text(image,
                              side_txt,
                              side_pos,
                              size= 0.75,
                              thickness= 1,
                              txt_color= 'black',
                              background= True)
    
    class_txt, class_pos = f'{self.curl_state}', (10, int(self.height * 0.9))
    image = self.drawing_utils.draw_text(image,
                              class_txt,
                              class_pos,
                              size= 0.75,
                              thickness= 1,
                              txt_color= 'white',
                              background= True,
                              bg_color= 'gray')
                              
    if state_class_index is not None and state_prob is not None:
      prob_txt, prob_pos = 'PROB: {:.2f}'.format(state_prob[state_class_index]), (10, 620)
      image = self.drawing_utils.draw_text(image,
                                  prob_txt,
                                  prob_pos,
                                  size= 0.75,
                                  thickness= 1,
                                  txt_color= 'black',
                                  background= True)
    return image
              
  def _get_csv_columns(self, side):
    CSV_COLUMNS = ['class']

    if side[0] == 'R':
      poses_of_interest = [
          'RIGHT_ELBOW',
          'RIGHT_WRIST',
          'RIGHT_HIP',
      ]
      for pose in poses_of_interest:
          CSV_COLUMNS += [f'{pose.lower()}_x', f'{pose.lower()}_y', f'{pose.lower()}_z', f'{pose.lower()}_v']
    elif side[0] == 'L':
      poses_of_interest = [
          'LEFT_ELBOW',
          'LEFT_WRIST',
          'LEFT_HIP',
      ]
      for pose in poses_of_interest:
          CSV_COLUMNS += [f'{pose.lower()}_x', f'{pose.lower()}_y', f'{pose.lower()}_z', f'{pose.lower()}_v']

    # additional features
    CSV_COLUMNS += [
        f'{side.lower()}_shoulder_ang',
        f'{side.lower()}_shoulder_hip_ang',
        f'{side.lower()}_elbow_hip_dist',
        f'{side.lower()}_shoulder_wrist_dist',
    ]
    return CSV_COLUMNS

  def _get_features(self, side: str) -> list:
    """Get a list of pose features based on the specified side of the body"""
    if side == 'RIGHT':
      right_shoulder = self.get_norm_point('right_shoulder')
      right_elbow = self.get_norm_point('right_elbow')
      right_wrist = self.get_norm_point('right_wrist')
      right_hip = self.get_norm_point('right_hip')

      try:
        right_elbow_ang = self.operation_utils.find_angle(right_shoulder, right_elbow, right_wrist)
        right_shoulder_ang = self.operation_utils.find_angle(right_elbow, right_shoulder, right_hip)
        right_shoulder_hip_ang = self.operation_utils.find_angle_2points(right_shoulder, right_hip)
        right_elbow_hip_dist = self.operation_utils.find_distance_h(right_hip, right_elbow)
        right_shoulder_wrist_dist = self.operation_utils.find_distance(right_shoulder, right_wrist)
      except Exception as e:
        print ( f'Exception in get_features(operation_utils) [R] -> {e}' )
        return
      
      return [
        right_elbow_ang,
        [right_shoulder_ang,  
        right_shoulder_hip_ang,
        right_elbow_hip_dist,
        right_shoulder_wrist_dist,]
      ]
    
    elif side == 'LEFT':
      left_shoulder = self.get_norm_point('left_shoulder')
      left_elbow = self.get_norm_point('left_elbow')
      left_wrist = self.get_norm_point('left_wrist')
      left_hip = self.get_norm_point('left_hip')

      try:
        left_elbow_ang = self.operation_utils.find_angle(left_shoulder, left_elbow, left_wrist)
        left_shoulder_ang = self.operation_utils.find_angle(left_elbow, left_shoulder, left_hip)
        left_shoulder_hip_ang = self.operation_utils.find_angle_2points(left_shoulder, left_hip)
        left_elbow_hip_dist = self.operation_utils.find_distance_h(left_hip, left_elbow)
        left_shoulder_wrist_dist = self.operation_utils.find_distance(left_shoulder, left_wrist)
      except Exception as e:
        print ( f'Exception in get_features(operation_utils) [L] -> {e}' )
        return

      return [
        left_elbow_ang,
        [left_shoulder_ang,
        left_shoulder_hip_ang,
        left_elbow_hip_dist,
        left_shoulder_wrist_dist,]
      ]
    
    else:
      return 'SIDE ERROR -> SIDE WAS NOT SET PROPERLY'
  
  def predict_class(self, norm_keypoints, side):
    if norm_keypoints is not None:
      row = []
      
      if side[0] == 'R':
        right_shoulder = self.get_norm_point('right_shoulder')
        poses_of_interest = [
          'RIGHT_ELBOW',
          'RIGHT_WRIST',
          'RIGHT_HIP',
        ]

        for pose in poses_of_interest:
          row += [
            *self.operation_utils.normalize_pose_points(
              right_shoulder, 
              [
                (norm_keypoints[f'{pose.lower()}']['x'],
                norm_keypoints[f'{pose.lower()}']['y']),
              ]
            ),
            norm_keypoints[f'{pose.lower()}']['z'],
            norm_keypoints[f'{pose.lower()}']['v'],
          ]

      elif side[0] == 'L':
        left_shoulder = self.get_norm_point('left_shoulder')
        poses_of_interest = [
          'LEFT_ELBOW',
          'LEFT_WRIST',
          'LEFT_HIP',
        ]

        for pose in poses_of_interest:
          row += [
            *self.operation_utils.normalize_pose_points(
              left_shoulder, 
              [
                (norm_keypoints[f'{pose.lower()}']['x'],
                norm_keypoints[f'{pose.lower()}']['y']),
              ]
            ),
            norm_keypoints[f'{pose.lower()}']['z'],
            norm_keypoints[f'{pose.lower()}']['v'],
          ]
    
    try:
      features = self._get_features(side)[1]
    except Exception as e:
      print(f'Exception in get_features(): {e}')
      return None
  
    for feature in features:
      row.append(feature)

    try:
      columns = self._get_csv_columns(side)
    except Exception as e:
      print(f'Exception in fetching CSV columns: {e}')
      return None
    
    features_df = pd.DataFrame([row], columns= columns[1:])
    body_language_prob = self.model.predict_proba(features_df)[0]
    body_language_class = np.argmax(body_language_prob)
    
    # self.curl_state = self.model.predict(features_df)[0]
    if body_language_prob[body_language_class] >= 0.6:
      if body_language_class == 1:
        self.curl_state = 'Correct'
      elif body_language_class == 2:
        self.curl_state = 'Elbow Displaced'
      elif body_language_class == 0:
        self.curl_state = 'Body Leaning'

    return body_language_class, body_language_prob

  def measure(self) -> None:
    """ Measure Bicep Curls """
    if not self.video_reader.is_opened():
        print("Error File Not Found.")
        return

    if not os.path.exists(VIDEOS_OUT_PATH):
      os.makedirs(VIDEOS_OUT_PATH)
      print(f'{VIDEOS_OUT_PATH} directory has been created')

    try:
      video = cv2.VideoWriter(
        get_output_filename(self.video_out), 
        self.fourcc, 
        self.video_fps, 
        (self.width, self.height)
      )
      print(f'{self.video_out} video has been created')
    except Exception as e:
      print(f'Error creating {self.video_out} -> {e}')
    
    while self.video_reader.is_opened():
      frame = self.video_reader.read_frame()

      if frame is None:
        break

      image, results = self.video_reader.process_frame(frame)
      
      if results.pose_landmarks is not None:
        self.keypoints = self.get_keypoints(results)
        self.norm_keypoints = self.get_norm_keypoints(results)

      try:
        side = self.check_posture_side('right_elbow', 'left_elbow')
      except Exception as e:
        print(f'Exception in check_posture_side(pt1, pt2): {e}')

      try:
        self.load_model(MODEL_PATH, side)
      except Exception as e:
        print(f'Exception in load_model(): {e}')
        return None
      
      state_class_index, state_prob = self.predict_class(self.norm_keypoints, side)

      elbow_angle = self._get_features(side)[0]

      # Drawing posture landmarks
      if self.curl_state == 'Correct':
        image = self._draw(image, side, 'green')
        
        # check if the bicep curl is being performed
        if elbow_angle < 90 and self.curl_state:
          if not self.curl_started:
            self.curl_started = True
            self.curl_stage = 'UP'
        if elbow_angle > 120 and self.curl_started:
          self.curl_count += 1
          self.curl_started = False
          self.curl_stage = 'DOWN'

      elif self.curl_state == 'Elbow Displaced':
        image = self._draw(image, side, 'red')
      elif self.curl_state == 'Body Leaning':
        image = self._draw(image, side, 'red')
      else:
        image = self._draw(image, side, 'white')
      
      # check if curl is complete
      if self.curl_count == 10:
        self.curl_count = 0
        self.curl_started = False
        image = self.show_text(image, 'HOORAY!\nYOU DID IT!', (60, 200), 'blue')


      # Show Curl Count and Curl Stage
      image = self._draw_text(image, side, state_class_index, state_prob)
      image = self.show_text(image, f'STAGE: {self.curl_stage}', (10, 40), 'blue')
      image = self.show_text(image, f'REPS: {self.curl_count}', (220, 40), 'blue')

      video.write(image)
      
      cv2.imshow('Bicep Curls Test', image)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    self.video_reader.release()


class Squat(Pose):
  """ Subclass for Squat """
  def __init__(self, video_reader, video_out) -> None:
    super().__init__(video_reader, video_out)
    self.out = video_out
    self.video_reader = video_reader
    self.squat_count = 0
    self.wrong_count = 0
    self.squat_state = 'UNKNOWN'
    self.squat_started = False
    self.squat_stage = 'DOWN'
    self.squat_correct = False

  def _draw(self, image):
    """ Draw lines between hip, knee and ankle """
    left_hip_knee_ankle = self.is_point_in_keypoints("left_hip") and \
                          self.is_point_in_keypoints("left_knee") and \
                          self.is_point_in_keypoints("left_ankle")
    
    right_hip_knee_ankle = self.is_point_in_keypoints("right_hip") and \
                          self.is_point_in_keypoints("right_knee") and \
                          self.is_point_in_keypoints("right_ankle")
    
    if left_hip_knee_ankle:
      image = self.drawing_utils.draw_line(image, self.get_point("left_hip"), self.get_point("left_knee"), 'pink', 4)
      image = self.drawing_utils.draw_line(image, self.get_point("left_knee"), self.get_point("left_ankle"), 'pink', 4)
      image = self.drawing_utils.draw_point(image, [self.get_point("left_knee"), self.get_point("left_hip"), self.get_point("left_ankle")])
    elif right_hip_knee_ankle:
      image = self.drawing_utils.draw_line(image, self.get_point("right_hip"), self.get_point("right_knee"), 'pink', 4)
      image = self.drawing_utils.draw_line(image, self.get_point("right_knee"), self.get_point("right_ankle"), 'pink', 4)
      image = self.drawing_utils.draw_point(image, [self.get_point("right_knee"), self.get_point("right_hip"), self.get_point("right_ankle")])
    else:
      pass

    return image

  def pose_algorithm(self, side):
    """ Squat algorithm """
    if side[0] == 'R':
      hip = self.get_point('right_hip')
      knee = self.get_point('right_knee')
      ankle = self.get_point('right_ankle')
    elif side[0] == 'L':
      hip = self.get_point('left_hip')
      knee = self.get_point('left_knee')
      ankle = self.get_point('left_ankle')
    else:	
      return

    if hip is None or knee is None or ankle is None:
      return
    
    hip_knee_angle = self.operation_utils.find_angle(hip, knee, ankle)
    
    if hip_knee_angle < 100:
      self.squat_stage = 'DOWN'
      self.squat_started = True 

    if hip_knee_angle > 160 and self.squat_started is True:
      self.squat_count += 1
      self.squat_stage = 'UP'
      self.squat_started = False
    
  def measure(self) -> None:
    """ Measure squats """
    if self.video_reader.is_opened() is False:
      print("Error File Not Found.")

    if not os.path.exists(VIDEOS_OUT_PATH):
      os.makedirs(VIDEOS_OUT_PATH)
      print(f'{VIDEOS_OUT_PATH} directory has been created')

    try:
      video = cv2.VideoWriter(
        get_output_filename(self.video_out), 
        self.fourcc, 
        self.video_fps, 
        (self.width, self.height)
      )
      print(f'{self.video_out} video has been created')
    except Exception as e:
      print(f'Error creating {self.video_out} -> {e}')


    while self.video_reader.is_opened():
      image = self.video_reader.read_frame()
      
      if image is None:
        break

      image, results = self.video_reader.process_frame(image)
      
      if results.pose_landmarks is not None:
          self.keypoints = self.get_keypoints(results)
          self.norm_keypoints = self.get_norm_keypoints(results)
          
          try:
            side = self.check_posture_side('right_elbow', 'left_elbow')
          except Exception as e:
            print(f'Error checking posture side -> {e}')

          self.pose_algorithm(side)

          image = self._draw(image)
          image = self.drawing_utils.draw_text(image, "Squat REPS: " + str(self.squat_count), (80, 80))

      # self.show_stage(image)
      # self.show_fps(image)

      video.write(image)
      cv2.imshow('Squats', image)
      if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    self.video_reader.release()
