import os, random
import time
import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

from utils.operation_utils import Operation
from utils.timer_utils import Timer
from utils.drawing_utils import Draw
from utils.pose_utils.const import POSE, PRESENCE_THRESHOLD, VISIBILITY_THRESHOLD

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence= 0.5, min_tracking_confidence= 0.5)

out_path = 'data/videos/out/'

class Pose():
  """ Base: Pose Class """
  def __init__(self, video_reader, out) -> None:
    self.out = out

    self.video_reader = video_reader
    self.operation = Operation()
    self.pushup_counter = self.plank_counter = self.squat_counter = self.pullup_counter = 0
    self.key_points = self.prev_pose = self.current_pose = None
    
    self.ang1_tracker = []
    self.ang4_tracker = []
    self.pose_tracker = []
    self.headpoint_tracker = []
    
    self.width = int(self.video_reader.get_frame_width())
    self.height = int(self.video_reader.get_frame_height())
    self.cTime = 0
    self.pTime = 0
    self.video_fps = self.video_reader.get_video_fps()
    self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    self.draw = Draw(self.width, self.height)

  # def pose_algorithm(self):
  #   """ Pose subclass algorithm """
  #   raise NotImplementedError("Requires Subclass implementation.")

  # def measure(self):
  #   """ Pose subclass measure pose """
  #   raise NotImplementedError("Requires Subclass implementation.")

  def show_stage(self, image):
    ''' Shows stage text in the video '''
    image = self.draw.draw_text(image,
                                text= f'STAGE: {self.stage}',
                                pos= (20, 940),
                                size= 2.2,
                                thick= 3,
                                color= (0, 0, 255))

    return image

  def show_fps(self, image):
    ''' Shows video fps in bottom right corner '''
    self.cTime = time.time()
    fps = int(1 / (self.cTime - self.pTime))
    self.pTime = self.cTime

    image = self.draw.draw_text(image, 
                                text= f'FPS: {str(fps)}', 
                                pos= (380, 940),
                                size= 1.2,
                                thick= 2,
                                color= (255, 0, 255))
    return image
                                    
  def get_keypoints(self, image, pose_result):
    """ Get keypoints """
    key_points = {}
    image_rows, image_cols, _ = image.shape
    
    for idx, landmark in enumerate(pose_result.pose_landmarks.landmark):
      if ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD) or
          (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
        continue
      landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                      image_cols, image_rows)
      if landmark_px:
        key_points[idx] = landmark_px
    
    return key_points

  def is_point_in_keypoints(self, str_point):
    """ Check if point is in keypoints """
    return POSE[str_point] in self.key_points

  def get_point(self, str_point):
    """ Get point from keypoints """
    return self.key_points[POSE[str_point]] if self.is_point_in_keypoints(str_point) else None

  def get_available_point(self, points):
    """
    Get highest priority keypoint from points list.
    i.e. first index is 1st priority, second index is 2nd priority, and so on.
    """
    available_point = None

    for point in points:
        if self.is_point_in_keypoints(point) and available_point is None:
            available_point = self.get_point(point)
            break
    
    return available_point

  def two_line_angle(self, str_point1, str_point2, str_point3):
    """ Angle between two lines """
    coord1 = self.get_point(str_point1)
    coord2 = self.get_point(str_point2)
    coord3 = self.get_point(str_point3)
    
    return self.operation.angle(coord1, coord2, coord3)

  def one_line_angle(self, str_point1, str_point2):
    """ Angle of a line """
    coord1 = self.get_point(str_point1)
    coord2 = self.get_point(str_point2)
    
    return self.operation.angle_of_singleline(coord1, coord2)

  def predict_pose(self):
    """ Predict pose """
    
    ang1 = ang2 = ang3 = ang4 = None
    is_pushup = is_plank = is_squat = is_jumping_jack = is_pullup = is_curl = False
    diff_head_hand_y = None

    # Calculate angle between lines shoulder-elbow, elbow-wrist
    if self.is_point_in_keypoints("left_shoulder") and \
      self.is_point_in_keypoints("left_elbow") and \
      self.is_point_in_keypoints("left_wrist"):
      ang1 = self.two_line_angle("left_shoulder", "left_elbow", "left_wrist")
    elif self.is_point_in_keypoints("right_shoulder") and \
      self.is_point_in_keypoints("right_elbow") and \
      self.is_point_in_keypoints("right_wrist"):
      ang1 = self.two_line_angle("right_shoulder", "right_elbow", "right_wrist")
    else:
      pass

    # Calculate angle between lines shoulder-hip, hip-ankle
    if self.is_point_in_keypoints("left_shoulder") and \
      self.is_point_in_keypoints("left_hip") and \
      self.is_point_in_keypoints("left_ankle"):
      ang2 = self.two_line_angle("left_shoulder", "left_hip", "left_ankle")
    elif self.is_point_in_keypoints("right_shoulder") and \
        self.is_point_in_keypoints("right_hip") and \
        self.is_point_in_keypoints("right_ankle"):
      ang2 = self.two_line_angle("right_shoulder", "right_hip", "right_ankle")
    else:
      pass

    # Calculate angle of line shoulder-ankle or hip-ankle
    left_shoulder_ankle = self.is_point_in_keypoints("left_shoulder") and self.is_point_in_keypoints("left_ankle")
    right_shoulder_ankle = self.is_point_in_keypoints("right_shoulder") and self.is_point_in_keypoints("right_ankle")
    left_hip_ankle = self.is_point_in_keypoints("left_hip") and self.is_point_in_keypoints("left_ankle")
    right_hip_ankle = self.is_point_in_keypoints("right_hip") and self.is_point_in_keypoints("right_ankle")
    if left_shoulder_ankle or right_shoulder_ankle:
      shoulder = "left_shoulder" if left_shoulder_ankle else "right_shoulder"
      ankle = "left_ankle" if left_shoulder_ankle else "right_ankle"
      ang3 = self.one_line_angle(shoulder, ankle)
    elif left_hip_ankle or right_hip_ankle:
      hip = "left_hip" if left_hip_ankle else "right_hip"
      ankle = "left_ankle" if left_hip_ankle else "right_ankle"
      ang3 = self.one_line_angle(hip, ankle)
    else:
      pass

    # Calculate angle of line elbow-wrist
    left_elbow_wrist = self.is_point_in_keypoints("left_elbow") and self.is_point_in_keypoints("left_wrist")
    right_elbow_wrist = self.is_point_in_keypoints("right_elbow") and self.is_point_in_keypoints("right_wrist")
    if left_elbow_wrist or right_elbow_wrist:
      elbow = "left_elbow" if left_elbow_wrist else "right_elbow"
      wrist = "left_wrist" if left_elbow_wrist else "right_wrist"
      ang4 = self.one_line_angle(elbow, wrist)
    else:
      pass

    # Calculate angle of line knee-ankle
    left_knee_ankle = self.is_point_in_keypoints("left_knee") and self.is_point_in_keypoints("left_ankle")
    right_knee_ankle = self.is_point_in_keypoints("right_knee") and self.is_point_in_keypoints("right_ankle")
    if left_knee_ankle or right_knee_ankle:
      knee = "left_knee" if left_knee_ankle else "right_knee"
      ankle = "left_ankle" if left_knee_ankle else "right_ankle"
      ang5 = self.one_line_angle(knee, ankle)
    else:
      pass

    # Calculate angle of line hip-knee
    left_hip_knee = self.is_point_in_keypoints("left_hip") and self.is_point_in_keypoints("left_knee")
    right_hip_knee = self.is_point_in_keypoints("right_hip") and self.is_point_in_keypoints("right_knee")
    if left_hip_knee or right_hip_knee:
      knee = "left_knee" if left_hip_knee else "right_knee"
      hip = "left_hip" if left_hip_knee else "right_hip"
      ang6 = self.one_line_angle(hip, knee)
    else:
      pass

    if ang3 is not None and ((0 <= ang3 <= 50) or (130 <= ang3 <= 180)):
      if (ang1 is not None or ang2 is not None) and ang4 is not None:
        if (160 <= ang2 <= 180) or (0 <= ang2 <= 20):
          self.pushup_counter += 1
          self.ang1_tracker.append(ang1)
          self.ang4_tracker.append(ang4)

    if self.pushup_counter >= 24 and len(self.ang1_tracker) == 24 and len(self.ang4_tracker) == 24:
      ang1_diff1 = abs(self.ang1_tracker[0] - self.ang1_tracker[12])
      ang1_diff2 = abs(self.ang1_tracker[12] - self.ang1_tracker[23])
      ang1_diff_mean = (ang1_diff1 + ang1_diff2) / 2
      ang4_mean = sum(self.ang4_tracker) / len(self.ang4_tracker)
      del self.ang1_tracker[0]
      del self.ang4_tracker[0]
      if ang1_diff_mean < 5 and not 75 <= ang4_mean <= 105:
        is_plank = True
        is_pushup = is_squat = is_jumping_jack = is_pullup = is_curl = False
      else:
        is_pushup = True
        is_plank = is_squat = is_jumping_jack = is_pullup = is_curl = False

    # Distance algorithm
    head_point = self.get_available_point(["nose", "left_ear", "right_ear", "left_eye", "right_eye"])
    hip = self.get_available_point(["left_hip", "right_hip"])
    knee = self.get_available_point(["left_knee", "right_knee"])
    foot = self.get_available_point(["left_foot_index", "right_foot_index", "left_heel", "right_heel", "left_ankle", "right_ankle"])
    hand_point = self.get_available_point(["left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index"])
    
    if head_point is not None and hand_point is not None:
      self.headpoint_tracker.append(head_point[1]) # height only
      diff_head_hand_y = head_point[1] - hand_point[1]
    if ang3 is not None and ang5 is not None and diff_head_hand_y is not None:
      if ((70 <= ang3 <= 110) or (70 <= ang5 <= 110)) and len(self.headpoint_tracker) == 24:
        height_mean = int(sum(self.headpoint_tracker) / len(self.headpoint_tracker))
        height_norm = self.operation.normalize(height_mean, head_point[1], foot[1])
        del self.headpoint_tracker[0]
        if height_norm < 0 and diff_head_hand_y < 0 and not 70 <= abs(ang6) <= 110:
          is_squat = True
          is_pushup = is_plank = is_jumping_jack = is_pullup = is_curl = False
        else:
          is_squat = False

    if diff_head_hand_y is not None and ang3 is not None:
      if diff_head_hand_y > 0 and 80 <= ang3 <= 100:
        is_jumping_jack = True
        is_pushup = is_plank = is_squat = is_pullup = is_curl = False
      if diff_head_hand_y < 0 and is_jumping_jack is True:
        is_jumping_jack = False

    if len(self.ang1_tracker) == 24:
      del self.ang1_tracker[0]
    if len(self.ang4_tracker) == 24:
      del self.ang4_tracker[0]
    if len(self.headpoint_tracker) == 24:
      del self.headpoint_tracker[0]

    if is_pushup:
      return "Pushup"
    elif is_plank:
      return "Plank"
    elif is_squat:
      return "Squat"
    elif is_jumping_jack:
      return "JumpingJack"
    elif is_pullup:
      return "PullUp"
    elif is_curl:
      return "Curl"

    return None

  def estimate(self) -> None:
    """ Estimate pose """
    if self.video_reader.is_opened() is False:
        print("Error File Not Found.")

    if not os.path.exists(out_path):
      os.makedirs(out_path)
      print(f'{out_path} directory has been created')

    name = self.out + '.mp4'
    out = cv2.VideoWriter(out_path + name, self.fourcc, self.video_fps, (self.width, self.height))
    print(f'{name} video has been created')
    pTime = 0
    
    while self.video_reader.is_opened():
        image = self.video_reader.read_frame()
        if image is None:
          print("Ignoring empty camera frame.")
          break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = self.draw.overlay(image)
        image = self.draw.skeleton(image, results)

        if results.pose_landmarks is not None:
          self.key_points = self.get_keypoints(image, results)
          estimated_pose = self.predict_pose()
          if estimated_pose is not None:
            self.current_pose = estimated_pose
            self.pose_tracker.append(self.current_pose)
            if len(self.pose_tracker) == 10 and len(set(self.pose_tracker[-6:])) == 1:
              image = self.draw.pose_text(image, "Prediction: " + estimated_pose)

        if len(self.pose_tracker) == 10:
          del self.pose_tracker[0]
          self.prev_pose = self.pose_tracker[-1]
        
        self.draw.draw_text(image,
                            text= f'Prediction: {estimated_pose}',
                            pos= (20, 150),
                            size= 2.2,
                            thick= 3,
                            color= (0, 0, 255))
        self.show_fps(image)

        out.write(image)

        cv2.imshow('Estimation of Exercise', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
          break

    self.video_reader.release()


class Pushup(Pose):
  """ Sub: Pushup class """
  def __init__(self, video_reader, out) -> None:
    super().__init__(video_reader, out)
    self.out = out
    self.video_reader = video_reader
    self.pushups_count = 0
    self.timer = Timer()
    self.is_pushup = False
    self.stage = "UP"

  def _draw(self, image):
    """ Draw lines between shoulder, wrist and foot """
    left_shoulder_wrist_foot = self.is_point_in_keypoints("left_shoulder") and \
                                self.is_point_in_keypoints("left_wrist") and \
                                self.is_point_in_keypoints("left_foot_index")
    
    right_shoulder_wrist_foot = self.is_point_in_keypoints("right_shoulder") and \
                                self.is_point_in_keypoints("right_wrist") and \
                                self.is_point_in_keypoints("right_foot_index")
    
    if left_shoulder_wrist_foot:
      image = self.draw.draw_line(image, self.get_point("left_shoulder"), self.get_point("left_wrist"))
      image = self.draw.draw_line(image, self.get_point("left_wrist"), self.get_point("left_foot_index"))
      image = self.draw.draw_line(image, self.get_point("left_foot_index"), self.get_point("left_shoulder"))
      image = self.draw.draw_point(image, self.get_point("left_shoulder"))
      # image = self.draw.draw_point(image, self.get_point("left_wrist"))
      # image = self.draw.draw_point(image, self.get_point("left_foot_index"))
    elif right_shoulder_wrist_foot:
      image = self.draw.draw_line(image, self.get_point("right_shoulder"), self.get_point("right_wrist"))
      image = self.draw.draw_line(image, self.get_point("right_wrist"), self.get_point("right_foot_index"))
      image = self.draw.draw_line(image, self.get_point("right_foot_index"), self.get_point("right_shoulder"))
      image = self.draw.draw_point(image, self.get_point("right_shoulder"))
      # image = self.draw.draw_point(image, self.get_point("right_wrist"))
      # image = self.draw.draw_point(image, self.get_point("right_foot_index"))
    else:
      pass

    return image

  def pose_algorithm(self):
    """ Pushup algorithm """
    # Distance algorithm
    head_point = self.get_available_point(["nose", "left_ear", "right_ear", "left_eye", "right_eye"])
    ankle = self.get_available_point(["left_ankle", "right_ankle"])
    if head_point is None or ankle is None:
      return

    diff_y = self.operation.dist_y(head_point, ankle)

    # Angle algorithm
    head_pos = self.operation.point_position(head_point, (self.width /2, 0), (self.width /2, self.height))
    wrist = self.get_available_point(["left_wrist", "right_wrist"])
    ang = self.operation.angle(head_point, ankle, wrist)
    # print(f'Angle: {ang}, Difference: {diff_y},\n')
    
    if diff_y < 60 and (ang < 18 and head_pos == "right") or (ang < 10 and head_pos == "left"):
      self.is_pushup = True
      self.stage = 'DOWN'
    if diff_y > 40 and ((ang > 22 and head_pos == "right") or (ang > 12 and head_pos == "left")) and self.is_pushup is True:
      self.pushups_count += 1
      self.is_pushup = False
      self.stage = 'UP'

  def measure(self) -> None:
    """ Measure pushups """
    if self.video_reader.is_opened() is False:
        print("Error File Not Found.")

    if not os.path.exists(out_path):
      os.makedirs(out_path)
      print(f'{out_path} directory has been created')

    name = self.out + '.mp4'
    out = cv2.VideoWriter(out_path + name, self.fourcc, self.video_fps, (self.width, self.height))
    print(f'{name} video has been created')
    
    pushup_count_prev = pushup_count_current = progress_counter = 0
    progress_bar_color = (255, 255, 255)
    
    while self.video_reader.is_opened():
      image = self.video_reader.read_frame()
      if image is None:
        print("Ignoring empty camera frame.")
        break

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = pose.process(image)

      # overlay
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      image = self.draw.overlay(image)

      # progress bar
      image = cv2.rectangle(image, (0, self.height//8 - 10), (self.width//10 * progress_counter, self.height//8),
                              progress_bar_color, cv2.FILLED)
      
      if results.pose_landmarks is not None:
        self.key_points = self.get_keypoints(image, results)
        self.pose_algorithm()
        image = self._draw(image)
        image = self.draw.pose_text(image, "PushUp REPS: " + str(self.pushups_count))
        pushup_count_prev = pushup_count_current
        pushup_count_current = self.pushups_count
        if self.pushups_count > 0 and abs(pushup_count_current - pushup_count_prev) == 1:
          progress_counter += 1
          if progress_counter == 10:
            progress_counter = 0
            progress_bar_color = random.choices(range(128, 256), k= 3)

      self.show_stage(image)
      self.show_fps(image)
     
      out.write(image)
      
      cv2.imshow('Pushups', image)
      if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    self.video_reader.release()


class Plank(Pose):
  """ Sub: Plank class """
  def __init__(self, video_reader, out) -> None:
    super().__init__(video_reader, out)
    self.out = out
    self.video_reader = video_reader
    self.timer = Timer()
    self.plank_counter = 0
    self.start_time = None
    self.total_time = 0

  def _draw(self, image):
    """ Draw lines between shoulder, wrist and foot """
    left_shoulder_wrist_foot = self.is_point_in_keypoints("left_shoulder") and \
                                self.is_point_in_keypoints("left_elbow") and \
                                self.is_point_in_keypoints("left_foot_index")
    
    right_shoulder_wrist_foot = self.is_point_in_keypoints("right_shoulder") and \
                                self.is_point_in_keypoints("right_elbow") and \
                                self.is_point_in_keypoints("right_foot_index")
    
    if left_shoulder_wrist_foot:
      image = self.draw.draw_line(image, self.get_point("left_shoulder"), self.get_point("left_elbow"))
      image = self.draw.draw_line(image, self.get_point("left_elbow"), self.get_point("left_foot_index"))
      image = self.draw.draw_line(image, self.get_point("left_foot_index"), self.get_point("left_shoulder"))
      image = self.draw.draw_point(image, self.get_point("left_shoulder"), self.get_point("left_elbow"), self.get_point("left_foot_index"))
    elif right_shoulder_wrist_foot:
      image = self.draw.draw_line(image, self.get_point("right_shoulder"), self.get_point("right_elbow"))
      image = self.draw.draw_line(image, self.get_point("right_elbow"), self.get_point("right_foot_index"))
      image = self.draw.draw_line(image, self.get_point("right_foot_index"), self.get_point("right_shoulder"))
      image = self.draw.draw_point(image, self.get_point("right_shoulder"), self.get_point("right_elbow"), self.get_point("right_foot_index"))

    else:
      pass
    
    return image

  def pose_algorithm(self):
    """ Plank algorithm """
    ang1 = ang2 = ang3 = ang4 = None

    # Calculate angle between lines shoulder-elbow, elbow-wrist
    if self.is_point_in_keypoints("left_shoulder") and \
        self.is_point_in_keypoints("left_elbow") and \
        self.is_point_in_keypoints("left_wrist"):
      ang1 = self.two_line_angle("left_shoulder", "left_elbow", "left_wrist")
    elif self.is_point_in_keypoints("right_shoulder") and \
          self.is_point_in_keypoints("right_elbow") and \
          self.is_point_in_keypoints("right_wrist"):
      ang1 = self.two_line_angle("right_shoulder", "right_elbow", "right_wrist")
    else:
      pass

    # Calculate angle between lines shoulder-hip, hip-ankle
    if self.is_point_in_keypoints("left_shoulder") and \
        self.is_point_in_keypoints("left_hip") and \
        self.is_point_in_keypoints("left_ankle"):
      ang2 = self.two_line_angle("left_shoulder", "left_hip", "left_ankle")
    elif self.is_point_in_keypoints("right_shoulder") and \
          self.is_point_in_keypoints("right_hip") and \
          self.is_point_in_keypoints("right_ankle"):
      ang2 = self.two_line_angle("right_shoulder", "right_hip", "right_ankle")
    else:
      pass

    # Calculate angle of line shoulder-ankle or hip-ankle
    left_shoulder_ankle = self.is_point_in_keypoints("left_shoulder") and self.is_point_in_keypoints("left_ankle")
    right_shoulder_ankle = self.is_point_in_keypoints("right_shoulder") and self.is_point_in_keypoints("right_ankle")
    left_hip_ankle = self.is_point_in_keypoints("left_hip") and self.is_point_in_keypoints("left_ankle")
    right_hip_ankle = self.is_point_in_keypoints("right_hip") and self.is_point_in_keypoints("right_ankle")
    
    if left_shoulder_ankle or right_shoulder_ankle:
      shoulder = "left_shoulder" if left_shoulder_ankle else "right_shoulder"
      ankle = "left_ankle" if left_shoulder_ankle else "right_ankle"
      ang3 = self.one_line_angle(shoulder, ankle)
    elif left_hip_ankle or right_hip_ankle:
      hip = "left_hip" if left_hip_ankle else "right_hip"
      ankle = "left_ankle" if left_hip_ankle else "right_ankle"
      ang3 = self.one_line_angle(hip, ankle)
    else:
      pass

    # Calculate angle of line elbow-wrist
    left_elbow_wrist = self.is_point_in_keypoints("left_elbow") and self.is_point_in_keypoints("left_wrist")
    right_elbow_wrist = self.is_point_in_keypoints("right_elbow") and self.is_point_in_keypoints("right_wrist")
    

    if left_elbow_wrist or right_elbow_wrist:
      elbow = "left_elbow" if left_elbow_wrist else "right_elbow"
      wrist = "left_wrist" if left_elbow_wrist else "right_wrist"
      ang4 = self.one_line_angle(elbow, wrist)
    else:
      pass
    
    # print(f'ang2= {ang2}, ang3= {ang3}, ang4= {ang4}')

    if ang3 is not None and ((0 <= ang3 <= 50) or (-180 <= ang3 <= -130)):
      if (ang1 is not None or ang2 is not None) and ang4 is not None:
        if (140 <= ang2 <= 160) or (0 <= ang2 <= 20):
          self.plank_counter += 1
          self.ang1_tracker.append(ang1)
          self.ang4_tracker.append(ang4)

    if self.plank_counter >= 24 and len(self.ang1_tracker) == 24 and len(self.ang4_tracker) == 24:
      ang1_diff1 = abs(self.ang1_tracker[0] - self.ang1_tracker[12])
      ang1_diff2 = abs(self.ang1_tracker[12] - self.ang1_tracker[23])

      ang1_diff_mean = (ang1_diff1 + ang1_diff2) / 2
      ang4_mean = sum(self.ang4_tracker) / len(self.ang4_tracker)
      del self.ang1_tracker[0]
      del self.ang4_tracker[0]

      # print(f'ang4 mean= {ang4_mean}')
      if ang1_diff_mean < 5 and not -180 <= ang4_mean <= -150:
        if self.start_time is None:
          self.timer.start()
        self.start_time = self.timer._start_time
      else:
        if self.start_time is not None:
          self.timer.end()
          self.total_time = self.timer._total_time
        self.start_time = None

  def measure(self) -> None:
    """ Measure planks """
    if self.video_reader.is_opened() is False:
        print("Error File Not Found.")

    if not os.path.exists(out_path):
      os.makedirs(out_path)
      print(f'{out_path} directory has been created')

    name = self.out + '.mp4'
    out = cv2.VideoWriter(out_path + name, self.fourcc, self.video_fps, (self.width, self.height))
    print(f'{name} video has been created')

    time_adjustment = 6 / self.video_fps # 6 is magic number
    progress_counter = 0
    progress_bar_color = (255, 255, 255)

    while self.video_reader.is_opened():
      image = self.video_reader.read_frame()
      if image is None:
        print("Ignoring empty camera frame.")
        break

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = pose.process(image)

      # overlay
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      image = self.draw.overlay(image)

      # progress bar
      image = cv2.rectangle(image, (0, self.height //8 -10), (self.width //60 * progress_counter, self.height //8),
                              progress_bar_color, cv2.FILLED)
      if results.pose_landmarks is not None:
        self.key_points = self.get_keypoints(image, results)
        self.pose_algorithm()
        image = self._draw(image)
        time = round(time_adjustment * (self.timer.get_current_time() + self.total_time), 2)
        h_m_s_ms_time = self.timer.convert_time(time) # convert Seconds to Hour : Minute : Second : Milli-Second format
        image = self.draw.pose_text(image, "Plank Timer: " + str(h_m_s_ms_time)[:10])
        progress_counter = int(int(time) % self.video_fps)

      # self.show_timer(image)
      # self.show_stage(image)
      self.show_fps(image)

      out.write(image)

      cv2.imshow('Planks', image)
      if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    self.video_reader.release()


class Squat(Pose):
  """ Sub: Squat class """
  def __init__(self, video_reader, out) -> None:
    super().__init__(video_reader, out)
    self.out = out
    self.video_reader = video_reader
    self.squats_count = 0
    self.is_squat = False
    self.stage = 'UP'

  def _draw(self, image):
    """ Draw lines between hip, knee and ankle """
    left_hip_knee_ankle = self.is_point_in_keypoints("left_hip") and \
                          self.is_point_in_keypoints("left_knee") and \
                          self.is_point_in_keypoints("left_ankle")
    
    right_hip_knee_ankle = self.is_point_in_keypoints("right_hip") and \
                          self.is_point_in_keypoints("right_knee") and \
                          self.is_point_in_keypoints("right_ankle")
    
    if left_hip_knee_ankle:
      image = self.draw.draw_line(image, self.get_point("left_hip"), self.get_point("left_knee"))
      image = self.draw.draw_line(image, self.get_point("left_knee"), self.get_point("left_ankle"))
      image = self.draw.draw_point(image, self.get_point("left_knee"))
      image = self.draw.draw_point(image, self.get_point("left_hip"))
      image = self.draw.draw_point(image, self.get_point("left_ankle"))
    elif right_hip_knee_ankle:
      image = self.draw.draw_line(image, self.get_point("right_hip"), self.get_point("right_knee"))
      image = self.draw.draw_line(image, self.get_point("right_knee"), self.get_point("right_ankle"))
      image = self.draw.draw_point(image, self.get_point("right_knee"))
      image = self.draw.draw_point(image, self.get_point("right_hip"))
      image = self.draw.draw_point(image, self.get_point("right_ankle"))
    else:
      pass

    return image

  def pose_algorithm(self):
    """ Squat algorithm """
    # Distance algorithm
    head_point = self.get_available_point(["nose", "left_ear", "right_ear", "left_eye", "right_eye"])
    ankle = self.get_available_point(["left_ankle", "right_ankle"])
    if head_point is None or ankle is None:
      return

    diff_y = self.operation.dist_y(head_point, ankle)
    norm_diff_y = self.operation.normalize(diff_y, 0, self.height)
    if norm_diff_y < 0.5:
        self.is_squat = True
        self.stage = 'DOWN'
    if norm_diff_y > 0.5 and self.is_squat is True:
        self.squats_count += 1
        self.is_squat = False
        self.stage = 'UP'

  def measure(self) -> None:
    """ Measure squats """
    if self.video_reader.is_opened() is False:
      print("Error File Not Found.")

    if not os.path.exists(out_path):
      os.makedirs(out_path)
      print(f'{out_path} directory has been created')

    name = self.out + '.mp4'
    out = cv2.VideoWriter(out_path + name, self.fourcc, self.video_fps, (self.width, self.height))
    print(f'{name} video has been created')
    
    squats_count_prev = squat_count_current = progress_counter = 0
    progress_bar_color = (255, 255, 255)
      
    while self.video_reader.is_opened():
      image = self.video_reader.read_frame()
      if image is None:
        print("Ignoring empty camera frame.")
        break

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      
      # Make detection
      results = pose.process(image)

      # Recolor back to BGR
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
      # overlay
      image = self.draw.overlay(image)
      image = self._draw(image)
      image = self.draw.skeleton(image, results)

      # progress bar
      image = cv2.rectangle(image, (0, self.height //8 -10), (self.width //10 * progress_counter, self.height //8),
                              progress_bar_color, cv2.FILLED)
      
      if results.pose_landmarks is not None:
          self.key_points = self.get_keypoints(image, results)
          self.pose_algorithm()
          image = self._draw(image)
          image = self.draw.pose_text(image, "Squat REPS: " + str(self.squats_count))
          squats_count_prev = squat_count_current
          squat_count_current = self.squats_count
          if self.squats_count > 0 and abs(squat_count_current - squats_count_prev) == 1:
            progress_counter += 1
            if progress_counter == 10:
              progress_counter = 0
              progress_bar_color = random.choices(range(128, 256), k= 3)

      self.show_stage(image)
      self.show_fps(image)

      out.write(image)
      cv2.imshow('Squats', image)
      if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    self.video_reader.release()


class Jumpingjack(Pose):
  """ Sub: JumpingJack class """
  def __init__(self, video_reader, out) -> None:
    super().__init__(video_reader, out)
    self.out = out
    self.video_reader = video_reader
    self.jumping_jack_count = 0
    self.is_jumping_jack = False
    self.stage = 'DOWN'

  def pose_algorithm(self):
    """ JumpingJack algorithm """
    # Distance algorithm
    hand_point = self.get_available_point(["left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index"])
    head_point = self.get_available_point(["nose", "left_ear", "right_ear", "left_eye", "right_eye"])
    
    if head_point is None or hand_point is None:
      return

    diff_y = head_point[1] - hand_point[1]
    norm_diff_y = self.operation.normalize(diff_y, 0, self.height)

    # Angle algorithm
    # Calculate angle of line shoulder-ankle or hip-ankle
    left_shoulder_ankle = self.is_point_in_keypoints("left_shoulder") and self.is_point_in_keypoints("left_ankle")
    right_shoulder_ankle = self.is_point_in_keypoints("right_shoulder") and self.is_point_in_keypoints("right_ankle")
    left_hip_ankle = self.is_point_in_keypoints("left_hip") and self.is_point_in_keypoints("left_ankle")
    right_hip_ankle = self.is_point_in_keypoints("right_hip") and self.is_point_in_keypoints("right_ankle")
    
    if left_shoulder_ankle or right_shoulder_ankle:
      shoulder = "left_shoulder" if left_shoulder_ankle else "right_shoulder"
      ankle = "left_ankle" if left_shoulder_ankle else "right_ankle"
      ang = self.one_line_angle(shoulder, ankle)
    elif left_hip_ankle or right_hip_ankle:
      hip = "left_hip" if left_hip_ankle else "right_hip"
      ankle = "left_ankle" if left_hip_ankle else "right_ankle"
      ang = self.one_line_angle(hip, ankle)
    else:
      pass
    
    # print(f'Norm Diff: {diff_y},\n Angle: {ang}')

    if norm_diff_y > 0 and (75 <= ang <= 90):
      self.is_jumping_jack = True
      self.stage = 'UP'
    if norm_diff_y < 0 and self.is_jumping_jack is True:
      self.jumping_jack_count += 1
      self.is_jumping_jack = False
      self.stage = 'DOWN'

  def measure(self) -> None:
    """ Measure JumpingJacks """
    if self.video_reader.is_opened() is False:
      print("Error File Not Found.")
    
    if not os.path.exists(out_path):
      os.makedirs(out_path)
      print(f'{out_path} directory has been created')

    name = self.out + '.mp4'
    out = cv2.VideoWriter(out_path + name, self.fourcc, self.video_fps, (self.width, self.height))
    print(f'{name} video has been created')
    
    jj_count_prev = jj_count_current = progress_counter = 0
    progress_bar_color = (255, 255, 255)
    
    while self.video_reader.is_opened():
      image = self.video_reader.read_frame()
      
      if image is None:
        print("Ignoring empty camera frame.")
        break

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = pose.process(image)

      # overlay
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      image = self.draw.overlay(image)
      image = self.draw.skeleton(image, results)

      # progress bar
      image = cv2.rectangle(image, (0, self.height//8 - 10), (self.width//10 * progress_counter, self.height//8),
                              progress_bar_color, cv2.FILLED)
      
      
      if results.pose_landmarks is not None:
        self.key_points = self.get_keypoints(image, results)
        
        try:
          self.pose_algorithm()
        except UnboundLocalError:
          raise Exception('Please make sure your full body appears in the frame')

        image = self.draw.pose_text(image, "JumpingJacks REPS: " + str(self.jumping_jack_count))
        
        jj_count_prev = jj_count_current
        jj_count_current = self.jumping_jack_count
        
        if self.jumping_jack_count > 0 and abs(jj_count_current - jj_count_prev) == 1:
          progress_counter += 1
          if progress_counter == 10:
            progress_counter = 0
            progress_bar_color = random.choices(range(128, 256), k= 3)

        self.show_stage(image)
        self.show_fps(image)

        out.write(image)

        cv2.imshow('JumpingJacks', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
          break

    self.video_reader.release()


class Pullup(Pose):
  """ Sub: PullUp class """
  def __init__(self, video_reader, out):
    super().__init__(video_reader, out)
    self.out = out
    self.video_reader = video_reader
    self.pullups_count = 0
    self.is_pullup = False
    self.stage = 'DOWN'

  def _draw(self, image):
    """ Draw lines between shoulder, elbow and wrist """
    left_shoulder_elbow_wrist = self.is_point_in_keypoints("left_shoulder") and \
                                self.is_point_in_keypoints("left_elbow") and \
                                self.is_point_in_keypoints("left_wrist")
    
    right_shoulder_elbow_wrist = self.is_point_in_keypoints("right_shoulder") and \
                                self.is_point_in_keypoints("right_elbow") and \
                                self.is_point_in_keypoints("right_wrist")
    
    if left_shoulder_elbow_wrist:
      image = self.draw.draw_line(image, self.get_point("left_shoulder"), self.get_point("left_elbow"))
      image = self.draw.draw_line(image, self.get_point("left_elbow"), self.get_point("left_wrist"))
      image = self.draw.draw_point(image, self.get_point("left_elbow"))
    if right_shoulder_elbow_wrist:
      image = self.draw.draw_line(image, self.get_point("right_shoulder"), self.get_point("right_elbow"))
      image = self.draw.draw_line(image, self.get_point("right_elbow"), self.get_point("right_wrist"))
      image = self.draw.draw_point(image, self.get_point("right_elbow"))
    
    return image

  def pose_algorithm(self):
    """ PullUp algorithm """
    # Distance algorithm
    shoulder = self.get_available_point(["left_shoulder", "right_shoulder"])
    elbow = self.get_available_point(["left_elbow", "right_elbow"])
    wrist = self.get_available_point(["left_wrist", "right_wrist"])
    if shoulder is None or elbow is None:
      return

    diff_y = self.operation.dist_y(shoulder, wrist)
    # Angle algorithm
    ang = self.operation.angle(shoulder, elbow, wrist)
    # print(f'diff_y: {diff_y}, ang: {ang}')

    if ang >= 100 or diff_y >= 100:
      self.stage = 'DOWN'
      self.is_pullup = True
      
    if (ang <= 50 or diff_y <= 15) and self.is_pullup == True:
      self.pullups_count += 1
      self.is_pullup = False
      self.stage = 'UP'

  def measure(self) -> None:
    """ Measure pullups """
    if self.video_reader.is_opened() is False:
        print("Error File Not Found.")

    if not os.path.exists(out_path):
      os.makedirs(out_path)
      print(f'{out_path} directory has been created')
    
    name = self.out + '.mp4'
    out = cv2.VideoWriter(out_path + name, self.fourcc, self.video_fps, (self.width, self.height))
    print(f'{name} video has been created')
    
    pullup_count_prev = pullup_count_current = progress_counter = 0
    progress_bar_color = (255, 255, 255)

    while self.video_reader.is_opened():
      image = self.video_reader.read_frame()
      
      if image is None:
        print("Ignoring empty camera frame.")
        break

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      # Make detection
      results = pose.process(image)

      # Recolor back to BGR
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
      # overlay
      image = self.draw.overlay(image)

      # progress bar
      image = cv2.rectangle(image, (0, self.height //8 -10), (self.width //10 * progress_counter, self.height //8),
                              progress_bar_color, cv2.FILLED)
      
      if results.pose_landmarks is not None:
        self.key_points = self.get_keypoints(image, results)
        
        self.pose_algorithm()
        
        image = self._draw(image)
        # image = self.draw.skeleton(image, results)
        image = self.draw.pose_text(image, "PullUp REPS: " + str(self.pullups_count))
        
        pullup_count_prev = pullup_count_current
        pullup_count_current = self.pullups_count
        
        if self.pullups_count > 0 and abs(pullup_count_current - pullup_count_prev) == 1:
          progress_counter += 1
          if progress_counter == 10:
            progress_counter = 0
            progress_bar_color = random.choices(range(128, 256), k= 3)

      self.show_stage(image)
      self.show_fps(image)
     
      out.write(image)
      
      cv2.imshow('Pullups', image)
      if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    self.video_reader.release()


class Curl(Pose):
  """ Sub: Bicep Curl class """
  def __init__(self, video_reader, out):
    super().__init__(video_reader, out)
    self.out = out
    self.video_reader = video_reader
    self.curls_count = 0
    self.is_curl = False
    self.stage = 'UP'

  def _draw(self, image):
    """ Draw lines between shoulder, elbow and wrist """
    left_shoulder_elbow_wrist = self.is_point_in_keypoints("left_shoulder") and \
                                self.is_point_in_keypoints("left_elbow") and \
                                self.is_point_in_keypoints("left_wrist")
    
    right_shoulder_elbow_wrist = self.is_point_in_keypoints("right_shoulder") and \
                                self.is_point_in_keypoints("right_elbow") and \
                                self.is_point_in_keypoints("right_wrist")
    
    if left_shoulder_elbow_wrist:
      image = self.draw.draw_line(image, self.get_point("left_shoulder"), self.get_point("left_elbow"))
      image = self.draw.draw_line(image, self.get_point("left_elbow"), self.get_point("left_wrist"))
      image = self.draw.draw_point(image, self.get_point("left_elbow"))
    if right_shoulder_elbow_wrist:
      image = self.draw.draw_line(image, self.get_point("right_shoulder"), self.get_point("right_elbow"))
      image = self.draw.draw_line(image, self.get_point("right_elbow"), self.get_point("right_wrist"))
      image = self.draw.draw_point(image, self.get_point("right_elbow"))
    
    return image

  def pose_algorithm(self):
    """ curl algorithm """
    # Distance algorithm
    shoulder = self.get_available_point(["left_shoulder", "right_shoulder"])
    elbow = self.get_available_point(["left_elbow", "right_elbow"])
    wrist = self.get_available_point(["left_wrist", "right_wrist"])

    if shoulder is None or elbow is None:
      return

    diff_y = self.operation.dist_y(shoulder, wrist)
    # Angle algorithm
    ang = self.operation.angle(shoulder, elbow, wrist)
    # print(f'diff_y: {diff_y}, ang: {ang}')

    if ang > 120 or diff_y > 140:
      self.stage = 'DOWN'
      self.is_curl = True
      
    if (ang < 60 or diff_y < 80) and self.is_curl == True:
      self.stage = 'UP'
      self.curls_count += 1
      self.is_curl = False

  def measure(self) -> None:
    """ Measure Bicep Curls """
    if self.video_reader.is_opened() is False:
        print("Error File Not Found.")

    if not os.path.exists(out_path):
      os.makedirs(out_path)
      print(f'{out_path} directory has been created')

    name = self.out + '.mp4'
    out = cv2.VideoWriter(out_path + name, self.fourcc, self.video_fps, (self.width, self.height))
    print(f'{name} video has been created')
    
    curl_count_prev = curl_count_current = progress_counter = 0
    progress_bar_color = (255, 255, 255)
    
    while self.video_reader.is_opened():
      image = self.video_reader.read_frame()
      
      if image is None:
        print("Ignoring empty camera frame.")
        break

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      # Make detection
      results = pose.process(image)

      # Recolor back to BGR
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
      # overlay
      image = self.draw.overlay(image)

      # progress bar
      image = cv2.rectangle(image, (0, self.height //8 -10), (self.width //10 * progress_counter, self.height //8),
                              progress_bar_color, cv2.FILLED)
      
      if results.pose_landmarks is not None:
        self.key_points = self.get_keypoints(image, results)
        
        self.pose_algorithm()
        
        image = self._draw(image)
        image = self.draw.pose_text(image, "Bicep Curl REPS: " + str(self.curls_count))
        # image = self.draw.skeleton(image, results)
        
        curl_count_prev = curl_count_current
        curl_count_current = self.curls_count
        
        if self.curls_count > 0 and abs(curl_count_current - curl_count_prev) == 1:
          progress_counter += 1
          if progress_counter == 10:
            progress_counter = 0
            progress_bar_color = random.choices(range(128, 256), k= 3)
      
      self.show_stage(image)
      self.show_fps(image)

      out.write(image)
      
      cv2.imshow('Curls', image)
      if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    self.video_reader.release()


class Abdominal(Pose):
  """ Sub: Abdominal class """
  def __init__(self, video_reader, out):
    super().__init__(video_reader, out)
    self.out = out
    self.video_reader = video_reader
    self.abdominals_count = 0
    self.is_abdominal = False
    self.stage = 'DOWN'
    
  def pose_algorithm(self):
    """ Abs algorithm """
    # Distance algorithm
    shoulder = self.get_available_point(["left_shoulder", "right_shoulder"])
    hip = self.get_available_point(["left_hip", "right_hip"])
    knee = self.get_available_point(["left_knee", "right_knee"])

    if shoulder is None or hip is None:
      return

    # diff_y = self.operation.dist_y(shoulder, knee)

    # Angle algorithm
    ang = self.operation.angle(shoulder, hip, knee)
    # print(f'diff_y: {diff_y}, ang: {ang}')

    if ang > 100:
      self.stage = 'DOWN'
      self.is_abdominal = True
      
    if ang < 90 and self.is_abdominal == True:
      self.stage = 'UP'
      self.abdominals_count += 1
      self.is_abdominal = False

  def measure(self) -> None:
    """ Measure Abdominals """
    if self.video_reader.is_opened() is False:
        print("Error File Not Found.")

    if not os.path.exists(out_path):
      os.makedirs(out_path)
      print(f'{out_path} directory has been created')

    name = self.out + '.mp4'
    out = cv2.VideoWriter(out_path + name, self.fourcc, self.video_fps, (self.width, self.height))
    print(f'{name} video has been created')
    
    abdominal_count_prev = abdominal_count_current = progress_counter = 0
    progress_bar_color = (255, 255, 255)
    
    while self.video_reader.is_opened():
      image = self.video_reader.read_frame()
      
      if image is None:
        print("Ignoring empty camera frame.")
        break

      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image.flags.writeable = False

      # Make detection
      results = pose.process(image)

      # Recolor back to BGR
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      
      # overlay
      image = self.draw.overlay(image)

      # progress bar
      image = cv2.rectangle(image, (0, self.height //8 -10), (self.width //10 * progress_counter, self.height //8),
                              progress_bar_color, cv2.FILLED)
      
      if results.pose_landmarks is not None:
        self.key_points = self.get_keypoints(image, results)
        
        self.pose_algorithm()
        
        # image = self._draw(image)
        image = self.draw.pose_text(image, "Abdominal REPS: " + str(self.abdominals_count))
        image = self.draw.skeleton(image, results)
        
        abdominal_count_prev = abdominal_count_current
        abdominal_count_current = self.abdominals_count
        
        if self.abdominals_count > 0 and abs(abdominal_count_current - abdominal_count_prev) == 1:
          progress_counter += 1
          if progress_counter == 10:
            progress_counter = 0
            progress_bar_color = random.choices(range(128, 256), k= 3)
      
      self.show_stage(image)
      self.show_fps(image)

      out.write(image)
      
      cv2.imshow('Toes To Bar', image)
      if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    self.video_reader.release()