import math
import numpy as np

class Operation():
  """ Helper class for operation utilities """
  def __init__(self) -> None:
    pass

  def angle_of_singleline(self, a, b):
    """ Calculate angle of a single line """
    x_diff = b[0] - a[0]
    y_diff = b[1] - a[1]
    return math.degrees(math.atan2(y_diff, x_diff))

  def angle(self, a, b, c):
    """ Calculate angle between two lines """
    if(a == (0,0) or b == (0,0) or c == (0,0)):
      return 0
    
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

  def dist_xy(self, a, b):
    """ Euclidean distance between two points (a, b) """
    diff_point1 = (a[0] - b[0]) ** 2
    diff_point2 = (a[1] - b[1]) ** 2
    return (diff_point1 + diff_point2) ** 0.5

  def dist_x(self, a, b):
    """ Distance between x coordinates of two points """
    return abs(b[0] - a[0])

  def dist_y(self, a, b):
    """ Distance between y coordinates of two points """
    return abs(b[1] - a[1])

  def point_position(self, point, line_pt_1, line_pt_2):
    value = (line_pt_2[0] - line_pt_1[0]) * (point[1] - line_pt_1[1]) - \
            (line_pt_2[1] - line_pt_1[1]) * (point[0] - line_pt_1[0])
    
    if value >= 0:
      return "left"
    return "right"

  def normalize(self, value, min_val, max_val):
    """ Normalize to [0, 1] range """
    return (value - min_val) / (max_val - min_val)
