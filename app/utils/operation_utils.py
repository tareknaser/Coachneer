import numpy as np

class OperationUtils:
  """Helper class for operation utilities"""
  def __init__(self):
    pass

  def find_angle(self, a, b, c, angle_360= False):
    """ Calculate angle between two lines """
    if any(point == (0, 0) for point in [a, b, c]):
        return 0

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if not angle_360:
      if angle > 180.0:
        angle = 360 - angle
  
    return angle

  def find_angle_2points(self, a, b):
    # Calculate the vector between the two pose landmarks
    vector_a_to_b = np.array([b[0] - a[0],
                              b[1] - a[1]])
    # Calculate the vertical vector to the second point
    vertical_vector = np.array([0, 1])
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector_a_to_b, vertical_vector)
    # Calculate the magnitudes of the two vectors
    magnitude_a_to_b = np.linalg.norm(vector_a_to_b)
    magnitude_vertical = np.linalg.norm(vertical_vector)
    # Calculate the angle in radians using the dot product formula
    angle_radians = np.arccos(dot_product / (magnitude_a_to_b * magnitude_vertical))
    # Convert the angle to degrees and return it
    return np.degrees(angle_radians)
  
  def find_distance(self, a, b):
    """ Euclidean distance between two points (a, b) """
    diff_point1 = (a[0] - b[0]) ** 2
    diff_point2 = (a[1] - b[1]) ** 2
    return (diff_point1 + diff_point2) ** 0.5

  def find_distance_h(self, a, b):
    """ Distance between x coordinates of two points """
    return (b[0] - a[0])

  def find_distance_v(self, a, b):
    """ Distance between y coordinates of two points """
    return (b[1] - a[1])
  
  def normalize(self, value, min_val, max_val):
    """ Normalize to [0, 1] range """
    return (value - min_val) / (max_val - min_val)
  
  def normalize_pose_points(self, ref: tuple, pose_points: list):
    # Get the coordinates of the shoulder pose point
    reference_x = ref[0]
    reference_y = ref[1]
    normalized_points = []

    for point in pose_points:
        # Calculate the normalized x and y coordinates
        normalized_x = point[0] - reference_x
        normalized_points.append(normalized_x)

        normalized_y = reference_y - point[1]
        normalized_points.append(normalized_y)

    return normalized_points