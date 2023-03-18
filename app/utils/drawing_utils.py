import cv2
import numpy as np
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class DrawingUtils():
  """ Helper class for drawing utilities """
  def __init__(self, width: int, height: int) -> None:
      self.width = width
      self.height = height

      # self.font = cv2.FONT_HERSHEY_SIMPLEX
      self.font = cv2.FONT_HERSHEY_COMPLEX
      self.font_size = 0.75
      self.font_thickness = 1
      
      # Define colors
      self.colors = {
          'green'      : (127, 255, 0),
          'dark_blue'  : (127, 20, 0),
          'light_green': (127, 233, 100),
          'pink'       : (255, 0, 255),
          'black'      : (0, 0, 0),
          'gray'       : (25, 25, 25),
          'white'      : (255, 255, 255),
          'blue'       : (255, 50, 50),
          'red'        : (50, 50, 255),
          'yellow'     : (0, 255, 255),
          'magenta'    : (255, 0, 255),
          'cyan'       : (0, 255, 255),
          'light_blue' : (102, 204, 255)
      }

  def draw_skeleton(self, image, pose_results):
    """ Draw skeleton with pose landmarks """
    mp_drawing.draw_landmarks(image,
                              pose_results.pose_landmarks,
                              mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=self.colors['red'],
                                                      thickness=2,
                                                      circle_radius=4),
                              mp_drawing.DrawingSpec(color=self.colors['green'],
                                                      thickness=2,
                                                      circle_radius=2))
    return image

  def draw_overlay(self, image):
    """Draws an overlay on the image"""
    alpha = 0.5
    overlay = np.zeros(image.shape, dtype='uint8')
    cv2.rectangle(overlay, (0, self.height // 16), (self.width, self.height // 8), self.colors['gray'], -1)
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return output	# Returning output instead of modifying image in place

  def draw_line(self, image, start: tuple, end: tuple, color: str, thickness: int):
    """Draws a line on the image from start to end points with the specified color and thickness"""
    cv2.line(image, 
              start,
              end, 
              self.colors[color], 
              thickness)
    return image

  def draw_circle(self, image, center: tuple, radius: int, color: str, thickness: int):
    """ Draw a circle with pose landmarks """
    cv2.circle(image,
                center,
                radius,
                self.colors[color],
                thickness)
    return image
    
  def draw_point(self, image, points: list):
    """ Draw a intersection point in an image """
    for coord in points:
      cv2.circle(image,
                  coord,
                  radius= 10,
                  color= (0, 0, 255),
                  thickness= -1)

      cv2.circle(image,
                  coord,
                  radius= 14,
                  color= (0, 0, 255),
                  thickness= 1)
    return image

  def draw_available_point(self, image, points: list):
    """ Draw a intersection point in an image """
    for point in points:
      center = point
      radius = 10
      thickness = -1
      color = self.colors['red']
      cv2.circle(image, center, radius, color, thickness)
      radius = 14
      thickness = 1
      cv2.circle(image, center, radius, color, thickness)
    return image

  def draw_text(self, image, text: str, pos: tuple, font= None, size= None, txt_color= None, thickness: int = None, background: bool = False, bg_color: tuple = None):
    """ Draw a intersection point in an image """
    if font is None: font = self.font
    if size is None: size = self.font_size
    if thickness is None: thickness = self.font_thickness
    if txt_color is None: txt_color = 'black'

    if background:
      if bg_color is None: bg_color = 'white'

      text_size, _ = cv2.getTextSize(text, font, size, thickness)
      text_w, text_h = text_size
      rect_w, rect_h = text_w + 10, text_h + 15
      rect_x, rect_y = pos[0] - 5, pos[1] - text_h - 5
      
      cv2.rectangle(image, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), self.colors[bg_color], -1)
      cv2.putText(image, text, pos, font, size, self.colors[txt_color], thickness, cv2.LINE_AA)
      
    else:
      cv2.putText(image, text, pos, font, size, self.colors[txt_color], thickness, cv2.LINE_AA)
    return image
		