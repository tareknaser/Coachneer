import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence= 0.5, min_tracking_confidence= 0.5)


class VideoReaderUtils:
    """Helper class for video utilities"""
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image, results
    
    def read_frame(self):
        """Read a frame"""
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or (frame is None):
                print( f"Ignoring Empty Frame.. {ret}" )
                return 
            self.current_frame += 1
        else:
            return "Error File Not Found"
        return frame

    def read_n_frames(self, num_frames=1):
        """Read n frames"""
        frames_list = []
        for _ in range(num_frames):
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or (frame is None):
                    return f"Ignoring Empty Frame.\nSUCCESS: {ret}"
                frames_list.append(frame)
                self.current_frame += 1
            else:
                return "Error File Not Found"
        return frames_list

    def is_opened(self):
        """Check if video capture is opened"""
        return self.cap.isOpened()

    def get_frame_width(self):
        """Get width of a frame"""
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    def get_frame_height(self):
        """Get height of a frame"""
        return self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame_size(self):
        """Get size of a frame"""
        return self.get_frame_width(), self.get_frame_height()
    
    def get_video_fps(self):
        """Get frames per second of video"""
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_current_frame(self):
        """Get current frame of video being read"""
        return self.current_frame

    def get_total_frames(self):
        """Get total frames of a video"""
        return self.total_frames

    def release(self):
        """Release video capture"""
        self.cap.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.release()