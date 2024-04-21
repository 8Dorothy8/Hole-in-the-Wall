"""
A game that uses pose tracking to play hole in the wall

@author: Dorothy Zhang
@version: April 2024

"""

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import random
import time

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Library Constants
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkPoints = mp.solutions.pose.PoseLandmark
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils

class Figure:
    """
    A class to represent a random circle
    enemy. It spawns randomly within 
    the given bounds.
    """
    def __init__(self, color, screen_width=1000, screen_height=600):
        self.color = color
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.x = 400
        self.y = 200
        self.size = 300
    
    def draw(self, image):
        """
        Enemy is drawn as a circle onto the image

        Args:
            image (Image): The image to draw the enemy onto
        """
        cv2.rectangle(image, (self.x, self.y), (self.x + self.size, self.y + self.size), self.color, 5)

class Game:
    def __init__(self):
        # Load game elements
        self.score = 0
        self.figures = [Figure(RED)]

        # Create the hand detector
        base_options = BaseOptions(model_asset_path='pose_landmarker_lite.task')
        options = PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=True)
        self.detector = PoseLandmarker.create_from_options(options)

        # TODO: Load video
        self.video = cv2.VideoCapture(0)

    
    def draw_landmarks_on_pose(self, image, detection_result):
        """
        Draws all the landmarks on the hand
        Args:
            image (Image): Image to draw on
            detection_result (HandLandmarkerResult): HandLandmarker detection results
        """
        # Get a list of the landmarks
        pose_landmarks_list = detection_result.pose_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Save the landmarks into a NormalizedLandmarkList
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])

            # Draw the landmarks on the hand
            solutions.drawing_utils.draw_landmarks(image,
                                       pose_landmarks_proto,
                                       solutions.pose.POSE_CONNECTIONS,
                                       solutions.drawing_styles.get_default_pose_landmarks_style(),
                                       #solutions.drawing_styles.get_default_pose_connections_style()
                                       )
    
    def check_intercept(self, x, y, enemy, enemy_list, image):
        """
        Determines if the person is within the figure
        """
        
        if (x > self.figures[0].x and x < self.figures[0].x + self.figures[0].size and y > self.figures[0].y and y < self.figures[0].y + self.figures[0].size):
            print("Yay")
            

    def check_enemy_kill(self, image, detection_result):
        """
        Draws a green circle on the index finger 
        and calls a method to check if we've intercepted
        with the enemy
        Args:
            image (Image): The image to draw on
            detection_result (HandLandmarkerResult): HandLandmarker detection results
        """
        # Get image details
        imageHeight, imageWidth = image.shape[:2]

        # Get a list of the landmarks
        pose_landmarks_list = detection_result.pose_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Get the coordinates for the index finger
            finger = pose_landmarks[PoseLandmarkPoints.LEFT_SHOULDER.value]

            # Map the coodrinates back to screen dimensions
            pixelCoordinates = DrawingUtil._normalized_to_pixel_coordinates(finger.x,
                                                                            finger.y,
                                                                            imageWidth,
                                                                            imageHeight)


            if pixelCoordinates:
                # Draw the circle around the index finger
                cv2.circle(image, (pixelCoordinates[0], pixelCoordinates[1]), 25, GREEN, 5)

                # Check if we intercept the enemy
                for figure in self.figures:
                    self.check_intercept(pixelCoordinates[0], pixelCoordinates[1], figure, self.figures, image)
        
    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        # Fun until we close the video  
        #self.level = int(input("Please enter which level you want to play (Normal = 0, Time it = 1, Infinite Spawining = 2, Kill them both! = 3: "))
        self.start_time = time.time()

        while self.video.isOpened():

            current_time = time.time()-self.start_time

            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Image comes mirrored, now flip it
            image = cv2.flip(image, 1)

             # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw the enemy on the image
            for figure in self.figures:
                figure.draw(image)
            
            # Draw the enemy on the image
            self.draw_landmarks_on_pose(image, results)

            cv2.putText(image, str(self.score), (50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=GREEN, thickness=2)
            cv2.putText(image, str(current_time), (50, 200), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=GREEN, thickness=2)

            self.check_enemy_kill(image, results)
            
            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Pose Tracking', image)

            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print(self.score)
                break
        
        # Release our video and close all windows
        self.video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":        
    g = Game()
    g.run()