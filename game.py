"""
A game that uses pose tracking to play hole in the wall

@author: Dorothy Zhang
@version: April 2024

"""

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import time
from shapely import Point
from shapely import Polygon 
import numpy as np
import random
import time


RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

SHOULDER = [300, 100]
HEAD = [400,300]
TORO = [50,200]
LEG = [250,300]

images = []

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
    def __init__(self, color, file, screen_width=1000, screen_height=700):
        self.color = color
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.x = 300
        self.y = 10

        img = cv2.imread(file) 
        # resize image to standard and to adjust to distortions
        img = cv2.resize(img, (500,500))
        img = cv2.resize(img, None, fx = 0.75, fy = 0.75)
  
        # converting image into grayscale image 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        
        # setting threshold of gray image 
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
        
        # using a findContours() function 
        self.contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        self.contours = [contour for contour in self.contours if cv2.contourArea(contour) < 100000 and cv2.contourArea(contour) > 20000]
        fig = np.squeeze(self.contours)

        x = self.x
        y = self.y
        self.pts = []
        scale = 1.8
        for idx in range (len(fig)):
            self.pts.append((self.x+fig[idx][0]*scale, self.y+fig[idx][1]*scale))

        self.poly = Polygon(self.pts)
    
    def draw(self, image):
        """
        Enemy is drawn as a circle onto the image

        Args:
            image (Image): The image to draw the enemy onto
        """
        cv2.polylines(image, [np.array(self.pts, np.int32)], 1, self.color, 2)

    def within(self, x, y):
        return self.poly.contains(Point(x,y))

class Game:
    def __init__(self):
        # Load game elements
        self.alive = True
        self.lives = 3
        self.score = 0

        # easy: 0-9
        # medium : 10-18
        # hard: 19-26

        self.figures = [Figure(RED, 'data/easy/easy.png'), Figure(RED, 'data/easy/circle.png'), Figure(RED, 'data/easy/square.png'),
        Figure(RED, 'data/easy/stand.png'), Figure(RED, 'data/easy/star.png'), Figure(RED, 'data/easy/spongebob.png'), Figure(RED, 'data/easy/patrick.png'),
        Figure(RED, 'data/easy/dab.png'), Figure(RED, 'data/easy/yoga.jpg'), Figure(RED, 'data/easy/curve.jpg'),
        Figure(RED, 'data/medium/arms.png'), Figure(RED, 'data/medium/heel.png'), Figure(RED, 'data/medium/jolly.jpg'),
        Figure(RED, 'data/medium/jumping.jpg'), (RED, 'data/medium/kick.png'), Figure(RED, 'data/medium/man.png'),
        Figure(RED, 'data/medium/plank.jpg'), Figure(RED, 'data/medium/sit.png'), Figure(RED, 'data/medium/stretch.png'),
        Figure(RED, 'data/hard/crunch.png'), Figure(RED, 'data/hard/curve.jpg'), Figure(RED, 'data/hard/dance.png'),
        Figure(RED, 'data/hard/Lsit.png'), Figure(RED, 'data/hard/scorpion.png'), Figure(RED, 'data/hard/split.png'),
        Figure(RED, 'data/hard/stag.png'), Figure(RED, 'data/hard/V.png')]

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
            

    def check_in_box(self, image, detection_result):
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
            contained = True

            for i in range(len(pose_landmarks)):
                pose = pose_landmarks[i]

                # Check if we intercept the enemy
                if( pose is not None):
                    coordinates = DrawingUtil._normalized_to_pixel_coordinates(
                    pose.x, pose.y, imageWidth, imageHeight)
                if(coordinates is not None and not self.figures[0].within(coordinates[0], coordinates[1])):
                    contained = False
            print(contained)
            return contained
               
        
    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        # Fun until we close the video  
        r1 = random.randint(0, len(self.figures)-1)
        self.time_limit = 20
        self.start_time = time.time()
        # self.level = int(input("Please enter which level you want to play (Easy = 0, Medium = 1, Hard = 2: "))
        # if(self.level == 0):
        #     r1 = random.randint(0, 9)
        # elif(self.level==1):
        #     r1 = random.randint(10, 18)
        # elif(self.level==2):
        #     r1 = random.randint(19, 26)

        while self.video.isOpened():

            current_time = int(time.time()-self.start_time)
            time_left = self.time_limit-current_time

            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Image comes mirrored, now flip it
            image = cv2.flip(image, 1)

             # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw the figure on the image

            self.figures[r1].draw(image)
            
            # Draw the enemy on the image
            self.draw_landmarks_on_pose(image, results)

            cv2.rectangle(image, (25, 15),(260, 125), (0, 0, 0), -1)

            cv2.putText(image, "time left: " + str(time_left), (50, 100), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=BLUE, thickness=2)
            cv2.putText(image, "score: " + str(self.score), (50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=BLUE, thickness=2)
            cv2.rectangle(image, (975, 15),(1150, 75), (0, 0, 0), -1)
            cv2.putText(image, "lives: " + str(self.lives), (1000, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=BLUE, thickness=2)

            if self.lives <= 0:
                cv2.rectangle(image, (0, 0), (1300, 800), (0, 0 , 0), -1)
                print("game over")
                cv2.putText(image, "YOU DIED", (650, 300), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=GREEN, thickness=2)
                cv2.putText(image, "press r to play again", (650, 300), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=GREEN, thickness=2)

            self.check_in_box(image, results)

            if self.check_in_box(image, results) == True:
                 self.score+=1
                 r1 = random.randint(0, len(self.figures))
            
            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Pose Tracking', image)

            if time_left<=0:
                self.lives-=1
                self.start_time = time.time()
                r1 = random.randint(0, len(self.figures))

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