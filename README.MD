# Coachneer: AI Personal Trainer at Home

## Introduction

This technical documentation provides detailed information about the internal workings of the Coachneer project, including its model architecture, design, implementation details, and mobile app integration. It is intended for developers and other technical stakeholders who want to understand how the software works and how to contribute to its development.

## Model Architecture

Our project uses Google MediaPipe pose estimation model to capture landmarks on the human body in real-time. From these landmarks, we perform feature enginnering to classify the current stage of the workout and count reps. These calculations include:

- Euclidean distance between two key points
- Vertical distance between two key points
- The horizontal distance between two key points
- Specifying point position
- Angle between three key points

Using these calculations, we predict the training workout and detect any bad posture during the workout in real time. We use a pipeline of Random Forest Classifier, Gradient Boosting Classifier, Logistic Regression, and Ridge Classifier to train our model and evaluate results. We choose the best output based on the evaluation metrics. 

Our model consists of four main components:

1. Input Layer: The input layer takes the user's video feed as input and processes it using the Google Mediapipe Pose Estimation model to detect and track keypoints in the user's body.

2. Feature Extraction Layer: The feature extraction layer takes the output from the input layer and performs several calculations on the captured landmarks.

3. Classification Layer: The classification layer takes the output from the feature extraction layer. Predicts the user's training workout using the AI algorithms on pose landmarks and extracted features. Classify bad posture in real-time.


## Structure

The repository is structured into two main folders:

- **app**: Contains the `main.py` file that runs our app, and two subfolders:
  - **data**: Contains the data we use, such as fonts, input videos, or outputs like output videos.
  - **utils**: Contains our app's main utilities files.
- **classification**: Contains notebooks for training our models, and three subfolders:
  - **coords**: The export path after capturing landmarks from training videos and being labeled.
  - **models**: The created model export path.
  - **videos**: Contains a sample of our training videos


## Future Work

In the future, we plan to integrate the Coachneer model with the mobile app to provide a seamless user experience. The mobile app will allow users to record their training sessions, track their progress, and receive real-time feedback on their posture. We will also explore new features such as personalized training plans and social sharing to enhance the user's training experience.

## Conclusion

The Coachneer project is an AI model that uses Google Mediapipe Pose Estimation model to detect and track keypoints in the user's body during training. It uses AI algorithms to predict the user's training workout and to train a classifier model to classify bad posture during the workout. The project is implemented using Python and is structured into an input layer, feature extraction layer, and classification layer. The mobile app will provide users with a convenient way to use the Coachneer model and achieve their training goals.