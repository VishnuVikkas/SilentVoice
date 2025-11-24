SilentVoice: AI-Powered ASL Interpreter & Multilingual Communicator

🌟 Project Overview

SilentVoice is a comprehensive desktop application that provides a full-circle communication solution for American Sign Language (ASL) users. It converts real-time hand gestures into predicted text, offers intelligent word suggestions, and translates/speaks the final sentence in various languages.

The project is structured into two complementary Python applications:

ASL Model Trainer (train.py): A dedicated GUI for collecting new gesture data and training the custom Machine Learning model.

Live Interpreter (runmodel.py): The main application for real-time sign recognition, word prediction, and multilingual speech output.

Developed by: VISHNUVIKKAS J K R, ASHUTOSH CHANDRAKANT PANDEY, SUJAL MISHRA, TANMOY SAHU, and MOHIT PRIYAJ GANDHI, DIVIT BOHRA.

✨ Core Features

SilentVoice is built on a seamless blend of Computer Vision, Machine Learning, and Natural Language Processing.

Real-Time ASL Interpretation: Uses Mediapipe and OpenCV to detect and track 21 points on the hand, feeding the 63 coordinates into a pre-trained MLPClassifier (Scikit-learn) for high-accuracy letter prediction.

Intelligent Word Prediction: Employs NLTK and the Brown Corpus to analyze statistical word transitions, providing context-aware suggestions to speed up communication.

Multilingual Translation: Translates the interpreted ASL sentence into multiple languages (e.g., Hindi, Spanish, Tamil) using the googletrans library.

Speech Output (TTS): The final output is read aloud using gTTS and playsound. The speech function runs asynchronously via threading so the live camera feed never freezes.

Robust GUI: Features a dynamic, dark-themed interface built using Tkinter, complete with a real-time accuracy progress bar and responsive controls.

🏗 System Workflow

The system operates in a two-stage pipeline:

Training Stage:

Gesture Capture: train.py records hand landmarks using Mediapipe.

Model Generation: An MLPClassifier is trained to map these landmarks to ASL letters and saved as a .pkl file.

Interpretation Stage:

Prediction Stability: runmodel.py uses a history buffer to ensure the predicted letter is stable (e.g., held for 10 frames) before adding it to the word, reducing visual noise.

Word Completion: The predicted words are cross-referenced with NLTK for smart, contextual suggestions.

Non-Blocking Output: Translation and Text-to-Speech are executed on separate threads, allowing the user to continue signing while the application speaks.

⚙ Setup and Installation

Prerequisites

Python 3.8+

A working Webcam

📦 Installation Steps

Clone the repository:

git clone [https://github.com/VishnuVikkas/VishnuVikkas.git](https://github.com/VishnuVikkas/SilentVoice.git)
cd SilentVoice


Create a Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate   # Linux/macOS
.\venv\Scripts\activate    # Windows


Install Required Libraries:

pip install opencv-python mediapipe numpy scikit-learn pillow tkinter gtts playsound nltk googletrans==3.1.0a0


Note: We lock the googletrans version for reliability.

NLTK Data:
The application will automatically download the required NLTK data (words and brown) on its first run.

🚀 Getting Started (Two-Step Process)

Step 1: Train Your Model (train.py)

You must train the gesture recognition model first using your specific hand signs.

Run the Trainer GUI:

python train.py


Collect Data: Use the GUI to input a letter (e.g., A) and the number of samples (e.g., 50). Click "▶ Start Collection" and perform the sign repeatedly.

Train: Once enough data is collected, click "🎯 Train Model". This generates the essential asl(1)_landmarks_model.pkl file.

Step 2: Launch the Live Interpreter (runmodel.py)

Ensure the trained model (asl(1)_landmarks_model.pkl) is in the project root.

Run the main application:

python runmodel.py


Click "▶ Start Live Interpreter" in the launcher window to begin real-time sign recognition.

⚠ Important Customization Notes

Asset Paths: The application uses hardcoded file paths for the icon and startup video. You must edit the ICON_PATH and STARTUP_VIDEO_PATH variables inside runmodel.py to point to valid local files on your system before running.

Dependencies: If you encounter issues with audio playback, ensure you have the necessary audio backend libraries for the playsound library installed on your operating system.
