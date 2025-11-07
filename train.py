# Standard library imports
import cv2
import mediapipe as mp
import csv
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import threading

# Define file paths for data storage and model
DATA_FILE = "asl_landmarks_data.csv"    # CSV file to store hand landmark data
MODEL_FILE = "asl(1)_landmarks_model.pkl"  # File to save trained model

# Initialize MediaPipe hands module for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

class TrainModelGUI:
    """
    Main GUI class for ASL Model Training application.
    Provides interface for data collection and model training.
    """
    def __init__(self, root):
        """
        Initialize the GUI application.
        Args:
            root: The root tkinter window
        """
        self.root = root
        self.root.title("ASL Model Trainer - SilentVoice")
        self.root.geometry("1000x700")
        self.root.configure(bg="#0f0f0f")
        
        # Initialize variables
        self.cap = None                  # Video capture object
        self.is_collecting = False       # Flag for data collection state
        self.current_letter = ""         # Current letter being collected
        self.samples_to_collect = 0      # Total samples to collect
        self.samples_collected = 0       # Current number of samples
        self.is_paused = False          # Pause state flag
        self.collecting_thread = None    # Thread for data collection
        
        # Setup UI components
        self.create_ui()

    def create_ui(self):
        """
        Create and setup all UI components including frames, buttons, and labels.
        Organizes the interface into main sections: video feed, controls, and logging.
        """
        header = tk.Label(self.root, text="ASL Model Trainer", font=("Segoe UI", 24, "bold"),fg="#ffffff", bg="#1c1c1c", pady=10)
        header.pack(fill=tk.X)
        main_frame = tk.Frame(self.root, bg="#0f0f0f")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        left_frame = tk.Frame(main_frame, bg="#1c1c1c", bd=2, relief="ridge")
        left_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 10))
        cam_label = tk.Label(left_frame, text="Camera Preview", font=("Segoe UI", 14, "bold"), fg="#00ff88", bg="#1c1c1c")
        cam_label.pack(pady=5)
        self.video_label = tk.Label(left_frame, bg="#0f0f0f")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)
        right_frame = tk.Frame(main_frame, bg="#0f0f0f", width=400)
        right_frame.pack(side=tk.RIGHT, fill="both", padx=(10, 0))
        collection_frame = tk.LabelFrame(right_frame, text="Data Collection", font=("Segoe UI", 12, "bold"), fg="#00ff88", bg="#1c1c1c", padx=15, pady=15)
        collection_frame.pack(fill="x", pady=(0, 15))
        tk.Label(collection_frame, text="Letter (A-Z):", font=("Segoe UI", 11), fg="#ffffff", bg="#1c1c1c").grid(row=0, column=0, sticky="w", pady=5)
        self.letter_entry = tk.Entry(collection_frame, font=("Segoe UI", 11), width=10)
        self.letter_entry.grid(row=0, column=1, pady=5, sticky="ew")
        tk.Label(collection_frame, text="Samples:", font=("Segoe UI", 11), fg="#ffffff", bg="#1c1c1c").grid(row=1, column=0, sticky="w", pady=5)
        self.samples_entry = tk.Entry(collection_frame, font=("Segoe UI", 11), width=10)
        self.samples_entry.insert(0, "50")
        self.samples_entry.grid(row=1, column=1, pady=5, sticky="ew")
        collection_frame.columnconfigure(1, weight=1)
        self.progress_label = tk.Label(collection_frame, text="Progress: 0/0", font=("Segoe UI", 10), fg="#f1c40f", bg="#1c1c1c")
        self.progress_label.grid(row=2, column=0, columnspan=2, pady=10)
        self.progress_bar = ttk.Progressbar(collection_frame, orient="horizontal", mode="determinate", maximum=100, length=300)
        self.progress_bar.grid(row=3, column=0, columnspan=2, pady=5)
        btn_frame = tk.Frame(collection_frame, bg="#1c1c1c")
        btn_frame.grid(row=4, column=0, columnspan=2, pady=15)
        self.start_btn = tk.Button(btn_frame, text="▶ Start Collection", font=("Segoe UI", 11,"bold"), bg="#27ae60", fg="white", padx=15, pady=8, command=self.start_collection)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = tk.Button(btn_frame, text="■ Stop", font=("Segoe UI", 11, "bold"), bg="#c0392b", fg="white", padx=15, pady=8, command=self.stop_collection, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.pause_btn = tk.Button(btn_frame, text="⏸ Pause", font=("Segoe UI", 11, "bold"), bg="#f1c40f", fg="white", padx=15, pady=8, command=self.pause_collection, state="normal")
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        self.resume_btn = tk.Button(btn_frame, text="▶ Resume", font=("Segoe UI", 11, "bold"), bg="#27ae60", fg="white", padx=15, pady=8, command=self.resume_collection, state="disabled")
        self.resume_btn.pack(side=tk.LEFT, padx=5)
        training_frame = tk.LabelFrame(right_frame, text="Model Training", font=("Segoe UI", 12, "bold"), fg="#00ff88", bg="#1c1c1c", padx=15, pady=15)
        training_frame.pack(fill="x", pady=(0, 15))
        self.train_btn = tk.Button(training_frame, text="🎯 Train Model", font=("Segoe UI", 12, "bold"),bg="#3498db", fg="white", padx=20, pady=10, command=self.train_model)
        self.train_btn.pack(pady=10)
        self.training_status = tk.Label(training_frame, text="Status: Not trained", font=("Segoe UI", 10), fg="#cccccc", bg="#1c1c1c")
        self.training_status.pack()
        log_frame = tk.LabelFrame(right_frame, text="Activity Log", font=("Segoe UI", 12, "bold"), fg="#00ff88", bg="#1c1c1c", padx=10, pady=10)
        log_frame.pack(fill="both", expand=True)
        self.log_text = scrolledtext.ScrolledText(log_frame, font=("Consolas", 9), bg="#0f0f0f", fg="#00ff88", height=10, wrap=tk.WORD)
        self.log_text.pack(fill="both", expand=True)
        self.init_camera()
        self.update_video()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def log(self, message):
        """
        Add a message to the log window and console.
        Args:
            message: The message to log
        """
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        print(message)

    def init_camera(self):
        """
        Initialize the webcam capture with specified resolution.
        Attempts to open the default camera (index 0).
        """
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.log("✅ Camera initialized")

    def update_video(self):
        """
        Update the video feed display.
        Processes each frame to detect and draw hand landmarks.
        Called repeatedly to maintain real-time video feed.
        """
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                # Convert BGR to RGB for MediaPipe processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Detect hand landmarks
                results = hands.process(frame_rgb)
                
                # Draw landmarks if hands are detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Convert and display the frame
                img = Image.fromarray(frame_rgb)
                img = img.resize((480, 360), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        
        # Schedule next frame update
        self.root.after(10, self.update_video)

    def append_to_csv(self, data, label):
        with open(DATA_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data + [label])

    def pause_collection(self):
        """
        Pause the data collection process.
        Changes the state to paused and updates the UI accordingly.
        """
        if self.is_collecting and not self.is_paused:
            self.is_paused = True
            self.pause_btn.config(state="disabled")
            self.resume_btn.config(state="normal")
            self.log("⏸ Paused collection")

    def resume_collection(self):
        """
        Resume the paused data collection process.
        Changes the state to collecting and updates the UI accordingly.
        """
        if self.is_collecting and self.is_paused:
            self.is_paused = False
            self.pause_btn.config(state="normal")
            self.resume_btn.config(state="disabled")
            self.log("▶ Resumed collection")

    def collection_worker(self):
        """
        Worker thread function for collecting hand landmark data.
        Runs continuously until required samples are collected or stopped manually.
        Handles data collection, progress updates, and error conditions.
        """
        self.samples_collected = 0
        while self.is_collecting and self.samples_collected < self.samples_to_collect:
            if self.is_paused:
                self.root.after(100, lambda: None)
                continue
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    
                    # Check for multiple hands
                    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1:
                        self.log("⚠️ Multiple hands detected! Training will stop.")
                        self.training_status.config(text="Status: Stopped - Multiple hands detected", fg="#e74c3c")
                        messagebox.showwarning("Warning", "Training stopped: Multiple hands detected.\nPlease use only ONE hand for training.")
                        self.stop_collection()
                        return

                    # Continue with single hand detection
                    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
                        for hand_landmarks in results.multi_hand_landmarks:
                            lm_list = []
                            for lm in hand_landmarks.landmark:
                                lm_list.extend([lm.x, lm.y, lm.z])
                            self.append_to_csv(lm_list, self.current_letter)
                            self.samples_collected += 1
                            progress = (self.samples_collected / self.samples_to_collect) * 100
                            self.progress_bar['value'] = progress
                            self.progress_label.config(text=f"Progress: {self.samples_collected}/{self.samples_to_collect}")
                            self.log(f"Collected sample {self.samples_collected}/{self.samples_to_collect} for '{self.current_letter}'")
                            if self.samples_collected >= self.samples_to_collect:
                                break
        self.is_collecting = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.pause_btn.config(state="normal")
        self.resume_btn.config(state="disabled")
        self.log(f"✅ Collection complete for letter '{self.current_letter}'")
        messagebox.showinfo("Success", f"Collected {self.samples_collected} samples for '{self.current_letter}'")

    def start_collection(self):
        """
        Start the data collection process.
        Validates input parameters and initializes the collection thread.
        Updates UI to reflect the collecting state.
        """
        letter = self.letter_entry.get().strip().upper()
        if not letter or len(letter) != 1 or not letter.isalpha():
            messagebox.showerror("Error", "Please enter a single letter (A-Z)")
            return
        try:
            samples = int(self.samples_entry.get())
            if samples <= 0:
                raise ValueError()
        except:
            messagebox.showerror("Error", "Please enter a valid number of samples")
            return
        self.current_letter = letter
        self.samples_to_collect = samples
        self.is_collecting = True
        self.is_paused = False
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.pause_btn.config(state="normal")
        self.resume_btn.config(state="disabled")
        self.log(f"🎬 Starting collection for letter '{letter}' ({samples} samples)")
        self.log("👋 Show your hand sign to the camera...")
        self.collecting_thread = threading.Thread(target=self.collection_worker, daemon=True)
        self.collecting_thread.start()

    def stop_collection(self):
        """
        Stop the data collection process.
        Updates the state and UI to reflect the stopped condition.
        """
        self.is_collecting = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.pause_btn.config(state="normal")
        self.resume_btn.config(state="disabled")
        self.log("⏹ Collection stopped")

    def train_model(self):
        """
        Initiates the model training process in a separate thread.
        Handles data loading, model training, evaluation, and saving.
        Updates UI with progress and results.
        """
        self.log("🎯 Starting model training...")
        self.training_status.config(text="Status: Training...", fg="#f1c40f")
        self.train_btn.config(state="disabled")
        def train_worker():
            try:
                X, y = [], []
                if not os.path.exists(DATA_FILE):
                    self.log("❌ No data file found! Collect data first.")
                    messagebox.showerror("Error", "No data found. Please collect data first.")
                    self.train_btn.config(state="normal")
                    self.training_status.config(text="Status: Failed", fg="#e74c3c")
                    return
                with open(DATA_FILE, "r") as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) > 0:
                            X.append([float(val) for val in row[:-1]])
                            y.append(row[-1])
                if len(X) == 0:
                    self.log("❌ No data found in CSV file!")
                    messagebox.showerror("Error", "CSV file is empty. Please collect data first.")
                    self.train_btn.config(state="normal")
                    self.training_status.config(text="Status: Failed", fg="#e74c3c")
                    return
                self.log(f"📊 Loaded {len(X)} samples")
                le = LabelEncoder()
                y_enc = le.fit_transform(y)
                X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
                self.log(f"📚 Training set: {len(X_train)} samples")
                self.log(f"📝 Test set: {len(X_test)} samples")
                model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
                model.fit(X_train, y_train)
                accuracy = model.score(X_test, y_test)
                self.log(f"✅ Model trained successfully!")
                self.log(f"📈 Accuracy: {accuracy * 100:.2f}%")
                with open(MODEL_FILE, "wb") as f:
                    pickle.dump((model, le), f)
                self.log(f"💾 Model saved as '{MODEL_FILE}'")
                self.training_status.config(text=f"Status: Trained (Accuracy: {accuracy * 100:.1f}%)", fg="#2ecc71")
                messagebox.showinfo("Success", f"Model trained successfully!\n\nAccuracy: {accuracy * 100:.2f}%\nSaved as: {MODEL_FILE}")
            except Exception as e:
                self.log(f"❌ Training failed: {str(e)}")
                messagebox.showerror("Error", f"Training failed:\n{str(e)}")
                self.training_status.config(text="Status: Failed", fg="#e74c3c")
            finally:
                self.train_btn.config(state="normal")
        threading.Thread(target=train_worker, daemon=True).start()

    def on_closing(self):
        """
        Cleanup method called when closing the application.
        Releases camera resources and closes the window.
        """
        self.is_collecting = False
        if self.cap:
            self.cap.release()
        hands.close()
        self.root.destroy()

# Entry point of the application
if __name__ == "__main__":
    root = tk.Tk()
    app = TrainModelGUI(root)
    root.mainloop()
