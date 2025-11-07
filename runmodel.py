"""
SilentVoice - ASL Interpreter
A real-time American Sign Language interpreter using computer vision and machine learning.
Features:
- Live hand gesture recognition
- Word formation and suggestions
- Multi-language translation
- Text-to-speech output
"""

# Standard library imports
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import pickle
from googletrans import Translator
from gtts import gTTS
import os
import playsound
import colorsys
import nltk
from nltk.corpus import words, brown
from nltk.util import ngrams
from collections import Counter, defaultdict
import asyncio
import tempfile

# Download and setup NLTK data for word prediction
try:
    nltk.data.find('corpora/words')
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('words')
    nltk.download('brown')

# Build word prediction model using Brown corpus
word_list = set(words.words())
word_transitions = defaultdict(Counter)
for sentence in brown.sents():
    for w1, w2 in zip(sentence[:-1], sentence[1:]):
        word_transitions[w1.lower()][w2.lower()] += 1

# Load the trained ASL recognition model
with open("asl(1)_landmarks_model.pkl", "rb") as f:
    model, le = pickle.load(f)

# Initialize translator for multi-language support
translator = Translator()

def get_word_suggestions(current_word, num_suggestions=5):
    """
    Generate AI-powered word suggestions based on current input.
    
    Args:
        current_word (str): The current word being typed
        num_suggestions (int): Number of suggestions to return
    
    Returns:
        list: List of suggested words
    """
    suggestions = []
    
    # Default suggestions for empty input
    if not current_word:
        return ["HELLO", "I", "THE", "PLEASE", "CAN"]
    
    # Get dictionary completions
    word_completions = [w for w in word_list if w.lower().startswith(current_word.lower())]
    suggestions.extend(word_completions[:2])
    
    # Add statistical predictions from Brown corpus
    if current_word.lower() in word_transitions:
        common_next = word_transitions[current_word.lower()].most_common(3)
        suggestions.extend(word for word, _ in common_next)
    
    return list(dict.fromkeys(suggestions))[:num_suggestions]

def translate_sentence(sentence, target_lang="hi"):
    """
    Translate text to target language using Google Translate.
    
    Args:
        sentence (str): Text to translate
        target_lang (str): Target language code
    
    Returns:
        str: Translated text or error message
    """
    try:
        result = translator.translate(sentence, dest=target_lang)
        if result is None:
            return "Translation error: No result returned."
        
        # Handle async translation results
        if asyncio.iscoroutine(result):
            try:
                coro_result = asyncio.run(translator.translate(sentence, dest=target_lang))
                if coro_result is None:
                    return "Translation error: No result returned from coroutine."
                return getattr(coro_result, 'text', str(coro_result))
            except RuntimeError:
                return sentence
        return getattr(result, 'text', str(result))
    except Exception as e:
        return f"Translation error: {e}"

def speak_multilang(text, lang="en"):
    """
    Convert text to speech using gTTS and play it.
    
    Args:
        text (str): Text to convert to speech
        lang (str): Language code for TTS
    """
    def run():
        try:
            # Create TTS object and generate audio
            tts = gTTS(text=text, lang=lang)
            
            # Create unique temp file for audio
            fd, filename = tempfile.mkstemp(prefix="sv_tts_", suffix=".mp3")
            os.close(fd)
            
            try:
                tts.save(filename)
            except Exception as e:
                # Clean up on save error
                if os.path.exists(filename):
                    os.remove(filename)
                raise

            # Play audio and handle cleanup
            try:
                playsound.playsound(filename)
            except Exception as e:
                print("TTS playback error:", e)

            # Attempt file removal with retries
            removed = False
            for attempt in range(5):
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                    removed = True
                    break
                except (PermissionError, OSError):
                    time.sleep(0.3)

            # Schedule delayed cleanup if immediate removal fails
            if not removed and os.path.exists(filename):
                def delayed_remove(path):
                    try:
                        if os.path.exists(path):
                            os.remove(path)
                    except Exception:
                        pass
                threading.Timer(2.0, delayed_remove, args=(filename,)).start()
                
        except Exception as e:
            print("TTS Error:", e)
            
    # Run TTS in separate thread to avoid blocking
    threading.Thread(target=run, daemon=True).start()

# Initialize MediaPipe hands detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Configuration paths
ICON_PATH = r"C:\Users\91932\Downloads\ChatGPT Image Sep 10, 2025, 05_04_37 PM.png"
STARTUP_VIDEO_PATH = r"C:\python\python projects\mediapipe\WhatsApp Video 2025-10-12 at 22.49.02_66357fbf.mp4"
BACKGROUND_IMAGE_PATH = ""

##########################################HITEN########################################
###########################################TANMOY###########################################
def load_window_icon(win, path):
    try:
        if not path or not os.path.exists(path):
            return
        if path.lower().endswith(".ico"):
            win.iconbitmap(path)
        else:
            img = Image.open(path)
            photo = ImageTk.PhotoImage(img)
            win.iconphoto(False, photo)
            if not hasattr(win, "_icon_refs"):
                win._icon_refs = []
            win._icon_refs.append(photo)
    except Exception as e:
        print("Failed to load window icon:", e)

# --------- Live window class -
class LiveWindow:
    def __init__(self, master=None):
        self._created_root = master is None
        if self._created_root:
            self.win = tk.Tk()
        else:
            self.win = tk.Toplevel(master)

        self.win.title("SilentVoice - ASL Interpreter (Live)")
        load_window_icon(self.win, ICON_PATH)
        self.win.geometry("1200x800")
        # use a solid background color (no transparency)
        self.base_bg = "#0f0f0f"
        self.win.configure(bg=self.base_bg)

        # ====== Top Title Bar ======
        self.header = tk.Label(
            self.win,
            text="SilentVoice",
            font=("Segoe UI", 28, "bold"),
            fg="#ffffff",
            bg="#1c1c1c",
            pady=12
        )
        self.header.pack(fill=tk.X)
        # start hue for RGB cycling
        self._title_hue = 0.0
        self.animate_title()

        # ====== Main Content (Camera + Sidebar) ======
        self.content_frame = tk.Frame(self.win, bg=self.base_bg)
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Camera Feed (LEFT side)
        CAM_BG_W, CAM_BG_H = 800, 600
        self.cam_frame = tk.Frame(self.content_frame, bg="#1c1c1c", bd=4, relief="ridge", width=CAM_BG_W, height=CAM_BG_H)
        self.cam_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.cam_frame.pack_propagate(False)

        self.bg_photo = None
        try:
            if os.path.exists(BACKGROUND_IMAGE_PATH):
                bg_img = Image.open(BACKGROUND_IMAGE_PATH).convert("RGB")
                bg_img = bg_img.resize((CAM_BG_W, CAM_BG_H), Image.Resampling.LANCZOS)
                self.bg_photo = ImageTk.PhotoImage(bg_img)
                self.bg_label = tk.Label(self.cam_frame, image=self.bg_photo)
                self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        except Exception as e:
            print("Background load error:", e)

        # 🔥 Camera feed covers entire frame
        self.lmain = tk.Label(self.cam_frame, bg="#0f0f0f")
        self.lmain.pack(fill="both", expand=True)

        # Sidebar (RIGHT side)
        self.sidebar = tk.Frame(self.content_frame, bg=self.base_bg, width=700)
        self.sidebar.pack(side=tk.RIGHT, fill="y", padx=10, pady=10)

        self.label_pred = tk.Label(self.sidebar, text="Letter: -", font=("Segoe UI", 40, "bold"),
                                   fg="#00ff88", bg=self.base_bg, anchor="w")
        self.label_pred.pack(pady=(20,8), fill="x", padx=10)

        self.label_word = tk.Label(self.sidebar, text="Word:", font=("Segoe UI", 22),
                                   fg="#f1c40f", bg=self.base_bg, wraplength=660, justify="left", anchor="w")
        self.label_word.pack(pady=(8,10), fill="x", padx=10)

        # Word suggestions frame
        self.suggestions_frame = tk.Frame(self.sidebar, bg=self.base_bg)
        self.suggestions_frame.pack(fill="x", padx=10, pady=5)
        self.suggestion_buttons = []
        for i in range(5):
            btn = tk.Button(self.suggestions_frame, text="", font=("Segoe UI", 12),
                          bg="#2c3e50", fg="white", relief="flat", command=lambda x=i: self.use_suggestion(x))
            btn.pack(side=tk.LEFT, padx=2, pady=5, expand=True, fill="x")
            self.suggestion_buttons.append(btn)

        # Define available languages early so translate dropdown can be placed near the Translate button
        self.languages = {
            "English": "en",
            "Hindi": "hi",
            "Tamil": "ta",
            "Telugu": "te",
            "Marathi": "mr"
        }
        # variable for translate dropdown (used by translate button)
        self.translate_lang = tk.StringVar(self.win)
        self.translate_lang.set("English")
        
        # Controls (Translate + language dropdown + Speak) for the detected word
        self.controls_frame = tk.Frame(self.sidebar, bg=self.base_bg)
        self.controls_frame.pack(fill="x", padx=10, pady=(8,0))

        # Translate button
        self.translate_speak_btn = tk.Button(self.controls_frame, text="Translate", 
                     font=("Segoe UI", 12, "bold"), bg="#2980b9", fg="white",
                     command=self.translate_and_speak)
        self.translate_speak_btn.pack(side=tk.LEFT, padx=(0,6))

        # Translate language dropdown (placed next to Translate button)
        self.translate_menu = tk.OptionMenu(self.controls_frame, self.translate_lang, *self.languages.keys())
        self.translate_menu.config(font=("Segoe UI", 11), bg="#2980b9", fg="white", relief="flat", width=10)
        try:
            tm = self.translate_menu.nametowidget(self.translate_menu.menuname)
            tm.config(bg="#2c2c2c", fg="white")
        except Exception:
            pass
        self.translate_menu.pack(side=tk.LEFT, padx=(0,8))

        self.word_speak_btn = tk.Button(self.controls_frame, text="🔊 Speak", 
                     font=("Segoe UI", 12, "bold"), bg="#e67e22", fg="white",
                     command=self.speak_word)
        self.word_speak_btn.pack(side=tk.LEFT, fill="x", expand=True)

        self.label_fps = tk.Label(self.sidebar, text="FPS: 0", font=("Segoe UI", 16),
                                  fg="#ff5555", bg=self.base_bg, anchor="w")
        self.label_fps.pack(pady=10, fill="x", padx=10)

        # ===== Bottom Control Bar =====
        self.bottom_bar = tk.Frame(self.win, bg=self.base_bg, pady=10)
        self.bottom_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_start = self.make_button(self.bottom_bar, "■ End", "#c0392b", self.toggle_camera)
        self.btn_start.pack(side=tk.LEFT, padx=12)
        # Remove the speak button from here
        self.btn_backspace = self.make_button(self.bottom_bar, "⌫ Backspace", "#7f8c8d", self.remove_last_char)
        self.btn_backspace.pack(side=tk.LEFT, padx=12)
        self.btn_pause = self.make_button(self.bottom_bar, "⏸ Pause", "#8e44ad", self.toggle_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=12)

        # Bottom-screen language dropdown removed. Keep a default selected_lang variable
        # so other methods that reference it continue to work.
        self.selected_lang = tk.StringVar(self.win)
        self.selected_lang.set("English")

        self.acc_panel = tk.Frame(self.bottom_bar, bg=self.base_bg)
        self.acc_panel.pack(side=tk.RIGHT, padx=(10,20), pady=4)
        
        self.label_acc = tk.Label(self.acc_panel, text="Accuracy", font=("Segoe UI", 10, "bold"),
                                  fg="#cccccc", bg=self.base_bg)
        self.label_acc.pack(anchor="e")

        style = ttk.Style()
        try:
            style.theme_use('default')
        except Exception:
            pass
        style.configure("Acc.Horizontal.TProgressbar", troughcolor="#2c2c2c", background="#2ecc71", thickness=12)
        self.acc_bar = ttk.Progressbar(self.acc_panel, orient="horizontal", mode="determinate",
                                       maximum=100, style="Acc.Horizontal.TProgressbar", length=220)
        self.acc_bar.pack(anchor="e", pady=(4,0))

        # Camera + Variables
        self.cap = None
        self.running = True
        self.paused = False
        self.prev_time = 0
        self.pred_history = []
        self.max_history = 5
        self.last_letter = ""
        self.word = ""
        self.letter_hold_count = 0
        self.hold_threshold = 10

        self.win.bind('<space>', self.add_space)
        self.win.bind('<BackSpace>', self.remove_last_char)
        self.win.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.update_frame()

    def make_button(self, parent, text, color, command):
        return tk.Button(parent, text=text, font=("Segoe UI", 14, "bold"),
                         bg=color, fg="white", relief="flat",
                         activebackground="#2c2c2c", padx=15, pady=8, command=command)

    def add_space(self, event=None):
        self.word += " "
        self.label_word.config(text=f"Word: {self.word}")

    def remove_last_char(self, event=None):
        if self.word:
            self.word = self.word[:-1]
            self.label_word.config(text=f"Word: {self.word}")

    def toggle_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        sentence = self.word.strip()
        if sentence:
            self.speak_sentence(sentence)
        self.win.destroy()

    def toggle_pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.config(text="▶ Resume", bg="#16a085")
        else:
            self.btn_pause.config(text="⏸ Pause", bg="#8e44ad")

    def speak_word(self):
        sentence = self.word.strip()
        if sentence:
            self.speak_sentence(sentence)
        else:
            messagebox.showinfo("Speak", "No word detected to speak.")

    def speak_sentence(self, sentence):
        # Speak the provided sentence as-is using the language selected in the translate dropdown.
        # Do NOT auto-translate here — user must click Translate first if they want the displayed word to be translated.
        lang_code = self.languages.get(self.translate_lang.get(), "en")
        # Update displayed word to whatever is being spoken (keeps UI consistent)
        self.label_word.config(text=f"Word: {sentence}")
        speak_multilang(sentence, lang=lang_code)

    def update_suggestions(self):
        """Update word suggestion buttons based on current word"""
        suggestions = get_word_suggestions(self.word.strip().split()[-1] if self.word.strip() else "")
        for btn, sugg in zip(self.suggestion_buttons, suggestions + [""] * 5):
            btn.config(text=sugg)
            
    def use_suggestion(self, index):
        """Use the selected word suggestion"""
        if index < len(self.suggestion_buttons):
            suggestion = self.suggestion_buttons[index]["text"]
            if suggestion:
                # Replace last word if exists, otherwise append
                words = self.word.strip().split()
                if words:
                    words[-1] = suggestion
                else:
                    words.append(suggestion)
                self.word = " ".join(words)
                self.label_word.config(text=f"Word: {self.word}")
                self.update_suggestions()

    def translate_and_speak(self):
        """Translate the currently detected word and update UI. Do NOT speak."""
        lang_code = self.languages.get(self.translate_lang.get(), "en")
        sentence = self.word.strip()
        if sentence:
            translated_sentence = translate_sentence(sentence, lang_code)
            # update internal word so Speak will speak the translated text
            self.word = translated_sentence
            self.label_word.config(text=f"Word: {translated_sentence}")
        else:
            messagebox.showinfo("Translate", "No detected word to translate.")

    def get_stable_prediction(self, pred_class):
        self.pred_history.append(pred_class)
        if len(self.pred_history) > self.max_history:
            self.pred_history.pop(0)
        return max(set(self.pred_history), key=self.pred_history.count)

    def update_frame(self):
        if not self.running:
            return

        if self.paused:
            self.win.after(10, self.update_frame)
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    lm_list = []
                    for lm in hand_landmarks.landmark:
                        lm_list.extend([lm.x, lm.y, lm.z])

                    try:
                        proba = model.predict_proba([lm_list])[0]
                        pred_idx = np.argmax(proba)
                        letter = le.inverse_transform([pred_idx])[0]
                        confidence = proba[pred_idx] * 100
                    except Exception:
                        letter = ""
                        confidence = 0.0

                    stable_letter = self.get_stable_prediction(letter) if letter else ""
                    self.label_pred.config(text=f"Letter: {stable_letter}")

                    if confidence > 80:
                        color = "#2ecc71"
                    elif confidence > 50:
                        color = "#f1c40f"
                    else:
                        color = "#e74c3c"
                    
                    self.label_acc.config(text=f"Accuracy: {confidence:.2f}%", fg=color)
                    try:
                        self.acc_bar['value'] = max(0.0, min(100.0, confidence))
                    except Exception:
                        pass

                    if stable_letter == self.last_letter:
                        self.letter_hold_count += 1
                    else:
                        self.letter_hold_count = 0
                        self.last_letter = stable_letter

                    if self.letter_hold_count == self.hold_threshold and stable_letter:
                        self.word += stable_letter
                        self.label_word.config(text=f"Word: {self.word}")
                        # Update word suggestions when word changes
                        self.update_suggestions()
            else:
                try:
                    self.acc_bar['value'] = 0
                except Exception:
                    pass

            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time + 1e-6)
            self.prev_time = curr_time
            self.label_fps.config(text=f"FPS: {int(fps)}")

            # 🔥 Resize frame to cover full cam_frame
            img = Image.fromarray(frame_rgb)
            w = self.cam_frame.winfo_width()
            h = self.cam_frame.winfo_height()
            if w > 1 and h > 1:
                img = img.resize((w, h), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)
            self.lmain.imgtk = imgtk
            self.lmain.configure(image=imgtk)

        self.win.after(10, self.update_frame)

    def on_closing(self):
        self.running = False
        if self.cap:
            self.cap.release()
        hands.close()
        self.win.destroy()

    # animate title color (HSV -> RGB cycling)
    def animate_title(self):
        try:
            r, g, b = colorsys.hsv_to_rgb(self._title_hue, 1.0, 1.0)
            hexcol = '#%02x%02x%02x' % (int(r * 255), int(g * 255), int(b * 255))
            self.header.config(fg=hexcol)
            self._title_hue += 0.008
            if self._title_hue >= 1.0:
                self._title_hue -= 1.0
        except Exception:
            pass
        self.win.after(50, self.animate_title)

# --------- Startup window (with dynamic video background) ----------
class StartupWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SilentVoice - Launcher")
        load_window_icon(self.root, ICON_PATH)
        self.root.geometry("600x400")
        self.root.configure(bg="#121212")

        # background video label (fills window)
        self.bg_label = tk.Label(self.root, bd=0)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # attempt to open startup video
        self._cap = None
        try:
            if STARTUP_VIDEO_PATH and os.path.exists(STARTUP_VIDEO_PATH):
                self._cap = cv2.VideoCapture(STARTUP_VIDEO_PATH)
                if not self._cap.isOpened():
                    self._cap = None
        except Exception:
            self._cap = None

        # fallback: if no video, keep solid background (previous title will show)
        # overlay UI (will appear above bg_label)
        # title = tk.Label(self.root, text="SILENTVOICE", font=("Segoe UI", 32, "bold"),
        #                  fg="white", bg="#121212")
        # title.place(relx=0.5, rely=0.18, anchor="center")

        # subtitle = tk.Label(self.root, text="The Symphony of Aphonics", font=("Segoe UI", 16),
        #                     fg="#cccccc", bg="#121212")
        # subtitle.place(relx=0.5, rely=0.30, anchor="center")

        start_btn = tk.Button(self.root, text="▶ Start Live Interpreter", font=("Segoe UI", 16, "bold"),
                              bg="#27ae60", fg="white", padx=20, pady=10, command=self.open_live)
        start_btn.place(relx=0.5, rely=0.80, anchor="center")

        # info = tk.Label(self.root, text="Press Start to open the live interpreter window.", fg="#aaaaaa", bg="#121212")
        # info.place(relx=0.5, rely=0.70, anchor="center")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # start video loop if available
        if self._cap:
            self._video_running = True
            self._update_startup_video()
        else:
            # show a static background color/image if you want; keep bg_label empty for solid bg
            pass

        self.root.mainloop()

    def _update_startup_video(self):
        if not self._video_running or not self._cap:
            return
        ret, frame = self._cap.read()
        if not ret:
            # loop video
            try:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
            except Exception:
                ret = False

        if ret:
            # convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # resize to current window size
            w = max(1, self.root.winfo_width())
            h = max(1, self.root.winfo_height())
            try:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass
            img = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(img)
            # keep reference to avoid GC
            self.bg_label.imgtk = photo
            self.bg_label.configure(image=photo)
            # ensure bg_label is at back
            self.bg_label.lower()

        # schedule next frame
        self.root.after(30, self._update_startup_video)

    def open_live(self):
        try:
            # stop and release startup video
            if hasattr(self, "_video_running"):
                self._video_running = False
            if self._cap:
                try:
                    self._cap.release()
                except:
                    pass
            self.root.destroy()
        except:
            pass
        LiveWindow(None)

    def on_close(self):
        try:
            if hasattr(self, "_video_running"):
                self._video_running = False
            if self._cap:
                try:
                    self._cap.release()
                except:
                    pass
        except:
            pass
        try:
            hands.close()
        except:
            pass
        self.root.destroy()

if __name__ == "__main__":
    StartupWindow()