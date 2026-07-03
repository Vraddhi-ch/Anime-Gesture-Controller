# 🎌 Anime Gesture Controller

> Control your computer using anime-inspired hand gestures.

Anime Gesture Controller is a real-time computer vision project that uses a webcam, MediaPipe hand tracking, and machine learning to recognize custom anime gestures.

Perform the **Gojo gesture** to activate a blue “Infinite Void” aura and open ChatGPT. Perform the **Naruto gesture**( yeah..more of like illuminati traiangle sign.. i am sorry for that (TT)) to activate an orange chakra aura and enter hand-controlled cursor mode.

## 🎥 Demo

![Anime Gesture Controller Demo](assets/demo1.gif)

---

## ✨ Features

* Real-time webcam hand tracking
* One-hand and two-hand gesture recognition
* Gojo-inspired blue aura visual effect
* “Domain Expansion: Infinite Void” animation
* Gojo gesture automatically opens ChatGPT
* Naruto-inspired orange chakra aura
* Naruto gesture activates hand-controlled cursor mode
* Move the real mouse cursor using your index fingertip
* Live gesture label and confidence score
* Protection against repeated accidental triggers

---

## 🧠 How It Works

```text
Webcam
  ↓
OpenCV Frame Capture
  ↓
MediaPipe Hand Landmark Detection
  ↓
21 Hand Landmarks Per Hand
  ↓
Gesture Recognition Model
  ↓
Gojo / Naruto / Idle
  ↓
Anime Effects + Desktop Actions
```

Each hand contains 21 landmark points. The application uses these landmarks to recognize custom gestures in real time.

---

## 🛠 Tech Stack

| Technology      | Purpose                                     |
| --------------- | ------------------------------------------- |
| Python          | Main programming language                   |
| OpenCV          | Webcam capture, drawing, and visual effects |
| MediaPipe Tasks | Real-time hand landmark detection           |
| Scikit-learn    | Gesture classification                      |
| Pandas & NumPy  | Landmark data processing                    |
| PyAutoGUI       | Mouse cursor control                        |
| Joblib          | Loading the trained gesture model           |

---

## 📂 Project Structure

```text
Anime-Gesture-Controller/
│
├── assets/
│   └── demo1.gif
│
├── models/
│   ├── hand_landmarker.task
│   └── gesture_classifier.joblib
│
├── scripts/
│   ├── camera.py
│   ├── handtracker.py
│   ├── live_predict.py
│   └── gesture_actions.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Vraddhi-ch/Anime-Gesture-Controller.git
cd Anime-Gesture-Controller
```

## 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

## 4. Confirm Required Model Files

Make sure these files exist inside the `models` folder:

```text
models/
├── hand_landmarker.task
└── gesture_classifier.joblib
```

---

# 🚀 How to Use

Run the full Anime Gesture Controller:

```powershell
python scripts\gesture_actions.py
```

A webcam window will open. Keep your hands clearly visible in the frame.

---

# 🎮 Gesture Controls

| Gesture / Key                  | Result                                                              |
| ------------------------------ | ------------------------------------------------------------------- |
| Gojo gesture                   | Blue aura appears → “Infinite Void” animation plays → ChatGPT opens |
| Naruto gesture                 | Orange chakra aura appears → Naruto cursor mode activates           |
| Index fingertip in cursor mode | Moves the real mouse cursor                                         |
| `M`                            | Turns off Naruto cursor mode                                        |
| `Q`                            | Closes the application                                              |

---

# 🖱 Naruto Cursor Mode

After the Naruto gesture is recognized:

1. An orange chakra animation plays.
2. Cursor mode activates automatically.
3. Move your index fingertip in front of the webcam.
4. The real mouse cursor follows your fingertip.
5. Press `M` anytime to exit cursor mode.

For the smoothest result, use your right-most visible hand and keep it inside the webcam frame.

---

# 💡 Tips for Best Results

* Use bright, even lighting.
* Keep your hand fully visible inside the webcam frame.
* Avoid covering one hand completely with the other.
* For Naruto, keep both hands visible while doing the gesture.
* Wait for the aura animation to finish before making another gesture.
* Return to a neutral hand position before triggering another action.

---

# 🔮 Future Improvements

* Pinch gesture for mouse click
* Two-finger gesture for scrolling
* More anime gestures such as Sasuke, Kakashi, Tanjiro, and Luffy
* Sound effects for each activation
* Custom gesture-to-action settings
* Full-body aura effects using person segmentation
* Desktop app interface
* Windows executable release

---

# 👩‍💻 Author

**Vraddhi Chakrapani**

Built as a computer vision, machine learning, and human-computer interaction project exploring anime-inspired desktop controls.

---

# 📜 License

This project is available for learning, experimentation, and personal use.
