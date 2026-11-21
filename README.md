# 🖐️ Sign to Speech — A Sign Language-to-Speech Conversion System

An artificial intelligence system that recognizes hand gestures via a webcam and converts them into spoken Arabic—built entirely using open-source tools, with no specialized hardware required.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![OpenCV](https://img.shields.io/badge/OpenCV-MediaPipe-orange)

---

## 💡 The Concept

The goal is to build a "bridge" that helps deaf individuals communicate with those who do not know sign language. You perform a hand gesture in front of the camera; the system recognizes it, displays it on the screen, and speaks it aloud.

> Note: This is a prototype that recognizes a set of gestures defined by the user; it is not a full-scale translation of a complete sign language.

---

## ⚙️ How It Works

1. **Hand Tracking:** The camera captures the hand, and MediaPipe maps it into 21 key points representing the joints.
2. **Feature Extraction:** The key points are normalized so that the data remains independent of the hand's position or size within the frame.
3. **Recognition:** A K-Nearest Neighbors (KNN) algorithm compares the gesture against the data the system has learned and identifies the closest match.
4. **Speech Synthesis:** The name of the gesture is spoken in Arabic using gTTS (Google Text-to-Speech).

Key Advantage: There is no need for a massive dataset—the system **learns gestures directly from you** (requiring only a few dozen samples per gesture). ---

## ✨ Features

- A dynamic interface alongside the video feed, displaying the recognized sign, confidence score, and the list of signs.
- Easy training of new signs directly from within the application window (no terminal input required).
- Full support for the Arabic language (both in display and text-to-speech).
- Works on any laptop with a webcam—no additional hardware required.

---

## 🛠️ Requirements

```bash
pip install opencv-python mediapipe numpy pillow arabic_reshaper python-bidi gTTS pygame
```

> Requires Python 3.12 (or an earlier version compatible with MediaPipe) and an internet connection for audio generation via gTTS.

---

## ▶️ Running the Application

```bash
python sign_studio.py
```

### Controls (within the camera window)
| Key | Function |
|-----|----------|
| `T` | Train a new sign (then select its number from the list) |
| `C` | Clear all saved signs |
| `S` | Save signs |
| `F` | Toggle full-screen mode |
| `Q` | Exit |

### Steps to Train a Sign
1. Press `T`, then select the sign number from the list.
2. Hold the sign steady in front of the camera until the progress bar fills up (40 samples).
3. The sign is saved automatically, and the system returns to recognition mode.

To modify the list of signs, edit the `LABELS` variable at the top of the file.

---

## 📂 Repository Contents

- `sign_studio.py` — The complete source code for the system.
- `شرح_المشروع.pdf` (Project_Explanation.pdf) — A detailed explanation of the concept and how it works. ---

## 🧰 Technologies Used

`Python` · `OpenCV` · `MediaPipe` · `NumPy` · `gTTS` · `Pygame` · `Pillow`

---

## 📜 License

This project is licensed under the MIT License — feel free to use, modify, and share it.

---

## 🤝 Contributing

Suggestions and improvements are welcome. Please open an issue or submit a pull request.

> Created with passion to bring technology closer to humanity. 🤍
