# Face Recognition Attendance + Signature Capture 🧠✍️

This project combines **real-time face recognition** with **virtual signature capture** using your webcam. It's designed for tracking attendance and capturing user input with color markers.

---

## 🚀 Features

- ✅ Face recognition using OpenCV + `face_recognition`
- ✍️ Signature drawing using color detection (e.g., colored pen cap or marker)
- 📝 Automatically logs attendance with timestamps
- 💾 Saves signature as an image
- 🔄 Switch modes with keyboard keys (`f`, `s`, `c`, `q`, `x`)

---

## 📁 Included Test Files

Inside the `ImagesAttendance/` folder, you’ll find **3 preloaded images**:
- `jackma.jpg`
- `bill gates.jpg`
- `elon musk.jpg`

Just open the webcam and **hold a phone screen** with one of their faces to test the recognition.

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `f` | Switch to Face Recognition mode |
| `s` | Switch to Signature Drawing mode |
| `c` | Clear canvas in Signature mode |
| `q` | Save signature and quit |
| `x` | Quit program entirely |

---

## 🛠 Requirements

Install the required packages using:

```bash
pip install opencv-python face_recognition numpy
```

## Running the project

```bash
python AttendanceProject.py
```

### ✅ Setup Requirements

Make sure you have:

- A **webcam connected**
- The folder **`ImagesAttendance/`** with test images
- **`Attendance.csv`** will be created automatically on the first run to store attendance logs

---

### 📸 How It Works

#### Face Recognition Mode
- Detects faces in real-time and matches them with known encodings
- Displays a green box and the person's name when matched
- Logs attendance to `Attendance.csv` with a timestamp

#### Signature Mode
- Detects specific colors in front of the camera (like a colored marker cap)
- Captures strokes drawn on a virtual canvas
- Saves the final signature as a **PNG** file

