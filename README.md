
# Hand Sign Detection System

This project is designed to detect hand signs using a webcam and convert them to text and audio. It includes the following features:
1. **Hand Sign Detection**: Using OpenCV and a deep learning model to detect hand signs in real-time.
2. **Text-to-Speech (TTS)**: The detected hand sign is converted into speech using a text-to-speech engine (pyttsx3).
3. **Model Training**: You can train the model using a dataset of hand signs collected via webcam.
4. **Real-time Detection**: The system detects and classifies hand signs in real-time.

---

## Features

- **Hand Detection**: Uses OpenCV's hand tracking module to detect and capture hand gestures.
- **Model for Classification**: A CNN model trained to classify hand signs.
- **Text-to-Speech**: Detected signs are spoken using pyttsx3 for accessibility.
- **Data Collection**: You can collect data by pressing the `s` key to save images for training.

---

## Prerequisites

- Python 3.8, 3.9, or 3.10 (TensorFlow 2.13+ supports these versions)
- OpenCV
- TensorFlow
- pyttsx3

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-repository/hand-sign-detection.git
cd hand-sign-detection
```

### 2. Create a Virtual Environment

It's a good practice to use a virtual environment for managing dependencies. Run the following commands to set up and activate a virtual environment:

#### Windows

```bash
python -m venv venv
.env\Scriptsctivate
```

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages

After activating the virtual environment, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 4. Dataset Collection

To collect data for training the model, run the `data_collection.py` file. This will allow you to capture hand sign images by pressing the `s` key.

```bash
python data_collection.py
```

The images will be saved in the specified folder (`Data/1`).

### 5. Train the Model

After collecting sufficient data, you can train the model by running the following command:

```bash
python train_model.py
```

The model will be saved in the `Model` directory as `keras_model.h5`.

### 6. Test the Model

Once the model is trained, you can test the hand sign detection system by running:

```bash
python test_model.py
```

This will open a webcam window that will detect hand signs and speak the detected signs.

---

## File Structure

```
hand-sign-detection/
│
├── Data/               # Folder containing collected images for training
│
├── Model/              # Folder to save the trained model and labels
│   ├── keras_model.h5  # The trained model
│   └── labels.txt      # File containing the labels
│
├── data_collection.py  # Script to collect hand sign data
├── train_model.py      # Script to train the model
├── test_model.py       # Script to test the hand sign detection
└── README.md           # This file
```

---

## Troubleshooting

- **No webcam access**: Ensure your webcam is correctly connected and not being used by other applications.
- **Missing dependencies**: If any libraries fail to install, try upgrading `pip` and reinstalling the dependencies:

  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

---

### requirements.txt

```
opencv-python==4.7.0.72
pyttsx3==2.90
tensorflow==2.13.0
numpy==1.23.5
cvzone==1.5.0
Pillow==9.5.0
```

---

### Notes

- Ensure you have a folder `Data/1` to store images during the data collection process.
- Adjust the paths in the code if you're using a different directory structure for storing data or models.
