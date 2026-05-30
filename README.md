# Early Fatigue Detection System

A computer vision-based driver drowsiness detection system that detects early signs of fatigue using facial behavior analysis.

The project contains two approaches:

- **Threshold Model** – uses manually defined threshold values for fatigue detection
- **Hybrid Model** – combines trained model-based prediction with rule-based/manual detection logic

---

## 🚀 Project Overview

Driver fatigue is one of the major causes of road accidents. This system monitors facial behavior and detects drowsiness/fatigue signs using Python and OpenCV.

When fatigue is detected, the system triggers an alarm. If the user does not turn off the alarm within a specific time, the emergency alert service is automatically invoked.

The emergency alert system uses **Twilio** to send emergency messages to the respective family members. In future improvements, this system can be extended to notify emergency services such as ambulance support.

---

## 🛠️ Tech Stack

- Python
- OpenCV
- Computer Vision
- Machine Learning
- Facial Feature Analysis
- LightGBM
- Twilio API

---

## ✨ Features

- Real-time fatigue/drowsiness detection
- Threshold-based fatigue detection
- Hybrid fatigue detection model
- Facial feature tracking
- Alarm alert system
- Emergency message alert using Twilio
- Automatic emergency notification if the user does not turn off the alarm within a specific time
- Location helper support
- WhatsApp notification support

---

## 🧠 Models / Approaches Used

### 1. Threshold Model

The threshold model uses manually defined threshold values to detect fatigue based on facial behavior and feature changes.

### 2. Hybrid Model

The hybrid model uses a trained LightGBM model along with rule-based logic to improve fatigue detection reliability.

---

## 📌 My Contributions

- Developed the threshold-based fatigue detection model
- Worked on the hybrid fatigue detection approach
- Implemented OpenCV-based facial feature analysis
- Contributed to fatigue detection logic and alert flow
- Integrated emergency alert logic using Twilio for sending messages to family members
- Implemented automatic emergency service invocation when the alarm is not turned off within the set time
- Tested and improved the real-time detection process

---

## 📂 Project Structure

```text
Early-Fatigue-Detection-System/
│
├── hybrid_model/
│   ├── alarm_manager.py
│   ├── alarm.wav
│   ├── config.py
│   ├── detector.py
│   ├── face_landmarker.task
│   ├── fatigue_lightgbm_feature_columns_final.pkl
│   ├── fatigue_lightgbm_final_threshold.pkl
│   ├── fatigue_lightgbm_model_final.pkl
│   ├── features.py
│   ├── get_location.html
│   ├── location_helper.py
│   ├── location.json
│   ├── main_hybrid3state_final.py
│   ├── main.py
│   ├── mild_alarm.wav
│   ├── predictor.py
│   ├── tracker.py
│   ├── ui.py
│   └── whatsapp_notifier.py
│
├── threshold_model/
│   ├── alarm_manager.py
│   ├── alarm.wav
│   ├── config.py
│   ├── detector.py
│   ├── face_landmarker.task
│   ├── fatigue_features_log.csv
│   ├── features.py
│   ├── get_location.html
│   ├── location_helper.py
│   ├── main.py
│   ├── mild_alarm.wav
│   └── whatsapp_notifier.py
│
├── .env.example
├── .gitignore
├── launcher.py
├── LICENSE
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/skoder404/Early-Fatigue-Detection-System.git
```

Move into the project folder:

```bash
cd Early-Fatigue-Detection-System
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the threshold model:

```bash
python threshold_model/main.py
```

Run the hybrid model:

```bash
python hybrid_model/main_hybrid3state_final.py
```

Or run using the launcher:

```bash
python launcher.py
```

---

## 🔐 Environment Setup

This project includes emergency alert features using Twilio.

Create a `.env` file based on `.env.example` and add the required Twilio credentials and phone numbers.

Example:

```text
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=your_twilio_number
EMERGENCY_CONTACT_NUMBER=family_member_number
```

> Do not upload your real `.env` file to GitHub.

---

## 🎯 Future Improvements

- Improve fatigue detection accuracy using larger datasets
- Add ambulance/emergency service integration
- Build a dashboard for fatigue history and emergency alerts
- Deploy the system as a desktop or web application
- Optimize real-time performance
- Improve alert customization and emergency response flow

---

## 📌 Note

This project is created for academic and learning purposes.

Some features such as Twilio emergency messaging, WhatsApp notification, and location support may require API configuration or environment variables.

---

## 👨‍💻 Author

**M A SUSHIL KUMAR**

- GitHub: [skoder404](https://github.com/skoder404)
- LinkedIn: [sushil006](https://www.linkedin.com/in/sushil006)
- Email: sushilmit28@gmail.com

---

## 📜 License

This project is for academic and learning purposes.
