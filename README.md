# AIRA — AI Respiratory Intelligence & Risk Assessment

AIRA is an AI-powered air quality forecasting and respiratory risk assessment platform that combines real-time AQI monitoring, deep learning forecasting, anomaly detection, and personalized health recommendations.

The system uses a GRU (Gated Recurrent Unit) neural network to forecast AQI 24 hours ahead and estimate respiratory health risks before hazardous conditions occur.

---

## Features

### Real-Time Air Quality Monitoring
- Live AQI retrieval using WAQI API
- Automatic geolocation detection
- Real-time pollutant monitoring:
  - PM2.5
  - PM10
  - NO₂
  - CO
  - SO₂
  - O₃

### 24-Hour AQI Forecasting
- GRU Neural Network
- Multi-step AQI prediction
- 24-hour forecast horizon
- Confidence interval estimation using Monte Carlo Dropout

### Pollution Anomaly Detection
- Detects unusual AQI spikes
- Early warning system for pollution events
- Forecast-based anomaly alerts

### Personalized Respiratory Risk Analysis
Health profiles supported:

- Normal
- Asthma
- Elderly
- Athlete

Risk estimation considers:
- Forecasted AQI
- Exposure duration
- Physiological sensitivity

### Forecast-Based Recommendations
Dynamic recommendations such as:

- Reduce outdoor exposure
- Close windows
- Wear masks
- Avoid strenuous exercise

### Interactive Dashboard
- Modern dark-themed UI
- AQI visualization
- Forecast charts
- Risk timeline graphs
- Confidence indicators
- Mobile responsive design

---

## Technology Stack

### Frontend
- HTML5
- CSS3
- JavaScript
- Chart.js

### Backend
- Python
- Flask
- Flask-CORS

### Machine Learning
- TensorFlow / Keras
- GRU Neural Networks
- Monte Carlo Dropout
- Scikit-Learn

### Data Processing
- NumPy
- Pandas
- Joblib

### APIs
- WAQI API

---

## Project Structure

```text
AIRA/
├── app/
│   └── api.py
│
├── frontend/
│   ├── server.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── app.js
│
├── models/
│   ├── aqi_gru_model.h5
│   ├── aqi_scaler.pkl
│   ├── aqi_target_scaler.pkl
│   ├── city_encoder.pkl
│   ├── train_gru.py
│   └── training_history.json
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Model Architecture

### GRU Neural Network

Input Features:

- PM2.5
- PM10
- NO₂
- CO
- SO₂
- O₃
- Current AQI
- City Encoding

Output:

- 24-hour AQI forecast
- Confidence interval
- Risk timeline
- Anomaly detection

---

## API Endpoints

### Health Check

```http
GET /api/health
```

### Fetch AQI

```http
GET /api/fetch-aqi?lat=xx&lon=yy
```

### Generate Forecast

```http
POST /api/forecast
```

### Example Request

```json
{
  "pm25": 109,
  "pm10": 52,
  "no2": 2.2,
  "co": 2.9,
  "so2": 2.4,
  "o3": 21.6,
  "current_aqi": 109,
  "profile": "normal",
  "exposure_hours": 3,
  "city": "Karnal"
}
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/AIRA.git
cd AIRA
```

### Create Virtual Environment

```bash
python -m venv venv
```

macOS/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\\Scripts\\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Start Backend

```bash
python app/api.py
```

Backend:

```text
http://localhost:5001
```

### Start Frontend

```bash
python frontend/server.py
```

Frontend:

```text
http://localhost:8080
```

---

## Future Improvements

- Weather integration
- Multi-city forecasting
- Mobile application
- AQI notification alerts
- Historical trend analytics
- Cloud deployment

---

## Author

Arnav Kakkar

Engineering Student, Thapar Institute of Engineering & Technology

---

## License

This project is developed for educational, research, and competition purposes.