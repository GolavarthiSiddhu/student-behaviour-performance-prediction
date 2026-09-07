# Student AI Flask App

## Features
- Login / Register (password hashed)
- Dashboard: input values in textboxes → predict behaviour + performance
- History: saves every prediction in SQLite and shows list + details
- Bootstrap UI

## Setup
1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Copy your trained models into `./models/`:
- `student_behavior_classifier.joblib`
- `student_performance_regressor.joblib`
- `student_performance_classifier.joblib`

3) Run:
```bash
python app.py
```

Open: `http://127.0.0.1:5000/`

## Database
- SQLite file auto-created: `app.db`
