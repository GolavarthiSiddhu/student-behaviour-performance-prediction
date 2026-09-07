# app_gradio_student_behavior_performance_textboxes.py
# Gradio UI (TEXTBOX inputs only) for:
#  - Student Behaviour (classification)
#  - Student Performance (regression + class)
#
# Make sure these exist (created by your notebook):
#   models/student_behavior_classifier.joblib
#   models/student_performance_regressor.joblib
#   models/student_performance_classifier.joblib
#
# Run:
#   pip install -U gradio pandas joblib scikit-learn
#   python app_gradio_student_behavior_performance_textboxes.py

from pathlib import Path
import pandas as pd
import joblib
import gradio as gr

MODELS_DIR = Path("models")
BEH_PATH = MODELS_DIR / "student_behavior_classifier.joblib"
SCORE_PATH = MODELS_DIR / "student_performance_regressor.joblib"
PERF_PATH = MODELS_DIR / "student_performance_classifier.joblib"

missing = [str(p) for p in [BEH_PATH, SCORE_PATH, PERF_PATH] if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Missing model files:\n- " + "\n- ".join(missing) +
        "\n\nRun your notebook first to generate the models in ./models"
    )

beh_model = joblib.load(BEH_PATH)
score_model = joblib.load(SCORE_PATH)
perf_model = joblib.load(PERF_PATH)

FEATURES = [
    "gender",
    "age",
    "parent_education",
    "family_income_band",
    "travel_time_mins",
    "study_hours_per_week",
    "attendance_rate",
    "past_failures",
    "tutoring",
    "internet_access",
    "sleep_hours",
    "screen_time_hours",
    "extracurricular_hours",
    "discipline_incidents",
    "late_submissions",
    "participation_score",
    "quiz_avg",
    "assignment_avg",
    "midterm_score",
]

ALLOWED = {
    "gender": {"M", "F", "Other"},
    "parent_education": {"none", "high_school", "bachelor", "master", "phd"},
    "family_income_band": {"low", "middle", "high"},
}

def _to_int(name, v):
    try:
        return int(str(v).strip())
    except:
        raise ValueError(f"{name} must be an integer. You entered: {v}")

def _to_float(name, v):
    try:
        return float(str(v).strip())
    except:
        raise ValueError(f"{name} must be a number. You entered: {v}")

def _to_str_allowed(name, v):
    s = str(v).strip()
    if s not in ALLOWED[name]:
        raise ValueError(f"{name} must be one of {sorted(ALLOWED[name])}. You entered: {s}")
    return s

def _behavior_proba_dict(model, x_df):
    try:
        proba = model.predict_proba(x_df)[0]
        classes = list(model.named_steps["clf"].classes_)
        return {c: float(p) for c, p in zip(classes, proba)}
    except Exception:
        return None

def predict_textboxes(
    gender, age, parent_education, family_income_band,
    travel_time_mins, study_hours_per_week, attendance_rate,
    past_failures, tutoring, internet_access, sleep_hours,
    screen_time_hours, extracurricular_hours, discipline_incidents,
    late_submissions, participation_score, quiz_avg, assignment_avg,
    midterm_score
):
    try:
        sample = {
            "gender": _to_str_allowed("gender", gender),
            "age": _to_int("age", age),
            "parent_education": _to_str_allowed("parent_education", parent_education),
            "family_income_band": _to_str_allowed("family_income_band", family_income_band),
            "travel_time_mins": _to_int("travel_time_mins", travel_time_mins),
            "study_hours_per_week": _to_float("study_hours_per_week", study_hours_per_week),
            "attendance_rate": _to_float("attendance_rate", attendance_rate),
            "past_failures": _to_int("past_failures", past_failures),
            "tutoring": _to_int("tutoring", tutoring),
            "internet_access": _to_int("internet_access", internet_access),
            "sleep_hours": _to_float("sleep_hours", sleep_hours),
            "screen_time_hours": _to_float("screen_time_hours", screen_time_hours),
            "extracurricular_hours": _to_float("extracurricular_hours", extracurricular_hours),
            "discipline_incidents": _to_int("discipline_incidents", discipline_incidents),
            "late_submissions": _to_int("late_submissions", late_submissions),
            "participation_score": _to_float("participation_score", participation_score),
            "quiz_avg": _to_float("quiz_avg", quiz_avg),
            "assignment_avg": _to_float("assignment_avg", assignment_avg),
            "midterm_score": _to_float("midterm_score", midterm_score),
        }

        # Basic range checks (optional but helpful)
        if not (0.30 <= sample["attendance_rate"] <= 1.0):
            raise ValueError("attendance_rate must be between 0.30 and 1.0")
        if not (15 <= sample["age"] <= 23):
            raise ValueError("age must be between 15 and 23")
        if sample["tutoring"] not in (0, 1):
            raise ValueError("tutoring must be 0 or 1")
        if sample["internet_access"] not in (0, 1):
            raise ValueError("internet_access must be 0 or 1")

        x_df = pd.DataFrame([sample], columns=FEATURES)

        behavior_label = beh_model.predict(x_df)[0]
        behavior_probs = _behavior_proba_dict(beh_model, x_df)

        final_score_pred = float(score_model.predict(x_df)[0])
        performance_label = perf_model.predict(x_df)[0]

        summary = (
            f"Behaviour: **{behavior_label}**\n"
            f"Predicted Final Score: **{final_score_pred:.2f} / 100**\n"
            f"Performance Label: **{performance_label}**"
        )

        return summary, behavior_probs, round(final_score_pred, 2), performance_label, ""

    except Exception as e:
        # Show error message in UI (and clear other outputs)
        return "", None, None, "", f"❌ Input Error: {e}"


with gr.Blocks(title="Student Behaviour & Performance (Textbox Inputs)") as demo:
    gr.Markdown(
        "# 🎓 Student Behaviour & Performance Prediction (Textbox Inputs)\n"
        "Enter values manually in textboxes.\n\n"
        "**Allowed categorical values:**\n"
        "- gender: `M` / `F` / `Other`\n"
        "- parent_education: `none` / `high_school` / `bachelor` / `master` / `phd`\n"
        "- family_income_band: `low` / `middle` / `high`\n"
        "- tutoring: `0` or `1`, internet_access: `0` or `1`\n"
    )

    with gr.Row():
        with gr.Column():
            gender = gr.Textbox(value="M", label="gender (M/F/Other)")
            age = gr.Textbox(value="18", label="age (15-23)")
            parent_education = gr.Textbox(value="high_school", label="parent_education")
            family_income_band = gr.Textbox(value="middle", label="family_income_band")

            travel_time_mins = gr.Textbox(value="20", label="travel_time_mins")
            study_hours_per_week = gr.Textbox(value="10", label="study_hours_per_week")
            attendance_rate = gr.Textbox(value="0.85", label="attendance_rate (0.30-1.0)")
            past_failures = gr.Textbox(value="0", label="past_failures (0-3)")

            tutoring = gr.Textbox(value="0", label="tutoring (0/1)")
            internet_access = gr.Textbox(value="1", label="internet_access (0/1)")

            sleep_hours = gr.Textbox(value="7.0", label="sleep_hours")
            screen_time_hours = gr.Textbox(value="4.0", label="screen_time_hours")
            extracurricular_hours = gr.Textbox(value="2.0", label="extracurricular_hours")

            discipline_incidents = gr.Textbox(value="0", label="discipline_incidents")
            late_submissions = gr.Textbox(value="1", label="late_submissions")

            participation_score = gr.Textbox(value="65", label="participation_score (0-100)")
            quiz_avg = gr.Textbox(value="60", label="quiz_avg (0-100)")
            assignment_avg = gr.Textbox(value="68", label="assignment_avg (0-100)")
            midterm_score = gr.Textbox(value="62", label="midterm_score (0-100)")

            btn = gr.Button("Predict")

        with gr.Column():
            summary_out = gr.Markdown(label="Summary")
            proba_out = gr.JSON(label="Behaviour Probabilities")
            score_out = gr.Number(label="Predicted Final Score")
            perf_out = gr.Textbox(label="Performance Label")
            err_out = gr.Markdown(label="Errors")

    btn.click(
        fn=predict_textboxes,
        inputs=[
            gender, age, parent_education, family_income_band,
            travel_time_mins, study_hours_per_week, attendance_rate,
            past_failures, tutoring, internet_access, sleep_hours,
            screen_time_hours, extracurricular_hours, discipline_incidents,
            late_submissions, participation_score, quiz_avg, assignment_avg,
            midterm_score
        ],
        outputs=[summary_out, proba_out, score_out, perf_out, err_out]
    )

if __name__ == "__main__":
    demo.launch()
