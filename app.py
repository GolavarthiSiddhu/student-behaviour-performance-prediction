import os
import json
import sqlite3
import io
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, send_file, abort
)
from werkzeug.security import generate_password_hash, check_password_hash

# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app.db"
MODELS_DIR = BASE_DIR / "models"

BEH_PATH = MODELS_DIR / "student_behavior_classifier.joblib"
SCORE_PATH = MODELS_DIR / "student_performance_regressor.joblib"
PERF_PATH = MODELS_DIR / "student_performance_classifier.joblib"

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

# -----------------------------
# DB helpers
# -----------------------------
def db():
    return sqlite3.connect(DB_PATH)

def ensure_column(con, table: str, col_name: str, col_ddl: str):
    """Add a column if it doesn't exist."""
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = {r[1] for r in cur.fetchall()}
    if col_name not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_ddl}")

def init_db():
    with db() as con:
        cur = con.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "name TEXT NOT NULL,"
            "email TEXT NOT NULL UNIQUE,"
            "password_hash TEXT NOT NULL,"
            "created_at TEXT NOT NULL,"
            "is_admin INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "user_id INTEGER NOT NULL,"
            "created_at TEXT NOT NULL,"
            "features_json TEXT NOT NULL,"
            "behavior_label TEXT NOT NULL,"
            "behavior_probs_json TEXT,"
            "final_score_pred REAL NOT NULL,"
            "performance_label TEXT NOT NULL,"
            "FOREIGN KEY(user_id) REFERENCES users(id)"
            ")"
        )
        # migrate old dbs
        ensure_column(con, "users", "is_admin", "is_admin INTEGER NOT NULL DEFAULT 0")
        con.commit()

# -----------------------------
# ML helpers
# -----------------------------
def load_models():
    missing = [str(p) for p in [BEH_PATH, SCORE_PATH, PERF_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files:\n- " + "\n- ".join(missing) +
            "\n\nCopy your trained model files into ./models folder."
        )
    beh_model = joblib.load(BEH_PATH)
    score_model = joblib.load(SCORE_PATH)
    perf_model = joblib.load(PERF_PATH)
    return {"beh": beh_model, "score": score_model, "perf": perf_model}

def _to_int(name, v):
    try:
        return int(str(v).strip())
    except Exception:
        raise ValueError(f"{name} must be an integer. You entered: {v}")

def _to_float(name, v):
    try:
        return float(str(v).strip())
    except Exception:
        raise ValueError(f"{name} must be a number. You entered: {v}")

def _to_str_allowed(name, v):
    s = str(v).strip()
    if s not in ALLOWED[name]:
        raise ValueError(f"{name} must be one of {sorted(ALLOWED[name])}. You entered: {s}")
    return s

def validate_and_cast(form: dict) -> dict:
    sample = {
        "gender": _to_str_allowed("gender", form["gender"]),
        "age": _to_int("age", form["age"]),
        "parent_education": _to_str_allowed("parent_education", form["parent_education"]),
        "family_income_band": _to_str_allowed("family_income_band", form["family_income_band"]),
        "travel_time_mins": _to_int("travel_time_mins", form["travel_time_mins"]),
        "study_hours_per_week": _to_float("study_hours_per_week", form["study_hours_per_week"]),
        "attendance_rate": _to_float("attendance_rate", form["attendance_rate"]),
        "past_failures": _to_int("past_failures", form["past_failures"]),
        "tutoring": _to_int("tutoring", form["tutoring"]),
        "internet_access": _to_int("internet_access", form["internet_access"]),
        "sleep_hours": _to_float("sleep_hours", form["sleep_hours"]),
        "screen_time_hours": _to_float("screen_time_hours", form["screen_time_hours"]),
        "extracurricular_hours": _to_float("extracurricular_hours", form["extracurricular_hours"]),
        "discipline_incidents": _to_int("discipline_incidents", form["discipline_incidents"]),
        "late_submissions": _to_int("late_submissions", form["late_submissions"]),
        "participation_score": _to_float("participation_score", form["participation_score"]),
        "quiz_avg": _to_float("quiz_avg", form["quiz_avg"]),
        "assignment_avg": _to_float("assignment_avg", form["assignment_avg"]),
        "midterm_score": _to_float("midterm_score", form["midterm_score"]),
    }

    # Range validations
    if not (15 <= sample["age"] <= 23):
        raise ValueError("age must be between 15 and 23")
    if not (0.30 <= sample["attendance_rate"] <= 1.0):
        raise ValueError("attendance_rate must be between 0.30 and 1.0")
    if sample["tutoring"] not in (0, 1):
        raise ValueError("tutoring must be 0 or 1")
    if sample["internet_access"] not in (0, 1):
        raise ValueError("internet_access must be 0 or 1")
    if not (0 <= sample["travel_time_mins"] <= 120):
        raise ValueError("travel_time_mins must be between 0 and 120")
    if not (0 <= sample["study_hours_per_week"] <= 40):
        raise ValueError("study_hours_per_week must be between 0 and 40")

    return sample

def behavior_proba_dict(beh_model, x_df):
    try:
        proba = beh_model.predict_proba(x_df)[0]
        classes = list(beh_model.named_steps["clf"].classes_)
        return {c: float(p) for c, p in zip(classes, proba)}
    except Exception:
        return None

def predict(models, sample: dict):
    x_df = pd.DataFrame([sample], columns=FEATURES)
    behavior_label = models["beh"].predict(x_df)[0]
    probs = behavior_proba_dict(models["beh"], x_df)
    final_score_pred = float(models["score"].predict(x_df)[0])
    performance_label = models["perf"].predict(x_df)[0]
    result = {
        "behavior_label": str(behavior_label),
        "final_score_pred": round(final_score_pred, 2),
        "performance_label": str(performance_label),
    }
    return result, probs

def save_history(user_id: int, features: dict, result: dict, probs):
    with db() as con:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO history(user_id, created_at, features_json, behavior_label, behavior_probs_json, final_score_pred, performance_label) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                user_id,
                datetime.utcnow().isoformat(),
                json.dumps(features, ensure_ascii=False),
                result["behavior_label"],
                json.dumps(probs, ensure_ascii=False) if probs else None,
                float(result["final_score_pred"]),
                result["performance_label"],
            )
        )
        con.commit()

# -----------------------------
# App factory
# -----------------------------
def create_app():
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "dev-change-this-secret")

    init_db()
    models = load_models()

    @app.context_processor
    def inject_now():
        return {"now_year": datetime.now().year}

    def current_user_id():
        return session.get("user_id")

    def login_required(view_func):
        from functools import wraps
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            if not current_user_id():
                return redirect(url_for("login", next=request.path))
            return view_func(*args, **kwargs)
        return wrapper

    def is_admin():
        uid = current_user_id()
        if not uid:
            return False
        with db() as con:
            cur = con.cursor()
            cur.execute("SELECT is_admin FROM users WHERE id = ?", (uid,))
            row = cur.fetchone()
        return bool(row and row[0] == 1)

    def admin_required(view_func):
        from functools import wraps
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            if not current_user_id():
                return redirect(url_for("login", next=request.path))
            if not is_admin():
                abort(403)
            return view_func(*args, **kwargs)
        return wrapper

    # -----------------------------
    # Routes
    # -----------------------------
    @app.route("/")
    def index():
        if current_user_id():
            return redirect(url_for("dashboard"))
        return render_template("index.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            if not name or not email or not password:
                flash("All fields are required.", "danger")
                return render_template("register.html", name=name, email=email)

            if len(password) < 6:
                flash("Password must be at least 6 characters.", "danger")
                return render_template("register.html", name=name, email=email)

            with db() as con:
                cur = con.cursor()
                cur.execute("SELECT id FROM users WHERE email = ?", (email,))
                if cur.fetchone():
                    flash("Email already registered. Please login.", "warning")
                    return redirect(url_for("login"))

                pw_hash = generate_password_hash(password)
                admin_emails = {e.strip().lower() for e in os.environ.get("ADMIN_EMAILS", "").split(",") if e.strip()}
                is_admin_val = 1 if email in admin_emails else 0

                cur.execute(
                    "INSERT INTO users(name, email, password_hash, created_at, is_admin) VALUES(?,?,?,?,?)",
                    (name, email, pw_hash, datetime.utcnow().isoformat(), is_admin_val)
                )
                con.commit()

            flash("Registration successful. Please login.", "success")
            return redirect(url_for("login"))

        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            with db() as con:
                cur = con.cursor()
                cur.execute("SELECT id, name, password_hash, is_admin FROM users WHERE email = ?", (email,))
                row = cur.fetchone()

            if not row or not check_password_hash(row[2], password):
                flash("Invalid email or password.", "danger")
                return render_template("login.html", email=email)

            session["user_id"] = row[0]
            session["user_name"] = row[1]
            session["is_admin"] = 1 if row[3] == 1 else 0

            flash("Logged in successfully.", "success")
            nxt = request.args.get("next")
            return redirect(nxt or url_for("dashboard"))

        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("You have been logged out.", "info")
        return redirect(url_for("index"))

    @app.route("/dashboard", methods=["GET", "POST"])
    @login_required
    def dashboard():
        defaults = {
            "gender": "M",
            "age": "18",
            "parent_education": "high_school",
            "family_income_band": "middle",
            "travel_time_mins": "20",
            "study_hours_per_week": "10",
            "attendance_rate": "0.85",
            "past_failures": "0",
            "tutoring": "0",
            "internet_access": "1",
            "sleep_hours": "7.0",
            "screen_time_hours": "4.0",
            "extracurricular_hours": "2.0",
            "discipline_incidents": "0",
            "late_submissions": "1",
            "participation_score": "65",
            "quiz_avg": "60",
            "assignment_avg": "68",
            "midterm_score": "62",
        }

        result = None
        probs = None

        if request.method == "POST":
            form = {k: request.form.get(k, "").strip() for k in FEATURES}
            defaults.update(form)
            try:
                sample = validate_and_cast(form)
                result, probs = predict(models, sample)
                save_history(current_user_id(), sample, result, probs)
                flash("Prediction completed and saved to history.", "success")
            except Exception as e:
                flash(f"Input Error: {e}", "danger")

        return render_template("dashboard.html", form=defaults, result=result, probs=probs)

    @app.route("/history")
    @login_required
    def history():
        with db() as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(
                "SELECT id, created_at, behavior_label, final_score_pred, performance_label "
                "FROM history WHERE user_id = ? ORDER BY id DESC LIMIT 200",
                (current_user_id(),)
            )
            rows = cur.fetchall()
        return render_template("history.html", rows=rows)

    @app.route("/history/<int:hid>")
    @login_required
    def history_detail(hid: int):
        with db() as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(
                "SELECT * FROM history WHERE id = ? AND user_id = ?",
                (hid, current_user_id())
            )
            row = cur.fetchone()

        if not row:
            flash("History record not found.", "warning")
            return redirect(url_for("history"))

        features = json.loads(row["features_json"]) if row["features_json"] else {}
        probs = json.loads(row["behavior_probs_json"]) if row["behavior_probs_json"] else None
        return render_template("history_detail.html", row=row, features=features, probs=probs)

    @app.route("/history/<int:hid>/delete", methods=["POST"])
    @login_required
    def history_delete(hid: int):
        with db() as con:
            cur = con.cursor()
            cur.execute("DELETE FROM history WHERE id = ? AND user_id = ?", (hid, current_user_id()))
            con.commit()
        flash("History record deleted.", "info")
        return redirect(url_for("history"))

    @app.route("/history/clear", methods=["POST"])
    @login_required
    def history_clear():
        with db() as con:
            cur = con.cursor()
            cur.execute("DELETE FROM history WHERE user_id = ?", (current_user_id(),))
            con.commit()
        flash("All your history has been deleted.", "info")
        return redirect(url_for("history"))

    @app.route("/history/export.csv")
    @login_required
    def history_export_csv():
        with db() as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(
                "SELECT created_at, behavior_label, final_score_pred, performance_label, features_json, behavior_probs_json "
                "FROM history WHERE user_id = ? ORDER BY id DESC",
                (current_user_id(),)
            )
            rows = cur.fetchall()

        df = pd.DataFrame([dict(r) for r in rows])
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        filename = f"history_user_{current_user_id()}_{datetime.utcnow().date().isoformat()}.csv"
        return send_file(
            io.BytesIO(csv_bytes),
            mimetype="text/csv",
            as_attachment=True,
            download_name=filename
        )

    # -----------------------------
    # Admin views
    # -----------------------------
    @app.route("/admin")
    @admin_required
    def admin_home():
        return redirect(url_for("admin_users"))

    @app.route("/admin/users")
    @admin_required
    def admin_users():
        with db() as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(
                "SELECT u.id, u.name, u.email, u.is_admin, u.created_at, "
                "(SELECT COUNT(1) FROM history h WHERE h.user_id = u.id) AS predictions "
                "FROM users u ORDER BY u.id DESC"
            )
            users = cur.fetchall()
        return render_template("admin_users.html", users=users)

    @app.route("/admin/history")
    @admin_required
    def admin_history():
        with db() as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(
                "SELECT h.id, h.created_at, h.user_id, u.name, u.email, "
                "h.behavior_label, h.final_score_pred, h.performance_label "
                "FROM history h JOIN users u ON u.id = h.user_id "
                "ORDER BY h.id DESC LIMIT 500"
            )
            rows = cur.fetchall()
        return render_template("admin_history.html", rows=rows)

    @app.route("/admin/history/export.csv")
    @admin_required
    def admin_history_export_csv():
        with db() as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(
                "SELECT h.created_at, u.id AS user_id, u.name, u.email, u.is_admin, "
                "h.behavior_label, h.final_score_pred, h.performance_label, "
                "h.features_json, h.behavior_probs_json "
                "FROM history h JOIN users u ON u.id = h.user_id "
                "ORDER BY h.id DESC"
            )
            rows = cur.fetchall()

        df = pd.DataFrame([dict(r) for r in rows])
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        filename = f"admin_all_history_{datetime.utcnow().date().isoformat()}.csv"
        return send_file(
            io.BytesIO(csv_bytes),
            mimetype="text/csv",
            as_attachment=True,
            download_name=filename
        )

    @app.route("/admin/users/<int:uid>/toggle-admin", methods=["POST"])
    @admin_required
    def admin_toggle(uid: int):
        # Prevent toggling yourself (safety)
        if uid == current_user_id():
            flash("You cannot change your own admin status here.", "warning")
            return redirect(url_for("admin_users"))

        with db() as con:
            cur = con.cursor()
            cur.execute("SELECT is_admin FROM users WHERE id = ?", (uid,))
            row = cur.fetchone()
            if not row:
                flash("User not found.", "warning")
                return redirect(url_for("admin_users"))

            new_val = 0 if row[0] == 1 else 1
            cur.execute("UPDATE users SET is_admin = ? WHERE id = ?", (new_val, uid))
            con.commit()

        flash("User admin status updated.", "success")
        return redirect(url_for("admin_users"))

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
