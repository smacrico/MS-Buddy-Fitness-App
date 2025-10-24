Purpose

Generates a daily health & recovery report by pulling HRV + Garmin sleep/stress + activity data from SQLite DBs, computing recovery/fatigue/training/fat-burn metrics, composing a text report, and sending it by email.
Key configuration

DB paths: HRV_DB_PATH and GARMIN_DB_PATH (placeholders in file; several local paths set).
Personal baselines and thresholds (HRV, RHR, FTP, LTHR, age, weight).
Email settings: SENDER_EMAIL, RECIPIENT_EMAIL, SMTP_SERVER/PORT and EMAIL_PASSWORD read from environment (load_dotenv() call present).
Main functions

fetch_hrv_data(db_path, target_date): queries hrv_table for date.
fetch_garmin_sleep_stress_data(db_path, target_date): queries garmin_sleep_stress_table for overnight sleep/stress metrics.
fetch_garmin_activity_data(db_path, target_date): queries garmin_activities_table and returns list of activities.
calculate_recovery_score(...): composite recovery score (0–100) using HRV, sleep, RHR, body battery.
calculate_fatigue_level(...), calculate_sleep_charge(...), calculate_recovery_ratio(...): derived metrics.
TSS / calories / fat: calculate_power_tss, calculate_hr_tss, calculate_pace_tss, get_mets_from_pace, calculate_total_calories_burned_mets, calculate_fat_calories_burned, calculate_grams_fat_burned.
generate_recommendation(recovery_score, fatigue_level): textual guidance.
send_email_report(...): SMTP_SSL send of the report.
Workflow (generate_daily_health_report)

Sets today/yesterday; fetches HRV (today), sleep/stress (yesterday), activities (today).
Applies defaults if data missing.
Computes recovery_score, sleep_charge, recovery_ratio.
Iterates activities to accumulate TSS, Garmin load, calories, fat calories, grams fat.
Computes fatigue_level and recommendation.
Builds a plain-text report string and calls send_email_report().
Assumptions and risks

Assumes specific table names and column schemas (hrv_table, garmin_sleep_stress_table, garmin_activities_table). Update queries to match your schema.
Database path placeholders must be set to actual files.
Email password must be set in environment variable EMAIL_PASSWORD (or .env). Current code prints errors on failure but will attempt send.
No error handling around malformed activity rows beyond simple fallbacks.
Uses simple science models / heuristics — tune weights, baselines and formulas to your data.
Quick recommendations

Verify and update DB paths and SQL table/column names.
Use secrets manager or OS env for email password (do not hardcode).
Add logging and better exception handling for production.
Consider HTML email or attachments if you want richer reports.