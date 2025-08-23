# -- Load and prepare data --
df = load_hrv_sessions(sqlite_db, last_days=30)  # Latest data
baseline_df = load_hrv_baselines(baseline_db)    # Your 90-day baseline stats

# -- Calculate Z-scores for each session/metric --
for metric in ['hrv_rmssd', 'sdnn', 'hrv_pnn50', 'mean_hr', 'stress_hrpa']:
    mean = baseline_df.at[metric, 'baseline_mean']
    std = baseline_df.at[metric, 'baseline_std']
    df[f'{metric}_z'] = (df[metric] - mean) / std if std > 0 else 0

# -- Alert Logic --
alerts = []
for _, row in df.iterrows():
    alert_msgs = []
    for metric in ['hrv_rmssd', 'sdnn', 'hrv_pnn50', 'mean_hr', 'stress_hrpa']:
        z = row[f'{metric}_z']
        if z <= -1.5:
            alert_msgs.append(f'⚠️ CRITICAL: {metric} is {z:.1f} SD below baseline')
        elif z <= -1.0:
            alert_msgs.append(f'⚠️ WARNING: {metric} is {z:.1f} SD below baseline')
    if alert_msgs:
        alerts.append({
            'timestamp': row['timestamp'],
            'activity_id': row['activity_id'],
            'alerts': alert_msgs
        })

# -- Trend-based alert (optional) --
# If any metric has Z < 0 for 3+ consecutive sessions, flag as "watch"
for metric in ['hrv_rmssd_z', 'sdnn_z', 'hrv_pnn50_z', 'mean_hr_z', 'stress_hrpa_z']:
    if len(df[df[metric] < 0]) >= 3:
        alerts.append({
            'timestamp': datetime.now().isoformat(),
            'activity_id': 'trend',
            'alerts': [f'👀 TREND: {metric[:-2]} below baseline for 3+ sessions']
        })

# -- Export/Notify Alerts --
for alert in alerts:
    print(f"[{alert['timestamp']}] {alert['activity_id']}:")
    for msg in alert['alerts']:
        print(f"  {msg}")

# Production-Grade Python Function 
##################


def generate_hrv_alerts(hrv_db_path, baseline_db_path):
    # ... load data as above ...
    alerts = []
    for _, row in df.iterrows():
        alert_msgs = []
        for metric in ['hrv_rmssd', 'sdnn', 'hrv_pnn50', 'mean_hr', 'stress_hrpa']:
            z = row[f'{metric}_z']
            if z <= -1.5:
                alert_msgs.append(f'CRITICAL: {metric} is {z:.1f} SD below baseline')
            elif z <= -1.0:
                alert_msgs.append(f'WARNING: {metric} is {z:.1f} SD below baseline')
        if alert_msgs:
            alerts.append({
                'timestamp': row['timestamp'],
                'activity_id': row['activity_id'],
                'alerts': alert_msgs
            })
    return alerts
