# Count placeholders and values in INSERT statement

insert_statement = '''
INSERT INTO metrics_breakdown VALUES (
    ?, ?, 
    ?, ?, ?, ?, 
    ?, ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?,
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
'''

placeholders = insert_statement.count('?')
print(f"Number of placeholders (?): {placeholders}")

# List of values being inserted
values = [
    "current_date",  # 1
    "overall_score",  # 2
    # running_economy - 4 values
    "running_economy_normalized", "running_economy_weighted", "running_economy_raw_mean", "running_economy_raw_std",  # 3-6
    # vo2max - 4 values
    "vo2max_normalized", "vo2max_weighted", "vo2max_raw_mean", "vo2max_raw_std",  # 7-10
    # distance - 4 values
    "distance_normalized", "distance_weighted", "distance_raw_mean", "distance_raw_std",  # 11-14
    # efficiency_score - 4 values
    "efficiency_score_normalized", "efficiency_score_weighted", "efficiency_score_raw_mean", "efficiency_score_raw_std",  # 15-18
    # heart_rate - 4 values
    "heart_rate_normalized", "heart_rate_weighted", "heart_rate_raw_mean", "heart_rate_raw_std",  # 19-22
    # trends - 2 values
    "running_economy_trend", "distance_progression",  # 23-24
    # NEW METRICS - 24 values (12 metrics * 2 each for mean/std)
    "avg_speed_mean", "avg_speed_std",  # 25-26
    "max_speed_mean", "max_speed_std",  # 27-28
    "speed_reserve_mean", "speed_reserve_std",  # 29-30
    "speed_consistency_mean", "speed_consistency_std",  # 31-32
    "pace_per_km_mean", "pace_per_km_std",  # 33-34
    "speed_efficiency_mean", "speed_efficiency_std",  # 35-36
    "economy_at_speed_mean", "economy_at_speed_std",  # 37-38
    "speed_vo2max_index_mean", "speed_vo2max_index_std",  # 39-40
    "hr_rs_deviation_mean", "hr_rs_deviation_std",  # 41-42
    "cardiac_drift_mean", "cardiac_drift_std",  # 43-44
    "physio_efficiency_mean", "physio_efficiency_std",  # 45-46
    "fatigue_index_mean", "fatigue_index_std",  # 47-48
]

print(f"Number of values: {len(values)}")

if placeholders == len(values):
    print("✓ MATCH! Placeholders and values count is correct")
else:
    print(f"✗ MISMATCH! Difference: {placeholders - len(values)}")
    
print(f"\nExpected columns in table: 48")
print(f"We are providing: {len(values)} values")
