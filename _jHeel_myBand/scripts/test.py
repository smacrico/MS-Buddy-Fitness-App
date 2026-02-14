
import pandas as pd

# Load the demo data
df = pd.read_csv('demo.csv')

# Parse timestamp column
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Display basic info
print(df.head())
print(f"\nData shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

### Parse RR intervals


import ast

# Convert RR intervals from string to list
df['rr_intervals'] = df['rr_intervals'].apply(ast.literal_eval)

# Access individual RR interval lists
print(df['rr_intervals'].iloc[0])

### Basic analysis examples


# Sleep analysis
sleep_data = df[df['sleep_stage'] != 'awake']
print(f"Average sleep heart rate: {sleep_data['heart_rate'].mean():.1f} BPM")

# Activity patterns
print(f"Average daily activity: {df['activity_level'].mean():.1f}")

# Heart rate zones
print(df['heart_rate'].describe())

### Visualization examples


import matplotlib.pyplot as plt

# Plot heart rate over time
plt.figure(figsize=(15, 5))
plt.plot(df['timestamp'], df['heart_rate'])
plt.xlabel('Time')
plt.ylabel('Heart Rate (BPM)')
plt.title('Heart Rate Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Sleep stages distribution
df['sleep_stage'].value_counts().plot(kind='bar')
plt.title('Sleep Stage Distribution')
plt.ylabel('Count')
plt.show()
