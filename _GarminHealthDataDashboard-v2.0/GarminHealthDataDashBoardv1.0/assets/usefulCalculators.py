## **Health API Metrics and Formulas**

### **Daily Summary Metrics**

**Activity Level Score**
```
Activity Score = (Steps / 10,000) × 0.4 + (Active Minutes / 60) × 0.6
```

**Caloric Efficiency**
```
Caloric Efficiency = Active Calories / Total Calories × 100
```

**Daily Movement Ratio**
```
Movement Ratio = Active Time / (24 × 60) × 100
```

### **Epoch Summary Metrics**

**Average Activity Intensity**
```
Avg Intensity = Σ(METs × Duration) / Total Duration
```

**Movement Consistency Index**
```
Consistency = (Active Epochs / Total Epochs) × 100
```

**Caloric Burn Rate**
```
Burn Rate = Active Kilocalories / (Active Time in Hours)
```

### **Sleep Summary Metrics**

**Sleep Efficiency**
```
Sleep Efficiency = (Total Sleep Time / Time in Bed) × 100
```

**Deep Sleep Percentage**
```
Deep Sleep % = (Deep Sleep Duration / Total Sleep Time) × 100
```

**REM Sleep Quality Score**
```
REM Quality = (REM Duration / Total Sleep Time) × 100
```

**Sleep Debt**
```
Sleep Debt = Target Sleep Hours - Actual Sleep Hours
```

### **Heart Rate Variability Metrics**

**HRV Recovery Score**
```
Recovery Score = (Current HRV / 7-Day HRV Average) × 100
```

**HRV Trend**
```
HRV Trend = (Today's HRV - Yesterday's HRV) / Yesterday's HRV × 100
```

### **Stress Detail Metrics**

**Average Daily Stress**
```
Avg Stress = Σ(Stress Values) / Number of Readings
```

**Stress Load Index**
```
Stress Load = Σ(Stress Value × Duration) / Total Monitoring Time
```

**Stress Recovery Rate**
```
Recovery Rate = (Peak Stress - Current Stress) / Time Elapsed
```

### **Body Composition Metrics**

**Body Fat Percentage Change**
```
BF Change = ((Current BF% - Previous BF%) / Previous BF%) × 100
```

**Lean Body Mass**
```
Lean Mass = Weight × (1 - Body Fat Percentage)
```

**BMI Classification Score**
```
BMI Score = Weight (kg) / (Height (m))²
```

### **Pulse Ox Metrics**

**Oxygen Saturation Variability**
```
SpO2 Variability = Standard Deviation of SpO2 Readings
```

**Average Nighttime SpO2**
```
Night SpO2 = Σ(Sleep Period SpO2 Values) / Number of Sleep Readings
```

### **Respiration Metrics**

**Respiratory Rate Variability**
```
RR Variability = (Max RR - Min RR) / Average RR × 100
```

**Sleep Breathing Stability**
```
Breathing Stability = 1 - (StdDev Sleep RR / Mean Sleep RR)
```

## **Activity API Metrics and Formulas**

### **Activity Summary Metrics**

**Training Intensity Factor**
```
Intensity Factor = Average Heart Rate / Threshold Heart Rate
```

**Pace Consistency**
```
Pace Consistency = 1 - (Pace Standard Deviation / Average Pace)
```

**Caloric Efficiency**
```
Cal Efficiency = Distance / Calories Burned
```

**Performance Index**
```
Performance Index = (Distance × Speed) / (Heart Rate × Duration)
```

### **Activity Details Metrics**

**Elevation Efficiency**
```
Elevation Efficiency = Total Ascent / Total Distance
```

**Heart Rate Drift**
```
HR Drift = (Final 30min Avg HR - Initial 30min Avg HR) / Initial HR × 100
```

**Power-to-Weight Ratio** (for cycling)
```
Power-to-Weight = Average Power (W) / Body Weight (kg)
```

**Training Stress Score**
```
TSS = (Duration × Average Power × Intensity Factor) / (FTP × 3600) × 100
```

**Cadence Efficiency**
```
Cadence Efficiency = Distance / (Cadence × Duration)
```

### **MoveIQ Activity Metrics**

**Auto-Detection Accuracy**
```
Detection Accuracy = Correctly Identified Activities / Total Activities × 100
```

**Activity Diversity Index**
```
Diversity = Number of Different Activity Types / Total Activities
```

## **Cross-Summary Composite Metrics**

### **Overall Wellness Score**
```
Wellness = (Sleep Quality × 0.3) + (HRV Score × 0.2) + (Activity Level × 0.3) + 
           (Stress Management × 0.2)
```

### **Recovery Readiness**
```
Recovery = (HRV × 0.4) + (Sleep Efficiency × 0.3) + (Resting HR × 0.3)
```

### **Fitness Progression Rate**
```
Progression = (Current Period Performance - Previous Period) / Previous Period × 100
```