Here is a suggested `README.md` for your run analysis Power BI + Python visuals project.

***

# Run Analysis – Power BI & Python Visuals

This project combines a SQLite-backed **RunningAnalysis** engine with **Python visuals in Power BI** to monitor training load, performance, and readiness over time. It uses data from the `running_sessions` table in `Apex.db` and derived tables such as `training_logs`, `metrics_breakdown`, and `monthly_summaries`.[3]

## Data model

The core data source is the `running_sessions` table, enriched into a `training_log` DataFrame with the following key fields:[3]

- `date` – Session date (datetime)  
- `running_economy` – Running economy metric  
- `vo2max` – VO2Max estimate  
- `distance` – Session distance (km)  
- `time` – Session duration (seconds or minutes, depending on your pipeline)  
- `heart_rate` – Average heart rate (bpm)  
- `efficiency_score` – `running_economy / vo2max`  
- `energy_cost` – `running_economy * (distance / time)`  
- `TRIMP` – Training load score derived from duration and heart rate  
- `resting_hr` / `rest_hr` – Resting heart rate, optional  
- `sleep_quality` – Subjective 1–5 score (optional)  
- `fatigue_level` – Subjective 1–10 score (optional)[3]

In Power BI, these fields are exposed as columns (e.g., `Running Economy`, `Efficiency Score`, `Energy Cost`, etc.) and are renamed inside the Python visuals to snake_case names expected by the scripts (for example, `Running Economy` → `running_economy`).[7]

***

## Basic trends visual

**Script:** `visualize_trends_basic(dataset)` (used inside a Python visual).  
**Purpose:** Show core performance relationships over time and by session.[1][7]

### Expected columns (Power BI → Python)

- `date` → `date`  
- `Running Economy` → `running_economy`  
- `Efficiency Score` → `efficiency_score`  
- `Energy Cost` → `energy_cost`  
- `Heart Rate` → `heart_rate`  
- `Distance` → `distance`[7]

Renaming pattern inside the visual:

```python
dataset = dataset.rename(columns={
    'Running Economy': 'running_economy',
    'Efficiency Score': 'efficiency_score',
    'Energy Cost': 'energy_cost',
    'Heart Rate': 'heart_rate',
    'Distance': 'distance'
})
```

### Plots

The function creates a 2×2 figure:[1][7]

1. **Running Economy Trend** – `date` vs `running_economy` (line with markers).  
2. **Efficiency Score Trend** – `date` vs `efficiency_score`.  
3. **Energy Cost vs Distance** – scatter of `distance` vs `energy_cost`.  
4. **Heart Rate vs Running Economy** – scatter of `heart_rate` vs `running_economy`.

Use this visual to monitor whether running economy and efficiency are improving and how energy cost and heart rate relate to session characteristics.[1]

***

## Training load & ACWR visual

**Script:** `visualize_training_load(dataset)` (Power BI Python visual).  
**Purpose:** Quantify training load over time and calculate the **Acute:Chronic Workload Ratio (ACWR)** based on TRIMP.[7][3]

### Expected columns (Power BI → Python)

- `date` → `date`  
- `Heart Rate` → `heart_rate`  
- `Time` → `time` (used to compute duration in minutes)  
- Optional: `TRIMP` if you precompute it in SQL; otherwise it is calculated.[3][7]

Typical renaming:

```python
dataset = dataset.rename(columns={
    'Running Economy': 'running_economy',
    'Efficiency Score': 'efficiency_score',
    'Energy Cost': 'energy_cost',
    'Heart Rate': 'heart_rate',
    'Distance': 'distance',
    'TRIMP': 'TRIMP',
    'Time': 'time',
    'resting_hr': 'rest_hr',
    'VO2Max': 'vo2max'
})
dataset['date'] = pd.to_datetime(dataset['date'])
```

### Computations

- **TRIMP per session**:  
  - `duration_min = time / 60`  
  - `hr_ratio = (heart_rate - rest_hr) / (max_hr - rest_hr)`  
  - `TRIMP = duration_min * hr_ratio`[7][3]
- **Weekly aggregation**:  
  - `week = date.dt.isocalendar().week`  
  - `weekly_trimp`: sum of TRIMP per ISO week  
  - `acute_load`: 1‑week rolling mean of weekly TRIMP  
  - `chronic_load`: 4‑week rolling mean  
  - `acwr = acute_load / chronic_load`[3]

### Plots

Two subplots:[3]

- **TRIMP per Session** – `date` vs `TRIMP`.  
- **Weekly Training Load & ACWR** – `week` vs:  
  - weekly TRIMP  
  - acute and chronic load  
  - ACWR, with reference lines at 0.8 and 1.3 for lower/upper thresholds.

This visual highlights spikes in load and periods where ACWR is in a higher injury‑risk range.[3]

***

## Recovery & readiness visual

**Script:** `visualize_recovery_readiness(dataset)` (Power BI Python visual).  
**Purpose:** Combine objective load and heart rate with subjective metrics into daily **recovery** and **readiness** scores between 0 and 1.[7][3]

### Expected columns (Power BI → Python)

- `date` → `date`  
- `TRIMP` → `TRIMP`  
- `resting_hr` (or `Resting HR`) → `rest_hr`  
- Optional:  
  - `sleep_quality` (1–5)  
  - `fatigue_level` (1–10)[3]

Typical renaming:

```python
dataset = dataset.rename(columns={
    'Running Economy': 'running_economy',
    'Efficiency Score': 'efficiency_score',
    'Energy Cost': 'energy_cost',
    'Heart Rate': 'heart_rate',
    'Distance': 'distance',
    'TRIMP': 'TRIMP',
    'resting_hr': 'rest_hr'
})
dataset['date'] = pd.to_datetime(dataset['date'])
```

### Computations

- Baselines:  
  - `rhr_baseline`: mean of `rest_hr` or 60 if missing.  
  - `trimp_baseline`: 4‑session rolling mean of TRIMP (default 50 if missing).[3]
- Component scores (0–1, higher is better):  
  - `rhr_score = 1 - ((rest_hr - rhr_baseline) / rhr_baseline)`  
  - `load_score = 1 - (TRIMP / (trimp_baseline + 1e-8))`  
  - `sleep_score = sleep_quality / 5` (default 3)  
  - `fatigue_score = 1 - (fatigue_level / 10)` (default 5).[3]
- Composite scores:  
  - `recovery_score = 0.3*rhr_score + 0.3*load_score + 0.2*sleep_score + 0.2*fatigue_score`  
  - `readiness_score = 0.5*recovery_score + 0.3*load_score + 0.2*sleep_score`.[3]

### Plot

Single time‑series chart:[3]

- `date` vs `recovery_score` and `readiness_score`.  
- Threshold lines at 0.7 (caution) and 0.5 (high‑risk zone).

Use this visual to decide when to push or back off in training based on combined physiological and subjective signals.[3]

***

## Training score trends visual

**Script:** `visualize_training_score_trends(dataset)` (Power BI Python visual).  
**Purpose:** Build a composite **training score (0–100)** from multiple performance metrics and display its evolution over time.[7][3]

### Expected columns (Power BI → Python)

- `date` → `date`  
- `Running Economy` → `running_economy`  
- `VO2Max` → `vo2max`  
- `Distance` → `distance`  
- `Efficiency Score` → `efficiency_score`  
- `Heart Rate` → `heart_rate`[7][3]

Renaming snippet:

```python
dataset = dataset.rename(columns={
    'Running Economy': 'running_economy',
    'Efficiency Score': 'efficiency_score',
    'Energy Cost': 'energy_cost',
    'Heart Rate': 'heart_rate',
    'Distance': 'distance',
    'VO2Max': 'vo2max',
    'TRIMP': 'TRIMP',
    'resting_hr': 'rest_hr'
})
dataset['date'] = pd.to_datetime(dataset['date'])
```

### Computations

Metrics and weights:[3]

- `running_economy` – weight 0.25, higher is better  
- `vo2max` – weight 0.20, higher is better  
- `distance` – weight 0.15, higher is better  
- `efficiency_score` – weight 0.20, higher is better  
- `heart_rate` – weight 0.20, lower is better

Steps:[7][3]

1. For each metric, normalize to 0–1 across available values.  
2. Invert heart_rate so lower HR at given performance scores higher.  
3. Multiply each normalized metric by its weight.  
4. Sum weighted metrics per row and multiply by 100 to get `training_score` (0–100).

### Plot

- `date` vs `training_score` (line + filled area).  
- Thresholds:  
  - 70 – “Good” training state.  
  - 50 – “Caution” zone.[3]

This visual summarizes overall training quality incorporating performance, load, and efficiency signals.[3]

***

## Monthly summaries and database tables

Outside Power BI, the `RunningAnalysis` class calculates monthly aggregates and stores them into a `monthly_summaries` table with one record per month including means, standard deviations, and session counts for all core metrics and scores (recovery and readiness when available). These summaries are ideal as data sources for additional Power BI visuals or tables.[2][3]

***

## Usage guidelines

- Each Python visual in Power BI should:  
  - Include the relevant imports.  
  - Rename columns from Power BI’s display names to the script’s snake_case names.  
  - Convert `date` to datetime.  
  - Define *one* visual function.  
  - End with a single call like `visualize_trends_basic(dataset)` (no extra code below).[7]
- In the visual’s **Values** pane, add all columns referenced by that script, set to *Don’t summarize* for row‑level behavior.[12]

This setup yields a cohesive dashboard covering performance trends, training load and risk (ACWR), recovery/readiness, and an overall training score, all driven by the same run analysis data model.

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/76865532/336082b6-c9fb-4d5e-872d-1f340d33b086/VisualiseBasicTrends.py)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/76865532/184ad287-e32b-477d-8936-d64569ffe208/RunningAnalysis_v6.5.py)
[3](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/76865532/a48d8cd2-67e3-4ddf-9abe-a191426a0e11/RunningAnalysis_v6.5.py)
[4](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/76865532/61d1ed20-076c-40b1-b573-957bfb2fcaa5/image.jpg)
[5](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/76865532/cc8a6d59-38c8-4f7b-8f4d-1f61125b261e/image.jpg)
[6](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/76865532/671aec5c-7a16-4e7d-ab9c-1e9c5b0352f6/image.jpg)
[7](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/76865532/50b4501f-c422-42d7-82d3-772776d0eee0/PowerBIVisuals_V1.0.py)
[8](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/76865532/69b1ede2-7963-4866-a6e7-a15fb24592db/image.jpg)
[9](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/76865532/32aa9a24-9381-4ea5-a25f-b67d4f867b06/image.jpg)
[10](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/76865532/3d26b62d-cf16-46c0-92d6-c986aeefd77b/image.jpg)
[11](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/76865532/e7fe00bb-75fd-4c47-86a2-4635f8c59739/image.jpg)
[12](https://learn.microsoft.com/en-us/power-bi/connect-data/desktop-python-visuals)
