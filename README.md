# API Abuse – AI Cyber Resilience Dashboard

This project detects suspicious API usage sessions using **behaviour metrics + API call-graph features** and visualizes the results in a **dark (black + purple/blue) Streamlit dashboard**.

## What you get

- `api_abuse_anomaly_detector.py`  
  Builds graph features, trains an Isolation Forest (semi-supervised style), scores sessions, and saves results to CSV.

- `dashboard.py`  
  A Streamlit dashboard that reads the CSV outputs and provides SOC-style views: posture, risk tiers, driver analysis, investigation workbench, and supervised validation.

---

## Folder structure

Keep this exact structure:

```
api_abuse/
  api_abuse_anomaly_detector.py
  dashboard.py
  requirements.txt
  README.md

  data/
    supervised_dataset.csv
    supervised_call_graphs.json
    remaining_behavior_ext.csv
    remaining_call_graphs.json

  outputs/
    supervised_scored.csv
    remaining_scored.csv
```

If `outputs/` doesn’t exist, create it.

---

## Requirements

### `requirements.txt`

Create (or use) this:

```txt
streamlit
pandas
numpy
plotly
scikit-learn
```

---

## Run locally (Mac / Linux)

### 1) Create & activate a virtual environment

From inside the `api_abuse/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Run the detector (generates outputs CSVs)

```bash
python api_abuse_anomaly_detector.py
```

Expected outputs:

- `outputs/supervised_scored.csv`
- `outputs/remaining_scored.csv`

### 3) Run the dashboard

```bash
streamlit run dashboard.py
```

Streamlit prints a URL like:

- `http://localhost:8501`

Open it in your browser.

---

## Run locally (Windows)

Open **PowerShell** in the `api_abuse/` folder:

### 1) Create & activate venv

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) Run detector

```powershell
python api_abuse_anomaly_detector.py
```

### 3) Run dashboard

```powershell
streamlit run dashboard.py
```

---

## Run on Google Colab

### 1) Upload project
Upload the **entire** `api_abuse/` folder (or a zip) to Colab, then:

```python
!pip -q install -r /content/api_abuse/requirements.txt
```

### 2) Run detector

```python
!python /content/api_abuse/api_abuse_anomaly_detector.py
```

### 3) Run Streamlit in Colab (with tunnel)

Colab cannot expose localhost directly; use a tunnel. For example:

```python
!pip -q install streamlit pyngrok
```

```python
from pyngrok import ngrok
public_url = ngrok.connect(8501)
public_url
```

In another cell:

```python
!streamlit run /content/api_abuse/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

Open the URL printed by `ngrok`.

---

## Notes on metrics

- `anomaly_score` and `is_anomaly` are produced by the detector.
- The dashboard is **read-only analytics**: it loads CSVs and creates filters, aggregations, risk tiers, and charts.
- Risk tiers are percentile-based (configurable in the sidebar).

---

This project applies **behavioural anomaly detection** to API usage sessions by combining:

- behaviour metrics (CSV)
- call-graph statistics (JSON → graph features)

The model (Isolation Forest) flags deviations from normal patterns and the dashboard provides SOC-friendly visualization and investigation workflows.
