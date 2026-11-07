# 💧 Smart Irrigation Analytics Dashboard

An end-to-end Big Data analytics and visualization dashboard built with **Streamlit**, **Plotly**, **scikit-learn**, and **NetworkX** to analyze sensor telemetry and irrigation pipeline network topologies.

---

## 🚀 Quick Start & Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## 📂 Data Sources & Local Loading

Place `system_data.csv` (sensor telemetry) and `irrigation_graph_edges.csv` (pipeline network topology) in the same directory as `app.py`.

* **Sidebar Options:** Select your data source via "Local file (path)", "Upload CSV", or "GitHub URL (raw)".
* **Data Processing:** Loaded via `pd.read_csv("system_data.csv")` and dynamically normalized using `RENAME_MAP`.

---

## ⚙️ Key Assumptions & Simulation Logic

* **Core Dataset:** The GitHub source dataset contains 5 primary attributes (*Soil Moisture*, *Temperature*, *Humidity*, *Light Level*, and *Valve Duration*).
* **Simulated Attributes:** If missing from the CSV, `timestamp` (hourly frequency), `zone_id` (round-robin assignment across Zones A–D), and `device_id` are **simulated**. If present, native columns are prioritized.
* **Water Volume Estimation:** Calculated as $\text{Water Volume} = \text{Valve Duration (s)} \times \text{Flow Rate (L/s)}$ (default flow rate: 2.0 L/s, configurable via the sidebar).

