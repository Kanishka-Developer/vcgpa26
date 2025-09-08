# CGPA Explorer (Streamlit)

An interactive Streamlit app to explore the VIT 2026 Batch Dataset.

Features:
- Robust CSV loading and cleaning
- Search by name, registration number, email
- Filter by campus, department, degree/specialization
- Filter by CGPA ranges
- Graphs: histograms, box plots, scatter, bar counts, heatmap
- Upload a list of names or registration numbers to compute CGPA stats client-side
- Download filtered dataset as CSV

## Quick start

1. Create a virtual environment (optional but recommended)
2. Install dependencies
3. Run the Streamlit app

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run src/app.py
```

The app expects a `data.csv` in the project root by default. You can also upload a file from the UI.

## Project structure

- `src/app.py` — Streamlit UI
- `src/data_utils.py` — data loading, cleaning, transformations
- `src/charts.py` — chart helpers using Altair
