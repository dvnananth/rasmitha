Maintenance Hub (Streamlit)

Run locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Notes:
- Data is persisted to CSVs under `data_store/`.
- Tabs: Work Order, Plans, Related Records, Actuals, Safety, Logs, Failure Reporting, Reports & Analytics.
- This is a scaffold. Integrate with your backend/ERP later by replacing CSV operations with API/database calls.
