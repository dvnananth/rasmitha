# Maintenance Work Orders (Flask)

A minimal website to track maintenance work orders using Flask and SQLite.

## Features

- Create, read, update, and delete work orders
- Filter by status and priority, basic keyword search
- SQLite database, no external dependencies

## Requirements

- Python 3.9+

## Setup

```bash
cd maintenance_workorder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

The app will start on `http://127.0.0.1:5000` (or `http://0.0.0.0:5000` for LAN access).

## Notes

- The database file `workorders.db` is created automatically on first run.
- Default secret key is for development only. Set `SECRET_KEY` in your environment for production.