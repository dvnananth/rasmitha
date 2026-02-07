## Composable MES PoC (FastAPI + Streamlit)

Quick start:

1. Create the `.env` (already present with defaults). Adjust if needed.
2. Build and run:

```bash
docker compose up -d --build
```

Services:
- API: http://localhost:8000/docs
- Builder (Streamlit): http://localhost:8501

Notes:
- DB auto-migrates via SQLAlchemy metadata on startup (PoC). For prod, introduce Alembic migrations.

# rasmitha