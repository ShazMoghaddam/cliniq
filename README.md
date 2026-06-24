# ClinIQ

**Clinical Trial Site Performance Intelligence**

ClinIQ is a Python-based B2B intelligence platform for monitoring clinical trial site performance, targeting mid-market CROs and biotech sponsors across the UK and EU.

[![CI](https://github.com/ShazMoghaddam/cliniq/actions/workflows/ci.yml/badge.svg)](https://github.com/ShazMoghaddam/cliniq/actions)

---

## Live Demo

**Dashboard:** `https://cliniq.onrender.com`

| Field | Value |
|---|---|
| User ID | `demo-sponsor` |
| Password | `cliniq-demo-2024` |
| Role | `sponsor_view` |

API docs: `https://cliniq.onrender.com/docs`

---

## Views

| View | Description |
|---|---|
| **Portfolio Heatmap** | Risk vs enrolment scatter, quadrant-coded by threshold |
| **Watchlist** | Sites ranked by composite risk score with alert flags |
| **Site Drill-Down** | Enrolment curves, lag trend, deviation timeline, risk breakdown |
| **Deviation Log** | Filterable log with NLP category tags and severity colour coding |
| **AI Assistant** | Claude-powered plain-English queries grounded in trial KPIs |

---

## Local setup

```bash
git clone https://github.com/ShazMoghaddam/cliniq.git
cd cliniq
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m cliniq.db.seeder          # seed 12-month synthetic dataset
python cliniq/dashboard/app.py      # → http://localhost:8050
```

FastAPI layer:
```bash
uvicorn cliniq.api.app:app --reload --port 8000
# → http://localhost:8000/docs
```

---

## Running the tests

```bash
PYTHONPATH=. python -m pytest tests/ -v
PYTHONPATH=. python -m pytest tests/ --cov=cliniq --cov-report=term-missing
```

| Phase | Layer | Tests |
|---|---|---|
| 1 | Data Foundation | 70 |
| 2 | Analytics & ML | 82 |
| 3 | API Layer | 100 |
| 4 | Dashboard & AI | 120 |
| 5 | E2E + Security | 124 |
| **Total** | | **496** |

---

## Architecture

```
cliniq/
├── db/           SQLAlchemy models · Alembic migrations · Seeder
├── analytics/    Velocity · Funnel · Lag · KPI snapshots
├── ml/           Risk model · NLP classifier · Dropout probability
├── api/          FastAPI · JWT auth · RBAC · OpenAPI
├── dashboard/    Plotly Dash · URL routing · Cache · Charts
├── ai/           Claude assistant · Context builder
├── rbac/         Token auth · Role hierarchy · Audit log
└── config/       Settings · Feature flags
```

**Stack:** Python 3.11 · FastAPI · Plotly Dash · SQLAlchemy 2.0 · scikit-learn · spaCy · Claude · Docker · Render

**Regulatory:** ICH E6(R3) · MHRA RBM · UK GDPR / DPA 2018 · EU CTR 536/2014

---

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | SQLite (dev) |
| `SECRET_KEY` | JWT signing secret (≥32 chars) | Dev default |
| `ANTHROPIC_API_KEY` | Claude API key | Empty |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT expiry | 60 |
| `DEBUG` | Debug mode | false |

---

## Deployment

```bash
# Docker
docker build -t cliniq .
docker run -p 8050:8050 -e SECRET_KEY=... -e ANTHROPIC_API_KEY=... cliniq

# Render — deploys automatically via render.yaml on push to main
# Set ANTHROPIC_API_KEY manually in Render dashboard
```

---

## Go-to-market

OEM/white-label to mid-market CROs (5–50 concurrent trials):
- Platform licence: £800–1,500/month
- White-label: £300/month additional
- Onboarding: 2-hour call + CSV import, no API integration required for v1

---

## Author

**Shaz Moghaddam** — London

- Website: [shazmoghaddam.github.io](https://shazmoghaddam.github.io/)
- LinkedIn: [linkedin.com/in/shazmoghaddam](https://www.linkedin.com/in/shazmoghaddam/)
- GitHub: [github.com/ShazMoghaddam](https://github.com/ShazMoghaddam)

*Build completely. Test thoroughly. Approach buyers only when the product can stand on its own.*
