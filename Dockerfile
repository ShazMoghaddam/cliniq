FROM python:3.11-slim AS base

# System deps for spaCy, psycopg2-binary, compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security hardening
RUN addgroup --system cliniq && adduser --system --ingroup cliniq cliniq

WORKDIR /app

# ---- deps layer (cached unless requirements.txt changes) ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- application source ----
COPY --chown=cliniq:cliniq . .

# Pre-seed SQLite dev DB so cold start is fast
RUN python -c "
import os
if not os.getenv('DATABASE_URL'):
    from cliniq.db.database import init_db, SessionLocal
    from cliniq.db.seeder import seed
    init_db()
    with SessionLocal() as s:
        result = seed(s)
        print('Dev DB:', result.get('status'))
" || true

# Health check: Dash serves HTML at /
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

USER cliniq

EXPOSE 8050

CMD ["python", "-m", "gunicorn", \
     "--bind", "0.0.0.0:8050", \
     "--workers", "1", \
     "--timeout", "120", \
     "cliniq.dashboard.wsgi:server"]
