"""
ClinIQ — WSGI entry point for gunicorn.
gunicorn cliniq.dashboard.wsgi:server
"""
from cliniq.dashboard.app import create_dashboard

_app = create_dashboard()
server = _app.server   # Underlying Flask app
