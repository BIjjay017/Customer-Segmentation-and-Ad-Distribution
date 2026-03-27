"""Vercel entrypoint that exposes the Flask WSGI app."""

import sys
import os

# Add the parent directory to the path so we can import app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app as app
