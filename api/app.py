"""Vercel Python function entrypoint for Flask app."""

import os
import sys

# Ensure project root is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import app as app
