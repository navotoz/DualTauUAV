#!/bin/bash
cd /home/pi/DualTauUAV
gunicorn /venv/bin/gunicorn -w 1 --timeout 1000 --bind 0.0.0.0:8080 app:app