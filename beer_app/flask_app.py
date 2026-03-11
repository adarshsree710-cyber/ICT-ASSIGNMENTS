# flask_app.py  — WSGI entry point for PythonAnywhere
# In PythonAnywhere dashboard set:
#   Source code:   /home/<your-username>/beer_app
#   Working dir:   /home/<your-username>/beer_app
#   WSGI file:     this file

import sys, os
sys.path.insert(0, '/home/<your-username>/beer_app')
os.chdir('/home/<your-username>/beer_app')

from app import app as application
