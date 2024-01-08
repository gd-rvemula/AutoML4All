#!/bin/bash
# Start the AutoML4All server
git pull
pip install -r requirements.txt
python -m streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
