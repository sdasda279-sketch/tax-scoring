#!/usr/bin/env bash
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Python3 topilmadi"; exit 1; }
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
[ -f .venv/tayyor.txt ] || { pip install -r requirements.txt && echo ok > .venv/tayyor.txt; } || exit 1
streamlit run app.py
