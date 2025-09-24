# Contributing

## Branching
Create feature branches from `main`: `feature/<short>` or `fix/<short>` and open a PR early (link the issue with `Closes #<id>`).

## Local dev
```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-gui.txt
python -u app.py
