#!/usr/bin/env bash
# One-command launcher: venv (first run) + deps (if missing) + web app.
set -e
cd "$(dirname "$0")"
PY=./.venv/bin/python

for bin in tesseract pdftotext; do
  command -v "$bin" >/dev/null 2>&1 || echo "⚠️  '$bin' not found — install with: brew install tesseract poppler"
done

if [ ! -d .venv ]; then
  echo "creating virtualenv (.venv) ..."
  python3 -m venv .venv
fi

if "$PY" -c "import fastapi, uvicorn, jinja2, anthropic, fitz, faker, pytesseract" >/dev/null 2>&1; then
  echo "dependencies present ✓"
else
  echo "installing dependencies (first run — may take a minute; do NOT Ctrl-C) ..."
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -r requirements.txt
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
  echo "API key configured ✓ (vision extraction enabled)"
else
  echo "ℹ️  No ANTHROPIC_API_KEY set — real uploads show a setup prompt; built-in samples"
  echo "    still demo on reference data. To enable extraction: export ANTHROPIC_API_KEY=sk-ant-..."
  echo "    before ./run.sh, or set it in-app as Admin (API Key, top-right)."
fi

echo "launching FinWise AI → http://localhost:8000   (Ctrl-C here stops the app)"
exec ./.venv/bin/uvicorn webapp.main:app --host 0.0.0.0 --port 8000 "$@"
