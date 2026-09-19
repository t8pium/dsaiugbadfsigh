#!/bin/sh
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  exec python3 bootstrap.py
elif command -v python >/dev/null 2>&1; then
  exec python bootstrap.py
else
  echo "64-bit Python 3.11-3.13 is required: https://www.python.org/downloads/"
  exit 1
fi
