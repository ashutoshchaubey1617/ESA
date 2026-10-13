#!/bin/zsh
cd "$(dirname "$0")"
if [[ -f face-dev/bin/activate ]]; then
  source face-dev/bin/activate
fi
python3 recognize.py
