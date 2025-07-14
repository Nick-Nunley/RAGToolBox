#!/bin/bash

set -euo pipefail
PYTHONPATH=src pytest --maxfail=1 --disable-warnings -q
