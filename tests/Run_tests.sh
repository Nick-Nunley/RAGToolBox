#!/bin/bash

set -euo pipefail
PYTHONPATH=. pytest --maxfail=1 --disable-warnings -q
