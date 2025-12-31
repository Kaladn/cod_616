#!/bin/bash
set -e

echo "Checking for forbidden patterns in runtime code..."

FORBIDDEN=$(grep -rn --include="*.py" \
    -e "stub" \
    -e "fake" \
    -e "mock" \
    -e "dry_run" \
    -e "test_mode" \
    loggers/ | grep -v "test_" | grep -v "__pycache__" || true)

if [ -n "$FORBIDDEN" ]; then
    echo "ERROR: Found forbidden patterns in runtime code:"
    echo "$FORBIDDEN"
    exit 1
fi

echo "✓ No forbidden patterns found"