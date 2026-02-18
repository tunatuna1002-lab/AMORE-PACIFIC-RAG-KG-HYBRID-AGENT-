#!/bin/sh
set -e

# Fix Railway volume mount permissions (volume is mounted as root)
if [ -d /data ]; then
    chown -R appuser:appuser /data 2>/dev/null || true
fi

# Create data directory if it doesn't exist
mkdir -p /data 2>/dev/null || true

# Drop privileges and run as appuser
exec gosu appuser "$@"
