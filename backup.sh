#!/bin/bash
# Dated backup script - excludes pycache and temp files

BACKUP_DIR="$HOME/backups"
DATE=$(date +%Y-%m-%d_%H%M)
DEST="${BACKUP_DIR}/${DATE}_backup"

mkdir -p "$DEST"

rsync -a \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.lgd*' \
    --exclude='.pytest_cache' \
    "$(pwd)/" "$DEST/"

echo "Backup created: $DEST"
