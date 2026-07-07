#!/bin/sh

# Automated Backup Script for Face Authentication System Database and Metadata Assets.
# Sweeps and packages enrollment templates, user databases, and transaction logs.

DATA_DIR="/app/data"
BACKUP_DIR="/app/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/biometric_backup_${TIMESTAMP}.tar.gz"

echo "Initializing backup process..."
mkdir -p "${BACKUP_DIR}"

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory ${DATA_DIR} does not exist. Aborting backup."
    exit 1
fi

# Package and compress the database and raw assets folders
echo "Compressing databases and metadata directories to ${BACKUP_FILE}..."
tar -czf "${BACKUP_FILE}" -C /app data

if [ $? -eq 0 ]; then
    echo "Backup completed successfully: ${BACKUP_FILE}"
else
    echo "ERROR: Compression failed. Backup aborted."
    exit 1
fi

# Retention Policy: Delete older backups, keeping only the 10 most recent files
echo "Enforcing 10-file backup retention policy..."
cd "${BACKUP_DIR}" && ls -tp | grep -v '/$' | tail -n +11 | xargs -I {} rm -- {}

echo "Backup maintenance cycle complete."
exit 0
