#!/bin/sh

# Ensure logs directories exist inside the container
mkdir -p /app/logs

echo "Starting FastAPI app behind Gunicorn process manager..."
# Boot Gunicorn with Uvicorn worker classes
# -w 4: Spawns 4 independent workers for high concurrency.
# --bind 0.0.0.0:8000: Binds to local network port 8000.
# --timeout 120: Gives extra timeout room for deep SVD matrix operations.
exec gunicorn myface.api.main:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:${PORT:-8000} \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
