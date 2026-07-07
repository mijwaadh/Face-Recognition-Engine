# Base stage using official slim Python runtime
FROM python:3.12-slim

# System environment configs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies for OpenCV and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python package dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application directories
COPY myface/ /app/myface/
COPY scripts/ /app/scripts/

# Create folders for data and logs with access rights
RUN mkdir -p /app/data /app/logs && chmod -R 755 /app/data /app/logs

# Make startup scripts executable
RUN chmod +x /app/scripts/start.sh

EXPOSE 8000

# Execute Gunicorn launcher
ENTRYPOINT ["sh", "/app/scripts/start.sh"]
