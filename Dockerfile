# Use Ubuntu 24 as the base image
FROM ubuntu:24.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python, pip, and venv support
RUN apt-get update \
    && apt-get install -y python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# create a virtual environment for package installation
RUN python3 -m venv /opt/venv

# prioritize venv binaries in PATH
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /api

# Copy all backend files into the container
COPY . /api

# Install Python dependencies inside the virtual environment
RUN pip install --no-cache-dir -r requirements-api-sensor.txt

# Expose port for uvicorn
EXPOSE 3000

# Default command
CMD ["uvicorn", "api.heatmap:app", "--host", "0.0.0.0", "--port", "3000"]
