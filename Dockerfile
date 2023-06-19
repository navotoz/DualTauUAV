FROM python:3.9-slim AS base

# Install dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends \ 
    gcc \
    build-essential \
    make \
    python3-dev
RUN apt-get install -y --no-install-recommends \
        libc-dev \
        libffi-dev \
        libnotify4 \
        libgtk2.0-dev \
        libgtkmm-3.0-1v5 \
        libatlas-base-dev \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv venv

# install the requirements into a build, to avoid reinstallation
FROM base as build-venv
COPY requirements.txt /requirements.txt
RUN /venv/bin/pip3 install -r requirements.txt  --no-cache-dir --index-url https://www.piwheels.org/simple

# Copy files into the build image
FROM build-venv as build
COPY . /app
WORKDIR /app

# Expose the default Flask port
EXPOSE 8080

# Set the entrypoint to start the Flask app
ENTRYPOINT ["/venv/bin/gunicorn", "-w", "1", "--timeout", "1000", "--bind", "0.0.0.0:8080", "app:app"]