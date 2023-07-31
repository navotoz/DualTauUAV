FROM python:3.9-slim AS base

# Install dependencies
RUN apt-get update
RUN apt-get install -y --no-install-recommends \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv venv

# install the requirements into a build, to avoid reinstallation
FROM base as build-venv
RUN /venv/bin/pip3 install flask gunicorn  pyproj --no-cache-dir  --index-url https://www.piwheels.org/simple --index-url https://pypi.org/simple

# Copy files into the build image
FROM build-venv as build
COPY . /app
WORKDIR /app

# Expose the default Flask port
EXPOSE 5000

# Set the entrypoint to start the Flask app
ENTRYPOINT ["/venv/bin/gunicorn", "-w", "1", "--timeout", "1000", "--bind", "0.0.0.0:5000", "app:app"]