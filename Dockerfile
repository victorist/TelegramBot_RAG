# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.12.2
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1
# Keeps Python from buffering stdout and stderr.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
# COPY . .
COPY ./config.json    congig.json
COPY ./telegram.py    telegram.py
COPY ./requirements.txt   requirements.txt
COPY ./store/        /store/


# Change ownership of the application files
RUN chown -R appuser /app

# Switch to the non-privileged user to run the application.
USER appuser

# Expose the port the app runs on
EXPOSE 

# Command to run the application
CMD ["python", "telegram.py"]
