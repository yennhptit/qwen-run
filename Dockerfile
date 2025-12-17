# Base image Python
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy files
COPY requirements.txt .
COPY app.py .
COPY .env .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
