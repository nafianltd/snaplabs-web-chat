# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create logs directory
RUN mkdir -p logs

# Create startup script
RUN echo '#!/bin/bash\n\
if [ "$USE_CHAINLIT" = "true" ]; then\n\
    exec chainlit run app.py --host 0.0.0.0 --port $PORT\n\
else\n\
    exec uvicorn app:app --host 0.0.0.0 --port $PORT\n\
fi\n\
' > /app/start.sh && chmod +x /app/start.sh

# Set the startup script as the entry point
ENTRYPOINT ["/app/start.sh"]
