FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Use environment variable for port with default fallback
ENV PORT=8080

# Run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:server