# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project
COPY . .

# Expose the port Railway will map
EXPOSE 8080

# Start with gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
