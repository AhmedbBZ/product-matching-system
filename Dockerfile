FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system deps needed by common packages (adjust as required)
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		build-essential \
		curl \
		libgomp1 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App files
COPY data/ ./data/
COPY models/ ./models/
COPY src/ ./src/
COPY index.html ./index.html

# Create a non-root user for runtime
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck (requires curl which we installed above)
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://127.0.0.1:8000/api/status || exit 1

# Use Uvicorn to run the FastAPI app and bind to 0.0.0.0
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]