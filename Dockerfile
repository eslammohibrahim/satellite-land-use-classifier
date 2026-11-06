FROM python:3.11-slim

WORKDIR /app

# Dependencies first, so this layer is cached when only the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app

# Bake the trained weights into the image, so the container starts without network access
RUN mkdir -p models && python -c "from urllib.request import urlretrieve; urlretrieve('https://github.com/eslammohibrahim/satellite-land-use-classifier/releases/download/v1.0/resnet18_pretrained.pt', 'models/resnet18_pretrained.pt')"

# Don't run the service as root
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
