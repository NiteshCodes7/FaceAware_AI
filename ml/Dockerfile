FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

ENV WEIGHTS_TAG=v2-custom-model \
    WEIGHTS_FILE=bisenet_celebamask_best.pt \
    WEIGHTS_DIR=/home/user/app/weights \
    MODEL_PATH=/home/user/app/weights/bisenet_celebamask_best.pt

RUN python scripts/download_weights.py

EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]