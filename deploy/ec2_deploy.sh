#!/usr/bin/env bash
set -euo pipefail

APP_NAME="${APP_NAME:-plant-diagnosis-api}"
IMAGE_NAME="${IMAGE_NAME:-}"
IMAGE_TAR="${IMAGE_TAR:-/tmp/plant-diagnosis-api.tar.gz}"
APP_PORT="${APP_PORT:-8000}"
CONTAINER_PORT="${CONTAINER_PORT:-8000}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[INFO] Docker nu exista. Instalez Docker..."
  sudo apt-get update
  sudo apt-get install -y docker.io
  sudo systemctl enable docker
  sudo systemctl start docker
fi

echo "[INFO] Incarc imaginea Docker din ${IMAGE_TAR}"
gunzip -c "${IMAGE_TAR}" | sudo docker load

if [[ -z "${IMAGE_NAME}" ]]; then
  IMAGE_NAME="$(sudo docker images --format '{{.Repository}}:{{.Tag}}' | grep "^${APP_NAME}:" | head -n 1)"
fi
if [[ -z "${IMAGE_NAME}" ]]; then
  echo "[ERROR] Nu am gasit imagine Docker cu prefixul ${APP_NAME}:"
  sudo docker images
  exit 1
fi

echo "[INFO] Opresc containerul vechi daca exista"
sudo docker rm -f "${APP_NAME}" >/dev/null 2>&1 || true

echo "[INFO] Pornesc ${IMAGE_NAME} pe portul ${APP_PORT}"
sudo docker run -d \
  --name "${APP_NAME}" \
  --restart unless-stopped \
  -p "${APP_PORT}:${CONTAINER_PORT}" \
  -e MODEL_PATH=/app/models/classifier/best_model.pt \
  -e REMOVE_BACKGROUND=true \
  -e MAX_CROPS=8 \
  "${IMAGE_NAME}"

echo "[OK] Deploy terminat"
sudo docker ps --filter "name=${APP_NAME}"
rm -f "${IMAGE_TAR}"
sudo docker image prune -f >/dev/null 2>&1 || true
