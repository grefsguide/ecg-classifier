#!/usr/bin/env sh
set -eu

echo "[data-init] starting"

: "${DATASET_NAME:=ecg_img_v1}"
: "${SHARED_DATA_DIR:=/shared-data}"
: "${SHARED_DATASET_DIR:=/shared-data/ecg_img}"

DVC_FILE="data_registry/${DATASET_NAME}.dvc"
SOURCE_DIR="${DATASET_NAME}"
TARGET_DIR="${SHARED_DATASET_DIR}"

echo "[data-init] DATASET_NAME=${DATASET_NAME}"
echo "[data-init] DVC_FILE=${DVC_FILE}"
echo "[data-init] SOURCE_DIR=${SOURCE_DIR}"
echo "[data-init] TARGET_DIR=${TARGET_DIR}"

if [ ! -f "${DVC_FILE}" ]; then
  echo "[data-init] DVC file not found: ${DVC_FILE}"
  exit 1
fi

echo "[data-init] pulling dataset via DVC"
dvc pull "${DVC_FILE}" --force

if [ ! -d "${SOURCE_DIR}" ]; then
  echo "[data-init] source dataset directory not found after dvc pull: ${SOURCE_DIR}"
  exit 1
fi

mkdir -p "${SHARED_DATA_DIR}"
rm -rf "${TARGET_DIR}"
cp -R "${SOURCE_DIR}" "${TARGET_DIR}"

echo "[data-init] dataset copied to ${TARGET_DIR}"
ls -lah "${SHARED_DATA_DIR}"
ls -lah "${TARGET_DIR}"