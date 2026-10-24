#!/usr/bin/env bash
set -euo pipefail

# Hive FL Containerized Test Runner
# 1 Host + 2 Contributors with Live Host & Contributor Web Dashboards

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Hive FL: Containerized 1-Host + 2-Contributor Verification ==="

NETWORK_NAME="hive-net"
IMAGE_NAME="hive-fl:latest"
HOST_CONTAINER="hive-host"
CLIENT1_CONTAINER="hive-client-1"
CLIENT2_CONTAINER="hive-client-2"
WORK_DIR="${ROOT_DIR}/.hive_run"

# Optional flag: KEEP_ALIVE=1 to keep containers running after verification
KEEP_ALIVE="${KEEP_ALIVE:-0}"

# 1. Cleanup old containers and network
echo "[1/7] Cleaning up previous containers..."
docker rm -f "${HOST_CONTAINER}" "${CLIENT1_CONTAINER}" "${CLIENT2_CONTAINER}" 2>/dev/null || true
docker network create "${NETWORK_NAME}" 2>/dev/null || true

docker run --rm --entrypoint rm -v "${ROOT_DIR}:/workspace" "${IMAGE_NAME}" -rf /workspace/.hive_run 2>/dev/null || rm -rf "${WORK_DIR}" || true
mkdir -p "${WORK_DIR}/host" "${ROOT_DIR}/data/client1" "${ROOT_DIR}/data/client2"

# 2. Build model package if not present
if [ ! -f "${ROOT_DIR}/examples/demo.hivepkg" ]; then
    echo "[2/7] Building sample .hivepkg model package..."
    docker run --rm -v "${ROOT_DIR}:/workspace" -w /workspace "${IMAGE_NAME}" \
        package build --src examples/demo_pkg --out examples/demo.hivepkg
else
    echo "[2/7] Using existing package: examples/demo.hivepkg"
fi

# 3. Generate client datasets if not present
if [ ! -f "${ROOT_DIR}/data/client1/data.pt" ] || [ ! -f "${ROOT_DIR}/data/client2/data.pt" ]; then
    echo "[3/7] Generating synthetic datasets..."
    docker run --rm -v "${ROOT_DIR}:/workspace" -w /workspace --entrypoint python "${IMAGE_NAME}" -c "
import torch
torch.manual_seed(101)
x1 = torch.randn(100, 4)
y1 = (x1[:, 0] + x1[:, 1] > 0).long()
torch.save({'x': x1, 'y': y1}, 'data/client1/data.pt')

torch.manual_seed(202)
x2 = torch.randn(120, 4)
y2 = (x2[:, 0] + x2[:, 1] > 0).long()
torch.save({'x': x2, 'y': y2}, 'data/client2/data.pt')
"
fi

# 4. Start Host Container
echo "[4/7] Launching Hive Host container (Port 8000: Admin UI, Port 9000: Public FL API)..."
docker run -d \
    --name "${HOST_CONTAINER}" \
    --network "${NETWORK_NAME}" \
    -p 8000:8000 \
    -p 9000:9000 \
    -v "${ROOT_DIR}:/workspace" \
    -w /workspace \
    -e HIVE_ADMIN_HOST=0.0.0.0 \
    "${IMAGE_NAME}" \
    host \
    --package examples/demo.hivepkg \
    --tunnel manual \
    --public-url "http://${HOST_CONTAINER}:9000" \
    --rounds 2 \
    --min-clients 2 \
    --workdir /workspace/.hive_run/host

# Wait for Host initialization and extract Join URI & Admin Token
echo "Waiting for Host to initialize..."
JOIN_URI=""
ADMIN_URL=""
for i in {1..30}; do
    LOGS=$(docker logs "${HOST_CONTAINER}" 2>&1 || true)
    JOIN_URI=$(echo "${LOGS}" | grep -o 'hive://[^[:space:]]*' | head -n 1 || true)
    ADMIN_URL=$(echo "${LOGS}" | grep -o 'http://127\.0\.0\.1:8000/?token=[^[:space:]]*' | head -n 1 || true)
    if [ -n "${JOIN_URI}" ] && [ -n "${ADMIN_URL}" ]; then
        break
    fi
    sleep 1
done

if [ -z "${JOIN_URI}" ]; then
    echo "ERROR: Failed to retrieve Join URI from host container."
    docker logs "${HOST_CONTAINER}"
    exit 1
fi

ADMIN_TOKEN=$(echo "${ADMIN_URL}" | sed -n 's/.*token=\([^&]*\).*/\1/p')
FP=$(echo "${JOIN_URI}" | sed -n 's/.*fp=\([^&]*\).*/\1/p' | sed 's/%3A/:/g')

# 5. Launch Contributors with Live Web Dashboards Enabled
echo "[5/7] Launching Contributor 1 (Port 8765) and Contributor 2 (Port 8766)..."
docker run -d \
    --name "${CLIENT1_CONTAINER}" \
    --network "${NETWORK_NAME}" \
    -p 8765:8765 \
    -v "${ROOT_DIR}:/workspace" \
    -w /workspace \
    -e HIVE_UI_HOST=0.0.0.0 \
    -e HIVE_ALLOWED_HOSTS="0.0.0.0,127.0.0.1,localhost,hive-client-1" \
    "${IMAGE_NAME}" \
    join \
    --uri "${JOIN_URI}" \
    --data-dir /workspace/data/client1 \
    --nickname Contributor-1 \
    --ui-port 8765 \
    --sandbox process \
    --no-only-idle \
    --no-pause-on-battery \
    --auto-approve-fingerprint "${FP}" \
    --auto-start

docker run -d \
    --name "${CLIENT2_CONTAINER}" \
    --network "${NETWORK_NAME}" \
    -p 8766:8766 \
    -v "${ROOT_DIR}:/workspace" \
    -w /workspace \
    -e HIVE_UI_HOST=0.0.0.0 \
    -e HIVE_ALLOWED_HOSTS="0.0.0.0,127.0.0.1,localhost,hive-client-2" \
    "${IMAGE_NAME}" \
    join \
    --uri "${JOIN_URI}" \
    --data-dir /workspace/data/client2 \
    --nickname Contributor-2 \
    --ui-port 8766 \
    --sandbox process \
    --no-only-idle \
    --no-pause-on-battery \
    --auto-approve-fingerprint "${FP}" \
    --auto-start

# Extract Contributor Web UI URLs & tokens
echo "Waiting for Contributor Dashboards to initialize..."
CLIENT1_TOKEN=""
CLIENT2_TOKEN=""
for i in {1..20}; do
    L1=$(docker logs "${CLIENT1_CONTAINER}" 2>&1 || true)
    L2=$(docker logs "${CLIENT2_CONTAINER}" 2>&1 || true)
    CLIENT1_TOKEN=$(echo "${L1}" | grep -o 'Contributor Local Web UI: http://[^[:space:]]*?token=[^[:space:]]*' | sed -n 's/.*token=\([^&]*\).*/\1/p' | head -n 1 || true)
    CLIENT2_TOKEN=$(echo "${L2}" | grep -o 'Contributor Local Web UI: http://[^[:space:]]*?token=[^[:space:]]*' | sed -n 's/.*token=\([^&]*\).*/\1/p' | head -n 1 || true)
    if [ -n "${CLIENT1_TOKEN}" ] && [ -n "${CLIENT2_TOKEN}" ]; then
        break
    fi
    sleep 1
done

HOST_DASHBOARD_URL="http://localhost:8000/?token=${ADMIN_TOKEN}"
CLIENT1_DASHBOARD_URL="http://localhost:8765/?token=${CLIENT1_TOKEN}"
CLIENT2_DASHBOARD_URL="http://localhost:8766/?token=${CLIENT2_TOKEN}"

echo ""
echo "======================================================================"
echo "🌐 ALL WEB DASHBOARDS LIVE & ACCESSIBLE FROM YOUR BROWSER:"
echo "======================================================================"
echo "  Host Admin Dashboard:      ${HOST_DASHBOARD_URL}"
echo "  Contributor 1 Dashboard:   ${CLIENT1_DASHBOARD_URL}"
echo "  Contributor 2 Dashboard:   ${CLIENT2_DASHBOARD_URL}"
echo "----------------------------------------------------------------------"
echo "  Host Join URI:             ${JOIN_URI}"
echo "======================================================================"
echo ""

# 6. Monitor FL Training Rounds until completion
echo "[6/7] Monitoring federated training progress across all nodes..."
SESSION_COMPLETED=false
for i in {1..60}; do
    SUMMARY=$(curl -s "http://127.0.0.1:8000/admin/api/summary?token=${ADMIN_TOKEN}" || echo "{}")
    STATUS=$(echo "${SUMMARY}" | grep -o '"status":"[^"]*"' | head -n 1 | cut -d'"' -f4 || true)
    CURRENT_ROUND=$(echo "${SUMMARY}" | grep -o '"current_round":[0-9]*' | head -n 1 | cut -d':' -f2 || true)
    
    echo "  [Poll $i] Host Session: '${STATUS}' | Current Round: ${CURRENT_ROUND}"
    
    if [ "${STATUS}" = "completed" ]; then
        SESSION_COMPLETED=true
        break
    fi
    sleep 2
done

if [ "${SESSION_COMPLETED}" = false ]; then
    echo "ERROR: Timed out waiting for session completion."
    docker logs "${HOST_CONTAINER}"
    exit 1
fi

echo ""
echo "=== Federated Learning Rounds Succeeded! ==="
echo ""

# 7. Collect and display metrics from Host AND Contributors
echo "[7/7] Telemetry Verification from Host & Contributor Dashboards:"
echo ""
echo "--- Host: Round Progression Summary ---"
curl -s "http://127.0.0.1:8000/admin/api/rounds?token=${ADMIN_TOKEN}"
echo ""

echo "--- Host: Contributor Nodes Status ---"
curl -s "http://127.0.0.1:8000/admin/api/nodes?token=${ADMIN_TOKEN}"
echo ""

echo "--- Host: Cryptographic Model Hash Verification ---"
curl -s -X POST "http://127.0.0.1:8000/model/verify?token=${ADMIN_TOKEN}"
echo ""

echo "--- Contributor 1: Live Status & Hardware Governor ---"
curl -s "http://127.0.0.1:8765/local/status?token=${CLIENT1_TOKEN}"
echo ""

echo "--- Contributor 1: 'What Leaves My Device' Payload Transparency ---"
curl -s "http://127.0.0.1:8765/local/payload-preview?token=${CLIENT1_TOKEN}"
echo ""

echo "--- Contributor 2: Live Status & Hardware Governor ---"
curl -s "http://127.0.0.1:8766/local/status?token=${CLIENT2_TOKEN}"
echo ""

echo "--- Disk Artifacts ---"
ls -lh "${WORK_DIR}/host/final_model.safetensors"
ls -la "${WORK_DIR}/host/checkpoints"
echo ""

if [ "${KEEP_ALIVE}" = "1" ]; then
    echo "======================================================================"
    echo "KEEP_ALIVE=1 detected. Containers will remain active."
    echo "Explore the dashboards in your browser:"
    echo "  Host:          ${HOST_DASHBOARD_URL}"
    echo "  Contributor 1: ${CLIENT1_DASHBOARD_URL}"
    echo "  Contributor 2: ${CLIENT2_DASHBOARD_URL}"
    echo "When done, stop them with: docker rm -f ${HOST_CONTAINER} ${CLIENT1_CONTAINER} ${CLIENT2_CONTAINER}"
    echo "======================================================================"
else
    echo "Containers are still active for 15 seconds so you can open the dashboards..."
    echo "  Host:          ${HOST_DASHBOARD_URL}"
    echo "  Contributor 1: ${CLIENT1_DASHBOARD_URL}"
    echo "  Contributor 2: ${CLIENT2_DASHBOARD_URL}"
    echo ""
    read -t 15 -p "Press [Enter] to teardown now (or auto-cleans after timeout)... " || true
    echo ""
    echo "Cleaning up containers..."
    docker rm -f "${HOST_CONTAINER}" "${CLIENT1_CONTAINER}" "${CLIENT2_CONTAINER}" >/dev/null 2>&1 || true
    echo "Done!"
fi
