#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# test_ollama_docker.sh — Verify Ollama is reachable from Docker
#
# Tests:
#   1. Ollama is running on the host
#   2. host.docker.internal resolves from inside a Docker container
#   3. Ollama API is reachable from inside Docker
#   4. The required model responds to a test prompt
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="${OLLAMA_MODEL:-qwen2.5-coder:7b-instruct-q4_K_M}"
OLLAMA_HOST_URL="http://localhost:11434"
OLLAMA_DOCKER_URL="http://host.docker.internal:11434"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
PASS="${GREEN}PASS${NC}"
FAIL="${RED}FAIL${NC}"
ERRORS=0

check() {
    local desc="$1"
    shift
    printf "  %-55s ... " "${desc}"
    if "$@" > /dev/null 2>&1; then
        printf "${PASS}\n"
    else
        printf "${FAIL}\n"
        ERRORS=$((ERRORS + 1))
    fi
}

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Ollama ↔ Docker Connectivity Test                  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Test 1: Ollama running on host ────────────────────────────────
echo "[host tests]"
check "Ollama server responding on localhost:11434" \
    curl -sf --max-time 5 "${OLLAMA_HOST_URL}/api/tags"

check "Model '${MODEL}' is available" \
    sh -c "curl -sf --max-time 5 '${OLLAMA_HOST_URL}/api/tags' | grep -q '${MODEL}'"

# ── Test 2: Docker connectivity ───────────────────────────────────
echo ""
echo "[docker tests]"

# Check Docker is available
check "Docker daemon is running" \
    docker info

# Run a test container that tries to reach Ollama via host.docker.internal
check "host.docker.internal resolves from Docker" \
    docker run --rm --add-host=host.docker.internal:host-gateway \
        alpine:3.19 sh -c "nslookup host.docker.internal || getent hosts host.docker.internal"

check "Ollama API reachable from Docker container" \
    docker run --rm --add-host=host.docker.internal:host-gateway \
        alpine:3.19 sh -c "apk add --no-cache curl >/dev/null 2>&1 && curl -sf --max-time 10 '${OLLAMA_DOCKER_URL}/api/tags'"

# ── Test 3: Model inference test ──────────────────────────────────
echo ""
echo "[inference test]"
printf "  %-55s ... " "Model responds to test prompt from Docker"

RESPONSE=$(docker run --rm --add-host=host.docker.internal:host-gateway \
    alpine:3.19 sh -c "
        apk add --no-cache curl >/dev/null 2>&1
        curl -sf --max-time 60 '${OLLAMA_DOCKER_URL}/api/generate' \
            -d '{\"model\": \"${MODEL}\", \"prompt\": \"Say hello in one word\", \"stream\": false}' \
            2>/dev/null
    " 2>/dev/null) || true

if echo "${RESPONSE}" | grep -q '"response"'; then
    printf "${PASS}\n"
    # Extract and show the response
    ANSWER=$(echo "${RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('response','').strip()[:80])" 2>/dev/null || echo "(could not parse)")
    echo "    └─ Model said: \"${ANSWER}\""
else
    printf "${FAIL}\n"
    ERRORS=$((ERRORS + 1))
fi

# ── Summary ───────────────────────────────────────────────────────
echo ""
if [ "${ERRORS}" -gt 0 ]; then
    echo -e "${RED}✗ ${ERRORS} test(s) failed${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Is Ollama running?          ollama serve"
    echo "  2. Is the model pulled?        ollama pull ${MODEL}"
    echo "  3. Is Docker running?          docker info"
    echo "  4. WSL2 networking issue?       Restart Docker Desktop"
    exit 1
else
    echo -e "${GREEN}✓ All tests passed — Ollama is fully reachable from Docker${NC}"
fi
