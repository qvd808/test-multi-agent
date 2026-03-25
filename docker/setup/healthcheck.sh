#!/bin/sh
set -e

PASS="\033[32mPASS\033[0m"
FAIL="\033[31mFAIL\033[0m"
WARN_C="\033[33mWARN\033[0m"
ERRORS=0

check() {
    DESC="$1"
    shift
    printf "[setup] %-50s ... " "$DESC"
    if "$@" > /dev/null 2>&1; then
        printf "$PASS\n"
    else
        printf "$FAIL\n"
        ERRORS=$((ERRORS + 1))
    fi
}

warn_check() {
    DESC="$1"
    shift
    printf "[setup] %-50s ... " "$DESC"
    if "$@" > /dev/null 2>&1; then
        printf "$PASS\n"
    else
        printf "$WARN_C (non-fatal)\n"
    fi
}

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  RL Trading Agent — Stack Health Check              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Check 1: Ollama reachable ──────────────────────────────────
OLLAMA_URL="${OLLAMA_BASE_URL:-http://host.docker.internal:11434}"
check "checking Ollama at ${OLLAMA_URL}" \
    curl -sf --max-time 10 "${OLLAMA_URL}/api/tags"

# ── Check 2: Correct model available ──────────────────────────
MODEL="${OLLAMA_MODEL:-qwen2.5-coder:7b-instruct-q4_K_M}"
check "checking model '${MODEL}' is pulled" \
    sh -c "curl -sf --max-time 10 '${OLLAMA_URL}/api/tags' | jq -e '.models[] | select(.name | contains(\"${MODEL}\"))'"

# ── Check 3: SQLite volume writable ───────────────────────────
check "checking SQLite volume writable" \
    sh -c "touch /data/db/test_write && rm /data/db/test_write"

# ── Check 4: GitHub token (non-fatal) ─────────────────────────
warn_check "checking GitHub token" \
    sh -c "[ -n '${GITHUB_TOKEN}' ] && [ '${GITHUB_TOKEN}' != 'your_personal_access_token' ] && curl -sf --max-time 10 -H 'Authorization: token ${GITHUB_TOKEN}' https://api.github.com/user"

echo ""

if [ "$ERRORS" -gt 0 ]; then
    echo "[setup] ✗ ${ERRORS} check(s) failed — agent will NOT start"
    echo "[setup] Fix the issue(s) above and re-run: docker compose up"
    exit 1
fi

echo "[setup] ✓ all checks passed — starting agent"
touch /tmp/setup-done
# Keep container alive briefly so healthcheck can read the done marker
sleep 5

