#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────
# install_ollama.sh — Install Ollama and pull the required model
#
# Works on Linux (x86_64, aarch64) and macOS.
# Designed to be idempotent — safe to run multiple times.
# ──────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="${OLLAMA_MODEL:-qwen2.5-coder:7b-instruct-q4_K_M}"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[ollama-install]${NC} $*"; }
warn()  { echo -e "${YELLOW}[ollama-install]${NC} $*"; }
error() { echo -e "${RED}[ollama-install]${NC} $*" >&2; }

# ── Detect OS ─────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
info "Detected: OS=${OS}, ARCH=${ARCH}"

# ── Check if Ollama is already installed ──────────────────────────
if command -v ollama &>/dev/null; then
    INSTALLED_VERSION="$(ollama --version 2>/dev/null || echo 'unknown')"
    warn "Ollama is already installed (${INSTALLED_VERSION})"
    warn "Skipping installation — will ensure model is pulled"
else
    info "Ollama not found — installing..."

    case "${OS}" in
        Linux)
            info "Installing via official install script..."
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        Darwin)
            if command -v brew &>/dev/null; then
                info "Installing via Homebrew..."
                brew install ollama
            else
                error "macOS detected but Homebrew not found."
                error "Install Homebrew first: https://brew.sh"
                error "Or install Ollama manually: https://ollama.com/download"
                exit 1
            fi
            ;;
        *)
            error "Unsupported OS: ${OS}"
            error "Install Ollama manually: https://ollama.com/download"
            exit 1
            ;;
    esac

    # Verify installation
    if ! command -v ollama &>/dev/null; then
        error "Installation completed but 'ollama' command not found in PATH"
        error "You may need to restart your shell or add Ollama to your PATH"
        exit 1
    fi

    info "Ollama installed successfully: $(ollama --version 2>/dev/null)"
fi

# ── Ensure Ollama server is running ───────────────────────────────
if curl -sf --max-time 3 http://localhost:11434/api/tags &>/dev/null; then
    info "Ollama server is already running"
else
    info "Starting Ollama server in background..."
    nohup ollama serve &>/dev/null &
    OLLAMA_PID=$!

    # Wait for server to be ready (up to 30 seconds)
    for i in $(seq 1 30); do
        if curl -sf --max-time 2 http://localhost:11434/api/tags &>/dev/null; then
            info "Ollama server started (PID: ${OLLAMA_PID})"
            break
        fi
        if [ "$i" -eq 30 ]; then
            error "Ollama server failed to start within 30 seconds"
            exit 1
        fi
        sleep 1
    done
fi

# ── Pull the model ────────────────────────────────────────────────
info "Ensuring model '${MODEL}' is available..."

# Check if model is already pulled
if ollama list 2>/dev/null | grep -q "${MODEL}"; then
    info "Model '${MODEL}' is already downloaded"
else
    info "Pulling model '${MODEL}' (~4.5GB download, this may take a while)..."
    ollama pull "${MODEL}"
    info "Model '${MODEL}' pulled successfully"
fi

# ── Final verification ────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Ollama Setup Complete                              ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
info "Version:  $(ollama --version 2>/dev/null)"
info "Server:   http://localhost:11434"
info "Model:    ${MODEL}"
info ""
info "To test Docker connectivity, run:"
info "  bash scripts/test_ollama_docker.sh"
