# Stage 1: Lean 4 Base Toolchain
FROM python:3.11-slim AS lean-base

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libgmp-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

ENV ELAN_HOME=/root/.elan
ENV PATH=$ELAN_HOME/bin:$PATH
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain stable

# Stage 2: App Builder
FROM lean-base AS builder

WORKDIR /app
RUN elan default stable

# Copy lake files first
COPY proofs/lakefile.toml proofs/lean-toolchain proofs/lake-manifest.json /app/proofs/
WORKDIR /app/proofs

# Download dependencies
RUN lake update

# ↓↓↓ THIS IS THE KEY: Download prebuilt Mathlib instead of compiling it ↓↓↓
RUN lake exe cache get

# Copy source code
COPY proofs /app/proofs

# Build only your project (now fast because Mathlib is cached)
RUN lake build

CMD ["sh", "-c", "cp /app/proofs/.lake/build/bin/margin_proofs /export/margin_proofs && echo '✓ Verified Binary Exported'"]