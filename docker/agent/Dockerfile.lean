# Stage 1: Lean 4 Base Toolchain
FROM python:3.11-slim AS lean-base

RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

ENV ELAN_HOME=/root/.elan
ENV PATH=$ELAN_HOME/bin:$PATH
RUN curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y --default-toolchain stable

# Stage 2: App Builder
FROM lean-base as builder

WORKDIR /app
# Pre-initialize a dummy project if needed to download the toolchain into the layer
RUN mkdir -p /tmp/cache && cd /tmp/cache && elan default stable && lean --version

# Now we are ready for the real project
COPY proofs /app/proofs
RUN cd proofs && lake build

CMD ["lake", "build"]
