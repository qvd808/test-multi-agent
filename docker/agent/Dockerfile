FROM python:3.11-slim

# System dependencies for opam + Rocq
RUN apt-get update && apt-get install -y --no-install-recommends \
    opam \
    m4 \
    bubblewrap \
    gcc \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Initialize opam and install Rocq (Coq)
RUN opam init --disable-sandboxing --yes && \
    eval $(opam env) && \
    opam install coq --yes

# Set up opam environment for all future RUN/CMD
ENV PATH="/root/.opam/default/bin:${PATH}"

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY state.py graph.py main.py ./
COPY agents/ ./agents/
COPY tools/ ./tools/
COPY env/ ./env/
COPY metrics/ ./metrics/
COPY db/ ./db/
# Ensure Python output is not buffered (real-time docker logs)
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "main.py"]
