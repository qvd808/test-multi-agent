FROM python:3.11-slim

# Minimal dependencies for RL execution
RUN pip install --no-cache-dir \
    gymnasium>=0.29.0 \
    numpy>=1.26.0 \
    pandas>=2.1.0 \
    yfinance>=0.2.30 \
    torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app

# Output directory is mounted read-only at runtime
# Generated RL code runs here in isolation
CMD ["python", "-u", "output/run_agent.py"]
