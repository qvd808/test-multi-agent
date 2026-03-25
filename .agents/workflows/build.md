---
description: Build the RL Trading Agent project — scaffold, validate, and commit with jj
---

// turbo-all

# Build RL Trading Agent

This workflow scaffolds, validates, and commits the entire RL Trading Agent project.

## Prerequisites

1. Ensure jj and Docker are available:
```bash
jj --version && docker --version
```

## Step 1 — Initialize jj repo (if not already initialized)

```bash
cd /root/Programming/MyProjects/multi-agents-project && jj status || jj git init
```

## Step 2 — Install Ollama (if not installed)

```bash
cd /root/Programming/MyProjects/multi-agents-project && bash scripts/install_ollama.sh
```

## Step 3 — Validate Python syntax on all .py files

```bash
cd /root/Programming/MyProjects/multi-agents-project && find . -name "*.py" -exec python3 -m py_compile {} \; && echo "All Python files OK"
```

## Step 4 — Validate docker-compose.yml

```bash
cd /root/Programming/MyProjects/multi-agents-project && docker compose config --quiet && echo "docker-compose.yml OK"
```

## Step 5 — Test Ollama ↔ Docker connectivity

```bash
cd /root/Programming/MyProjects/multi-agents-project && bash scripts/test_ollama_docker.sh
```

## Step 6 — Build Docker images

```bash
cd /root/Programming/MyProjects/multi-agents-project && docker compose build
```

## Step 7 — Commit with jj

```bash
cd /root/Programming/MyProjects/multi-agents-project && jj describe -m "Initial scaffold: Docker stack, agents, tools, env, metrics"
```

## Step 8 — Verify final state

```bash
cd /root/Programming/MyProjects/multi-agents-project && jj log --limit 5
```
