"""Researcher agent — gathers context on RL trading strategies via Tavily search.

This agent runs first in the pipeline. It queries Tavily for relevant papers,
code examples, and strategy descriptions. The aggregated context is passed to
the coder agent so it can generate an informed RL trading agent.
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama
from tavily import TavilyClient

from state import AgentState


SYSTEM_PROMPT = """You are a research assistant specialising in reinforcement learning
for stock trading. Your job is to gather the most relevant, actionable context that
a code-generation agent will use to write a paper trading RL agent.

Focus on:
- Which RL algorithms work well for trading (DQN, PPO, A2C, etc.)
- State representation (OHLCV features, technical indicators)
- Reward shaping for trading (risk-adjusted returns, Sharpe-based rewards)
- Position sizing and risk management constraints
- Common pitfalls and best practices

Return a structured markdown summary of your findings. Be concise but complete."""


def researcher_node(state: AgentState) -> AgentState:
    """Search for RL trading context and summarise findings."""
    print("[researcher] Starting research phase")

    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    model_name = os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b-instruct-q4_K_M")

    # Step 1: Search for context via Tavily
    query = "reinforcement learning paper trading agent gymnasium Python DQN"
    context_chunks: list[str] = []

    if tavily_key:
        try:
            client = TavilyClient(api_key=tavily_key)
            results = client.search(query, max_results=5, search_depth="advanced")
            for r in results.get("results", []):
                context_chunks.append(f"### {r['title']}\n{r['content']}\nSource: {r['url']}")
            print(f"[researcher] Got {len(context_chunks)} search results")
        except Exception as e:
            print(f"[researcher] Tavily search failed: {e}")
            context_chunks.append("(Search unavailable — using LLM knowledge only)")
    else:
        print("[researcher] No TAVILY_API_KEY — using LLM knowledge only")
        context_chunks.append("(No search API key — using LLM knowledge only)")

    raw_context = "\n\n".join(context_chunks)

    # Step 2: Ask the LLM to synthesise a structured research summary
    llm = ChatOllama(
        model=model_name,
        base_url=ollama_url,
        temperature=0.3,
    )

    prompt = (
        f"Here are search results about RL trading agents:\n\n{raw_context}\n\n"
        "Synthesise these into a concise, structured research brief that a code "
        "generator can use to build a Gymnasium-based paper trading RL agent in Python. "
        "Include recommended algorithm, state features, reward function design, "
        "and risk management constraints."
    )

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])

    research_context = response.content
    print(f"[researcher] Research summary: {len(research_context)} chars")

    return {
        **state,
        "research_query": query,
        "research_context": research_context,
        "current_phase": "code",
    }
