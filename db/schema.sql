-- SQLite schema for the RL Trading Agent pipeline
-- Auto-created on first run by main.py

CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL,
    phase           TEXT NOT NULL DEFAULT 'unknown',
    proof_passed    INTEGER NOT NULL DEFAULT 0,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    total_return    REAL,
    sharpe_ratio    REAL,
    max_drawdown    REAL,
    win_rate        REAL,
    github_pushed   INTEGER NOT NULL DEFAULT 0,
    github_commit_url TEXT DEFAULT '',
    error           TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS agent_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER REFERENCES runs(id),
    timestamp   TEXT NOT NULL,
    agent       TEXT NOT NULL,        -- 'researcher', 'coder', 'verifier'
    action      TEXT NOT NULL,        -- what the agent did
    detail      TEXT DEFAULT '',      -- additional context
    duration_ms INTEGER               -- wall clock time for this action
);

CREATE TABLE IF NOT EXISTS metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER REFERENCES runs(id),
    timestamp       TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    total_return    REAL,
    sharpe_ratio    REAL,
    max_drawdown    REAL,
    win_rate        REAL,
    cumulative_reward REAL,
    n_episodes      INTEGER,
    n_trades        INTEGER
);
