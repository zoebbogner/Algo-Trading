# Algo-Trading Project Blueprint

This document defines the architecture, file structure, configuration, logging, and workflow for the Algo-Trading Agent.  
The system is built for crypto (BTC, ETH) on 1-minute bars, accounting in USD, and runs in three modes: SIM, PAPER, LIVE.

--------------------------------------------------------------------------------
1. Goals
--------------------------------------------------------------------------------
- Build a fast trading bot for crypto (1m bars).
- Use data + table analytics as the foundation.
- Minimize the gap between demo (SIM/PAPER) and real (LIVE).
- Rely on a local LLM for planning and explanations.
- Apply best practices: Poetry, pre-commit hooks, structured logs, GitHub Actions CI.

--------------------------------------------------------------------------------
2. LLM Setup
--------------------------------------------------------------------------------
Instruction model:
  /Users/zoe/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf

Embeddings model:
  /Applications/gpt4all/bin/gpt4all.app/Contents/Resources/nomic-embed-text-v1.5.f16.gguf

Role of LLM:
- Reads compact tables (metrics, exposure, drawdown, slippage).
- Suggests policy deltas (reduce risk, switch strategy, pause).
- Logs all suggestions to JSONL files (auditable).
- Never executes trades directly.

--------------------------------------------------------------------------------
3. Repository Structure
--------------------------------------------------------------------------------
Algo-Trading/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ .env.example
│
├─ configs/
│  ├─ base.yaml
│  ├─ broker.sim.yaml
│  ├─ broker.paper.yaml
│  ├─ broker.live.yaml
│  ├─ strategy.default.yaml
│  ├─ risk.default.yaml
│  └─ llm.policy.yaml
│
├─ src/
│  ├─ app/              
│  ├─ adapters/         
│  │  ├─ broker/        
│  │  └─ data/          
│  ├─ core/
│  │  ├─ data_models/   
│  │  ├─ engine/        
│  │  ├─ features/      
│  │  ├─ risk/          
│  │  ├─ strategy/      
│  │  ├─ execution/     
│  │  └─ reporting/     
│  ├─ llm/
│  │  ├─ runtime/       
│  │  ├─ policies/      
│  │  └─ memos/         
│  ├─ utils/            
│  └─ validations/      
│
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ features/
│  └─ cache/
│
├─ state/
│  ├─ portfolios/
│  ├─ checkpoints/
│  └─ registry.json
│
├─ logs/
│  ├─ app.log
│  ├─ trades.log.jsonl
│  ├─ risk.log.jsonl
│  └─ llm_decisions/
│
├─ reports/
│  ├─ daily/
│  └─ runs/
│
├─ scripts/
├─ tests/
└─ .github/
   └─ workflows/
      └─ ci.yml

--------------------------------------------------------------------------------
4. Configuration
--------------------------------------------------------------------------------
Environment variables (.env):

MODE=SIM                 # SIM | PAPER | LIVE
BROKER_KEY_ID=...
BROKER_SECRET=...
BROKER_BASE_URL=...

DATA_CACHE_DIR=Algo-Trading/data/cache

LLM_INSTRUCT_MODEL_PATH=/Users/zoe/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf
EMBEDDINGS_MODEL_PATH=/Applications/gpt4all/bin/gpt4all.app/Contents/Resources/nomic-embed-text-v1.5.f16.gguf

TZ=Asia/Jerusalem

YAML configs:
Algo-Trading/configs/base.yaml
Algo-Trading/configs/broker.sim.yaml
Algo-Trading/configs/broker.paper.yaml
Algo-Trading/configs/broker.live.yaml
Algo-Trading/configs/strategy.default.yaml
Algo-Trading/configs/risk.default.yaml
Algo-Trading/configs/llm.policy.yaml

--------------------------------------------------------------------------------
5. Data & Tables
--------------------------------------------------------------------------------
Core tables stored under Algo-Trading/data:

- raw/          : immutable source bars
- processed/    : aligned bars
- features/     : engineered features
- cache/        : request/result caches

Schemas:
- bars              : ts, symbol, open, high, low, close, volume, spread
- features          : indicators, returns, volatility
- orders            : ts, run_id, symbol, side, qty, type, price, reason_code
- fills             : ts, order_id, price, qty, fee, slippage_bps
- positions         : ts, symbol, qty, avg_cost, unrealized_pnl, realized_pnl
- portfolio         : ts, cash, equity, exposure_gross, exposure_net, drawdown
- metrics_window    : window, sharpe, sortino, turnover, hit_rate
- risk_events       : ts, type, details, action_taken
- llm_policy_decisions : all LLM memos

--------------------------------------------------------------------------------
6. Execution & Risk
--------------------------------------------------------------------------------
- Trades on the next bar (no look-ahead).
- Market orders (default) or limit+timeout.
- Slippage = half-spread + impact function.

Risk guardrails:
- Per-trade stop/time stop.
- Max position per symbol.
- Max gross exposure.
- Daily MDD cut (flatten, cooldown).
- Circuit breaker (spread stress).

--------------------------------------------------------------------------------
7. Logging
--------------------------------------------------------------------------------
Console:
- INFO: mode, bars, orders, positions.
- WARN: data gaps, slippage, near limits.
- ERROR: rejects, adapter failures.

File logs (Algo-Trading/logs):
- app.log              : rotating console+file log
- trades.log.jsonl     : fills, pnl deltas
- risk.log.jsonl       : breaches, circuit events
- llm_decisions/*.jsonl: LLM memos

--------------------------------------------------------------------------------
8. Testing
--------------------------------------------------------------------------------
Tests in Algo-Trading/tests:

- Unit: bar alignment, pnl accounting, risk.
- Property: randomized gaps, monotonic prices.
- Integration: 7-day SIM replay, reproducible pnl.
- Parity: SIM vs PAPER drift < threshold.

--------------------------------------------------------------------------------
9. CI & Pre-Commit
--------------------------------------------------------------------------------
Poetry project files:
- Algo-Trading/pyproject.toml
- Algo-Trading/poetry.lock

Pre-commit config:
- Algo-Trading/.pre-commit-config.yaml

CI workflow:
- Algo-Trading/.github/workflows/ci.yml

Hooks:
- ruff, black, isort, mypy, bandit, detect-secrets.

CI:
- lint, type check, tests (SIM), reports, coverage.

--------------------------------------------------------------------------------
10. Runbook
--------------------------------------------------------------------------------
Bootstrap:
1. Clone repo, install Poetry.
2. Copy .env.example → .env.
3. Run in MODE=SIM: backfill → simulate → verify logs.
4. Enable LLM runtime: check memos in Algo-Trading/logs/llm_decisions/.
5. Switch to MODE=PAPER for testnet.

Go-Live:
- Update broker config to LIVE.
- Keep low risk caps.
- Enable kill switch.
- Monitor logs + reports in Algo-Trading/reports/.

--------------------------------------------------------------------------------
11. Defaults
--------------------------------------------------------------------------------
- Market: BTC, ETH.
- Bars: 1m.
- Risk: per-trade 0.5%, max pos 10%, gross 50%, daily MDD 5% (60m cooldown).
- Logs retention: 14 days.
- Alerts: disabled (ready in config).




### commit all the steps, like each change/ add to the agent