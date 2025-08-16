# Algo-Trading

Fast crypto trading bot with LLM-powered decision making for crypto (BTC, ETH) on 1-minute bars, accounting in USD, and running in three modes: SIM, PAPER, LIVE.

## 🚀 Features

- **Fast Trading Engine**: Optimized for 1-minute bar processing
- **LLM-Powered Decisions**: Local LLM integration for policy recommendations
- **Multiple Trading Modes**: SIM (simulation), PAPER (paper trading), LIVE (real trading)
- **Comprehensive Risk Management**: Circuit breakers, position limits, drawdown protection
- **Structured Data Pipeline**: Raw data → processed → features → decisions
- **Professional Development**: Poetry, pre-commit hooks, CI/CD, comprehensive testing

## 🏗️ Architecture

```
Algo-Trading/
├── configs/           # Configuration files (YAML)
├── src/               # Source code
│   ├── app/           # Main application
│   ├── adapters/      # External integrations
│   ├── core/          # Core trading engine
│   ├── llm/           # LLM integration
│   └── utils/         # Utilities and helpers
├── data/              # Data storage
├── state/             # System state
├── logs/              # Structured logging
├── reports/           # Performance reports
└── tests/             # Test suite
```

## 🛠️ Technology Stack

- **Python 3.11+**: Modern Python with type hints
- **Poetry**: Dependency management
- **Pydantic**: Data validation and serialization
- **Structlog**: Structured logging
- **Rich**: Beautiful terminal output
- **Click**: CLI framework
- **CCXT**: Cryptocurrency exchange integration
- **Local LLM**: Meta-Llama-3-8B-Instruct for decision making

## 📋 Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)
- Local LLM models (see LLM Setup section)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Algo-Trading
poetry install
```

### 2. Environment Configuration

```bash
cp env.example .env
# Edit .env with your configuration
```

### 3. System Setup

```bash
poetry run python -m src.app.main setup
```

### 4. Validate Configuration

```bash
poetry run python -m src.app.main validate-config
```

### 5. Run Backtesting

```bash
poetry run python -m src.app.main backtest --mode SIM --symbols "BTC/USD,ETH/USD" --duration 1d
```

### 6. Start Live Trading (SIM mode)

```bash
poetry run python -m src.app.main live --mode SIM --symbols "BTC/USD,ETH/USD"
```

## 🔧 Configuration

### Trading Modes

- **SIM**: Full simulation with historical data
- **PAPER**: Real market data with simulated execution
- **LIVE**: Real trading with actual capital

### Configuration Files

- `configs/base.yaml`: Common settings
- `configs/broker.{mode}.yaml`: Broker-specific settings
- `configs/strategy.default.yaml`: Trading strategy parameters
- `configs/risk.default.yaml`: Risk management settings
- `configs/llm.policy.yaml`: LLM configuration

### Environment Variables

Key environment variables in `.env`:

```bash
MODE=SIM                    # Trading mode
BROKER_KEY_ID=...          # Broker API key
BROKER_SECRET=...          # Broker API secret
DATA_CACHE_DIR=...         # Data cache directory
LLM_INSTRUCT_MODEL_PATH=... # LLM model path
```

## 🤖 LLM Setup

### Required Models

1. **Instruction Model**: Meta-Llama-3-8B-Instruct.Q4_0.gguf
   - Path: `/Users/zoe/Library/Application Support/nomic.ai/GPT4All/Meta-Llama-3-8B-Instruct.Q4_0.gguf`

2. **Embeddings Model**: nomic-embed-text-v1.5.f16.gguf
   - Path: `/Applications/gpt4all/bin/gpt4all.app/Contents/Resources/nomic-embed-text-v1.5.f16.gguf`

### LLM Role

The LLM provides policy recommendations based on:
- Market data and features
- Portfolio metrics and risk indicators
- Historical performance and decisions
- Market regime analysis

**Important**: The LLM never executes trades directly - it only provides recommendations.

## 📊 Data Flow

```
Market Data → Feature Engineering → Strategy Analysis → LLM Decision → Risk Check → Execution
     ↓              ↓                    ↓              ↓            ↓          ↓
  Raw Bars    Technical Features   Signal Generation  Policy Recs  Validation  Order Placement
```

## 🛡️ Risk Management

### Circuit Breakers

- **Daily Loss**: Triggers at 3% daily loss
- **Drawdown**: Triggers at 10% portfolio drawdown
- **Volatility**: Triggers at 5% volatility spike
- **Spread Stress**: Triggers at 2% spread widening

### Position Limits

- **Per Symbol**: Maximum 10% of portfolio
- **Gross Exposure**: Maximum 50% of portfolio
- **Cash Reserve**: Minimum 10% cash buffer

## 🧪 Testing

### Run Tests

```bash
# All tests
poetry run pytest

# With coverage
poetry run pytest --cov=src --cov-report=html

# Specific test file
poetry run pytest tests/test_data_models.py
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Property Tests**: Randomized data validation
- **Integration Tests**: End-to-end system testing
- **Parity Tests**: SIM vs PAPER consistency

## 🔍 Code Quality

### Pre-commit Hooks

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run manually
poetry run pre-commit run --all-files
```

### Code Formatting

- **Black**: Code formatting
- **Ruff**: Fast linting
- **isort**: Import sorting
- **mypy**: Type checking

### Security

- **Bandit**: Security linting
- **detect-secrets**: Secret detection

## 📈 Performance Monitoring

### Logs

- `logs/app.log`: Application logs
- `logs/trades.log.jsonl`: Trade execution logs
- `logs/risk.log.jsonl`: Risk management events
- `logs/llm_decisions/`: LLM decision logs

### Reports

- `reports/daily/`: Daily performance reports
- `reports/runs/`: Trading run summaries

## 🚨 Safety Features

- **Kill Switch**: Emergency stop capability
- **Position Monitoring**: Real-time position tracking
- **Circuit Breakers**: Automatic risk mitigation
- **Audit Trail**: Complete decision and execution logging

## 📚 Development

### Project Structure

```
src/
├── app/              # Main application and CLI
├── adapters/         # External system adapters
│   ├── broker/       # Broker integrations
│   └── data/         # Data source adapters
├── core/             # Core trading engine
│   ├── data_models/  # Data structures
│   ├── engine/       # Trading engine
│   ├── features/     # Feature engineering
│   ├── risk/         # Risk management
│   ├── strategy/     # Trading strategies
│   ├── execution/    # Order execution
│   └── reporting/    # Performance reporting
├── llm/              # LLM integration
│   ├── runtime/      # LLM runtime
│   ├── policies/     # Policy generation
│   └── memos/        # Decision memos
└── utils/            # Utilities and helpers
```

### Adding New Features

1. **Data Models**: Extend `src/core/data_models/`
2. **Strategies**: Implement in `src/core/strategy/`
3. **Risk Rules**: Add to `src/core/risk/`
4. **LLM Policies**: Extend `src/llm/policies/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Setup pre-commit hooks
poetry run pre-commit install

# Run quality checks
poetry run pre-commit run --all-files
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk of loss. The authors are not responsible for any financial losses incurred through the use of this software.

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: See `inst.md` for detailed architecture
- **Configuration**: Check `configs/` directory examples

## 🔄 Roadmap

- [ ] Real-time market data streaming
- [ ] Advanced portfolio optimization
- [ ] Multi-exchange support
- [ ] Web dashboard
- [ ] Mobile notifications
- [ ] Advanced backtesting engine
- [ ] Machine learning feature engineering
- [ ] Social trading features
