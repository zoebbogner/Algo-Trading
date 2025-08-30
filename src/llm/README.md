# LLM Module for Algo-Trading System

This module provides a **pluggable interface** for Large Language Models to act as trading orchestrators and advisors. The LLM is designed to be an **intelligent assistant** that analyzes market data and provides structured recommendations, while keeping all trading execution deterministic and safe.

## 🎯 **Design Philosophy**

### **What the LLM SHOULD Do:**
- **Orchestrate** data analysis and choose which analytics to run
- **Summarize** market conditions and label regimes
- **Generate hypotheses** about market behavior
- **Provide structured recommendations** in JSON format
- **Explain backtest results** and performance metrics
- **Create checklists** and policies for risk management

### **What the LLM MUST NOT Do:**
- **Predict exact prices** or price movements
- **Decide trade fills** or execution timing
- **Override risk limits** or position sizing rules
- **Place real orders** or execute trades
- **Make decisions** about stop-loss or take-profit levels

## 🏗️ **Architecture**

```
src/llm/
├── contracts/           # JSON schemas for validation
├── prompts/            # System and task templates
├── policy/             # Decision policies and guardrails
├── backends/           # LLM provider implementations
├── runtime/            # Core runtime components
└── README.md           # This file
```

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
pip install gpt4all jsonschema
```

### **2. Set Environment Variables**
```bash
# Copy and modify the example
cp env.example .env

# Set your local Llama model path
export LLM_INSTRUCT_MODEL_PATH="/path/to/your/model.gguf"
```

### **3. Basic Usage**
```python
from src.llm import get_llm_client

# Get LLM client
client = get_llm_client(run_id="my_trading_session")

# Generate trading recommendation
result = client.generate(
    "Analyze BTCUSDT on 1m timeframe and provide a trading recommendation",
    json_mode=True,
    system="You are a trading advisor. Provide JSON recommendations only."
)

print(result.text)
```

## 🔧 **Configuration**

### **LLM Configuration File (`configs/llm.yaml`)**
```yaml
backend: "local_llama"
model: "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
temperature: 0.2
max_tokens: 512
json_mode: true
timeout_s: 30
```

### **Environment Variables**
```bash
# Backend selection
LLM_BACKEND=local_llama

# Local model path
LLM_INSTRUCT_MODEL_PATH=/path/to/model.gguf

# Generation parameters
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=512
LLM_TIMEOUT_S=30
```

## 🎭 **Available Backends**

### **1. Local Llama (Default)**
- **Provider**: GPT4All
- **Model**: Any GGUF format Llama model
- **Use Case**: Privacy, offline operation, cost-effective
- **Requirements**: `LLM_INSTRUCT_MODEL_PATH` environment variable

### **2. OpenAI-like APIs (Planned)**
- **Providers**: OpenAI, Together.ai, local Ollama
- **Use Case**: Higher quality, faster responses
- **Requirements**: API key and endpoint configuration

### **3. AWS Bedrock (Planned)**
- **Provider**: Amazon Web Services
- **Use Case**: Enterprise integration, managed models
- **Requirements**: AWS credentials and region

## 📊 **Output Schema**

The LLM must output structured JSON following this schema:

```json
{
  "version": "1.0.0",
  "intent": "propose_trade",
  "symbol": "BTCUSDT",
  "timeframe": "1m",
  "rationale": "Strong momentum with volume confirmation",
  "action": {
    "type": "enter_long",
    "confidence": 0.75,
    "constraints": {
      "max_position_size": 0.03,
      "stop_loss_pct": 0.02,
      "take_profit_pct": 0.06
    }
  }
}
```

## 🛡️ **Safety & Validation**

### **Policy Guardrails**
- **SIMULATION ONLY**: No real trading allowed
- **Risk Limits**: Maximum position size enforced
- **Schema Validation**: All outputs validated against JSON schema
- **Policy Gates**: Actions must pass policy validation

### **Error Handling**
- **Retry Logic**: Automatic retry with exponential backoff
- **Timeout Protection**: Configurable timeouts prevent hanging
- **Fallback Responses**: Graceful degradation on errors
- **Circuit Breakers**: Automatic fallback on repeated failures

## 📈 **Performance & Monitoring**

### **Metrics Tracked**
- **Latency**: Response time in milliseconds
- **Token Usage**: Input/output token counts
- **Success Rate**: Percentage of successful generations
- **Error Types**: Classification of failures

### **Audit Logging**
- **Structured Logs**: JSON format for easy parsing
- **Prompt/Response**: Full conversation history
- **Performance Data**: Latency and token usage
- **Policy Decisions**: Allow/deny/review outcomes

## 🧪 **Testing**

### **Run the Test Suite**
```bash
python3 src/cli/test_llm.py
```

### **Test Coverage**
- Backend initialization
- Text generation
- JSON mode validation
- Health checks
- Error handling
- Performance metrics

## 🔄 **Adding New Backends**

### **1. Implement the Interface**
```python
from .base import LLMClient

class MyNewBackend(LLMClient):
    def generate(self, prompt, **kwargs):
        # Your implementation here
        pass
    
    def healthcheck(self):
        # Health check implementation
        pass
```

### **2. Register the Backend**
```python
# In src/llm/runtime/loader.py
SUPPORTED_BACKENDS = {
    'local_llama': LocalLlamaClient,
    'my_new_backend': MyNewBackend,  # Add here
}
```

### **3. Add Configuration**
```yaml
# In configs/llm.yaml
backend: "my_new_backend"
my_new_backend:
  api_key: "${MY_API_KEY}"
  endpoint: "https://api.myprovider.com"
```

## 🚨 **Troubleshooting**

### **Common Issues**

#### **1. Model Not Found**
```bash
Error: Model file not found
```
**Solution**: Check `LLM_INSTRUCT_MODEL_PATH` environment variable

#### **2. GPT4All Not Installed**
```bash
Error: GPT4All not installed
```
**Solution**: `pip install gpt4all`

#### **3. JSON Parsing Errors**
```bash
Error: Invalid JSON in json_mode
```
**Solution**: Check model output format, adjust temperature

#### **4. Slow Responses**
```bash
Warning: Response time > 10 seconds
```
**Solution**: Reduce `max_tokens`, check model size, consider GPU

### **Debug Mode**
```bash
export LOG_LEVEL=DEBUG
python3 src/cli/test_llm.py
```

## 📚 **Advanced Usage**

### **Custom Prompts**
```python
# Load system prompt
with open("src/llm/prompts/system_orchestrator.txt", "r") as f:
    system_prompt = f.read()

# Use in generation
result = client.generate(
    "Analyze market conditions",
    system=system_prompt,
    json_mode=True
)
```

### **Streaming Responses**
```python
# Get streaming response
chunks = client.chat(
    messages=[{"role": "user", "content": "Analyze BTCUSDT"}],
    stream=True
)

for chunk in chunks:
    print(chunk.delta, end="")
    if chunk.done:
        break
```

### **Batch Processing**
```python
# Process multiple symbols
symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
results = []

for symbol in symbols:
    result = client.generate(
        f"Analyze {symbol} and provide trading recommendation",
        json_mode=True
    )
    results.append(result)
```

## 🔮 **Future Enhancements**

### **Planned Features**
- **RAG Integration**: Retrieve relevant market data and documentation
- **Multi-Modal Support**: Analyze charts and technical indicators
- **Fine-tuning**: Custom training on trading-specific data
- **Ensemble Models**: Combine multiple LLM outputs
- **Real-time Streaming**: Live market analysis and alerts

### **Performance Optimizations**
- **Model Quantization**: Reduce memory usage and improve speed
- **GPU Acceleration**: CUDA support for faster inference
- **Caching**: Intelligent caching of common queries
- **Load Balancing**: Multiple model instances for high availability

## 📄 **License & Compliance**

- **Open Source**: MIT License
- **Privacy**: Local models keep data on your machine
- **Compliance**: Follows financial trading regulations
- **Audit**: Full audit trail for compliance requirements

## 🤝 **Contributing**

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes
4. **Add** tests and documentation
5. **Submit** a pull request

## 📞 **Support**

- **Issues**: GitHub Issues
- **Documentation**: This README and inline code docs
- **Examples**: `src/cli/test_llm.py`
- **Community**: Project discussions and forums

---

**Remember**: The LLM is your intelligent trading advisor, but the trading engine handles execution. Keep it safe, structured, and deterministic! 🚀
