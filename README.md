# Needle-Skill

A **26M parameter function calling bridge** for AI agents. Built on [Needle](https://github.com/cactus-compute/needle), it runs entirely on CPU with no GPU required.

Needle-Skill wraps Needle into a clean HTTP server + CLI, exposing a single `/v1/call` endpoint. AI agents (like opencode) load the bundled skill definition and invoke `needle-skill call` whenever natural language needs to be converted into structured function call JSON.

```
"查詢台北天氣"  ──▶  ./needle-skill call  ──▶  {"name":"get_weather","arguments":{"location":"Taipei"}}
"寄信給 john"     ──▶  ./needle-skill call  ──▶  {"name":"send_email","arguments":{"to":"john@example.com"}}
```

---

## Install

```bash
# 1. Install the package (from GitHub)
pip install git+https://github.com/raybird/needle-skill.git

# 2. One-time setup — installs Needle, downloads the 51MB checkpoint
needle-skill setup

# 3. Register as an opencode skill
needle-skill skill install /path/to/your/workspace
```

After step 3, restart opencode. The skill will appear in `available_skills`.

---

## Usage

### Start the server

```bash
needle-skill serve &
```

The server starts at `http://127.0.0.1:3918`. On first startup it loads the 26M parameter model (~1 second).

### Call from CLI

```bash
needle-skill call \
  --query "What's the weather in San Francisco?" \
  --tools '[{"name":"get_weather","description":"Get weather for a location","parameters":{"location":"string"}}]'
```

Output:
```json
{"ok": true, "result": "[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"San Francisco\"}}]"}
```

### Call from HTTP / curl

```bash
curl -s -X POST http://127.0.0.1:3918/v1/call \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Calculate 15 times 4",
    "tools": "[{\"name\":\"calculate\",\"description\":\"Perform arithmetic\",\"parameters\":{\"expression\":\"string\"}}]"
  }'
```

### Other commands

```bash
needle-skill health              # Check server status (JSON)
needle-skill status              # Check if server is running
needle-skill stop                # Gracefully stop the server
```

---

## API Reference

### POST `/v1/call`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | Natural language query |
| `tools` | string | yes | JSON-encoded array of tool definitions |
| `max_gen_len` | integer | no | Max output tokens (default: 512) |
| `constrained` | boolean | no | Enable grammar-constrained decoding (default: false) |
| `seed` | integer | no | Random seed (default: 0) |

Response:
```json
{
  "ok": true,
  "result": "[{\"name\":\"tool_name\",\"arguments\":{...}}]"
}
```

### GET `/v1/health`

```json
{
  "ok": true,
  "model": "needle.pkl",
  "params": 26315421,
  "uptime": 1234,
  "requests": 42
}
```

---

## Tools Format

Needs Needle-compatible tool definitions (not OpenAI format):

```json
[
  {
    "name": "tool_name",
    "description": "What the tool does",
    "parameters": {
      "param1": "type",
      "param2": "type"
    }
  }
]
```

Parameter types should use simple type names: `string`, `number`, `boolean`.

---

## Architecture

```
┌─────────────┐     HTTP / CLI      ┌──────────────────┐     JAX      ┌──────────┐
│  AI Agent   │ ──────────────────▶ │  needle-skill    │ ──────────▶ │  Needle  │
│  (opencode) │ ◀────────────────── │  server:3918     │ ◀────────── │  26M CPU  │
└─────────────┘    JSON response    └──────────────────┘             └──────────┘
```

- **needle_skill/engine.py** — Model lifecycle (load / generate / unload), lazy-imports Needle
- **needle_skill/server.py** — stdlib `http.server` backed HTTP API
- **needle_skill/cli.py** — Command-line interface (7 sub-commands)
- **needle_skill/config.py** — `~/.needle/` directory management

---

## `~/.needle/` Layout

```
~/.needle/
├── config.yaml          # Server & model configuration
├── cache/
│   └── needle.pkl       # 51MB model checkpoint (auto-downloaded)
├── logs/
└── server.pid
```

Edit `~/.needle/config.yaml` to change default port or checkpoint path.

---

## Limitations

- **26M parameters** — Best for single-shot function calling. Not suitable for multi-turn conversation or open-ended reasoning.
- **CPU-only** — Each call takes 10-30 seconds depending on query and tools length.
- **Stateless** — No conversation history. Each call is independent.
- **May miss optional arguments** — The model may sometimes omit non-required parameters.
- **Non-ASCII / Multilingual Issues** — Tiny model parameter size and English-dominated corpus training cause unicode escape sequence spelling hallucinations when given non-ASCII (e.g., Chinese) queries. See [RESEARCH.md](./RESEARCH.md) for details and pre-processing recommendations.

---

## Dependencies

- Python >= 3.11
- [Needle](https://github.com/cactus-compute/needle) (auto-installed by `needle-skill setup`)
  - JAX, Flax, SentencePiece, HuggingFace Hub
- pyyaml (lightweight config)

---

## License

MIT

---

## Credits

Built on Needle by [Cactus Compute](https://github.com/cactus-compute/needle).