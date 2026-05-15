---
name: needle-skill
description: Use when the AI agent needs to convert natural language into structured function call JSON. Triggered by keywords like "function call", "tool invocation", "call tool", "invoke function", or when the user asks to interact with external APIs/tools. Use ONLY when explicit function calling is needed - the user asks you to call a specific function or you need to determine which tool to invoke. The skill provides access to Needle, a 26M parameter model specialized in single-shot function calling that runs on CPU.
---

# Needle Function Calling Bridge

Needle is a 26M parameter function calling model (distilled from Gemini 3.1) that runs locally on CPU. It converts natural language queries into structured JSON function calls.

## Prerequisites

Before first use, run:
```bash
needle-skill setup
```

## Usage

### Start the server (once per session)
```bash
needle-skill serve &
```

### Check server status
```bash
needle-skill status
```

### Call a function
```bash
needle-skill call \
  --query "What's the weather in San Francisco?" \
  --tools '[{"name":"get_weather","description":"Get weather for a location","parameters":{"location":"string"}}]'
```

Returns:
```json
{"ok": true, "result": "[{\"name\":\"get_weather\",\"arguments\":{\"location\":\"San Francisco\"}}]"}
```

### Stop the server
```bash
needle-skill stop
```

## Tools Format

Tools must be provided as a JSON array with the following structure:
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

## Limitations

- 26M parameters - best for single-shot function calls, not complex multi-turn conversations
- Runs on CPU - each call takes 10-30 seconds depending on query length
- No built-in conversation history - each call is stateless
- The model may occasionally miss optional parameters or produce empty results

## Best Practices

1. Keep tool descriptions clear and concise
2. Parameter types should use simple type names (string, number, boolean)
3. If the result is empty (`[]`), the model determined no tool call is needed
4. For batch operations, prepare multiple queries and call sequentially