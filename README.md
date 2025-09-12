---
title: LibreChat Code Interpreter
emoji: 🐍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
private: true
---
 
# LibreChat Code Interpreter
 
A secure code execution environment for LibreChat using MCP (Model Context Protocol). Based on the graet work by Erick Santana https://ersantana.com/tools/librechat/implementing_custom_code_interpreter_hf
 
## Features
 
- Execute Python, JavaScript, and Bash code safely
- Matplotlib visualization support
- Session management for persistent context
- Security restrictions and sandboxing
- SSE integration with LibreChat
 
## Security
 
This space implements multiple security measures:
- Code analysis for dangerous patterns
- Execution timeouts (30 seconds)
- Output length limits
- Blocked dangerous imports and commands
- Isolated execution environment
 
## Usage
 
This space is designed to work with LibreChat's MCP integration. Configure your LibreChat instance to connect via SSE.

```
version: 1.2.4
 
# Your existing configuration...
 
mcpServers:
  code_interpreter:
    type: sse
    url: https://YOUR_USERNAME-librechat-code-interpreter.hf.space/gradio_api/mcp/sse
    headers:
      Authorization: "Bearer ${HF_TOKEN}"
      Content-Type: "application/json"
    serverInstructions: |
      LibreChat Code Interpreter Instructions:
      
      Available Functions:
      - execute_python_code(code, session_id=None): Execute Python code with matplotlib support
      - execute_javascript_code(code, session_id=None): Execute JavaScript with Node.js
      - execute_bash_command(command, session_id=None): Execute safe bash commands
      - create_execution_session(): Create persistent session for stateful execution
      - get_execution_history(session_id): View execution history
      - get_system_info(): Get available packages and system information
      
      Security Features:
      - 30-second execution timeout
      - Blocked dangerous imports/commands
      - Output length limits (10KB)
      - Isolated execution environment
      
      Visualization Support:
      - Matplotlib plots automatically captured as base64 images
      - Use plt.show() to generate visualizations
      - Supports numpy, pandas for data processing
      
      Session Management:
      - Create sessions for persistent variable state
      - Sessions auto-expire after 1 hour
      - Maximum 10 concurrent sessions per space
      
      Usage Tips:
      1. Always use create_execution_session() for multi-step code execution
      2. Sessions maintain variable state between executions
      3. Use get_system_info() to check available packages
      4. Matplotlib plots are automatically captured and returned as base64
```
