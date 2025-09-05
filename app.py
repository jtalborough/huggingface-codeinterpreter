#!/usr/bin/env python3
"""
LibreChat Custom Code Interpreter - HuggingFace Space
A secure code execution server using Gradio + MCP with SSE support
"""
 
import json
import subprocess
import tempfile
import os
import sys
import uuid
import time
import base64
import io
from pathlib import Path
from typing import Dict, Any, Optional, List
import gradio as gr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
class SecureCodeExecutor:
    def __init__(self):
        self.sessions = {}
        self.max_execution_time = 30
        self.max_output_length = 10000
        self.allowed_languages = ["python", "javascript", "bash"]
        
        # Security: List of blocked commands/imports
        self.blocked_imports = [
            'subprocess', 'os', 'sys', 'shutil', 'glob', 'pickle', 
            'marshal', 'imp', 'importlib', '__import__'
        ]
        self.blocked_bash_commands = [
            'rm', 'sudo', 'chmod', 'chown', 'dd', 'mkfs', 'fdisk',
            'curl', 'wget', 'ssh', 'scp', 'nc', 'netcat'
        ]
 
    def create_session(self) -> str:
        """Create a new execution session"""
        session_id = str(uuid.uuid4())[:8]  # Shorter ID for HF
        self.sessions[session_id] = {
            'created_at': time.time(),
            'variables': {},
            'history': [],
            'files': {}
        }
        return session_id
 
    def cleanup_old_sessions(self):
        """Remove sessions older than 1 hour"""
        current_time = time.time()
        old_sessions = [
            sid for sid, session in self.sessions.items()
            if current_time - session['created_at'] > 3600
        ]
        for sid in old_sessions:
            del self.sessions[sid]
 
    def is_code_safe(self, code: str, language: str) -> tuple[bool, str]:
        """Check if code is safe to execute"""
        if language == "python":
            # Check for blocked imports
            for blocked in self.blocked_imports:
                if blocked in code:
                    return False, f"Blocked import/function: {blocked}"
            
            # Check for dangerous patterns
            dangerous_patterns = ['exec(', 'eval(', 'open(', 'file(', '__']
            for pattern in dangerous_patterns:
                if pattern in code:
                    return False, f"Dangerous pattern detected: {pattern}"
        
        elif language == "bash":
            # Check for blocked commands
            for blocked in self.blocked_bash_commands:
                if blocked in code.lower():
                    return False, f"Blocked command: {blocked}"
        
        return True, ""
 
    def execute_python_code(self, code: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute Python code with visualization support"""
        # Security check
        is_safe, reason = self.is_code_safe(code, "python")
        if not is_safe:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Security violation: {reason}",
                "execution_time": time.time()
            }
 
        # Prepare execution environment
        setup_code = '''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import math
import random
import base64
import io
from datetime import datetime, timedelta
 
# Custom print function to capture output
_output_buffer = []
_original_print = print
def print(*args, **kwargs):
    _output_buffer.append(' '.join(str(arg) for arg in args))
 
# Function to save plots as base64
def save_current_plot():
    if plt.get_fignums():  # Check if there are any figures
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        return base64.b64encode(plot_data).decode()
    return None
'''
 
        # Combine setup and user code
        full_code = setup_code + "\n" + code + "\n"
        
        # Add plot capture if plotting commands detected
        if any(cmd in code for cmd in ['plt.', 'plot(', 'scatter(', 'bar(', 'hist(']):
            full_code += "\n_plot_data = save_current_plot()\nif _plot_data: _output_buffer.append('PLOT_DATA:' + _plot_data)\n"
 
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(full_code)
                temp_file = f.name
 
            # Execute with timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,
                cwd=tempfile.gettempdir()
            )
 
            # Process output
            stdout = result.stdout
            stderr = result.stderr
            plot_data = None
 
            # Extract plot data if present
            if 'PLOT_DATA:' in stdout:
                lines = stdout.split('\n')
                clean_lines = []
                for line in lines:
                    if line.startswith('PLOT_DATA:'):
                        plot_data = line.replace('PLOT_DATA:', '')
                    else:
                        clean_lines.append(line)
                stdout = '\n'.join(clean_lines)
 
            # Limit output length
            if len(stdout) > self.max_output_length:
                stdout = stdout[:self.max_output_length] + "\n... (output truncated)"
 
            execution_result = {
                "success": result.returncode == 0,
                "stdout": stdout.strip(),
                "stderr": stderr.strip() if stderr else "",
                "execution_time": time.time(),
                "return_code": result.returncode
            }
 
            if plot_data:
                execution_result["plot"] = plot_data
 
            return execution_result
 
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Execution timed out (30s limit)",
                "execution_time": time.time()
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "execution_time": time.time()
            }
        finally:
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
                    pass
 
    def execute_javascript_code(self, code: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute JavaScript code using Node.js"""
        # Security check
        is_safe, reason = self.is_code_safe(code, "javascript")
        if not is_safe:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Security violation: {reason}",
                "execution_time": time.time()
            }
 
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                temp_file = f.name
 
            result = subprocess.run(
                ['node', temp_file],
                capture_output=True,
                text=True,
                timeout=self.max_execution_time
            )
 
            stdout = result.stdout
            if len(stdout) > self.max_output_length:
                stdout = stdout[:self.max_output_length] + "\n... (output truncated)"
 
            return {
                "success": result.returncode == 0,
                "stdout": stdout.strip(),
                "stderr": result.stderr.strip() if result.stderr else "",
                "execution_time": time.time(),
                "return_code": result.returncode
            }
 
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Execution timed out (30s limit)",
                "execution_time": time.time()
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "execution_time": time.time()
            }
        finally:
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
                    pass
 
    def execute_bash_command(self, command: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute bash commands with security restrictions"""
        # Security check
        is_safe, reason = self.is_code_safe(command, "bash")
        if not is_safe:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Security violation: {reason}",
                "execution_time": time.time()
            }
 
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.max_execution_time,
                cwd=tempfile.gettempdir()
            )
 
            stdout = result.stdout
            if len(stdout) > self.max_output_length:
                stdout = stdout[:self.max_output_length] + "\n... (output truncated)"
 
            return {
                "success": result.returncode == 0,
                "stdout": stdout.strip(),
                "stderr": result.stderr.strip() if result.stderr else "",
                "execution_time": time.time(),
                "return_code": result.returncode
            }
 
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out (30s limit)",
                "execution_time": time.time()
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "execution_time": time.time()
            }
 
    def execute_code(self, code: str, language: str = "python", session_id: Optional[str] = None) -> str:
        """Main execution function - returns JSON for MCP compatibility"""
        # Cleanup old sessions periodically
        if len(self.sessions) > 10:
            self.cleanup_old_sessions()
 
        if language not in self.allowed_languages:
            return json.dumps({
                "success": False,
                "error": f"Language '{language}' not supported. Allowed: {', '.join(self.allowed_languages)}"
            })
 
        # Create session if needed
        if session_id and session_id not in self.sessions:
            session_id = self.create_session()
        elif not session_id:
            session_id = self.create_session()
 
        # Execute based on language
        if language == "python":
            result = self.execute_python_code(code, session_id)
        elif language == "javascript":
            result = self.execute_javascript_code(code, session_id)
        elif language == "bash":
            result = self.execute_bash_command(code, session_id)
        else:
            result = {
                "success": False,
                "error": f"Execution handler for {language} not implemented"
            }
 
        # Store in session history
        if session_id in self.sessions:
            self.sessions[session_id]['history'].append({
                'code': code,
                'language': language,
                'result': result,
                'timestamp': time.time()
            })
 
        result['session_id'] = session_id
        return json.dumps(result, indent=2)
 
# Global executor instance
executor = SecureCodeExecutor()
 
# MCP Functions
def execute_python_code(code: str, session_id: str = None) -> str:
    """
    Execute Python code safely with visualization support.
    
    Args:
        code (str): Python code to execute
        session_id (str, optional): Session ID for persistent context
    
    Returns:
        str: JSON string with execution results
    """
    return executor.execute_code(code, "python", session_id)
 
def execute_javascript_code(code: str, session_id: str = None) -> str:
    """
    Execute JavaScript code using Node.js.
    
    Args:
        code (str): JavaScript code to execute
        session_id (str, optional): Session ID for persistent context
    
    Returns:
        str: JSON string with execution results
    """
    return executor.execute_code(code, "javascript", session_id)
 
def execute_bash_command(command: str, session_id: str = None) -> str:
    """
    Execute bash commands with security restrictions.
    
    Args:
        command (str): Bash command to execute
        session_id (str, optional): Session ID for persistent context
    
    Returns:
        str: JSON string with execution results
    """
    return executor.execute_code(command, "bash", session_id)
 
def create_execution_session() -> str:
    """
    Create a new execution session for maintaining state.
    
    Returns:
        str: JSON string containing new session ID
    """
    session_id = executor.create_session()
    return json.dumps({"session_id": session_id, "created_at": time.time()})
 
def list_execution_sessions() -> str:
    """
    List all active execution sessions.
    
    Returns:
        str: JSON string containing session information
    """
    return json.dumps({
        "sessions": list(executor.sessions.keys()),
        "count": len(executor.sessions),
        "timestamp": time.time()
    })
 
def get_execution_history(session_id: str) -> str:
    """
    Get execution history for a specific session.
    
    Args:
        session_id (str): Session ID to get history for
    
    Returns:
        str: JSON string containing execution history
    """
    if session_id not in executor.sessions:
        return json.dumps({"error": "Session not found"})
    
    return json.dumps({
        "session_id": session_id,
        "history": executor.sessions[session_id]['history'],
        "created_at": executor.sessions[session_id]['created_at']
    })
 
def get_system_info() -> str:
    """
    Get system information and available packages.
    
    Returns:
        str: JSON string containing system information
    """
    return json.dumps({
        "python_version": sys.version,
        "available_packages": [
            "numpy", "pandas", "matplotlib", "json", "math", 
            "random", "datetime", "base64", "io"
        ],
        "execution_limits": {
            "max_time": executor.max_execution_time,
            "max_output": executor.max_output_length
        },
        "supported_languages": executor.allowed_languages
    })
 
# Gradio Interface
def gradio_execute_code(code: str, language: str, session_id: str = ""):
    """Gradio interface for code execution"""
    if not session_id:
        session_id = None
    
    result_json = executor.execute_code(code, language.lower(), session_id)
    result = json.loads(result_json)
    
    output = ""
    if result.get("success"):
        if result.get("stdout"):
            output += f"Output:\n{result['stdout']}\n\n"
        if result.get("stderr"):
            output += f"Warnings:\n{result['stderr']}\n\n"
        if result.get("plot"):
            output += f"Plot generated (base64): {result['plot'][:100]}...\n\n"
    else:
        output += f"Error:\n{result.get('stderr', result.get('error', 'Unknown error'))}\n\n"
    
    output += f"Session ID: {result.get('session_id', 'N/A')}"
    
    return output
 
# Create Gradio interface
with gr.Blocks(title="LibreChat Code Interpreter") as demo:
    gr.Markdown("# LibreChat Code Interpreter")
    gr.Markdown("Execute Python, JavaScript, and Bash code safely through MCP integration.")
    
    with gr.Row():
        with gr.Column():
            code_input = gr.Textbox(
                placeholder="Enter your code here...",
                lines=10,
                label="Code"
            )
            language_dropdown = gr.Dropdown(
                choices=["Python", "JavaScript", "Bash"],
                value="Python",
                label="Language"
            )
            session_input = gr.Textbox(
                placeholder="Optional: Session ID for persistent context",
                label="Session ID"
            )
            execute_btn = gr.Button("Execute Code", variant="primary")
        
        with gr.Column():
            output_display = gr.Textbox(
                lines=15,
                label="Execution Result",
                interactive=False
            )
    
    # Examples
    gr.Markdown("## Examples")
    
    example_python = gr.Code("""
import matplotlib.pyplot as plt
import numpy as np
 
# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x) * np.exp(-x/10)
 
# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='Damped Sine Wave')
plt.title('Example Visualization')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
 
print("Visualization created successfully!")
""", language="python", label="Python Example with Visualization")
    
    example_js = gr.Code("""
// Data processing example
const data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
 
const sum = data.reduce((acc, val) => acc + val, 0);
const mean = sum / data.length;
const variance = data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / data.length;
const stdDev = Math.sqrt(variance);
 
console.log(`Dataset: [${data.join(', ')}]`);
console.log(`Sum: ${sum}`);
console.log(`Mean: ${mean}`);
console.log(`Standard Deviation: ${stdDev.toFixed(3)}`);
 
// JSON processing
const result = {
    dataset: data,
    statistics: {
        sum, mean, variance, stdDev
    },
    timestamp: new Date().toISOString()
};
 
console.log('\\nResult:');
console.log(JSON.stringify(result, null, 2));
""", language="javascript", label="JavaScript Example")
    
    execute_btn.click(
        fn=gradio_execute_code,
        inputs=[code_input, language_dropdown, session_input],
        outputs=[output_display]
    )
 
if __name__ == "__main__":
    # Launch with MCP server enabled
    demo.launch(
        mcp_server=True,
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
