import sys
import json
import subprocess
import os

def main():
    # Path to the PETSc MCP example executable
    # Assuming it's built in the same directory or provided as argument
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_ex1>")
        sys.exit(1)
        
    executable = sys.argv[1]
    
    # Start the MCP server process
    process = subprocess.Popen(
        [executable],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1  # Line buffered
    )
    
    print(f"Started MCP server: {executable}")
    
    # 1. Initialize
    init_request = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "petsc-mcp-client", "version": "0.1.0"}
        },
        "id": 1
    }
    
    print(f"Sending: {json.dumps(init_request)}")
    process.stdin.write(json.dumps(init_request) + "\n")
    process.stdin.flush()
    
    response = process.stdout.readline()
    print(f"Received: {response.strip()}")
    
    # 2. Initialized Notification
    notify_init = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    }
    print(f"Sending: {json.dumps(notify_init)}")
    process.stdin.write(json.dumps(notify_init) + "\n")
    process.stdin.flush()
    
    # 3. List Tools
    list_tools = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 2
    }
    print(f"Sending: {json.dumps(list_tools)}")
    process.stdin.write(json.dumps(list_tools) + "\n")
    process.stdin.flush()
    
    response = process.stdout.readline()
    print(f"Received: {response.strip()}")
    
    # 4. Call Tool: petsc_get_version
    call_version = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "petsc_get_version",
            "arguments": {}
        },
        "id": 3
    }
    print(f"Sending: {json.dumps(call_version)}")
    process.stdin.write(json.dumps(call_version) + "\n")
    process.stdin.flush()
    
    response = process.stdout.readline()
    print(f"Received: {response.strip()}")
    
    # 5. Call Tool: petsc_options_view
    call_options = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "petsc_options_view",
            "arguments": {}
        },
        "id": 4
    }
    print(f"Sending: {json.dumps(call_options)}")
    process.stdin.write(json.dumps(call_options) + "\n")
    process.stdin.flush()
    
    response = process.stdout.readline()
    print(f"Received: {response.strip()}")

    # Terminate
    process.terminate()

if __name__ == "__main__":
    main()
