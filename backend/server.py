"""Entry point helpers for running the gRPC or MCP servers."""

import subprocess
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_mcp_server():
    """
    Run MCP server on stdio
    This is for MCP clients like Claude Desktop
    """
    logger.info("MCP Server: Starting on stdio...")
    subprocess.run([sys.executable, "mcp_server.py"], cwd=Path(__file__).parent)

def run_grpc_server():
    """Run gRPC server for local desktop client."""
    logger.info("gRPC Server: Starting on grpc://localhost:50051...")
    subprocess.run([sys.executable, "grpc_server.py"], cwd=Path(__file__).parent)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "mcp":
            run_mcp_server()
        elif mode == "grpc":
            run_grpc_server()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python server.py [mcp|grpc]")
            sys.exit(1)
    else:
        # Default: run gRPC server for desktop client
        run_grpc_server()
