"""Helper entry point for running the gRPC backend server."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run() -> None:
    """Start the gRPC backend using the current Python interpreter."""
    logger.info("Starting gRPC server on grpc://localhost:50051 ...")
    subprocess.run([sys.executable, "grpc_server.py"], cwd=Path(__file__).parent, check=False)


if __name__ == "__main__":
    run()
