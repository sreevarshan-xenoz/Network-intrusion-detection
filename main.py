#!/usr/bin/env python3
"""
Main entry point for the Network Intrusion Detection System.
"""
import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.config import config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point."""
    logger.info("Starting Network Intrusion Detection System")
    logger.info(f"Configuration loaded from: {config.config_path}")
    logger.info("System initialized successfully")


if __name__ == "__main__":
    main()