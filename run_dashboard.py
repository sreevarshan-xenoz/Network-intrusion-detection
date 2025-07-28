#!/usr/bin/env python3
"""
Script to run the Network Intrusion Detection Dashboard.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.dashboard import main

if __name__ == "__main__":
    main()