"""
Main Streamlit App for Pneumonia Detection
This is the entry point for Streamlit deployment
"""

import streamlit as st
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main app
from webapp.streamlit_app import main

if __name__ == "__main__":
    main()
