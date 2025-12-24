# Entry point for running the Lifecycle Investing Calculator
# Usage: streamlit run run.py

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from lifecycle_app.main import main

if __name__ == "__main__":
    main()
