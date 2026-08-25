"""
06_simulation_demo/simulate_full_flow.py
-----------------------------------------
Delegates directly to combined_demo.py (The Unified Master System Engine).
"""

import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from combined_demo import run_master_demo


def run_demo(run_name=None):
    """Delegates to the master system engine."""
    run_master_demo()


if __name__ == "__main__":
    run_demo()



