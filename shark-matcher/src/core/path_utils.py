"""Path manipulation utilities: sys.path helpers for optional sibling checkouts."""
import os
import sys

from src.core import constants


def ensure_pairx_in_path():
    """Ensure that the pairx repository is in sys.path."""
    base_pairx_dir = os.path.dirname(constants.PAIRX_DIR)
    if base_pairx_dir not in sys.path:
        if os.path.exists(constants.PAIRX_DIR):
            sys.path.append(base_pairx_dir)
        else:
            print(f"WARNING: pairx directory not found at {constants.PAIRX_DIR}.")
            print("Please ensure pairx is cloned as a sibling to 'shark-matcher'")
            print("or set the PAIRX_ROOT environment variable.")


def ensure_shark_model_in_path():
    """Ensure that shark-matcher is in sys.path."""
    if constants.THIS_DIR not in sys.path:
        sys.path.append(constants.THIS_DIR)
