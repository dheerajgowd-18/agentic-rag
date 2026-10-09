import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Set test environment defaults
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["CHECKPOINT_PERSISTENCE"] = "memory"
os.environ["GUARDRAILS_FAIL_CLOSED"] = "true"
