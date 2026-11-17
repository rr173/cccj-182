import os
import sys

# Tests run with "python3 -I"; isolated mode drops cwd/PYTHONPATH, so make the
# workspace package importable explicitly without weakening isolation.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
