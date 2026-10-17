"""Make the repo root importable however pytest is started.

`gym env test` puts the repo root on PYTHONPATH and `python -m pytest` adds the cwd, but a plain
`pytest` from the repo root or the server dir does neither. The tests import the server as
`resources_servers.web_agent.app` (a namespace package shared with Gym's built-in servers).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
