"""Allow ``python -m proto_language`` as an alternate CLI entry point.

Same surface as the ``proto-language`` console script registered in
``pyproject.toml`` — see ``proto_language.cli.main`` for the verbs.
"""

from __future__ import annotations

import sys

from proto_language.cli import main

if __name__ == "__main__":
    sys.exit(main())
