from __future__ import annotations

import argparse
from pathlib import Path

from openrouter_mode.dashboard import serve_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a local browser dashboard for one OpenRouter run directory.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory under project/openrouter_runs.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the local server.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind the local server.")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open the browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serve_dashboard(
        Path(args.run_dir).resolve(),
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
    )


if __name__ == "__main__":
    main()
