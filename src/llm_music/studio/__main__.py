"""Run the composer studio: ``python -m llm_music.studio`` or ``llm-music-studio``."""

from __future__ import annotations

import argparse
import sys

from . import config as cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Composer studio web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8321)
    parser.add_argument("--proxy-headers", action="store_true",
                        help="trust X-Forwarded-* (use behind Caddy/nginx)")
    args = parser.parse_args(argv)

    if not cfg.password():
        print("STUDIO_PASSWORD is not set — refusing to start an open studio.\n"
              "Set it in the environment or in .env.", file=sys.stderr)
        return 1

    import uvicorn

    uvicorn.run("llm_music.studio.app:app", host=args.host, port=args.port,
                proxy_headers=args.proxy_headers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
