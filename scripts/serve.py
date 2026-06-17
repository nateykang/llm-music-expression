#!/usr/bin/env python3
"""Local dev server for docs/ with HTTP Range support.

Python's stdlib http.server does NOT honor Range requests, which breaks audio
seeking and makes Ogg/Vorbis files report a "growing" duration (the browser
can't read the length from the file's tail). Real static hosts (GitHub Pages)
support ranges, so this only matters for local preview. Usage:

    python scripts/serve.py [port]      # defaults to 8000, serves ../docs
"""

from __future__ import annotations

import functools
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class RangeHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + minimal single-range (bytes=start-end) support.

    Also disables caching: this is a dev server, so we always want the freshest
    edits without forcing a hard refresh. (Real static hosts cache normally.)
    """

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def send_head(self):
        rng = self.headers.get("Range")
        if not rng or not rng.startswith("bytes="):
            return super().send_head()

        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()

        try:
            start_s, end_s = rng[len("bytes="):].split("-", 1)
            size = os.path.getsize(path)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else size - 1
            end = min(end, size - 1)
            if start > end:
                self.send_error(416, "Requested Range Not Satisfiable")
                return None
        except ValueError:
            return super().send_head()

        f = open(path, "rb")
        f.seek(start)
        self._range_remaining = end - start + 1
        self.send_response(206, "Partial Content")
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self._range_remaining))
        self.end_headers()
        return _LimitedReader(f, self._range_remaining)


class _LimitedReader:
    """File wrapper that yields at most `remaining` bytes (for copyfile)."""

    def __init__(self, fp, remaining: int):
        self.fp = fp
        self.remaining = remaining

    def read(self, n: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        if n < 0 or n > self.remaining:
            n = self.remaining
        data = self.fp.read(n)
        self.remaining -= len(data)
        return data

    def close(self):
        self.fp.close()


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    docs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
    handler = functools.partial(RangeHandler, directory=docs)
    print(f"Serving {docs} at http://localhost:{port}  (Range requests supported)")
    try:
        HTTPServer(("", port), handler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
