#!/usr/bin/env python3
"""Serve a live, auto-refreshing dashboard for TensorBoard scalar event logs."""

from __future__ import annotations

import argparse
import json
import math
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:
    raise SystemExit(
        "TensorBoard is required. Run this script with the Isaac Lab environment, for example:\n"
        "  conda run --no-capture-output -n env_isaaclab_2 python "
        "scripts/tools/tensorboard_live_dashboard.py"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs/rsl_rl/g1_assist_exoskeleton_v2_ppo"
DEFAULT_HTML = Path(__file__).with_name("tensorboard_live_dashboard.html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6007)
    parser.add_argument(
        "--max-points",
        type=int,
        default=1500,
        help="Maximum number of points returned for each scalar series.",
    )
    return parser.parse_args()


class EventStore:
    """Discover runs and incrementally reload their TensorBoard scalar data."""

    def __init__(self, log_dir: Path, max_points: int):
        self.log_dir = log_dir.expanduser().resolve()
        self.max_points = max_points
        self._accumulators: dict[str, EventAccumulator] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._cache_lock = threading.Lock()

    def runs(self) -> list[dict[str, object]]:
        discovered: dict[str, dict[str, object]] = {}
        if not self.log_dir.is_dir():
            return []
        for event_file in self.log_dir.rglob("events.out.tfevents.*"):
            run_dir = event_file.parent
            relative = run_dir.relative_to(self.log_dir).as_posix()
            try:
                mtime = event_file.stat().st_mtime
            except FileNotFoundError:
                continue
            item = discovered.setdefault(relative, {"name": relative, "mtime": 0.0})
            item["mtime"] = max(float(item["mtime"]), mtime)
        return sorted(discovered.values(), key=lambda item: float(item["mtime"]), reverse=True)

    def _resolve_run(self, name: str) -> Path:
        available = {str(item["name"]) for item in self.runs()}
        if name not in available:
            raise ValueError(f"Unknown run: {name}")
        run_dir = (self.log_dir / name).resolve()
        if self.log_dir != run_dir and self.log_dir not in run_dir.parents:
            raise ValueError("Run path escapes the configured log directory")
        return run_dir

    def _accumulator(self, name: str) -> tuple[EventAccumulator, threading.Lock]:
        run_dir = self._resolve_run(name)
        with self._cache_lock:
            if name not in self._accumulators:
                self._accumulators[name] = EventAccumulator(
                    str(run_dir), size_guidance={"scalars": 0}
                )
                self._locks[name] = threading.Lock()
            return self._accumulators[name], self._locks[name]

    def metadata(self, name: str) -> dict[str, object]:
        accumulator, lock = self._accumulator(name)
        with lock:
            accumulator.Reload()
            tags = sorted(accumulator.Tags().get("scalars", []))
            latest: dict[str, dict[str, float | int]] = {}
            for tag in tags:
                events = accumulator.Scalars(tag)
                if not events:
                    continue
                event = events[-1]
                latest[tag] = {
                    "step": int(event.step),
                    "value": float(event.value),
                    "wall_time": float(event.wall_time),
                }
        return {"run": name, "tags": tags, "latest": latest}

    def scalars(self, name: str, tags: list[str]) -> dict[str, object]:
        accumulator, lock = self._accumulator(name)
        with lock:
            accumulator.Reload()
            available = set(accumulator.Tags().get("scalars", []))
            series: dict[str, list[list[float | int]]] = {}
            for tag in tags:
                if tag not in available:
                    continue
                events = accumulator.Scalars(tag)
                if not events:
                    continue
                stride = max(1, math.ceil(len(events) / self.max_points))
                selected = events[::stride]
                if selected[-1].step != events[-1].step:
                    selected.append(events[-1])
                series[tag] = [
                    [int(event.step), float(event.value), float(event.wall_time)]
                    for event in selected
                ]
        return {"run": name, "series": series}


class DashboardHandler(BaseHTTPRequestHandler):
    store: EventStore
    html_path: Path

    def log_message(self, format_string: str, *args: object) -> None:
        return

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        try:
            body = self.html_path.read_bytes()
        except FileNotFoundError:
            self._send_json(
                {"error": f"Dashboard HTML not found: {self.html_path}"},
                HTTPStatus.NOT_FOUND,
            )
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        request = urlparse(self.path)
        query = parse_qs(request.query)
        try:
            if request.path in {"/", "/index.html"}:
                self._send_html()
            elif request.path == "/api/runs":
                self._send_json({"logdir": str(self.store.log_dir), "runs": self.store.runs()})
            elif request.path == "/api/metadata":
                run = query.get("run", [""])[0]
                self._send_json(self.store.metadata(run))
            elif request.path == "/api/scalars":
                run = query.get("run", [""])[0]
                tags = [tag for tag in query.get("tag", []) if tag]
                self._send_json(self.store.scalars(run, tags))
            else:
                self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, OSError, KeyError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # keep the dashboard alive if an event file is mid-write
            self._send_json({"error": f"Unable to read TensorBoard data: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    args = parse_args()
    if args.max_points <= 1:
        raise ValueError("--max-points must be greater than one")
    store = EventStore(args.logdir, args.max_points)
    DashboardHandler.store = store
    DashboardHandler.html_path = args.html.expanduser().resolve()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"TensorBoard log directory: {store.log_dir}")
    print(f"Live dashboard: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
