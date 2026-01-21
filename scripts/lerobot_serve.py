"""Simple policy server with RTC support for MMK."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingTCPServer
from typing import Any
import importlib
import importlib.util


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class Policy:
    def reset(self) -> None:
        return

    def act(self, observation: dict[str, Any]) -> Any:
        raise NotImplementedError


@dataclass
class DummyPolicy(Policy):
    default_value: float = 0.0

    def act(self, observation: dict[str, Any]) -> dict[str, float]:
        action_names = observation.get("action_names") or []
        return {f"{name}.pos": self.default_value for name in action_names}


@dataclass
class CallablePolicy(Policy):
    model: Any

    def reset(self) -> None:
        reset_fn = getattr(self.model, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def act(self, observation: dict[str, Any]) -> Any:
        act_fn = getattr(self.model, "act", None)
        if callable(act_fn):
            return act_fn(observation)
        if callable(self.model):
            return self.model(observation)
        raise TypeError("Loaded policy does not expose an act() method or __call__.")


@dataclass
class PolicyServerState:
    policy: Policy
    pending_actions: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def reset(self, client_id: str) -> None:
        self.pending_actions.pop(client_id, None)
        self.policy.reset()

    def next_actions(
        self,
        client_id: str,
        observation: dict[str, Any],
        chunk_size: int,
        action_names: list[str],
    ) -> list[dict[str, Any]]:
        if client_id not in self.pending_actions:
            self.pending_actions[client_id] = []
        queue = self.pending_actions[client_id]
        if not queue:
            observation = dict(observation)
            observation.setdefault("action_names", action_names)
            raw_actions = self.policy.act(observation)
            queue.extend(_normalize_actions(raw_actions, action_names))
        if chunk_size <= 0:
            chunk_size = 1
        actions = queue[:chunk_size]
        del queue[:chunk_size]
        return actions


def _load_policy(policy_path: str | None) -> Policy:
    if not policy_path:
        return DummyPolicy()
    path = Path(policy_path)
    if not path.exists():
        raise FileNotFoundError(f"Policy path does not exist: {path}")
    if importlib.util.find_spec("torch") is None:
        raise ImportError("torch is required to load a policy from disk")
    torch = importlib.import_module("torch")
    model = None
    if path.suffix in {".pt", ".pth"}:
        model = torch.jit.load(str(path))
    if model is None:
        model = torch.load(str(path), map_location="cpu")
    return CallablePolicy(model=model)


def _normalize_actions(raw_actions: Any, action_names: list[str]) -> list[dict[str, Any]]:
    if raw_actions is None:
        return []
    if isinstance(raw_actions, dict):
        return [raw_actions]
    if isinstance(raw_actions, list):
        if raw_actions and isinstance(raw_actions[0], dict):
            return raw_actions
        return [_action_from_list(raw_actions, action_names)]
    numpy_spec = importlib.util.find_spec("numpy")
    if numpy_spec is not None:
        np = importlib.import_module("numpy")
        if isinstance(raw_actions, np.ndarray):
            if raw_actions.ndim == 1:
                return [_action_from_list(raw_actions.tolist(), action_names)]
            if raw_actions.ndim == 2:
                return [
                    _action_from_list(row.tolist(), action_names) for row in raw_actions
                ]
    return [_action_from_list(list(raw_actions), action_names)]


def _action_from_list(values: list[float], action_names: list[str]) -> dict[str, float]:
    action = {}
    for name, value in zip(action_names, values, strict=False):
        action[f"{name}.pos"] = float(value)
    return action


class PolicyRequestHandler(BaseHTTPRequestHandler):
    server_version = "MMKPolicyServer/0.1"

    def _handle_health(self) -> None:
        _json_response(self, HTTPStatus.OK, {"status": "ok", "time": time.time()})

    def _handle_act(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            _json_response(self, HTTPStatus.BAD_REQUEST, {"error": "Empty payload"})
            return
        payload = json.loads(self.rfile.read(length))
        client_id = str(payload.get("client_id") or "default")
        if payload.get("reset"):
            self.server.state.reset(client_id)
        observation = payload.get("observation") or {}
        chunk_size = int(payload.get("chunk_size") or 1)
        action_names = list(payload.get("action_names") or [])
        actions = self.server.state.next_actions(
            client_id, observation, chunk_size, action_names
        )
        _json_response(self, HTTPStatus.OK, {"actions": actions})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._handle_health()
            return
        _json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/act":
            self._handle_act()
            return
        _json_response(self, HTTPStatus.NOT_FOUND, {"error": "Not found"})


class PolicyHTTPServer(ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: PolicyServerState):
        super().__init__(address, PolicyRequestHandler)
        self.state = state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start MMK policy server with RTC.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--policy_path", default=None)
    parser.add_argument("--config_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    policy_path = args.policy_path
    if not policy_path:
        config_path = args.config_path or "examples/evaluate.yaml"
        cfg_path = Path(config_path)
        if cfg_path.exists():
            config = _load_yaml(cfg_path)
            policy_path = config.get("policy", {}).get("path")
    policy = _load_policy(policy_path)
    state = PolicyServerState(policy=policy)
    server = PolicyHTTPServer((args.host, args.port), state)
    print(f"Policy server listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
