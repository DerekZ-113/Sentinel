#!/usr/bin/env python
"""
Continuous local demo streamer for Sentinel.

This script is for local demos only. It sends deterministic synthetic
notifications through the live prediction API and is not production ingestion.

Usage:
    python -m scripts.stream_demo --count 10 --interval 1
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import json as jsonlib
import os
import random
import sys
import time
from typing import Callable, Iterable, Iterator, TextIO
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import requests
except ModuleNotFoundError:
    requests = None


ROAD_PROFILES = {
    "highway": {"base_speed": 65.0, "weight": 0.08, "pedestrian_base": 0.0},
    "main_road": {"base_speed": 45.0, "weight": 0.38, "pedestrian_base": 0.3},
    "residential": {"base_speed": 30.0, "weight": 0.24, "pedestrian_base": 0.4},
    "downtown": {"base_speed": 35.0, "weight": 0.22, "pedestrian_base": 0.7},
    "school_zone": {"base_speed": 15.0, "weight": 0.08, "pedestrian_base": 0.5},
}

TRAFFIC_MODIFIERS = {
    "light": 1.0,
    "moderate": 0.7,
    "heavy": 0.3,
    "standstill": 0.05,
}

CONSTRUCTION_MODIFIERS = {
    "none": 1.0,
    "temporary": 0.6,
    "persistent": 0.5,
    "flagger": 0.0,
}

CONSTRUCTION_WEIGHTS = (
    ("none", 0.65),
    ("temporary", 0.18),
    ("persistent", 0.10),
    ("flagger", 0.07),
)

NOTIFICATION_WEIGHTS = (
    ("verification_request", 0.36),
    ("emergency_vehicle_alert", 0.18),
    ("stuck", 0.20),
    ("speed_anomaly", 0.14),
    ("impact_l0", 0.08),
    ("passenger_assist", 0.04),
)

VERIFICATION_SUBTYPE_WEIGHTS = (
    ("object_query", 0.72),
    ("traffic_signal_verify", 0.17),
    ("lane_mapping_verify", 0.11),
)

INTERVENTION_PROBABILITY = {
    "verification_request": 0.22,
    "emergency_vehicle_alert": 0.30,
    "stuck": 0.34,
    "speed_anomaly": 0.48,
    "impact_l0": 0.60,
    "passenger_assist": 1.0,
}

FATAL_STATUS_CODES = {401, 403, 422}


class TransportError(OSError):
    """Network transport error raised by the stdlib HTTP fallback."""


if requests is not None:
    REQUEST_EXCEPTIONS = (requests.RequestException, TransportError)
else:
    REQUEST_EXCEPTIONS = (TransportError,)


class StreamDemoError(RuntimeError):
    """Base error for local demo streaming failures."""


class FatalStreamError(StreamDemoError):
    """A response error that should stop the stream immediately."""


class HealthCheckError(StreamDemoError):
    """Raised when the API is unreachable or not ready."""


class UrlLibResponse:
    """Minimal response adapter matching the subset used by this script."""

    def __init__(self, status_code: int, reason: str, body: bytes):
        self.status_code = status_code
        self.reason = reason
        self._body = body

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return jsonlib.loads(self._body.decode("utf-8"))


def _urllib_json_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    payload: dict | None = None,
):
    body = None if payload is None else jsonlib.dumps(payload).encode("utf-8")
    request = urllib_request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib_request.urlopen(request, timeout=timeout) as response:
            return UrlLibResponse(
                status_code=response.status,
                reason=response.reason,
                body=response.read(),
            )
    except urllib_error.HTTPError as exc:
        return UrlLibResponse(
            status_code=exc.code,
            reason=exc.reason,
            body=exc.read(),
        )
    except urllib_error.URLError as exc:
        raise TransportError(str(exc.reason)) from exc


def get_json(session, url: str, *, headers: dict[str, str], timeout: float):
    if session is not None:
        return session.get(url, headers=headers, timeout=timeout)
    return _urllib_json_request("GET", url, headers=headers, timeout=timeout)


def post_json(
    session,
    url: str,
    *,
    payload: dict,
    headers: dict[str, str],
    timeout: float,
):
    if session is not None:
        return session.post(url, json=payload, headers=headers, timeout=timeout)
    return _urllib_json_request("POST", url, headers=headers, timeout=timeout, payload=payload)


@dataclass
class StreamSummary:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    interrupted: bool = False


def _weighted_choice(rng: random.Random, weighted_items: Iterable[tuple[str, float]]) -> str:
    items = list(weighted_items)
    total = sum(weight for _item, weight in items)
    target = rng.uniform(0, total)
    running = 0.0
    for item, weight in items:
        running += weight
        if target <= running:
            return item
    return items[-1][0]


def _traffic_weights_for_hour(hour: int) -> tuple[tuple[str, float], ...]:
    if hour in {6, 7, 8, 16, 17, 18}:
        return (
            ("light", 0.05),
            ("moderate", 0.30),
            ("heavy", 0.50),
            ("standstill", 0.15),
        )
    if hour in {5, 9, 10, 11, 12, 13, 14, 15, 19, 20}:
        return (
            ("light", 0.35),
            ("moderate", 0.35),
            ("heavy", 0.20),
            ("standstill", 0.10),
        )
    return (
        ("light", 0.70),
        ("moderate", 0.10),
        ("heavy", 0.10),
        ("standstill", 0.10),
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _expected_speed(road_type: str, traffic_condition: str, construction_zone: str) -> float:
    road_speed = ROAD_PROFILES[road_type]["base_speed"]
    speed = (
        road_speed
        * TRAFFIC_MODIFIERS[traffic_condition]
        * CONSTRUCTION_MODIFIERS[construction_zone]
    )
    return round(speed, 1)


def _pedestrian_density(rng: random.Random, road_type: str, hour: int) -> float:
    base_density = ROAD_PROFILES[road_type]["pedestrian_base"]
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        time_modifier = 1.3
    elif 10 <= hour <= 15:
        time_modifier = 1.0
    elif 19 <= hour <= 22:
        time_modifier = 0.7
    else:
        time_modifier = 0.2
    density = base_density * time_modifier + rng.uniform(-0.08, 0.08)
    return round(_clamp(density, 0.0, 1.0), 2)


def _base_speed_for_context(
    rng: random.Random,
    expected_speed: float,
    traffic_condition: str,
) -> float:
    if traffic_condition == "standstill":
        return round(rng.uniform(0.0, min(4.0, expected_speed + 3.0)), 1)
    return round(_clamp(expected_speed + rng.uniform(-4.0, 5.0), 0.0, 80.0), 1)


def _build_payload(
    rng: random.Random,
    vehicle_id: str,
    sim_time: datetime,
) -> dict:
    road_type = _weighted_choice(
        rng,
        ((road_type, profile["weight"]) for road_type, profile in ROAD_PROFILES.items()),
    )
    traffic_condition = _weighted_choice(rng, _traffic_weights_for_hour(sim_time.hour))
    construction_zone = _weighted_choice(rng, CONSTRUCTION_WEIGHTS)
    notification_type = _weighted_choice(rng, NOTIFICATION_WEIGHTS)
    expected_speed = _expected_speed(road_type, traffic_condition, construction_zone)
    speed = _base_speed_for_context(rng, expected_speed, traffic_condition)
    pedestrian_density = _pedestrian_density(rng, road_type, sim_time.hour)
    notification_subtype = None
    ev_distance = None
    object_in_path = False
    time_since_stop = 0.0

    needs_intervention = rng.random() < INTERVENTION_PROBABILITY[notification_type]

    if notification_type == "verification_request":
        notification_subtype = _weighted_choice(rng, VERIFICATION_SUBTYPE_WEIGHTS)
        object_in_path = (
            notification_subtype == "object_query"
            and rng.random() < (0.45 if needs_intervention else 0.12)
        )
        if notification_subtype == "traffic_signal_verify":
            pedestrian_density = max(pedestrian_density, round(rng.uniform(0.25, 0.75), 2))
    elif notification_type == "emergency_vehicle_alert":
        if needs_intervention:
            ev_distance = 0.0 if rng.random() < 0.08 else round(rng.uniform(10.0, 95.0), 1)
        else:
            ev_distance = round(rng.uniform(110.0, 500.0), 1)
    elif notification_type == "stuck":
        speed = round(rng.uniform(0.0, 2.5), 1)
        time_since_stop = round(
            rng.uniform(120.0, 600.0) if needs_intervention else rng.uniform(20.0, 180.0),
            1,
        )
    elif notification_type == "speed_anomaly":
        if needs_intervention:
            speed = round(_clamp(expected_speed + rng.uniform(15.0, 28.0), 0.0, 85.0), 1)
        else:
            speed = round(_clamp(expected_speed + rng.uniform(6.0, 14.0), 0.0, 80.0), 1)
    elif notification_type == "impact_l0":
        speed = round(rng.uniform(0.0, min(30.0, max(expected_speed, 5.0))), 1)
        object_in_path = rng.random() < 0.35
    elif notification_type == "passenger_assist":
        speed = 0.0
        time_since_stop = round(rng.uniform(30.0, 300.0), 1)
        pedestrian_density = max(pedestrian_density, round(rng.uniform(0.2, 0.8), 2))

    return {
        "vehicle_id": vehicle_id,
        "speed": speed,
        "expected_speed": expected_speed,
        "road_type": road_type,
        "traffic_condition": traffic_condition,
        "construction_zone": construction_zone,
        "notification_type": notification_type,
        "notification_subtype": notification_subtype,
        "ev_distance": ev_distance if ev_distance is not None else None,
        "pedestrian_density": pedestrian_density,
        "object_in_path": object_in_path,
        "time_since_stop": time_since_stop,
        "hour_of_day": sim_time.hour,
        "needs_intervention_actual": needs_intervention,
    }


def generate_payloads(
    *,
    seed: int,
    vehicles: int,
    start_time: datetime,
) -> Iterator[dict]:
    """Yield deterministic synthetic notification payloads indefinitely."""
    if vehicles <= 0:
        raise ValueError("vehicles must be greater than 0")

    rng = random.Random(seed)
    vehicle_ids = [f"stream_vehicle_{index:03d}" for index in range(vehicles)]
    sim_time = start_time

    while True:
        vehicle_id = rng.choice(vehicle_ids)
        yield _build_payload(rng, vehicle_id, sim_time)
        sim_time += timedelta(seconds=rng.randint(5, 90))


def build_headers(api_key: str | None) -> dict[str, str]:
    """Build JSON headers, adding X-API-Key only when configured."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def _api_url(api_base: str, path: str) -> str:
    return f"{api_base.rstrip('/')}{path}"


def _format_detail(detail) -> str:
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict):
                loc = item.get("loc")
                msg = item.get("msg") or item.get("message")
                if loc and msg:
                    parts.append(f"{'.'.join(str(part) for part in loc)}: {msg}")
                elif msg:
                    parts.append(str(msg))
                else:
                    parts.append(jsonlib.dumps(item, sort_keys=True))
            else:
                parts.append(str(item))
        return "; ".join(parts)
    if isinstance(detail, dict):
        return jsonlib.dumps(detail, sort_keys=True)
    return str(detail)


def _response_detail(response) -> str:
    try:
        body = response.json()
    except ValueError:
        return ""

    if isinstance(body, dict):
        for key in ("detail", "message", "error"):
            if key in body:
                return _format_detail(body[key])
        return jsonlib.dumps(body, sort_keys=True)
    return _format_detail(body)


def format_response_error(response, url: str) -> str:
    """Format a non-2xx response with status and backend detail when present."""
    status_code = getattr(response, "status_code", "unknown")
    reason = getattr(response, "reason", "") or ""
    status = f"{status_code} {reason}".strip()
    message = f"Request to {url} failed with {status}"
    detail = _response_detail(response)
    if detail:
        message = f"{message}: {detail}"
    return message


def check_api_health(
    *,
    session,
    api_base: str,
    api_key: str | None,
    timeout: float,
) -> dict:
    """Verify that the API is reachable and the model/DB are ready."""
    url = _api_url(api_base, "/health")
    try:
        response = get_json(session, url, headers=build_headers(api_key), timeout=timeout)
    except REQUEST_EXCEPTIONS as exc:
        raise HealthCheckError(f"API health check failed for {url}: {exc}") from exc

    if not response.ok:
        raise HealthCheckError(format_response_error(response, url))

    try:
        data = response.json()
    except ValueError as exc:
        raise HealthCheckError(f"API health check returned non-JSON response from {url}") from exc

    if not data.get("model_loaded") or not data.get("db_connected"):
        raise HealthCheckError(
            "API health check is not ready: "
            f"model_loaded={data.get('model_loaded')} "
            f"db_connected={data.get('db_connected')}"
        )
    return data


def _print_progress(index: int, payload: dict, prediction: dict, status_code: int, output: TextIO) -> None:
    confidence = prediction.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence_text = f"{confidence:.3f}"
    else:
        confidence_text = "n/a"
    print(
        f"{index}: vehicle_id={payload['vehicle_id']} "
        f"notification_type={payload['notification_type']} "
        f"needs_intervention={prediction.get('needs_intervention')} "
        f"confidence={confidence_text} status={status_code}",
        file=output,
        flush=True,
    )


def stream_predictions(
    *,
    payloads: Iterator[dict],
    session,
    api_base: str,
    api_key: str | None,
    count: int,
    interval: float,
    timeout: float,
    sleep_fn: Callable[[float], None] = time.sleep,
    output: TextIO = sys.stdout,
) -> StreamSummary:
    """Post generated payloads to the prediction API."""
    summary = StreamSummary()
    url = _api_url(api_base, "/api/predict")
    headers = build_headers(api_key)
    finite = count > 0

    try:
        while not finite or summary.attempts < count:
            payload = next(payloads)
            summary.attempts += 1

            try:
                response = post_json(
                    session,
                    url,
                    payload=payload,
                    headers=headers,
                    timeout=timeout,
                )
            except REQUEST_EXCEPTIONS as exc:
                summary.failures += 1
                print(f"Request to {url} failed: {exc}", file=output, flush=True)
            else:
                if response.ok:
                    try:
                        prediction = response.json()
                    except ValueError as exc:
                        summary.failures += 1
                        print(
                            f"Request to {url} returned non-JSON success response: {exc}",
                            file=output,
                            flush=True,
                        )
                    else:
                        summary.successes += 1
                        _print_progress(summary.attempts, payload, prediction, response.status_code, output)
                else:
                    summary.failures += 1
                    message = format_response_error(response, url)
                    print(message, file=output, flush=True)
                    if response.status_code in FATAL_STATUS_CODES:
                        raise FatalStreamError(message)

            if finite and summary.attempts >= count:
                break
            if interval > 0:
                sleep_fn(interval)
    except KeyboardInterrupt:
        summary.interrupted = True
        print("\nInterrupted by Ctrl-C.", file=output, flush=True)

    return summary


def parse_start_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "start time must use ISO format, e.g. 2024-12-01T06:00:00"
        ) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream deterministic synthetic Sentinel notifications through /api/predict.",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("SENTINEL_API_BASE", "http://localhost:8000"),
        help="Base URL for the Sentinel API.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY", ""),
        help="Optional API key. Defaults to API_KEY from the environment.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Post attempts; 0 means run until interrupted.",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between post attempts.")
    parser.add_argument("--vehicles", type=int, default=25, help="Number of synthetic vehicle IDs.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic payload seed.")
    parser.add_argument(
        "--start-time",
        type=parse_start_time,
        default=parse_start_time("2024-12-01T06:00:00"),
        help="Synthetic simulation start time in ISO format.",
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP request timeout in seconds.")
    args = parser.parse_args(argv)

    if args.count < 0:
        parser.error("--count must be greater than or equal to 0")
    if args.interval < 0:
        parser.error("--interval must be greater than or equal to 0")
    if args.vehicles <= 0:
        parser.error("--vehicles must be greater than 0")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    return args


def _print_summary(summary: StreamSummary, output: TextIO) -> None:
    print(
        "Summary: "
        f"attempts={summary.attempts} "
        f"successes={summary.successes} "
        f"failures={summary.failures} "
        f"interrupted={summary.interrupted}",
        file=output,
        flush=True,
    )


def run(argv: list[str] | None = None, output: TextIO = sys.stdout, error: TextIO = sys.stderr) -> int:
    args = parse_args(argv)
    session = requests.Session() if requests is not None else None

    try:
        health = check_api_health(
            session=session,
            api_base=args.api_base,
            api_key=args.api_key,
            timeout=args.timeout,
        )
    except HealthCheckError as exc:
        print(str(exc), file=error, flush=True)
        return 1

    print(
        "API ready: "
        f"model_loaded={health.get('model_loaded')} "
        f"db_connected={health.get('db_connected')}",
        file=output,
        flush=True,
    )

    payloads = generate_payloads(
        seed=args.seed,
        vehicles=args.vehicles,
        start_time=args.start_time,
    )

    try:
        summary = stream_predictions(
            payloads=payloads,
            session=session,
            api_base=args.api_base,
            api_key=args.api_key,
            count=args.count,
            interval=args.interval,
            timeout=args.timeout,
            output=output,
        )
    except FatalStreamError as exc:
        print(str(exc), file=error, flush=True)
        return 1

    _print_summary(summary, output)
    if args.count > 0 and summary.failures > 0:
        return 1
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
