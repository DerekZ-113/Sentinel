"""
Tests for the local continuous demo streaming script.
"""

from datetime import datetime
from itertools import islice
from io import StringIO

import pytest

from api.models import (
    ConstructionZone,
    NotificationPayload,
    NotificationSubtype,
    NotificationType,
    RoadType,
    TrafficCondition,
)
from scripts import stream_demo


class FakeResponse:
    def __init__(self, status_code=200, reason="OK", json_data=None):
        self.status_code = status_code
        self.reason = reason
        self._json_data = json_data if json_data is not None else {}

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if isinstance(self._json_data, Exception):
            raise self._json_data
        return self._json_data


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []

    def post(self, url, json, headers, timeout):
        self.posts.append({
            "url": url,
            "json": json,
            "headers": headers,
            "timeout": timeout,
        })
        return self.responses.pop(0)


def _payloads(seed=42, count=5):
    start_time = datetime.fromisoformat("2024-12-01T06:00:00")
    return list(islice(
        stream_demo.generate_payloads(seed=seed, vehicles=8, start_time=start_time),
        count,
    ))


def _success_response(payload, confidence=0.82):
    return FakeResponse(json_data={
        "vehicle_id": payload["vehicle_id"],
        "notification_type": payload["notification_type"],
        "needs_intervention": True,
        "confidence": confidence,
        "raw_score": confidence,
        "timestamp": "2024-12-01T06:00:00Z",
    })


def test_same_seed_produces_same_first_payloads():
    assert _payloads(seed=42, count=10) == _payloads(seed=42, count=10)
    assert _payloads(seed=43, count=10) != _payloads(seed=42, count=10)


def test_generated_payloads_validate_against_api_model():
    for payload in _payloads(seed=42, count=25):
        NotificationPayload(**payload)


def test_generated_payloads_use_valid_shapes_and_enum_values():
    road_types = {item.value for item in RoadType}
    traffic_conditions = {item.value for item in TrafficCondition}
    construction_zones = {item.value for item in ConstructionZone}
    notification_types = {item.value for item in NotificationType}
    notification_subtypes = {item.value for item in NotificationSubtype}

    for payload in _payloads(seed=7, count=40):
        assert payload["vehicle_id"].startswith("stream_vehicle_")
        assert payload["speed"] >= 0
        assert payload["expected_speed"] >= 0
        assert payload["road_type"] in road_types
        assert payload["traffic_condition"] in traffic_conditions
        assert payload["construction_zone"] in construction_zones
        assert payload["notification_type"] in notification_types
        assert 0.0 <= payload["pedestrian_density"] <= 1.0
        assert payload["time_since_stop"] >= 0
        assert 0 <= payload["hour_of_day"] <= 23
        assert isinstance(payload["object_in_path"], bool)
        assert isinstance(payload["needs_intervention_actual"], bool)
        if payload["notification_subtype"] is not None:
            assert payload["notification_type"] == "verification_request"
            assert payload["notification_subtype"] in notification_subtypes
        if payload["ev_distance"] is not None:
            assert payload["notification_type"] == "emergency_vehicle_alert"
            assert payload["ev_distance"] >= 0


def test_build_headers_includes_api_key_only_when_configured():
    assert stream_demo.build_headers("") == {"Content-Type": "application/json"}
    assert stream_demo.build_headers(None) == {"Content-Type": "application/json"}
    assert stream_demo.build_headers("secret-key") == {
        "Content-Type": "application/json",
        "X-API-Key": "secret-key",
    }


def test_count_zero_help_text_is_clear(capsys):
    with pytest.raises(SystemExit) as exc:
        stream_demo.parse_args(["--help"])

    assert exc.value.code == 0
    assert "0 means run until interrupted" in capsys.readouterr().out


def test_error_formatting_includes_status_and_backend_detail():
    response = FakeResponse(
        status_code=422,
        reason="Unprocessable Entity",
        json_data={
            "detail": [
                {
                    "loc": ["body", "speed"],
                    "msg": "Input should be greater than or equal to 0",
                }
            ]
        },
    )

    message = stream_demo.format_response_error(
        response,
        "http://localhost:8000/api/predict",
    )

    assert "422" in message
    assert "Unprocessable Entity" in message
    assert "body.speed: Input should be greater than or equal to 0" in message


def test_finite_streaming_posts_exactly_count_attempts():
    payloads = _payloads(seed=5, count=3)
    session = FakeSession([_success_response(payload) for payload in payloads])
    sleeps = []
    output = StringIO()

    summary = stream_demo.stream_predictions(
        payloads=iter(payloads),
        session=session,
        api_base="http://localhost:8000",
        api_key="",
        count=3,
        interval=0.25,
        timeout=5,
        sleep_fn=sleeps.append,
        output=output,
    )

    assert len(session.posts) == 3
    assert summary.attempts == 3
    assert summary.successes == 3
    assert summary.failures == 0
    assert sleeps == [0.25, 0.25]
    assert "status=200" in output.getvalue()


def test_api_base_trailing_slash_does_not_double_slash_predict_url():
    payload = _payloads(seed=6, count=1)[0]
    session = FakeSession([_success_response(payload)])

    stream_demo.stream_predictions(
        payloads=iter([payload]),
        session=session,
        api_base="http://localhost:8000/",
        api_key="",
        count=1,
        interval=0,
        timeout=5,
        sleep_fn=lambda _seconds: None,
        output=StringIO(),
    )

    assert session.posts[0]["url"] == "http://localhost:8000/api/predict"


@pytest.mark.parametrize(
    ("status_code", "reason", "detail"),
    [
        (401, "Unauthorized", "Invalid or missing API key"),
        (
            422,
            "Unprocessable Entity",
            [{"loc": ["body", "speed"], "msg": "Input should be greater than or equal to 0"}],
        ),
    ],
)
def test_auth_or_validation_errors_exit_clearly(status_code, reason, detail):
    payload = _payloads(seed=9, count=1)[0]
    session = FakeSession([
        FakeResponse(
            status_code=status_code,
            reason=reason,
            json_data={"detail": detail},
        )
    ])

    with pytest.raises(stream_demo.FatalStreamError) as exc:
        stream_demo.stream_predictions(
            payloads=iter([payload]),
            session=session,
            api_base="http://localhost:8000",
            api_key="secret",
            count=5,
            interval=0,
            timeout=5,
            sleep_fn=lambda _seconds: None,
            output=StringIO(),
        )

    assert len(session.posts) == 1
    assert str(status_code) in str(exc.value)
    assert reason in str(exc.value)
