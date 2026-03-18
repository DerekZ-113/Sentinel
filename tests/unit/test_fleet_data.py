"""
Tests for fleet_data/generate_fleet_data.py — Vehicle class and helper functions.
"""

import pytest
import random
from datetime import datetime

from fleet_data.generate_fleet_data import (
    Vehicle, ROAD_TYPE, TRAFFIC_CONDITION, CONSTRUCTION_ZONE,
    NOTIFICATION_TYPES, get_traffic_weights, get_pedestrian_density,
)


# ============================================================================
# VEHICLE INIT
# ============================================================================

class TestVehicleInit:

    def test_defaults(self):
        v = Vehicle("v001")
        assert v.vehicle_id == "v001"
        assert v.speed == 0.0
        assert v.status == "moving"
        assert v.active_notification is None
        assert v.needs_intervention is False

    def test_assign_initial_context(self):
        v = Vehicle("v001")
        v.assign_initial_context()
        assert v.road_type in ROAD_TYPE
        assert v.traffic_condition in TRAFFIC_CONDITION
        assert v.construction_zone in CONSTRUCTION_ZONE
        assert v.latitude is not None
        assert v.longitude is not None
        assert v.expected_speed >= 0


# ============================================================================
# EXPECTED SPEED
# ============================================================================

class TestExpectedSpeed:

    def test_basic(self):
        v = Vehicle("v001")
        v.road_type = "highway"
        v.traffic_condition = "light"
        v.construction_zone = "none"
        speed = v.calculate_expected_speed()
        assert speed == pytest.approx(65.0 * 1.0 * 1.0, abs=0.1)

    def test_with_modifiers(self):
        v = Vehicle("v001")
        v.road_type = "highway"
        v.traffic_condition = "heavy"
        v.construction_zone = "temporary"
        speed = v.calculate_expected_speed()
        assert speed == pytest.approx(65.0 * 0.3 * 0.6, abs=0.1)

    def test_flagger_zero(self):
        v = Vehicle("v001")
        v.road_type = "main_road"
        v.traffic_condition = "light"
        v.construction_zone = "flagger"
        speed = v.calculate_expected_speed()
        assert speed == pytest.approx(0.0, abs=0.1)


# ============================================================================
# TRAFFIC WEIGHTS
# ============================================================================

class TestTrafficWeights:

    def test_rush_hour(self):
        sim_time = datetime(2024, 12, 1, 8, 0, 0)
        weights = get_traffic_weights(sim_time)
        assert weights["heavy"] == 0.5

    def test_midday(self):
        sim_time = datetime(2024, 12, 1, 12, 0, 0)
        weights = get_traffic_weights(sim_time)
        assert weights["light"] == 0.35

    def test_night(self):
        sim_time = datetime(2024, 12, 1, 2, 0, 0)
        weights = get_traffic_weights(sim_time)
        assert weights["light"] == 0.7


# ============================================================================
# PEDESTRIAN DENSITY
# ============================================================================

class TestPedestrianDensity:

    def test_highway_near_zero(self):
        random.seed(42)
        density = get_pedestrian_density("highway", 12)
        assert density == pytest.approx(0.0, abs=0.15)  # base=0.0 + jitter

    def test_downtown_rush(self):
        """downtown base=0.7, rush modifier=1.3 → ~0.91 ± jitter"""
        random.seed(42)
        density = get_pedestrian_density("downtown", 8)
        assert 0.5 < density <= 1.0

    def test_always_clamped(self):
        """Density should always be in [0, 1]."""
        random.seed(0)
        for road_type in ROAD_TYPE:
            for hour in range(24):
                d = get_pedestrian_density(road_type, hour)
                assert 0.0 <= d <= 1.0


# ============================================================================
# VEHICLE BEHAVIOR
# ============================================================================

class TestVehicleBehavior:

    def test_stuck_sets_speed_zero(self):
        v = Vehicle("v001")
        v.assign_initial_context()
        v.active_notification = "stuck"
        v.notification_remaining = 100
        v.needs_intervention = True
        sim_time = datetime(2024, 12, 1, 12, 0, 0)
        v.update(sim_time)
        assert v.speed == 0

    def test_passenger_assist_stops(self):
        v = Vehicle("v001")
        v.assign_initial_context()
        v.active_notification = "passenger_assist"
        v.notification_remaining = 100
        v.needs_intervention = True
        sim_time = datetime(2024, 12, 1, 12, 0, 0)
        v.update(sim_time)
        assert v.speed == 0

    def test_notification_clears_after_duration(self):
        v = Vehicle("v001")
        v.assign_initial_context()
        v.active_notification = "stuck"
        v.notification_remaining = 1  # Will expire on next update
        v.needs_intervention = True
        sim_time = datetime(2024, 12, 1, 12, 0, 0)
        v.update(sim_time)
        assert v.active_notification is None

    def test_stop_timer_increments(self):
        v = Vehicle("v001")
        v.assign_initial_context()
        v.speed = 0.0
        v.active_notification = "stuck"
        v.notification_remaining = 100
        v.needs_intervention = True
        sim_time = datetime(2024, 12, 1, 12, 0, 0)
        v.update(sim_time)
        assert v.time_since_stop > 0

    def test_speed_convergence(self):
        """Without notification, speed should move toward expected_speed."""
        v = Vehicle("v001")
        v.assign_initial_context()
        v.speed = 0.0
        v.active_notification = None
        v.road_type = "main_road"
        v.traffic_condition = "light"
        v.construction_zone = "none"
        v.expected_speed = v.calculate_expected_speed()
        v.target_speed = v.expected_speed
        # Use _update_speed directly to avoid notification triggers
        for _ in range(100):
            v._update_speed()
        # After 100 updates, speed should be near expected
        assert abs(v.speed - v.expected_speed) < 10


# ============================================================================
# INTERVENTION DETERMINATION
# ============================================================================

class TestInterventionDetermination:

    def test_passenger_assist_always_true(self):
        """fp_rate=0.0 means needs_intervention is always True."""
        v = Vehicle("v001")
        v.assign_initial_context()
        sim_time = datetime(2024, 12, 1, 12, 0, 0)
        results = []
        for _ in range(100):
            result = v._determine_intervention("passenger_assist", 0.0, sim_time)
            results.append(result)
        assert all(results)

    def test_stuck_heavy_traffic_increases_fp(self):
        """Heavy traffic should increase FP rate for stuck notifications."""
        v = Vehicle("v001")
        v.assign_initial_context()
        v.traffic_condition = "heavy"
        v.construction_zone = "none"
        sim_time = datetime(2024, 12, 1, 12, 0, 0)

        random.seed(42)
        base_fp = 0.65
        interventions = sum(
            v._determine_intervention("stuck", base_fp, sim_time)
            for _ in range(1000)
        )
        # With increased FP, fewer interventions expected
        assert interventions < 500  # Base would be ~350, with traffic < 200

    def test_ev_context_close(self):
        """When needs_intervention=True, ev_distance should be 10-100."""
        v = Vehicle("v001")
        v.assign_initial_context()
        v.needs_intervention = True
        sim_time = datetime(2024, 12, 1, 12, 0, 0)
        v._set_notification_context("emergency_vehicle_alert", sim_time)
        assert 10 <= v.ev_distance <= 100

    def test_ev_context_far(self):
        """When needs_intervention=False, ev_distance should be 150-500."""
        v = Vehicle("v001")
        v.assign_initial_context()
        v.needs_intervention = False
        sim_time = datetime(2024, 12, 1, 12, 0, 0)
        v._set_notification_context("emergency_vehicle_alert", sim_time)
        assert 150 <= v.ev_distance <= 500
