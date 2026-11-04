
"""Regression tests for MoA Omega communication-time prediction.

Tests:
  - Uniform directed-link timing
  - Heterogeneous directed-link timing
  - Different outgoing-port constraints
  - Singleton machines with zero communication
  - Independent prediction audit
  - Invalid link parameters and missing links
  - Adversarial corruption of prediction certificates

The model predicts conditional communication time on an abstract
shaped machine. These are not hardware timing measurements.
"""

import unittest

from dataclasses import replace
from math import inf, nan

from moa_machine import Work

from moa_attention_omega_topology import (
    derive_omega_topology,
)

from moa_attention_port_schedule import (
    schedule_omega_ports,
)

from moa_attention_communication_time import (
    OmegaLinkParameters,
    uniform_omega_links,
    predict_omega_communication_time,
)

from moa_attention_communication_time_audit import (
    audit_omega_communication_time,
)

from test_attention_omega_topology import make_plan


ALPHA = 2.0e-6
BETA = 10.0e9


def make_case(
    work=None,
    machine_shape=(2, 3),
    query_counts=(1, 2, 1, 1, 1, 1),
    outgoing_ports=1,
):
    if work is None:
        work = Work(7, 5, 4, 3)

    plan = make_plan(
        work,
        machine_shape,
        query_counts,
    )

    topology = derive_omega_topology(plan)

    schedule = schedule_omega_ports(
        topology,
        outgoing_ports,
    )

    links = uniform_omega_links(
        topology,
        ALPHA,
        BETA,
    )

    return plan, topology, schedule, links


def predict_and_audit(plan, topology, schedule, links):
    prediction = predict_omega_communication_time(
        plan,
        topology,
        schedule,
        links,
    )

    assert audit_omega_communication_time(
        plan,
        topology,
        schedule,
        links,
        prediction,
    )

    return prediction


class TestOmegaCommunicationTime(unittest.TestCase):

    def test_uniform_prediction(self):
        plan, topology, schedule, links = make_case()

        prediction = predict_and_audit(
            plan, topology, schedule, links
        )

        self.assertEqual(prediction.stage_count, 11)
        self.assertEqual(prediction.total_link_bytes, 840)

        self.assertAlmostEqual(
            prediction.total_time * 1.0e6,
            22.036,
            places=8,
        )

    def test_uniform_prediction_matches_stage_formula(self):
        plan, topology, schedule, links = make_case()

        prediction = predict_and_audit(
            plan, topology, schedule, links
        )

        expected = []

        for stage in schedule.stages:
            expected.append(
                max(
                    (
                        ALPHA
                        + transfer.payload_bytes / BETA
                        for transfer in stage
                    ),
                    default=0.0,
                )
            )

        self.assertEqual(
            len(prediction.stage_times),
            len(expected),
        )

        for observed, computed in zip(
            prediction.stage_times,
            expected,
        ):
            self.assertAlmostEqual(
                observed,
                computed,
                places=14,
            )

    def test_heterogeneous_slow_link(self):
        plan, topology, schedule, links = make_case()

        baseline = predict_and_audit(
            plan, topology, schedule, links
        )

        changed_links = dict(links)

        edge = next(iter(changed_links))

        changed_links[edge] = OmegaLinkParameters(
            8.0e-6,
            1.0e9,
        )

        changed = predict_and_audit(
            plan,
            topology,
            schedule,
            changed_links,
        )

        self.assertGreater(
            changed.total_time,
            baseline.total_time,
        )

        self.assertEqual(
            changed.total_link_bytes,
            baseline.total_link_bytes,
        )

        self.assertAlmostEqual(
            changed.total_time * 1.0e6,
            34.0744,
            places=7,
        )

    def test_ports_change_stages_and_time(self):
        results = {}

        for ports in (1, 2, 100):
            plan, topology, schedule, links = make_case(
                outgoing_ports=ports
            )

            results[ports] = predict_and_audit(
                plan, topology, schedule, links
            )

        self.assertEqual(
            [results[p].stage_count for p in (1, 2, 100)],
            [11, 6, 3],
        )

        self.assertEqual(
            {
                results[p].total_link_bytes
                for p in (1, 2, 100)
            },
            {840},
        )

        self.assertGreater(
            results[1].total_time,
            results[2].total_time,
        )

        self.assertGreater(
            results[2].total_time,
            results[100].total_time,
        )

    def test_singleton_machine(self):
        plan, topology, schedule, links = make_case(
            work=Work(4, 3, 2, 5),
            machine_shape=(1,),
            query_counts=(4,),
        )

        prediction = predict_and_audit(
            plan, topology, schedule, links
        )

        self.assertEqual(links, {})
        self.assertEqual(prediction.stage_count, 0)
        self.assertEqual(prediction.stage_times, ())
        self.assertEqual(prediction.total_link_bytes, 0)
        self.assertEqual(prediction.total_time, 0.0)

    def test_idle_machine_components(self):
        plan, topology, schedule, links = make_case(
            work=Work(3, 6, 2, 4),
            machine_shape=(2, 3),
            query_counts=(3, 0, 0, 0, 0, 0),
        )

        prediction = predict_and_audit(
            plan, topology, schedule, links
        )

        self.assertEqual(prediction.stage_count, 3)
        self.assertEqual(prediction.total_link_bytes, 480)

    def test_multidimensional_machine(self):
        plan, topology, schedule, links = make_case(
            work=Work(8, 4, 3, 2),
            machine_shape=(2, 2, 2),
            query_counts=(1, 1, 1, 1, 1, 1, 1, 1),
            outgoing_ports=2,
        )

        prediction = predict_and_audit(
            plan, topology, schedule, links
        )

        self.assertEqual(prediction.stage_count, 6)
        self.assertEqual(prediction.total_link_bytes, 896)


class TestInvalidLinkParameters(unittest.TestCase):

    def setUp(self):
        (
            self.plan,
            self.topology,
            self.schedule,
            self.links,
        ) = make_case()

    def test_negative_latency(self):
        with self.assertRaises(ValueError):
            uniform_omega_links(
                self.topology, -1.0, BETA
            )

    def test_zero_bandwidth(self):
        with self.assertRaises(ValueError):
            uniform_omega_links(
                self.topology, ALPHA, 0.0
            )

    def test_negative_bandwidth(self):
        with self.assertRaises(ValueError):
            uniform_omega_links(
                self.topology, ALPHA, -1.0
            )

    def test_nan_latency(self):
        with self.assertRaises(ValueError):
            uniform_omega_links(
                self.topology, nan, BETA
            )

    def test_infinite_bandwidth(self):
        with self.assertRaises(ValueError):
            uniform_omega_links(
                self.topology, ALPHA, inf
            )

    def test_boolean_latency(self):
        with self.assertRaises(ValueError):
            uniform_omega_links(
                self.topology, True, BETA
            )

    def test_boolean_bandwidth(self):
        with self.assertRaises(ValueError):
            uniform_omega_links(
                self.topology, ALPHA, True
            )

    def test_missing_link(self):
        changed = dict(self.links)
        changed.pop(next(iter(changed)))

        with self.assertRaises(ValueError):
            predict_omega_communication_time(
                self.plan,
                self.topology,
                self.schedule,
                changed,
            )

    def test_unexpected_link(self):
        changed = dict(self.links)

        changed[
            ((999, 999), (999, 998))
        ] = OmegaLinkParameters(ALPHA, BETA)

        with self.assertRaises(ValueError):
            predict_omega_communication_time(
                self.plan,
                self.topology,
                self.schedule,
                changed,
            )

    def test_invalid_link_object(self):
        changed = dict(self.links)
        changed[next(iter(changed))] = (ALPHA, BETA)

        with self.assertRaises(TypeError):
            predict_omega_communication_time(
                self.plan,
                self.topology,
                self.schedule,
                changed,
            )

    def test_nonfinite_transfer_duration(self):
        changed = dict(self.links)
        edge = next(iter(changed))

        changed[edge] = OmegaLinkParameters(
            1.0e308,
            1.0e-308,
        )

        with self.assertRaises(ValueError):
            predict_omega_communication_time(
                self.plan,
                self.topology,
                self.schedule,
                changed,
            )


class TestAdversarialPredictionAudit(unittest.TestCase):

    def setUp(self):
        (
            self.plan,
            self.topology,
            self.schedule,
            self.links,
        ) = make_case()

        self.prediction = predict_and_audit(
            self.plan,
            self.topology,
            self.schedule,
            self.links,
        )

    def reject(self, changed):
        with self.assertRaises(ValueError):
            audit_omega_communication_time(
                self.plan,
                self.topology,
                self.schedule,
                self.links,
                changed,
            )

    def test_wrong_machine_shape(self):
        self.reject(
            replace(
                self.prediction,
                machine_shape=(3, 2),
            )
        )

    def test_wrong_port_count(self):
        self.reject(
            replace(
                self.prediction,
                outgoing_ports=100,
            )
        )

    def test_wrong_stage_count(self):
        self.reject(
            replace(
                self.prediction,
                stage_count=10,
            )
        )

    def test_missing_stage_time(self):
        self.reject(
            replace(
                self.prediction,
                stage_times=self.prediction.stage_times[:-1],
            )
        )

    def test_wrong_link_bytes(self):
        self.reject(
            replace(
                self.prediction,
                total_link_bytes=841,
            )
        )

    def test_corrupted_stage_time(self):
        changed = list(self.prediction.stage_times)

        changed[0] += 1.0e-6

        self.reject(
            replace(
                self.prediction,
                stage_times=tuple(changed),
            )
        )

    def test_corrupted_total_time(self):
        self.reject(
            replace(
                self.prediction,
                total_time=self.prediction.total_time + 1.0e-6,
            )
        )

    def test_negative_stage_time(self):
        changed = list(self.prediction.stage_times)

        changed[0] = -1.0

        self.reject(
            replace(
                self.prediction,
                stage_times=tuple(changed),
            )
        )

    def test_nan_stage_time(self):
        changed = list(self.prediction.stage_times)

        changed[0] = nan

        self.reject(
            replace(
                self.prediction,
                stage_times=tuple(changed),
            )
        )

    def test_infinite_total_time(self):
        self.reject(
            replace(
                self.prediction,
                total_time=inf,
            )
        )

    def test_boolean_stage_count(self):
        self.reject(
            replace(
                self.prediction,
                stage_count=True,
            )
        )

    def test_stage_times_wrong_type(self):
        self.reject(
            replace(
                self.prediction,
                stage_times=list(self.prediction.stage_times),
            )
        )

    def test_corrupt_stage_and_total_together(self):
        changed = list(self.prediction.stage_times)

        changed[0] += 1.0e-6

        # Internal accounting remains consistent, but the
        # independently reconstructed model must reject it.
        self.reject(
            replace(
                self.prediction,
                stage_times=tuple(changed),
                total_time=self.prediction.total_time + 1.0e-6,
            )
        )

    def test_wrong_link_model(self):
        changed_links = dict(self.links)

        edge = next(iter(changed_links))

        changed_links[edge] = OmegaLinkParameters(
            8.0e-6,
            1.0e9,
        )

        with self.assertRaises(ValueError):
            audit_omega_communication_time(
                self.plan,
                self.topology,
                self.schedule,
                changed_links,
                self.prediction,
            )


if __name__ == "__main__":
    unittest.main()
