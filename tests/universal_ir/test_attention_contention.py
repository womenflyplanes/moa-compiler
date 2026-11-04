"""Step 103H: regression tests for directed-link contention.

Tests mathematical accounting, heterogeneous links, critical-stage
behavior, determinism, and independent certificate auditing.
"""

from dataclasses import replace
import math
import unittest

from moa_machine import Work
from test_attention_omega_topology import make_plan
from moa_attention_omega_topology import derive_omega_topology
from moa_attention_port_schedule import schedule_omega_ports
from moa_attention_communication_time import (
    OmegaLinkParameters,
    uniform_omega_links,
    predict_omega_communication_time,
)
from moa_attention_contention import predict_omega_contention
from moa_attention_contention_audit import audit_omega_contention


class TestOmegaContention(unittest.TestCase):

    def setUp(self):
        self.plan = make_plan(
            Work(7, 5, 4, 3),
            (2, 3),
            (1, 2, 1, 1, 1, 1),
        )
        self.topology = derive_omega_topology(self.plan)
        self.schedule = schedule_omega_ports(self.topology, 2)
        self.links = uniform_omega_links(
            self.topology, 2e-6, 10e9
        )

    def predict(self, links=None):
        if links is None:
            links = self.links
        return predict_omega_contention(
            self.plan,
            self.topology,
            self.schedule,
            links,
        )

    def audit(self, prediction, links=None):
        if links is None:
            links = self.links
        try:
            return audit_omega_contention(
                self.plan,
                self.topology,
                self.schedule,
                links,
                prediction,
            )
        except ValueError:
            return False

    def test_uniform_prediction_and_audit(self):
        prediction = self.predict()

        self.assertTrue(self.audit(prediction))
        self.assertEqual(
            prediction.stage_count,
            self.schedule.stage_count,
        )
        self.assertEqual(
            prediction.total_link_bytes,
            self.schedule.total_link_bytes,
        )
        self.assertEqual(prediction.total_link_bytes, 840)
        self.assertTrue(
            math.isclose(
                prediction.total_seconds,
                12.0216e-6,
                rel_tol=1e-10,
                abs_tol=1e-15,
            )
        )

    def test_contended_link_identification(self):
        prediction = self.predict()

        shared = [
            link
            for link in prediction.stages[1]
            if link.parent == (1, 1)
            and link.child == (1, 2)
        ]

        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0].transfer_count, 2)
        self.assertEqual(shared[0].payload_bytes, 48)

        self.assertTrue(
            math.isclose(
                shared[0].predicted_seconds,
                2.0048e-6,
                rel_tol=1e-10,
                abs_tol=1e-15,
            )
        )

    def test_uniform_contention_does_not_extend_stage(self):
        original = predict_omega_communication_time(
            self.plan,
            self.topology,
            self.schedule,
            self.links,
        )
        contention = self.predict()

        self.assertTrue(self.audit(contention))
        self.assertTrue(
            math.isclose(
                original.total_time,
                contention.total_seconds,
                rel_tol=1e-10,
                abs_tol=1e-15,
            )
        )

    def test_heterogeneous_contention_extends_stage(self):
        links = dict(self.links)
        links[((1, 1), (1, 2))] = OmegaLinkParameters(
            2e-6, 1e9
        )

        original = predict_omega_communication_time(
            self.plan,
            self.topology,
            self.schedule,
            links,
        )
        contention = self.predict(links)

        self.assertTrue(self.audit(contention, links))
        self.assertGreater(
            contention.total_seconds,
            original.total_time,
        )

        self.assertTrue(
            math.isclose(
                original.total_time,
                12.0408e-6,
                rel_tol=1e-10,
                abs_tol=1e-15,
            )
        )
        self.assertTrue(
            math.isclose(
                contention.total_seconds,
                12.0648e-6,
                rel_tol=1e-10,
                abs_tol=1e-15,
            )
        )

    def test_every_stage_has_correct_link_accounting(self):
        prediction = self.predict()

        for stage_index, stage in enumerate(prediction.stages):
            transfers = self.schedule.stages[stage_index]

            expected_bytes = sum(
                transfer.payload_bytes
                for transfer in transfers
            )
            actual_bytes = sum(
                link.payload_bytes for link in stage
            )

            self.assertEqual(actual_bytes, expected_bytes)
            self.assertEqual(
                sum(link.transfer_count for link in stage),
                len(transfers),
            )

            expected_stage_time = max(
                (link.predicted_seconds for link in stage),
                default=0.0,
            )
            self.assertTrue(
                math.isclose(
                    prediction.stage_times[stage_index],
                    expected_stage_time,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            )

    def test_prediction_is_deterministic(self):
        first = self.predict()
        second = self.predict()

        self.assertEqual(first, second)
        self.assertTrue(self.audit(first))

    def test_audit_rejects_corrupted_link_count(self):
        prediction = self.predict()
        stages = [list(stage) for stage in prediction.stages]

        index = next(
            i for i, link in enumerate(stages[1])
            if link.transfer_count > 1
        )

        stages[1][index] = replace(
            stages[1][index],
            transfer_count=3,
        )

        corrupted = replace(
            prediction,
            stages=tuple(tuple(stage) for stage in stages),
        )
        self.assertFalse(self.audit(corrupted))

    def test_audit_rejects_corrupted_payload(self):
        prediction = self.predict()
        stages = [list(stage) for stage in prediction.stages]

        stages[1][0] = replace(
            stages[1][0],
            payload_bytes=stages[1][0].payload_bytes + 8,
        )

        corrupted = replace(
            prediction,
            stages=tuple(tuple(stage) for stage in stages),
        )
        self.assertFalse(self.audit(corrupted))

    def test_audit_rejects_corrupted_link_time(self):
        prediction = self.predict()
        stages = [list(stage) for stage in prediction.stages]

        stages[1][0] = replace(
            stages[1][0],
            predicted_seconds=(
                stages[1][0].predicted_seconds + 1e-6
            ),
        )

        corrupted = replace(
            prediction,
            stages=tuple(tuple(stage) for stage in stages),
        )
        self.assertFalse(self.audit(corrupted))

    def test_audit_rejects_corrupted_stage_time(self):
        prediction = self.predict()
        times = list(prediction.stage_times)
        times[1] += 1e-6

        corrupted = replace(
            prediction,
            stage_times=tuple(times),
        )
        self.assertFalse(self.audit(corrupted))

    def test_audit_rejects_corrupted_total_time(self):
        prediction = self.predict()

        corrupted = replace(
            prediction,
            total_seconds=prediction.total_seconds + 1e-6,
        )
        self.assertFalse(self.audit(corrupted))

    def test_audit_rejects_corrupted_total_bytes(self):
        prediction = self.predict()

        corrupted = replace(
            prediction,
            total_link_bytes=prediction.total_link_bytes + 8,
        )
        self.assertFalse(self.audit(corrupted))

    def test_audit_rejects_missing_link(self):
        prediction = self.predict()
        stages = [list(stage) for stage in prediction.stages]

        stages[1].pop()

        corrupted = replace(
            prediction,
            stages=tuple(tuple(stage) for stage in stages),
        )
        self.assertFalse(self.audit(corrupted))

    def test_audit_rejects_duplicate_link(self):
        prediction = self.predict()
        stages = [list(stage) for stage in prediction.stages]

        stages[1].append(stages[1][0])

        corrupted = replace(
            prediction,
            stages=tuple(tuple(stage) for stage in stages),
        )
        self.assertFalse(self.audit(corrupted))

    def test_audit_rejects_wrong_link_model(self):
        prediction = self.predict()
        modified_links = dict(self.links)

        modified_links[((1, 1), (1, 2))] = (
            OmegaLinkParameters(2e-6, 1e9)
        )

        self.assertFalse(
            self.audit(prediction, modified_links)
        )

    def test_prediction_never_underestimates_original_model(self):
        cases = [
            self.links,
            {
                **self.links,
                ((1, 1), (1, 2)):
                    OmegaLinkParameters(2e-6, 1e9),
            },
        ]

        for links in cases:
            with self.subTest(links=links):
                original = predict_omega_communication_time(
                    self.plan,
                    self.topology,
                    self.schedule,
                    links,
                )
                contention = self.predict(links)

                self.assertTrue(self.audit(contention, links))
                self.assertGreaterEqual(
                    contention.total_seconds + 1e-15,
                    original.total_time,
                )


if __name__ == "__main__":
    unittest.main()
