"""Step 109: scheduled MPI emission regression tests."""

import re
import unittest

from moa_machine import Machine, Work, gamma, iota
from moa_attention_resource_planner import modeled_attention_bytes
from moa_attention_communication_time import OmegaLinkParameters
from moa_attention_integrated_108 import compile_attention_108
from moa_attention_scheduled_mpi_109 import emit_scheduled_mpi_109


class TestScheduledMPI109(unittest.TestCase):

    def compile(self, shape, mode):
        machine = Machine(shape)
        work = Work(machine.tau, 5, 4, 3)

        capacities = tuple(
            modeled_attention_bytes(work, 1)
            for _ in range(machine.tau)
        )

        coords = tuple(iota(machine.shape))

        links = {
            (a, b): OmegaLinkParameters(
                alpha=1e-6, beta=1e9
            )
            for a in coords
            for b in coords
            if sum(abs(x-y) for x, y in zip(a, b)) == 1
        }

        integrated = compile_attention_108(
            work, machine, capacities, links,
            outgoing_ports=2, mode=mode
        )

        self.assertTrue(integrated.verify())

        emission = emit_scheduled_mpi_109(integrated)

        self.assertTrue(emission.verify())

        return machine, integrated, emission

    def test_generated_transfer_correspondence(self):
        cases = (
            ((1,), "heuristic"),
            ((2,), "heuristic"),
            ((2,), "exact"),
            ((3,), "heuristic"),
            ((3,), "exact"),
            ((2, 2), "heuristic"),
            ((2, 2), "exact"),
            ((2, 3), "heuristic"),
        )

        for shape, mode in cases:
            with self.subTest(shape=shape, mode=mode):
                machine, integrated, emission = self.compile(
                    shape, mode
                )

                expected = [
                    (
                        gamma(t.owner, machine.shape),
                        gamma(t.parent, machine.shape),
                        gamma(t.child, machine.shape),
                        t.payload_bytes // 8,
                    )
                    for stage in emission.schedule.stages
                    for t in stage
                ]

                match = re.search(
                    r"static const int transfers\[\]\[4\]"
                    r"\s*=\s*\{(.*?)\};",
                    emission.source,
                    re.S,
                )

                self.assertIsNotNone(match)

                actual = [
                    tuple(map(int, values))
                    for values in re.findall(
                        r"\{\s*(\d+)\s*,\s*(\d+)\s*,"
                        r"\s*(\d+)\s*,\s*(\d+)\s*\}",
                        match.group(1),
                    )
                ]

                # The emitter uses one inert table entry when
                # there are no scheduled transfers.
                if not expected:
                    self.assertEqual(actual, [(0, 0, 0, 0)])
                else:
                    self.assertEqual(actual, expected)

                offsets_match = re.search(
                    r"static const int stage_offsets\[\]"
                    r"\s*=\s*\{([^}]+)\};",
                    emission.source,
                )

                self.assertIsNotNone(offsets_match)

                actual_offsets = tuple(
                    int(x.strip())
                    for x in offsets_match.group(1).split(",")
                )

                expected_offsets = [0]

                for stage in emission.schedule.stages:
                    expected_offsets.append(
                        expected_offsets[-1] + len(stage)
                    )

                self.assertEqual(
                    actual_offsets, tuple(expected_offsets)
                )

                for owner, sender, receiver, count in expected:
                    self.assertEqual(
                        count, emission.counts[owner]
                    )
                    self.assertTrue(
                        0 <= owner < machine.tau
                    )
                    self.assertTrue(
                        0 <= sender < machine.tau
                    )
                    self.assertTrue(
                        0 <= receiver < machine.tau
                    )

                self.assertNotIn(
                    "MPI_Allgatherv(", emission.source
                )

                for operation in (
                    "MPI_Irecv(",
                    "MPI_Isend(",
                    "MPI_Waitall(",
                ):
                    self.assertIn(operation, emission.source)

    def test_stage_buffer_safety(self):
        cases = (
            ((2,), "heuristic"),
            ((2,), "exact"),
            ((3,), "heuristic"),
            ((3,), "exact"),
            ((2, 2), "heuristic"),
            ((2, 2), "exact"),
            ((2, 3), "heuristic"),
        )

        for shape, mode in cases:
            with self.subTest(shape=shape, mode=mode):
                machine, _, emission = self.compile(
                    shape, mode
                )

                for stage in emission.schedule.stages:
                    for rank in range(machine.tau):
                        reads = []
                        writes = []

                        for transfer in stage:
                            owner = gamma(
                                transfer.owner, machine.shape
                            )
                            sender = gamma(
                                transfer.parent, machine.shape
                            )
                            receiver = gamma(
                                transfer.child, machine.shape
                            )

                            begin = emission.displacements[owner]
                            end = begin + emission.counts[owner]

                            if sender == rank:
                                reads.append((begin, end))

                            if receiver == rank:
                                writes.append((begin, end))

                        for a, b in reads:
                            for c, d in writes:
                                self.assertTrue(
                                    b <= c or d <= a
                                )

                        for i, (a, b) in enumerate(writes):
                            for c, d in writes[i+1:]:
                                self.assertTrue(
                                    b <= c or d <= a
                                )


if __name__ == "__main__":
    unittest.main()
