
"""Step 106: deterministic adversarial search for MoA Omega scheduling.

Find a small legal Omega communication instance for which the
Step 105 exact optimizer predicts a strictly smaller communication
time than the Step 104 contention-aware greedy heuristic.

All times refer to the Step 103 logical directed-link contention
model, not measured physical network performance.

No changes are made to the schedulers or existing tests.
"""

import argparse
import random
from math import isclose

from moa_machine import Work
from test_attention_omega_topology import make_plan
from moa_attention_omega_topology import derive_omega_topology
from moa_attention_port_schedule import schedule_omega_ports
from moa_attention_port_schedule_audit import audit_omega_port_schedule
from moa_attention_communication_time import (
    OmegaLinkParameters,
)
from moa_attention_contention import predict_omega_contention
from moa_attention_contention_audit import audit_omega_contention
from moa_attention_contention_schedule import schedule_omega_contention
from moa_attention_exact_schedule import (
    OmegaExactSearchLimit,
    schedule_omega_exact,
)


SHAPES = (
    (2,),
    (3,),
    (2, 2),
    (2, 3),
)

OWNERSHIP_PATTERNS = (
    "uniform",
    "first_heavy",
    "last_heavy",
    "alternating",
)

PORT_CHOICES = (1, 2)

ALPHAS = (
    0.0,
    0.1e-6,
    0.5e-6,
    2e-6,
    5e-6,
    20e-6,
)

BETAS = (
    0.1e9,
    0.5e9,
    1e9,
    5e9,
    10e9,
    50e9,
)


def ownership_counts(shape, pattern):
    """Construct deterministic logical ownership distributions."""

    size = 1
    for dimension in shape:
        size *= dimension

    if pattern == "uniform":
        return (1,) * size

    if pattern == "first_heavy":
        return (4,) + (1,) * (size - 1)

    if pattern == "last_heavy":
        return (1,) * (size - 1) + (4,)

    if pattern == "alternating":
        return tuple(
            3 if index % 2 == 0 else 1
            for index in range(size)
        )

    raise ValueError("unknown ownership pattern")


def make_instance(shape, counts):
    """Construct a MoA plan and derived Omega topology."""

    plan = make_plan(
        Work(sum(counts), 5, 4, 3),
        shape,
        counts,
    )

    topology = derive_omega_topology(plan)

    return plan, topology


def heterogeneous_links(topology, seed, trial):
    """Assign reproducible positive-bandwidth directed links."""

    rng = random.Random(
        seed * 1000003 + trial * 9176
    )

    edges = sorted({
        (parent, child)
        for route in topology.routes
        for parent, child in route.edges
    })

    links = {}

    for edge in edges:
        links[edge] = OmegaLinkParameters(
            rng.choice(ALPHAS),
            rng.choice(BETAS),
        )

    return links


def verify_schedule(plan, topology, links, schedule):
    """Independently audit legality and predicted time."""

    audit_omega_port_schedule(
        plan,
        topology,
        schedule,
    )

    prediction = predict_omega_contention(
        plan,
        topology,
        schedule,
        links,
    )

    audit_omega_contention(
        plan,
        topology,
        schedule,
        links,
        prediction,
    )

    return prediction.total_seconds


def display_schedule(label, schedule, links):
    """Print the exact transfers and stage contention costs."""

    print(label)
    print("  stage_count:", schedule.stage_count)

    for stage_number, stage in enumerate(schedule.stages):
        traffic = {}

        for transfer in stage:
            edge = (transfer.parent, transfer.child)
            traffic[edge] = (
                traffic.get(edge, 0)
                + transfer.payload_bytes
            )

        stage_seconds = max(
            (
                links[edge].alpha
                + nbytes / links[edge].beta
                for edge, nbytes in traffic.items()
            ),
            default=0.0,
        )

        print(
            "  stage:",
            stage_number,
            "time_us:",
            stage_seconds * 1e6,
        )

        for transfer in stage:
            print(
                "    owner:",
                transfer.owner,
                "parent:",
                transfer.parent,
                "child:",
                transfer.child,
                "bytes:",
                transfer.payload_bytes,
            )


def search(seed, trials, max_transfers, max_states,
           max_transitions):
    """Search bounded configurations for a strict improvement."""

    examined = 0
    skipped = 0
    equal = 0

    for shape in SHAPES:
        for pattern in OWNERSHIP_PATTERNS:
            counts = ownership_counts(shape, pattern)

            plan, topology = make_instance(
                shape, counts
            )

            for ports in PORT_CHOICES:
                baseline = schedule_omega_ports(
                    topology, ports
                )

                transfer_count = sum(
                    len(stage)
                    for stage in baseline.stages
                )

                if transfer_count > max_transfers:
                    skipped += trials
                    continue

                for trial in range(trials):
                    links = heterogeneous_links(
                        topology, seed, trial
                    )

                    try:
                        heuristic = schedule_omega_contention(
                            plan,
                            topology,
                            links,
                            outgoing_ports=ports,
                        )

                        exact = schedule_omega_exact(
                            plan,
                            topology,
                            links,
                            outgoing_ports=ports,
                            max_transfers=max_transfers,
                            max_states=max_states,
                            max_transitions=max_transitions,
                        )

                    except OmegaExactSearchLimit:
                        skipped += 1
                        continue

                    examined += 1

                    heuristic_time = verify_schedule(
                        plan,
                        topology,
                        links,
                        heuristic.selected,
                    )

                    exact_time = verify_schedule(
                        plan,
                        topology,
                        links,
                        exact.optimal,
                    )

                    tolerance = max(
                        1e-15,
                        1e-12 * max(
                            abs(heuristic_time),
                            abs(exact_time),
                        ),
                    )

                    if exact_time > heuristic_time + tolerance:
                        raise AssertionError(
                            "exact search worse than heuristic"
                        )

                    if exact_time < heuristic_time - tolerance:
                        print("\nSTRICT_IMPROVEMENT_FOUND")
                        print("seed:", seed)
                        print("trial:", trial)
                        print("shape:", shape)
                        print("counts:", counts)
                        print("outgoing_ports:", ports)
                        print("transfers:", transfer_count)
                        print("heuristic_strategy:", heuristic.strategy)
                        print("heuristic_us:", heuristic_time * 1e6)
                        print("exact_us:", exact_time * 1e6)
                        print(
                            "improvement_us:",
                            (heuristic_time - exact_time) * 1e6,
                        )
                        print(
                            "relative_improvement_percent:",
                            100.0 * (
                                heuristic_time - exact_time
                            ) / heuristic_time,
                        )
                        print(
                            "states_examined:",
                            exact.states_examined,
                        )
                        print(
                            "transitions_examined:",
                            exact.transitions_examined,
                        )

                        print("directed_links:")
                        for edge in sorted(links):
                            parameter = links[edge]
                            print(
                                " ", edge,
                                "alpha:", parameter.alpha,
                                "beta:", parameter.beta,
                            )

                        display_schedule(
                            "HEURISTIC_SCHEDULE",
                            heuristic.selected,
                            links,
                        )

                        display_schedule(
                            "EXACT_SCHEDULE",
                            exact.optimal,
                            links,
                        )

                        print("RESULT: PASS")
                        return True

                    if isclose(
                        exact_time,
                        heuristic_time,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    ):
                        equal += 1

    print("\nSEARCH_COMPLETE")
    print("examined:", examined)
    print("skipped:", skipped)
    print("equal_within_tolerance:", equal)
    print("strict_improvement_found:", False)

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Step 106 MoA Omega adversarial search"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=106,
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max-transfers",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--max-transitions",
        type=int,
        default=1000000,
    )

    args = parser.parse_args()

    if (
        args.trials < 1
        or args.max_transfers < 1
        or args.max_states < 1
        or args.max_transitions < 1
    ):
        parser.error("all search limits must be positive")

    search(
        args.seed,
        args.trials,
        args.max_transfers,
        args.max_states,
        args.max_transitions,
    )


if __name__ == "__main__":
    main()
