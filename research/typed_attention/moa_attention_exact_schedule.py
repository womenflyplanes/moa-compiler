
"""Step 105: exact small-instance MoA Omega schedule optimization.

Find a minimum predicted communication-time schedule over every legal
nonempty-stage schedule of the fixed Omega topology transfer multiset.

Model:
  - Exact Omega owners, routes, transfers, and bytes are preserved.
  - Receive-before-forward is enforced at stage boundaries.
  - Each sender obeys its outgoing-port constraint.
  - All nonempty feasible subsets of eligible transfers are considered.
  - Stage cost follows Step 103 directed-link contention.
  - Step 104 supplies an independently audited upper bound.
  - Dynamic programming solves the finite completed-transfer state space.

Optimality is conditional on the specified logical communication model,
not on physical MPI or DGX execution time.

A search-limit failure raises an exception and never returns an
optimality claim.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import fsum, isfinite

from moa_attention_port_schedule import (
    OmegaPortSchedule,
    OmegaTransfer,
)
from moa_attention_port_schedule_audit import (
    audit_omega_port_schedule,
)
from moa_attention_contention import (
    predict_omega_contention,
)
from moa_attention_contention_audit import (
    audit_omega_contention,
)
from moa_attention_contention_schedule import (
    schedule_omega_contention,
)


class OmegaExactSearchLimit(RuntimeError):
    """Exact search could not finish within its stated limits."""


@dataclass(frozen=True)
class OmegaExactScheduleResult:
    """Completed exact-search result and verification metadata."""

    baseline: OmegaPortSchedule
    heuristic: OmegaPortSchedule
    optimal: OmegaPortSchedule

    baseline_prediction: object
    heuristic_prediction: object
    optimal_prediction: object

    states_examined: int
    transitions_examined: int
    optimality_proven: bool
    search_space: str

    @property
    def optimal_seconds(self):
        return self.optimal_prediction.total_seconds

    @property
    def heuristic_seconds(self):
        return self.heuristic_prediction.total_seconds

    @property
    def improvement_seconds(self):
        return self.heuristic_seconds - self.optimal_seconds

    def verify(self):
        if not self.optimality_proven:
            raise ValueError("incomplete exact search")

        if self.states_examined < 1:
            raise ValueError("no states examined")

        if self.transitions_examined < 0:
            raise ValueError("invalid transition count")

        for prediction in (
            self.baseline_prediction,
            self.heuristic_prediction,
            self.optimal_prediction,
        ):
            if not isfinite(prediction.total_seconds):
                raise ValueError("nonfinite predicted time")

        tolerance = 1.0e-12 * max(
            self.heuristic_seconds,
            self.optimal_seconds,
            1.0e-12,
        )

        if self.optimal_seconds > self.heuristic_seconds + tolerance:
            raise ValueError("exact result worse than heuristic")

        if (
            self.optimal.total_link_bytes
            != self.heuristic.total_link_bytes
        ):
            raise ValueError("communication volume changed")

        return True


def _transfer_key(transfer):
    """Canonical deterministic transfer ordering."""

    return (
        transfer.owner,
        transfer.parent,
        transfer.child,
        transfer.payload_bytes,
    )


def _stage_cost(transfers, links):
    """Step 103 contention cost for one nonempty stage."""

    grouped = {}

    for transfer in transfers:
        link = (transfer.parent, transfer.child)

        grouped[link] = (
            grouped.get(link, 0)
            + transfer.payload_bytes
        )

    durations = []

    for link in sorted(grouped):
        parameters = links[link]

        duration = (
            parameters.alpha
            + grouped[link] / parameters.beta
        )

        durations.append(duration)

    return max(durations, default=0.0)


def _legal_stage(indices, transfers, outgoing_ports):
    """Check outgoing-port and duplicate-delivery constraints."""

    outgoing = {}
    deliveries = set()

    for index in indices:
        transfer = transfers[index]

        parent = transfer.parent
        delivery = (transfer.owner, transfer.child)

        if delivery in deliveries:
            return False

        deliveries.add(delivery)

        outgoing[parent] = outgoing.get(parent, 0) + 1

        if outgoing[parent] > outgoing_ports:
            return False

    return True


def _make_schedule(
    machine_shape,
    outgoing_ports,
    stages,
):
    """Construct a certificate from exact Omega transfers."""

    stages = tuple(tuple(stage) for stage in stages)

    schedule = OmegaPortSchedule(
        machine_shape=machine_shape,
        outgoing_ports=outgoing_ports,
        stages=stages,
        stage_count=len(stages),
        total_link_bytes=sum(
            transfer.payload_bytes
            for stage in stages
            for transfer in stage
        ),
    )

    schedule.verify()

    return schedule


def schedule_omega_exact(
    plan,
    topology,
    links,
    outgoing_ports=1,
    max_transfers=16,
    max_states=100000,
    max_transitions=1000000,
):
    """Find the exact minimum predicted-time Omega schedule.

    Exhaustively searches legal nonempty-stage schedules of the
    fixed Omega transfer multiset.

    Search states are completed-transfer bitmasks. Each transition
    selects a legal nonempty subset of transfers eligible at stage
    entry. State cost is the minimum predicted remaining time.

    Limits are strict. Exceeding any limit raises
    OmegaExactSearchLimit; no incomplete result is certified.
    """

    for name, value in (
        ("max_transfers", max_transfers),
        ("max_states", max_states),
        ("max_transitions", max_transitions),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")

    heuristic_result = schedule_omega_contention(
        plan,
        topology,
        links,
        outgoing_ports=outgoing_ports,
    )

    baseline = heuristic_result.baseline
    heuristic = heuristic_result.selected

    transfers = tuple(
        sorted(
            (
                transfer
                for stage in baseline.stages
                for transfer in stage
            ),
            key=_transfer_key,
        )
    )

    count = len(transfers)

    if count > max_transfers:
        raise OmegaExactSearchLimit(
            "transfer count exceeds exact-search limit: "
            + str(count)
        )

    if len(set(_transfer_key(t) for t in transfers)) != count:
        raise ValueError(
            "duplicate transfer occurrences require "
            "an occurrence-aware exact-search representation"
        )

    owners = tuple(
        sorted(route.owner for route in topology.routes)
    )

    full_mask = (1 << count) - 1

    states_examined = 0
    transitions_examined = 0

    # A completed transfer makes its child eligible as a
    # subsequent-stage forwarding parent for the same owner.
    def eligible_indices(mask):
        reached = {
            owner: {owner}
            for owner in owners
        }

        for index, transfer in enumerate(transfers):
            if mask & (1 << index):
                reached[transfer.owner].add(transfer.child)

        eligible = []

        for index, transfer in enumerate(transfers):
            if mask & (1 << index):
                continue

            if transfer.parent not in reached[transfer.owner]:
                continue

            if transfer.child in reached[transfer.owner]:
                continue

            eligible.append(index)

        return tuple(eligible)

    @lru_cache(maxsize=None)
    def solve(mask):
        nonlocal states_examined
        nonlocal transitions_examined

        states_examined += 1

        if states_examined > max_states:
            raise OmegaExactSearchLimit(
                "exact-search state limit exceeded"
            )

        if mask == full_mask:
            return (0.0, ())

        eligible = eligible_indices(mask)

        if not eligible:
            raise ValueError(
                "incomplete Omega state has no eligible transfer"
            )

        best_cost = float("inf")
        best_stages = None

        # Enumerate every nonempty eligible subset.
        # The deterministic iteration order also supplies
        # a reproducible tie-breaking rule.
        for size in range(1, len(eligible) + 1):
            for subset in combinations(eligible, size):
                if not _legal_stage(
                    subset,
                    transfers,
                    outgoing_ports,
                ):
                    continue

                transitions_examined += 1

                if transitions_examined > max_transitions:
                    raise OmegaExactSearchLimit(
                        "exact-search transition limit exceeded"
                    )

                next_mask = mask

                for index in subset:
                    next_mask |= 1 << index

                stage = tuple(
                    transfers[index]
                    for index in subset
                )

                stage_seconds = _stage_cost(stage, links)

                remaining_seconds, remaining_stages = solve(
                    next_mask
                )

                total_seconds = fsum(
                    (stage_seconds, remaining_seconds)
                )

                if total_seconds < best_cost:
                    best_cost = total_seconds
                    best_stages = (
                        (stage,) + remaining_stages
                    )

        if best_stages is None:
            raise ValueError(
                "no legal continuation of Omega schedule"
            )

        return (best_cost, best_stages)

    computed_seconds, optimal_stages = solve(0)

    optimal = _make_schedule(
        topology.machine_shape,
        outgoing_ports,
        optimal_stages,
    )

    # Independent legality audit.
    audit_omega_port_schedule(
        plan,
        topology,
        optimal,
    )

    # Existing Step 103 prediction and independent audit.
    optimal_prediction = predict_omega_contention(
        plan,
        topology,
        optimal,
        links,
    )

    audit_omega_contention(
        plan,
        topology,
        optimal,
        links,
        optimal_prediction,
    )

    # Reconcile the dynamic-programming objective with the
    # existing independently audited prediction.
    tolerance = 1.0e-12 * max(
        abs(computed_seconds),
        abs(optimal_prediction.total_seconds),
        1.0e-12,
    )

    if (
        abs(
            computed_seconds
            - optimal_prediction.total_seconds
        ) > tolerance
    ):
        raise ValueError(
            "exact-search objective differs from "
            "audited Step 103 prediction"
        )

    result = OmegaExactScheduleResult(
        baseline=baseline,
        heuristic=heuristic,
        optimal=optimal,
        baseline_prediction=(
            heuristic_result.baseline_prediction
        ),
        heuristic_prediction=(
            heuristic_result.selected_prediction
        ),
        optimal_prediction=optimal_prediction,
        states_examined=states_examined,
        transitions_examined=transitions_examined,
        optimality_proven=True,
        search_space=(
            "all legal nonempty-stage schedules of the "
            "fixed Omega transfer multiset under the "
            "Step 103 directed-link contention model"
        ),
    )

    result.verify()

    return result
