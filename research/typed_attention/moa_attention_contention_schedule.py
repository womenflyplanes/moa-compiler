
"""Step 104: contention-aware MoA Omega communication scheduling.

Construct deterministic candidate schedules for an existing Omega
communication topology, independently audit their legality, and
compare predicted communication completion times.

The original port-constrained schedule remains the baseline.

Model:
  - Exact Omega topology transfers and payload ownership are preserved.
  - Outgoing-port constraints are preserved.
  - Receive-before-forward dependencies are preserved.
  - Candidate stages are evaluated using Step 103 directed-link
    contention and heterogeneous alpha/beta parameters.
  - Only strictly improving candidates replace the baseline.

This is a deterministic heuristic, not a global optimality proof.
Logical directed links are not a physical DGX network model.
"""

from dataclasses import dataclass
import math

from moa_machine import Machine, gamma

from moa_attention_omega_topology import OmegaTopology
from moa_attention_port_schedule import (
    OmegaPortSchedule,
    schedule_omega_ports,
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


@dataclass(frozen=True)
class OmegaContentionScheduleResult:
    """Audited comparison between baseline and selected schedules."""

    baseline: OmegaPortSchedule
    selected: OmegaPortSchedule
    baseline_prediction: object
    selected_prediction: object
    strategy: str
    candidates_examined: int

    @property
    def baseline_seconds(self):
        return self.baseline_prediction.total_seconds

    @property
    def selected_seconds(self):
        return self.selected_prediction.total_seconds

    @property
    def improvement_seconds(self):
        return self.baseline_seconds - self.selected_seconds

    def verify(self):
        """Check the result's internal selection accounting."""

        if self.candidates_examined < 1:
            raise ValueError("no scheduling candidates examined")

        if not isinstance(self.baseline, OmegaPortSchedule):
            raise ValueError("invalid baseline schedule")

        if not isinstance(self.selected, OmegaPortSchedule):
            raise ValueError("invalid selected schedule")

        if not math.isfinite(self.baseline_seconds):
            raise ValueError("invalid baseline prediction")

        if not math.isfinite(self.selected_seconds):
            raise ValueError("invalid selected prediction")

        if self.selected_seconds > self.baseline_seconds:
            raise ValueError("selected schedule is worse than baseline")

        if (
            self.selected.total_link_bytes
            != self.baseline.total_link_bytes
        ):
            raise ValueError("communication volume changed")

        return True


def _transfer_order(transfer, machine, links, strategy):
    """Deterministic priority derived from machine coordinates.

    This priority orders currently eligible transfers. It does not
    modify their ownership, dependencies, or communication volume.
    """

    owner = gamma(transfer.owner, machine.shape)
    parent = gamma(transfer.parent, machine.shape)
    child = gamma(transfer.child, machine.shape)

    link = links[(transfer.parent, transfer.child)]

    duration = (
        link.alpha
        + transfer.payload_bytes / link.beta
    )

    if strategy == "large_first":
        return (
            -transfer.payload_bytes,
            owner,
            parent,
            child,
        )

    if strategy == "small_first":
        return (
            transfer.payload_bytes,
            owner,
            parent,
            child,
        )

    if strategy == "slow_first":
        return (
            -duration,
            owner,
            parent,
            child,
        )

    if strategy == "fast_first":
        return (
            duration,
            owner,
            parent,
            child,
        )

    if strategy == "link_first":
        return (
            parent,
            child,
            owner,
        )

    if strategy == "owner_first":
        return (
            owner,
            parent,
            child,
        )

    raise ValueError("unknown scheduling strategy")


def _candidate_schedule(
    topology,
    baseline,
    links,
    strategy,
    avoid_shared_links,
):
    """Build one legal candidate from the exact baseline transfers.

    The stage-entry availability snapshot prevents a payload
    received in a stage from being forwarded in that same stage.

    When avoid_shared_links is True, at most one transfer uses
    each directed link in a stage.

    This additional restriction may increase stage count, so
    candidate quality must be evaluated over the complete schedule.
    """

    machine = Machine(topology.machine_shape)

    remaining = [
        transfer
        for stage in baseline.stages
        for transfer in stage
    ]

    reached = {
        route.owner: {route.owner}
        for route in topology.routes
    }

    stages = []

    while remaining:
        available = {
            owner: set(coordinates)
            for owner, coordinates in reached.items()
        }

        eligible = [
            transfer
            for transfer in remaining
            if (
                transfer.parent in available[transfer.owner]
                and transfer.child not in available[transfer.owner]
            )
        ]

        eligible.sort(
            key=lambda transfer: _transfer_order(
                transfer,
                machine,
                links,
                strategy,
            )
        )

        stage = []
        outgoing = {}
        occupied_links = set()
        deliveries = set()

        for transfer in eligible:
            parent = transfer.parent
            child = transfer.child
            owner = transfer.owner
            directed_link = (parent, child)

            if outgoing.get(parent, 0) >= baseline.outgoing_ports:
                continue

            if (
                avoid_shared_links
                and directed_link in occupied_links
            ):
                continue

            if (owner, child) in deliveries:
                continue

            stage.append(transfer)

            outgoing[parent] = outgoing.get(parent, 0) + 1
            occupied_links.add(directed_link)
            deliveries.add((owner, child))

        if not stage:
            raise ValueError(
                "candidate scheduler made no legal progress"
            )

        for transfer in stage:
            reached[transfer.owner].add(transfer.child)

        stages.append(tuple(stage))

        # Remove exactly the selected transfer occurrences.
        # The independent auditor subsequently checks the full
        # expected-versus-observed transfer multiset.
        selected = list(stage)

        deferred = []

        for transfer in remaining:
            if transfer in selected:
                selected.remove(transfer)
            else:
                deferred.append(transfer)

        remaining = deferred

    candidate = OmegaPortSchedule(
        machine_shape=baseline.machine_shape,
        outgoing_ports=baseline.outgoing_ports,
        stages=tuple(stages),
        stage_count=len(stages),
        total_link_bytes=sum(
            transfer.payload_bytes
            for stage in stages
            for transfer in stage
        ),
    )

    candidate.verify()

    return candidate


def schedule_omega_contention(
    plan,
    topology,
    links,
    outgoing_ports=1,
):
    """Derive and compare audited Omega communication schedules.

    Returns an OmegaContentionScheduleResult.

    Selection procedure:
      1. Construct and independently audit the original schedule.
      2. Predict its directed-link contention time.
      3. Construct deterministic candidate schedules.
      4. Independently audit each candidate.
      5. Predict each candidate's complete communication time.
      6. Select only a strictly faster candidate.

    The original schedule is always available as a fallback.
    """

    if not isinstance(topology, OmegaTopology):
        raise TypeError("OmegaTopology required")

    baseline = schedule_omega_ports(
        topology,
        outgoing_ports,
    )

    audit_omega_port_schedule(
        plan,
        topology,
        baseline,
    )

    baseline_prediction = predict_omega_contention(
        plan,
        topology,
        baseline,
        links,
    )

    audit_omega_contention(
        plan,
        topology,
        baseline,
        links,
        baseline_prediction,
    )

    selected = baseline
    selected_prediction = baseline_prediction
    selected_strategy = "baseline"

    strategies = (
        "large_first",
        "small_first",
        "slow_first",
        "fast_first",
        "link_first",
        "owner_first",
    )

    candidates_examined = 1

    for avoid_shared_links in (False, True):
        for strategy in strategies:
            candidate = _candidate_schedule(
                topology,
                baseline,
                links,
                strategy,
                avoid_shared_links,
            )

            # Independent legality audit.
            audit_omega_port_schedule(
                plan,
                topology,
                candidate,
            )

            prediction = predict_omega_contention(
                plan,
                topology,
                candidate,
                links,
            )

            # Independent prediction audit.
            audit_omega_contention(
                plan,
                topology,
                candidate,
                links,
                prediction,
            )

            candidates_examined += 1

            if (
                prediction.total_seconds
                < selected_prediction.total_seconds
            ):
                selected = candidate
                selected_prediction = prediction

                suffix = (
                    "distinct_links"
                    if avoid_shared_links
                    else "shared_links_allowed"
                )

                selected_strategy = strategy + ":" + suffix

    result = OmegaContentionScheduleResult(
        baseline=baseline,
        selected=selected,
        baseline_prediction=baseline_prediction,
        selected_prediction=selected_prediction,
        strategy=selected_strategy,
        candidates_examined=candidates_examined,
    )

    result.verify()

    return result
