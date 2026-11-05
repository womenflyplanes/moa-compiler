"""Step 108: restricted MoA attention compilation and prediction.

Composes existing verified components without modifying them.

Scope:
  - typed forward-attention specification
  - stable online reduction and gamma lowering
  - resource-feasible explicit Omega ownership
  - logical Allgatherv communication
  - deterministic candidate topology
  - audited heuristic or bounded exact communication scheduling

Predictions describe the specified logical communication model,
not measured hardware performance or actual MPI collective routing.
"""

from dataclasses import dataclass

from moa_machine import Machine, Work
from moa_typed_ir import (
    specification,
    stable_online_fold,
    gamma_lower,
)
from moa_attention_resource_planner import plan_attention_omega
from moa_attention_resource_communication import (
    derive_resource_communication,
)
from moa_attention_omega_topology import derive_omega_topology
from moa_attention_topology_audit import audit_omega_topology
from moa_attention_communication_audit import (
    audit_resource_communication_plan,
)
from moa_attention_port_schedule_audit import (
    audit_omega_port_schedule,
)
from moa_attention_contention_audit import (
    audit_omega_contention,
)
from moa_attention_contention_schedule import (
    schedule_omega_contention,
)
from moa_attention_exact_schedule import schedule_omega_exact


@dataclass(frozen=True)
class IntegratedAttention108:
    plan: object
    communication: object
    topology: object
    scheduling: object
    mode: str
    links: dict
    outgoing_ports: int

    def verify(self):
        self.plan.verify()
        self.communication.verify()
        self.topology.verify()
        self.scheduling.verify()

        audit_resource_communication_plan(
            self.plan, self.communication
        )

        audit_omega_topology(
            self.plan, self.topology
        )

        if (
            self.communication.aggregate_nonlocal_bytes
            != self.topology.total_link_bytes
        ):
            raise ValueError(
                "communication/topology volume mismatch"
            )

        if self.mode == "heuristic":
            schedule = self.scheduling.selected
            prediction = self.scheduling.selected_prediction
        elif self.mode == "exact":
            schedule = self.scheduling.optimal
            prediction = self.scheduling.optimal_prediction
        else:
            raise ValueError("invalid scheduling mode")

        if schedule.outgoing_ports != self.outgoing_ports:
            raise ValueError("outgoing-port mismatch")

        audit_omega_port_schedule(
            self.plan, self.topology, schedule
        )

        audit_omega_contention(
            self.plan,
            self.topology,
            schedule,
            self.links,
            prediction,
        )

        return True


def compile_attention_108(
    work,
    machine,
    capacities,
    links,
    outgoing_ports=1,
    mode="heuristic",
):
    """Derive and predict one resource-feasible attention mapping."""

    if not isinstance(work, Work):
        raise TypeError("Work required")

    if not isinstance(machine, Machine):
        raise TypeError("Machine required")

    expr = gamma_lower(
        stable_online_fold(specification())
    )

    plan = plan_attention_omega(
        expr, work, machine, capacities
    )

    communication = derive_resource_communication(plan)
    topology = derive_omega_topology(plan)

    if mode == "heuristic":
        scheduling = schedule_omega_contention(
            plan, topology, links,
            outgoing_ports=outgoing_ports,
        )
    elif mode == "exact":
        scheduling = schedule_omega_exact(
            plan, topology, links,
            outgoing_ports=outgoing_ports,
        )
    else:
        raise ValueError("mode must be heuristic or exact")

    result = IntegratedAttention108(
        plan=plan,
        communication=communication,
        topology=topology,
        scheduling=scheduling,
        mode=mode,
        links=dict(links),
        outgoing_ports=outgoing_ports,
    )
    result.verify()
    return result
