#!/usr/bin/env python3

import os
from pathlib import Path

from machetli import pddl, tools

PLANNER_REPO = Path(os.environ["DOWNWARD_REPO"])
PLANNER = PLANNER_REPO / "fast-downward.py"


def evaluate(domain, problem):
    reference_command = [
        PLANNER, domain, problem, "--search", "astar(lmcut())",
        "--translate-options", "--relaxed",
    ]
    result_reference = tools.run(reference_command, cpu_time_limit=20,
                                 memory_limit=3000, text=True)
    cost = tools.parse(result_reference.stdout, r"Plan cost: (\d+)")

    mip_command = [
        PLANNER, domain, problem, "--search",
        "astar(operatorcounting([delete_relaxation_constraints("
        "use_time_vars=true, use_integer_vars=true)], "
        "use_integer_operator_counts=True), bound=0)",
    ]
    result_mip = tools.run(mip_command, cpu_time_limit=20, memory_limit=3000,
                           text=True)
    initial_h = tools.parse(result_mip.stdout,
                            r"Initial heuristic value .* (\d+)")

    if cost is None or initial_h is None:
        return False
    return cost != initial_h

if __name__ == "__main__":
    pddl.run_evaluator(evaluate)
