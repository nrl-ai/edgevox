"""Forwarder for the Plan-Execute-Evaluate demo.

Real implementation lives in
``edgevox.examples.agents.plan_execute_evaluate_demo`` so it ships
inside the installed ``edgevox`` package.

Equivalent invocations::

    python -m edgevox.examples.agents.plan_execute_evaluate_demo --scripted
    python examples/agents/plan_execute_evaluate_demo.py --scripted
"""

from edgevox.examples.agents.plan_execute_evaluate_demo import main

if __name__ == "__main__":
    raise SystemExit(main())
