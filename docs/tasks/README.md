# Task specs

Implementation specs for workstream C (Patrick), plus Kyle's overnight loop (stream B),
written to be handed to a coding agent. Start every task by reading `AGENTS.md`, then
[contracts.md](contracts.md). The contracts are the interfaces that let the tasks run in
parallel. Where each task stands right now: [../status.md](../status.md).

| Task | What | Needed by |
| --- | --- | --- |
| [T1-app.md](T1-app.md) | Sandbox target app (12 verifiable tasks) + the GUI assist console | Sandbox ~4pm Sat; console ~10pm Sat |
| [T2-orchestrator.md](T2-orchestrator.md) | `common/grounding.py` (first 30 min), Lightning agent core, NeMo Gym `web_agent` env | grounding.py ASAP; loop ~5pm; Gym env ~8pm |
| [T3-lightning-data-engine.md](T3-lightning-data-engine.md) | Lightning rewrites raw labels into training instructions; publishable method | Dry run ~4pm; full run ~6pm |
| [B-night-loop.md](B-night-loop.md) | Overnight NeMo Gym improvement loop (profile → GRPO → dev gate), Kyle | Starts ~8pm Sat; hard end 05:40 |

Dependencies:
- T2 and T3 both read T1's `sandbox/tasks.json` and `checks.py` contract. Build
  against the contract right away.
- T1's console builds against T2's `run_task` contract, with a mock until T2 lands.
- T3 needs Charlie's manifest.
