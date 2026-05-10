# Control Machine Runbook (ALOHA + Remote OpenPI Server)

This document is for the AI agent running on the control machine.

Goal:
- Run robot-side control on an ALOHA-like setup.
- Query a remote OpenPI policy server for action chunks.
- Keep robot interface adaptations strictly on the control machine side.


## 1) Known server-side status

The remote policy server has been validated as working with:
- policy config: `pi0_aloha`
- checkpoint: `pi0_base` (local path on server)
- websocket bind: `0.0.0.0:8000`
- server IP: `192.168.3.103`

Expected remote endpoint from control machine:
- host: `192.168.3.103`
- port: `8000`


## 2) Action and observation conventions (ALOHA)

Use ALOHA policy conventions:
- State shape: `(14,)`, layout `6+1+6+1`
- Action chunk shape: `(50, 14)` from policy (for `pi0_aloha`)
- Image convention expected by aloha policy utilities:
  - per camera image shape: `(3, 224, 224)` uint8 if using ALOHA transform helpers
  - if your local runtime uses HWC, convert consistently before policy call

Required semantic fields:
- prompt: task string
- state: robot proprioception with expected ordering
- images:
  - `cam_high`
  - `cam_left_wrist`
  - `cam_right_wrist`
  - `cam_low` (optional depending on runtime path; can be a valid placeholder)


## 3) Control machine startup checklist

1. Activate environment on control machine.
2. Confirm network route to server:
	- `ping 192.168.3.103`
	- `nc -vz 192.168.3.103 8000` (or equivalent)
3. Confirm local robot middleware and camera streams are healthy.
4. Confirm control frequency target (ALOHA commonly 20 Hz/50 Hz depending stack; keep your local stack authoritative).


## 4) Minimal client pattern (control side)

Use websocket client to query remote server and execute locally.

Pseudo-flow:
1. Build observation dict from local sensors.
2. Send to websocket policy server.
3. Receive `actions` chunk.
4. Execute chunk locally with safety clipping/rate limits.
5. Re-query every N control steps.

Client keys should remain stable over time; do not rename keys per step.


## 5) Robot interface boundary (important)

Only control machine is responsible for robot-specific integration.

Allowed to modify on control machine side:
- camera acquisition and calibration mapping
- joint/gripper readback parsing
- action scaling/clipping/safety guards
- control-loop timing and watchdogs

Not required to modify remote server for robot hardware changes.


## 6) Safety guardrails (must-have)

Before enabling full autonomy:
1. Dry-run with execution disabled: only log incoming action chunk.
2. Enable low-gain/small-scale execution.
3. Add hard clipping on each joint/gripper channel.
4. Add E-stop and command timeout fallback.
5. Add stale-frame/stale-observation detection.


## 7) Common failure modes and fixes

1. Connection refused / timeout:
	- Server not running or not listening on `8000`.
	- Firewall or route issue.

2. Shape mismatch:
	- state not `(14,)` or action parser not expecting `(50, 14)`.
	- camera tensor format mismatch (CHW vs HWC).

3. High latency / jitter:
	- reduce image size or compress transmission path on control side.
	- reduce query frequency and execute chunk open-loop.

4. Robot unsafe motion:
	- scaling/clipping not applied.
	- wrong joint order mapping.


## 8) Hand-off note for control machine AI

Treat remote policy output as model-space action.
Always map and validate against hardware-space constraints before execution.
If unsure about channel semantics, default to no-op for unknown dimensions.

