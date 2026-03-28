import os
import sys
import traceback

import wandb


def mask_key(key: str) -> str:
    if not key:
        return "<empty>"
    if len(key) <= 12:
        return "*" * len(key)
    return f"{key[:8]}...{key[-6:]}"


def main() -> int:
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if not key:
        print("[FAIL] WANDB_API_KEY is empty")
        return 2

    print(f"[INFO] WANDB_API_KEY detected: {mask_key(key)}")
    print(f"[INFO] wandb version: {wandb.__version__}")

    try:
        ok = wandb.login(key=key, relogin=True)
        print(f"[INFO] wandb.login returned: {ok}")

        run = wandb.init(
            project="dsrl_lx_wandb_auth_smoke",
            name="auth_smoke",
            config={"source": "dsrl_pi0_lx/test"},
            settings=wandb.Settings(start_method="thread"),
        )
        wandb.log({"smoke": 1})
        run_url = getattr(run, "url", None)
        print(f"[OK] wandb.init succeeded. run_url={run_url}")
        wandb.finish()
        return 0
    except Exception as exc:
        print("[FAIL] wandb online init failed")
        print(f"[FAIL] exception={type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
