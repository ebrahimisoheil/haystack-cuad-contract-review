from __future__ import annotations

import argparse
import json

from contract_review_agent.app.config import Settings
from contract_review_agent.app.pipeline import run_review


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    mode = "live" if args.live else "deterministic"
    print(json.dumps(run_review(args.contract, Settings.from_env(mode=mode)), indent=2))


if __name__ == "__main__":
    main()
