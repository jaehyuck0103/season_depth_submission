import argparse
import os
import pprint
from pathlib import Path

from box import Box

from etri_depth.agents import get_agent


def main():
    agent = get_agent(cfg.agent.name, cfg)
    agent.train()


if __name__ == "__main__":
    # ------------
    # Argparse
    # ------------
    parser = argparse.ArgumentParser()

    parser.add_argument("config_path", type=Path)

    args = parser.parse_args()

    # --------------------------
    # Load and update settings
    # --------------------------
    cfg = Box.from_toml(filename=args.config_path)

    cfg.agent.log_dir = os.path.join(cfg.agent.log_dir, args.config_path.stem)

    # summarize settings
    if "RANK" not in os.environ or int(os.environ["RANK"]) == 0:
        pprint.pprint(cfg.to_dict(), width=88, sort_dicts=False)

    # -------
    # Run
    # -------
    main()
