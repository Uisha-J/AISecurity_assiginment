"""Run the full attack-defense simulation.

Usage:
    python -m voice_defense.scripts.run_simulation
    python -m voice_defense.scripts.run_simulation --config configs/simulation.yaml
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from ..pipeline.simulate import run_simulation


def main():
    parser = argparse.ArgumentParser(description="Run attack-defense simulation")
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--output-root", default="./outputs")
    args = parser.parse_args()

    run_simulation(args.config, args.data_root, args.output_root)


if __name__ == "__main__":
    main()
