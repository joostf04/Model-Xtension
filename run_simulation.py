import argparse
import config
from src.simulation import build_grid, run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-Xtension simulation runner")
    parser.add_argument("--M",           type=int, default=config.M,
                        help="Monte Carlo replications per cell")
    parser.add_argument("--n-jobs",      type=int, default=-1,
                        help="Parallel workers (-1 = all cores)")
    parser.add_argument("--output",      type=str, default="results/raw.parquet")
    parser.add_argument("--checkpoints", type=str, default="results/checkpoints")
    parser.add_argument("--cells",       type=int, nargs="+", default=None,
                        help="Run only these grid-cell indices (0-based)")
    args = parser.parse_args()

    grid = build_grid()
    if args.cells is not None:
        grid = [grid[i] for i in args.cells]

    print(f"{len(grid)} cells × {args.M} replications × 6 methods")
    run_simulation(grid, args.M, args.n_jobs, args.checkpoints, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
