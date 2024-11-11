#!/usr/bin/env python3
import argparse
import time
import numpy as np


def initialize(N):
    print("Initializing stencil grid...")
    grid = np.zeros((N + 2, N + 2))
    grid[:, 0] = -273.15
    grid[:, -1] = -273.15
    grid[-1, :] = -273.15
    grid[0, :] = 40.0
    return grid


def run_stencil(N, I, warmup, timing):  # noqa: E741
    grid = initialize(N)

    print("Running Jacobi stencil...")
    center = grid[1:-1, 1:-1]
    north = grid[0:-2, 1:-1]
    east = grid[1:-1, 2:]
    west = grid[1:-1, 0:-2]
    south = grid[2:, 1:-1]

    np.set_printoptions(edgeitems=30, linewidth=120)
    start = time.time()
    for i in range(I + warmup):
        if i == warmup:
            start = time.time()
        average = center + north + east + west + south
        print(average)
        work = 0.2 * average
        center[:] = work
    total = time.time() - start

    if timing:
        print(grid.ravel())
        print(f"Elapsed Time: {total} ms")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=100,
        dest="I",
        help="number of iterations to run",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=5,
        dest="warmup",
        help="warm-up iterations",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=100,
        dest="N",
        help="number of elements in one dimension",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )

    args = parser.parse_args()

    run_stencil(args.N, args.I, args.warmup, args.timing)
