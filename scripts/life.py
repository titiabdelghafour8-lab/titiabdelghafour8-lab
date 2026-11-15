"""Render Conway's Game of Life as a banner GIF for the profile README.

Usage: python scripts/life.py [--seed N] [--out assets/life.gif]

Without --seed the seed is the current ISO year and week, so the weekly
workflow draws a new board each week. The board is a torus (edges wrap).
Live cells are drawn bright; cells that just died fade out over a few frames.
"""

import argparse
import datetime

import numpy as np
from PIL import Image

W, H = 128, 32          # board size in cells
CELL = 5                # pixels per cell (gap of 1 px included)
FRAMES = 180            # generations in the GIF
FRAME_MS = 70
DENSITY = 0.28

BG = (13, 17, 23)       # GitHub dark background
GRID = (22, 27, 34)
# Live cell, then the trail of a dead cell from fresh to faint.
LIVE = (88, 166, 255)
TRAIL = [(56, 110, 180), (38, 74, 122), (29, 50, 80), (24, 34, 50)]


def step(board: np.ndarray) -> np.ndarray:
    n = sum(np.roll(np.roll(board, dy, 0), dx, 1)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    return (n == 3) | (board & (n == 2))


def run(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    board = rng.random((H, W)) < DENSITY
    boards = [board]
    for _ in range(FRAMES - 1):
        board = step(board)
        boards.append(board)
    return boards


def lively(boards: list[np.ndarray]) -> bool:
    """A board is worth showing if it is still changing at the end."""
    last, prev = boards[-1], boards[-5]
    return last.sum() > W * H * 0.04 and (last != prev).sum() > 20


def render(boards: list[np.ndarray]) -> list[Image.Image]:
    palette = [BG, GRID, LIVE, *TRAIL]
    flat = [c for rgb in palette for c in rgb]
    age = np.full((H, W), len(TRAIL) + 1)   # generations since the cell was alive
    frames = []
    for board in boards:
        age = np.where(board, 0, age + 1)
        # Index into the palette: 1 = empty grid cell, 2 = live, 3.. = trail.
        idx = np.where(age == 0, 2, np.where(age <= len(TRAIL), 2 + age, 1)).astype(np.uint8)
        big = np.kron(idx, np.ones((CELL, CELL), dtype=np.uint8))
        big[CELL - 1::CELL, :] = 0          # 1 px gaps between cells
        big[:, CELL - 1::CELL] = 0
        img = Image.fromarray(big, mode="P")
        img.putpalette(flat)
        frames.append(img)
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out", default="assets/life.gif")
    args = ap.parse_args()

    if args.seed is None:
        y, w, _ = datetime.date.today().isocalendar()
        args.seed = y * 100 + w

    seed = args.seed
    boards = run(seed)
    while not lively(boards):
        seed += 1000
        boards = run(seed)

    frames = render(boards)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=FRAME_MS, loop=0, optimize=True)
    print(f"seed {seed}: {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
