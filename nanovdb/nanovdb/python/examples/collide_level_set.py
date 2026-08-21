# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Collide falling particles against a level-set surface on the CPU.

Python port of the host path of ex_collide_level_set: particles fall
under gravity; each step transforms the candidate position to index
space with grid.worldToIndexF, tests the narrow band with
tree.isActive, reads the signed distance through a ReadAccessor, and
on penetration reflects the velocity about the surface normal. The C++
original builds the normal from 6-tap finite differences of the SDF;
here the bound sampler.gradient() provides the same quantity in one
call (index-space units — normalized before use).

Run with: python collide_level_set.py
"""
import random

import nanovdb

NUM_PARTICLES = 500
NUM_STEPS = 40
DT = 0.1
GRAVITY = -9.8


def main():
    handle = nanovdb.tools.createLevelSetSphere(radius=100.0, name="sphere")
    grid = handle.grid()
    tree = grid.tree()
    acc = grid.getAccessor()
    sampler = nanovdb.math.createTrilinearSampler(grid)

    # Seed particles above the north pole of the sphere, falling down.
    rng = random.Random(42)
    particles = []
    for _ in range(NUM_PARTICLES):
        p = [rng.uniform(-30.0, 30.0), rng.uniform(115.0, 140.0),
             rng.uniform(-30.0, 30.0)]
        particles.append((p, [0.0, -20.0, 0.0]))

    total_collisions = 0
    for step in range(NUM_STEPS):
        collisions = 0
        for p, v in particles:
            v[1] += GRAVITY * DT
            next_p = [p[i] + v[i] * DT for i in range(3)]

            ijk = nanovdb.math.Coord.Floor(grid.worldToIndexF(
                nanovdb.math.Vec3f(next_p[0], next_p[1], next_p[2])))
            if tree.isActive(ijk):  # inside the narrow band?
                d = acc.getValue(ijk)
                if d <= 0.0:  # inside the level set?
                    ipos = grid.worldToIndexF(
                        nanovdb.math.Vec3f(next_p[0], next_p[1], next_p[2]))
                    n = sampler.gradient(ipos)
                    n.normalize()
                    # Project the position back to the surface and
                    # reflect the velocity, as in the C++ example.
                    for i in range(3):
                        next_p[i] -= d * n[i]
                    v_dot_n = sum(v[i] * n[i] for i in range(3))
                    for i in range(3):
                        v[i] -= 2.0 * v_dot_n * n[i]
                    collisions += 1
            p[:] = next_p
        total_collisions += collisions
        if collisions:
            print(f"step {step:2d}: {collisions} collisions")

    print(f"total collisions over {NUM_STEPS} steps: {total_collisions}")
    assert total_collisions > 0

    # No particle should end up inside the surface.
    worst = 0.0
    for p, _ in particles:
        ijk = nanovdb.math.Coord.Floor(grid.worldToIndexF(
            nanovdb.math.Vec3f(p[0], p[1], p[2])))
        if tree.isActive(ijk):
            worst = min(worst, acc.getValue(ijk))
    print(f"deepest final penetration: {worst:.3f} world units")
    assert worst > -2.0 * grid.voxelSize()[0]


if __name__ == "__main__":
    main()
