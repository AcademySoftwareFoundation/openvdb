# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Walk the tree hierarchy and pretty-print per-node child counts.

node_manager.py visits nodes as flat per-level lists — every leaf,
every lower, every upper — with no notion of which upper owns which
lower, or which lower owns which leaf. This example does a real
top-down walk using Upper.getChild(n) / Lower.getChild(n): each
internal node's childMask() marks which of its table slots hold a
child rather than a tile value, isChild(n) reads one bit of that
mask, and getChild(n) returns the child at slot n (or None if that
slot is a tile).

Run with: python tree_stats.py
"""
import nanovdb


def child_slots(node):
    """Yield (n, child) for every slot in `node`'s table that holds a
    child rather than a tile value. childMask().bitCount() is the
    table size (32768 for Upper, 4096 for Lower) — the same constant
    isChild(n)/getChild(n) range-check against."""
    table_size = node.childMask().bitCount()
    for n in range(table_size):
        if node.isChild(n):
            yield n, node.getChild(n)


def main():
    handle = nanovdb.tools.createLevelSetSphere(radius=50.0, name="sphere")
    grid = handle.grid()
    tree = grid.tree()

    # A NodeManager gives us the root's upper nodes as a flat list to
    # start the walk from; everything below that is real parent -> child
    # descent through getChild(), not NodeManager indexing.
    nm = nanovdb.createNodeManager(grid)
    print(f"Tree for {grid.gridName()!r}: "
          f"{nm.upperCount()} upper, {tree.nodeCount(1)} lower, "
          f"{tree.nodeCount(0)} leaf nodes")

    total_lowers = 0
    total_leaves = 0
    for u in range(nm.upperCount()):
        upper = nm.upper(u)
        lowers = list(child_slots(upper))
        print(f"upper[{u}] origin={upper.origin()} "
              f"lower children={len(lowers)}")
        total_lowers += len(lowers)

        for n, lower in lowers:
            leaves = list(child_slots(lower))
            print(f"  lower[slot {n}] origin={lower.origin()} "
                  f"leaf children={len(leaves)}")
            total_leaves += len(leaves)

    # Every lower/leaf reached by the walk is reachable exactly once
    # (each has one parent), so the totals must match the tree's flat
    # per-level counts.
    print(f"walked {total_lowers} lower / {total_leaves} leaf nodes "
          f"(tree.nodeCount: {tree.nodeCount(1)} / {tree.nodeCount(0)})")
    assert total_lowers == tree.nodeCount(1)
    assert total_leaves == tree.nodeCount(0)

    # getChild(n) returns None for a tile slot (no child there), and
    # raises IndexError for n outside the table.
    upper0 = nm.upper(0)
    tile_slot = next(n for n in range(upper0.childMask().bitCount())
                      if not upper0.isChild(n))
    assert upper0.getChild(tile_slot) is None
    try:
        upper0.getChild(upper0.childMask().bitCount())
        raise AssertionError("expected IndexError")
    except IndexError:
        pass
    print("getChild(): None for a tile slot, IndexError out of range — OK")


if __name__ == "__main__":
    main()
