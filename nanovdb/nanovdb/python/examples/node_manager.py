# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Iterate a grid's nodes linearly through a NodeManager.

Python port of the host half of ex_nodemanager_cuda: createNodeManager
builds linearized arrays of the tree's leaf, lower, and upper nodes so
they can be visited by index instead of by tree traversal. Each node
exposes its origin, per-node stats, and (on leaves) the raw 512-value
buffer. For bulk NumPy analytics over every leaf at once, see
bulk_leaf_numpy.py's grid.leaf_values() instead.

Run with: python node_manager.py
"""
import nanovdb


def main():
    handle = nanovdb.tools.createLevelSetSphere(radius=50.0, name="sphere")
    grid = handle.grid()
    tree = grid.tree()

    nmh = nanovdb.createNodeManager(grid)
    nm = nmh.mgr()
    print(f"NodeManager over {grid.gridName()!r} (linear={nm.isLinear()}):")
    print(f"  leaves={nm.leafCount()}, lower={nm.lowerCount()}, "
          f"upper={nm.upperCount()}")

    # The counts mirror the tree's per-level node counts.
    assert nm.leafCount() == tree.nodeCount(0)
    assert nm.lowerCount() == tree.nodeCount(1)
    assert nm.upperCount() == tree.nodeCount(2)

    # Linear access agrees with tree traversal.
    assert nm.leaf(0).origin() == tree.getFirstLeaf().origin()

    # Visit a few leaves by index: origin, activity, and value range.
    for i in range(min(5, nm.leafCount())):
        leaf = nm.leaf(i)
        print(f"  leaf[{i}] origin={leaf.origin()} "
              f"on={leaf.valueMask().countOn()} "
              f"min={leaf.minimum():.3f} max={leaf.maximum():.3f}")

    # Internal nodes are reachable the same way.
    lower = nm.lower(0)
    print(f"  lower[0] origin={lower.origin()} "
          f"children={lower.childMask().countOn()}")


if __name__ == "__main__":
    main()
