# vdb_tool

## Bug Fixes

- Preserve rebuilt CSG results when replacing input grids.
- Return a failing exit status after action errors and stop immediately on
  control-flow errors instead of entering invalid scopes.
- Preserve input grids after failed VDB/NanoVDB writes and avoid changing
  the grid stack or names when voxel-kernel variable validation fails.
- Preserve modulo expressions and loop counters in configuration files.
  Inline comments now use '#'; '%' still supports full-line comments.
- Reject truncated or malformed PLY data instead of hanging or reading
  incomplete records.
- Support descending loops and reject zero or non-progressing steps.
- Keep binary stdout separate from diagnostic logs in both logging modes.
