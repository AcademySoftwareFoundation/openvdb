# TEACHME

This directory contains interactive lesson documents designed to be loaded by
an LLM coding agent (Claude Code, Cursor, or similar) to teach a user NanoVDB
interactively.

The lesson teaches how to *use* NanoVDB to read, build, transform, and render
sparse volumetric data on CPU and GPU. It is a single self-contained markdown
file the agent reads at the start of a session.

## Getting started

The easiest way to use this lesson is with [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
(CLI or IDE extension) from the root of an OpenVDB checkout. The agent can
read the lesson files *and* the NanoVDB source code, so it can verify API
details and help you debug exercises in real time.

**Example prompts:**

To learn how to use NanoVDB:

```
Read doc/nanovdb/TEACHME/nanovdb_user_lesson.md and teach me how to use NanoVDB.
```

You can also give the agent context about your background so it can tailor
the lesson:

```
Read doc/nanovdb/TEACHME/nanovdb_user_lesson.md and teach me NanoVDB. I have
C++ and CUDA experience but I've never used OpenVDB or any sparse 3D data
structure before.
```

Keep the matching cheatsheet (`*_cheatsheet.md`) open in your editor while
working through exercises — it's a quick reference for the APIs and
invariants covered in the lesson.

## How it works

The lesson is a self-contained markdown file that serves as both a
curriculum and an instructor prompt. The LLM acts as an interactive
instructor: teaching concepts module by module, quizzing the student, and
adapting to their responses.

The lesson includes:

- Teacher instructions (persona, pacing, scope)
- Module-by-module curriculum with embedded concepts and code examples
- Quiz questions and an answer key
- Exercises with progressive difficulty
- A capstone project
- A reference table at the end

## Available lessons

| Lesson | Cheat sheet | Covers |
|---|---|---|
| [nanovdb_user_lesson.md](nanovdb_user_lesson.md) | [nanovdb_user_cheatsheet.md](nanovdb_user_cheatsheet.md) | Reading `.nvdb` files, `ReadAccessor`, `NodeManager`, math + sampling, HDDA ray-march, GPU kernels with `Grid`, GPU topology builders (`PointsToGrid`, `MeshToGrid`, `DilateGrid`, etc.), `IndexGrid` + `VoxelBlockManager`, OpenVDB ↔ NanoVDB conversion |

## Not covered

- The OpenVDB side (the dynamic, mutable tree used at film/sim time) — this
  lesson treats OpenVDB only at the conversion boundary.
- Houdini / Maya integration — outside the scope.
- Gaussian splatting and fVDB-specific topics — see the fVDB repo's own
  TEACHME at <https://github.com/openvdb/fvdb-core/tree/main/docs/TEACHME>.
