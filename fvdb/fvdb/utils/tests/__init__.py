# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import functools
import site
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import git
import git.repo
import torch
from git.exc import InvalidGitRepositoryError
from parameterized import parameterized

from .grid_utils import (
    gridbatch_from_dense_cube,
    make_dense_grid_and_point_data,
    make_gridbatch_and_point_data,
    random_drop_points_if_mutable,
)

git_tag_for_data = "main"


def set_testing_git_tag(git_tag):
    global git_tag_for_data
    git_tag_for_data = git_tag


def _is_editable_install() -> bool:
    # check we're not in a site package
    module_path = Path(__file__).resolve()
    for site_path in site.getsitepackages():
        if str(module_path).startswith(site_path):
            return False
    # check if we're in the source directory
    module_dir = module_path.parent.parent.parent.parent
    return (module_dir / "setup.py").is_file()


def _get_local_repo_path() -> Path:
    if _is_editable_install():
        external_dir = Path(__file__).resolve().parent.parent.parent.parent / "external"
        if not external_dir.exists():
            external_dir.mkdir()
        local_repo_path = external_dir
    else:
        local_repo_path = Path(tempfile.gettempdir())

    local_repo_path = local_repo_path / "fvdb_example_data"
    return local_repo_path


def _clone_fvdb_test_data() -> Tuple[Path, git.repo.Repo]:
    global git_tag_for_data

    def is_git_repo(repo_path: str) -> bool:
        is_repo = False
        try:
            _ = git.repo.Repo(repo_path)
            is_repo = True
        except InvalidGitRepositoryError:
            is_repo = False

        return is_repo

    git_url = "https://github.com/voxel-foundation/fvdb-test-data.git"
    repo_path = _get_local_repo_path()

    if repo_path.exists() and repo_path.is_dir():
        if is_git_repo(str(repo_path)):
            repo = git.repo.Repo(repo_path)
        else:
            raise ValueError(f"A path {repo_path} exists but is not a git repo")
    else:
        repo = git.repo.Repo.clone_from(git_url, repo_path)
    repo.remotes.origin.fetch()
    repo.git.checkout(git_tag_for_data)

    return repo_path, repo


def get_fvdb_test_data_path() -> Path:
    repo_path, _ = _clone_fvdb_test_data()
    return repo_path / "unit_tests"


# Hack parameterized to use the function name and the expand parameters as the test name
expand_tests = functools.partial(
    parameterized.expand,
    name_func=lambda f, n, p: f'{f.__name__}_{parameterized.to_safe_name("_".join(str(x) for x in p.args))}',
)


def probabilistic_test(
    iterations,
    pass_percentage: float = 80,
    conditional_args: Optional[List[List]] = None,
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if the condition argument is present and matches the condition value
            do_repeat = True
            if conditional_args is None:
                do_repeat = False
            else:
                for a, condition_values in enumerate(conditional_args):
                    if args[a + 1] in condition_values:
                        continue
                    else:
                        do_repeat = False
                        break
            if do_repeat:
                passed = 0
                for _ in range(iterations):
                    try:
                        func(*args, **kwargs)
                        passed += 1
                    except AssertionError:
                        pass
                pass_rate = (passed / iterations) * 100
                assert pass_rate >= pass_percentage, f"Test passed only {pass_rate:.2f}% of the time"
            else:
                # If condition is not met, just run the function once
                return func(*args, **kwargs)

        return wrapper

    return decorator


def dtype_to_atol(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 1e-1
    if dtype == torch.float16:
        return 1e-1
    if dtype == torch.float32:
        return 1e-5
    if dtype == torch.float64:
        return 1e-5
    raise ValueError("dtype must be a valid torch floating type")


__all__ = [
    "set_testing_git_tag",
    "get_fvdb_test_data_path",
    "gridbatch_from_dense_cube",
    "random_drop_points_if_mutable",
    "make_dense_grid_and_point_data",
    "make_gridbatch_and_point_data",
    "dtype_to_atol",
    "expand_tests",
]
