# fVDB CI/CD Pipeline

## Getting Started

1. **Obtaining Github token**: Go to your Github account settings, click on `Developer Settings`, go to `Personal access tokens`, and create a `Tokens (classic)`. The token should have `repo` access. Keep this token as `<YOUR TOKEN>`. This token will be used to create runners and clone private repos on your behalf.
2. **Setting up NGC**: You should set up your NGC account properly, so you can push images to `nvcr.io` and launch NGC nodes.

## Create docker image

(Note that this is an one-off command already run by Jiahui, and should be run for all target CUDA versions)
```bash
# Compile to fvdb-ci-cu121:latest
docker build -t nvcr.io/nvidian/ct-toronto-ai/fvdb-ci-cu121 -f ./ci/Dockerfile.runner --build-arg CUDA_VERSION=12.1.1 --build-arg CUDNN_VERSION=8 .
ngc registry image push nvcr.io/nvidian/ct-toronto-ai/fvdb-ci-cu121:latest
```

## Spin up the Github Action runner

```bash
# Locally
docker run -it --rm --env GITHUB_ACCESS_TOKEN=<YOUR TOKEN> nvcr.io/nvidian/ct-toronto-ai/fvdb-ci-cu121:latest /tmp/main.sh
# Or on NGC...
ngc batch run --name fvdb-ci --result /result --instance cpu.x86.tiny --image nvidian/ct-toronto-ai/fvdb-ci-cu121:latest --commandline "/tmp/main.sh" --priority HIGH --env-var GITHUB_ACCESS_TOKEN:<YOUR TOKEN>
```

Check out the running instance under Github action runners under this repo. You should be able to trigger the workflow and it will start running.

## Target Pipeline

1. Upon release (or manual trigger), trigger the Github Action at `.github/workflows/building.yml`.
2. Spin up NGC nodes as Github Action runners.
3. The completed job will upload artifacts onto `voxel-foundation/fvdb-wheels`.
4. The wheels will be mirrored into an internal GitLab site for hosting the artifacts.
