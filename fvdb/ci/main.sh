# Running a Github Action Runner, the first argument

# Starting a dockerd
/usr/bin/dockerd &

# Get a register token for a GitHub runner
# https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28
token=$(curl -L \
  -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_ACCESS_TOKEN"\
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/voxel-foundation/fvdb/actions/runners/registration-token)

# Extract the part between : " and "
token=$(echo $token | sed -e 's/^.*\"token\":\s*\"\([^\"]*\)\".*$/\1/')

# Run the runner
CUDA_MAJOR="$(echo ${CUDA_VERSION} | cut -d'.' -f1)" && \
CUDA_MINOR="$(echo ${CUDA_VERSION} | cut -d'.' -f2)" && \
export CUDA_TAG="$(echo ${CUDA_MAJOR}${CUDA_MINOR})"

cd /tmp/actions-runner
./config.sh --url https://github.com/voxel-foundation/fvdb --token $token --labels self-hosted,x64,Linux,cu$CUDA_TAG --name `hostname` --replace --unattended
./run.sh
