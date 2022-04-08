#!/bin/sh
module load singularity

mkdir -p ${{SLURM_TMPDIR}}/singularity/{{tmp,workdir,home}}
mkdir -p ${{SLURM_TMPDIR}}/data

export HTTP_PROXY="{proxy_url}"
export HTTPS_PROXY="{proxy_url}"
export SINGULARITY_DOCKER_USERNAME="{docker_username}"
export SINGULARITY_DOCKER_PASSWORD="{docker_password}"
export SINGULARITY_CACHEDIR="/scratch/${{USER}}/singularity"
export SINGULARITY_TMPDIR="${{SLURM_TMPDIR}}/singularity/tmp"

singularity \
  run \
  --containall \
  --cleanenv \
  --nv \
  --bind "/project/{account}/${{USER}}":/mnt \
  --workdir "${{SLURM_TMPDIR}}/singularity/workdir" \
  --home "${{SLURM_TMPDIR}}/singularity/home" \
  --env MLFLOW_TRACKING_URI="https://mlflow.multimedialabsfu.xyz" \
  --env MLFLOW_TRACKING_USERNAME="{mlflow_username}" \
  --env MLFLOW_TRACKING_PASSWORD="{mlflow_password}" \
  --env MLFLOW_S3_ENDPOINT_URL="https://io.multimedialabsfu.xyz" \
  --env AWS_ACCESS_KEY_ID="{aws_access_key_id}" \
  --env AWS_SECRET_ACCESS_KEY="{aws_secret_access_key}" \
  --env S3_ENDPOINT_URL="https://io.multimedialabsfu.xyz" \
  --env DATA_PATH="/mnt" \
  --env DATA_EPHEMERAL_PATH="/data" \
  --env SLACK_URL="{slack_url}" \
  --env SLACK_USER="{slack_user}" \
  --env HTTP_PROXY="{proxy_url}" \
  --env HTTPS_PROXY="{proxy_url}" \
  "docker://{training_image}" \
  {job_arguments}
