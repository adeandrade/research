#!/bin/bash

launch_job(){{
	if [[ -n "${{PID}}" ]]; then
		kill -INT "$(pgrep -P "$(pgrep -P "${{PID}}")")"
		wait "${{PID}}"
	fi

	sbatch \
		--job-name {job_name} \
		--account {account_id} \
		--cpus-per-task {cpus} \
		--mem {memory} \
		--gpus-per-node {gpu} \
		--time {time} \
		--signal {signal} \
		<<- EOM
			#!/bin/bash

			$(type launch_job | sed '1d')
			trap launch_job USR1

			module load singularity

			mkdir -p "/scratch/\${{USER}}/data"
			mkdir -p "\${{SLURM_TMPDIR}}/data"
			mkdir -p "\${{SLURM_TMPDIR}}/singularity/"{{tmp,workdir,home}}

			export HTTP_PROXY="{proxy_url}"
			export HTTPS_PROXY="{proxy_url}"
			export SINGULARITY_DOCKER_USERNAME="{docker_username}"
			export SINGULARITY_DOCKER_PASSWORD="{docker_password}"
			export SINGULARITY_CACHEDIR="/scratch/${{USER}}/singularity"
			export SINGULARITY_TMPDIR="\${{SLURM_TMPDIR}}/singularity/tmp"

			singularity \
				run \
				--contain \
				--cleanenv \
				--nv \
				--workdir "\${{SLURM_TMPDIR}}/singularity/workdir" \
				--home "\${{SLURM_TMPDIR}}/singularity/home" \
				--bind "/scratch/\${{USER}}/data":/data-remote \
				--bind "\${{SLURM_TMPDIR}}/data":/data-local \
				--env MLFLOW_TRACKING_URI="https://mlflow.multimedialabsfu.xyz" \
				--env MLFLOW_TRACKING_USERNAME="{mlflow_username}" \
				--env MLFLOW_TRACKING_PASSWORD="{mlflow_password}" \
				--env MLFLOW_S3_ENDPOINT_URL="https://io.multimedialabsfu.xyz" \
				--env AWS_ACCESS_KEY_ID="{aws_access_key_id}" \
				--env AWS_SECRET_ACCESS_KEY="{aws_secret_access_key}" \
				--env S3_ENDPOINT_URL="https://io.multimedialabsfu.xyz" \
				--env DATA_PATH="/data-remote" \
				--env DATA_EPHEMERAL_PATH="/data-local" \
				--env SLACK_URL="{slack_url}" \
				--env SLACK_USER="{slack_user}" \
				--env HTTP_PROXY="{proxy_url}" \
				--env HTTPS_PROXY="{proxy_url}" \
				--env DEVELOPMENT_MODE="{development_mode}" \
				--env JOB_ID="\${{SLURM_JOB_ID}}" \
				"docker://{training_image}" \
				{job_arguments} \
				&

			PID="\$!"

			wait
		EOM
}}

launch_job
