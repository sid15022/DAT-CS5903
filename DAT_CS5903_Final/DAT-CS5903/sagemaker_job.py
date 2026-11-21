"""Render a single-host SageMaker Training job definition for the GPU Docker image."""

import argparse
import json
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="Dockerfile.gpu image URI in ECR")
    p.add_argument("--role", required=True, help="SageMaker execution role ARN")
    p.add_argument("--s3-output", required=True, help="S3 prefix for results")
    p.add_argument("--mode", choices=["pilot", "suite"], default="pilot")
    p.add_argument("--workers", type=int, choices=[1, 2], default=1)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--instance-type", default="ml.g5.12xlarge")
    p.add_argument("--rate", type=float, default=2.6)
    p.add_argument(
        "--duration",
        type=float,
        default=180.0,
        help="Seconds per trial (suite minimum 120)",
    )
    p.add_argument("--max-seconds", type=int, default=25200)
    p.add_argument("--with-overhead", action="store_true")
    a = p.parse_args()
    if (
        not a.s3_output.startswith("s3://")
        or a.rate <= 0
        or a.reps < 1
        or (a.mode == "suite" and a.duration < 120)
        or a.max_seconds < 1
    ):
        p.error(
            "Supply an S3 prefix, positive settings and at least 120 seconds per suite trial"
        )
    if a.with_overhead and a.mode != "suite":
        p.error("Select suite for an observer-overhead study")
    args = [
        a.mode,
        "--backend",
        "vllm-http",
        "--workers",
        str(a.workers),
        "--out",
        "/opt/ml/model/failslow",
        "--spool",
        "/opt/ml/kv-spool",
        "--rate",
        str(a.rate),
        "--duration",
        str(a.duration),
        "--reps",
        str(a.reps),
    ]
    if a.with_overhead:
        args.append("--with-overhead")
    spec = dict(
        TrainingJobName="dat-pd-"
        + a.mode
        + "-"
        + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        RoleArn=a.role,
        AlgorithmSpecification=dict(
            TrainingImage=a.image,
            TrainingInputMode="File",
            ContainerEntrypoint=["python3", "-m", "failslow"],
            ContainerArguments=args,
        ),
        ResourceConfig=dict(
            InstanceType=a.instance_type, InstanceCount=1, VolumeSizeInGB=100
        ),
        OutputDataConfig=dict(S3OutputPath=a.s3_output),
        StoppingCondition=dict(MaxRuntimeInSeconds=a.max_seconds),
    )
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
