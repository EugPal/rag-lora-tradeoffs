from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TERMINAL_STATUSES = {"SUCCESS", "FAILED", "CANCELLED", "ERROR"}


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, text=True, capture_output=True)


def _execute_job(project_id: str, config_path: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_path = Path(tmp.name)
    cmd = [
        "datasphere",
        "project",
        "job",
        "execute",
        "-p",
        project_id,
        "-c",
        config_path,
        "--async",
        "-o",
        str(out_path),
    ]
    print("Running:", " ".join(cmd), flush=True)
    _run(cmd)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    job_id = payload.get("id")
    if not job_id:
        raise RuntimeError(f"Cannot read job id from {out_path}")
    return str(job_id)


def _get_job(job_id: str) -> dict:
    cmd = ["datasphere", "project", "job", "get", "--id", job_id, "--format", "json"]
    res = _run(cmd)
    return json.loads(res.stdout)


def _download(job_id: str, output_root: Path, with_logs: bool) -> Path:
    output_dir = output_root / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "datasphere",
        "project",
        "job",
        "download-files",
        "--id",
        job_id,
        "--output-dir",
        str(output_dir),
    ]
    if with_logs:
        cmd.append("--with-logs")
    print("Running:", " ".join(cmd), flush=True)
    _run(cmd)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute DataSphere job and always download artifacts to experiments/datasphere/<job_id>."
    )
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="experiments/datasphere")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--no-wait", action="store_true", help="Do not wait for completion.")
    parser.add_argument("--no-logs", action="store_true", help="Do not download logs.")
    args = parser.parse_args()

    job_id = _execute_job(project_id=args.project_id, config_path=args.config)
    print(f"Job created: {job_id}", flush=True)

    if not args.no_wait:
        while True:
            job = _get_job(job_id)
            status = str(job.get("status", "")).upper()
            print(f"Status: {status}", flush=True)
            if status in TERMINAL_STATUSES:
                break
            time.sleep(max(1, args.poll_seconds))
        if status != "SUCCESS":
            print(f"Job finished with status={status}", file=sys.stderr)

    output_dir = _download(
        job_id,
        output_root=Path(args.output_root),
        with_logs=not args.no_logs,
    )
    print(f"Artifacts saved to: {output_dir.as_posix()}", flush=True)

    if not args.no_wait and status != "SUCCESS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
