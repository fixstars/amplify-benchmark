# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import json
import os
import shutil
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import boto3

from ..client_config.base import ClientConfig
from ..problem.base import Problem
from ..result import BenchmarkResult
from ..runner import ParallelRunner, Runner
from .parser import parse_input_data


def cli_benchmark_run(input_json: str, label: str, output: str, parallel: int, aws_profile: str, dry_run: bool):
    if output is None:
        output = str(Path(input_json).parent)

    # json 出力用の一時ディレクトリを作成
    with TemporaryDirectory() as temp_d:
        cli_benchmark_impl(Path(input_json), label, Path(temp_d), parallel, dry_run)

        # output pathのプロトコルを確認
        if urlparse(output).scheme == "":
            # 結果をローカルに保存
            _save_result_local(temp_d, output)

        elif urlparse(output).scheme == "s3":
            try:
                # AWS S3 に出力ファイルをアップロード
                session = _get_session(aws_profile)
                _push_s3(temp_d, output, session)
            except Exception:
                # AWS S3への保存に失敗した場合はローカルに保存
                alt_output = str(Path(input_json).parent.resolve())
                warnings.warn(f"Failed to save to S3 bucket. {output}\nSave to {alt_output} instead.")

                _save_result_local(temp_d, alt_output)


def cli_benchmark_impl(input_json: Path, label: str, output: Path, n_parallel: int, dry_run: bool):
    start_datetime: str = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    print(f"cli_benchmark_impl() {start_datetime}")

    job_template_list = parse_input_data(input_json)
    total_num_jobs = sum([num_samples for _, _, num_samples in job_template_list])

    # dry run の場合は QUBO model を作って終了
    if dry_run:
        [problem.make_model() for problem, _, _ in job_template_list]
        return
    results = _run_benchmark(job_template_list, n_parallel, label)

    _save_result_json(results, input_json, output, start_datetime, total_num_jobs)


def _run_benchmark(
    job_template_list: List[Tuple[Problem, ClientConfig, int]],
    n_parallel: int,
    label: str,
) -> BenchmarkResult:
    # define runner
    if n_parallel == 1:
        runner = Runner()
    elif n_parallel > 1:
        runner = ParallelRunner(nproc=n_parallel)
    else:
        raise TypeError(f"n_parallel: {n_parallel} must be a positive integer.")

    # register jobs
    for problem, client_config, num_samples in job_template_list:
        runner.register(problem, client_config, num_samples, label)

    # run
    results = runner.run()

    return results


def _save_result_json(
    results: BenchmarkResult,
    input: Path,
    output: Path,
    start_datetime: str,
    total_num_jobs: int,
):
    def output_file(input_file: Path, time: str):
        return input_file.stem + "_" + time + ".json"

    def error_logfile(input_file: Path, time: str):
        return input_file.stem + "_" + time + "_error.json"

    def report_by_json(file: Path, result: List[dict]):
        with open(file, "w") as f:
            f.write(json.dumps(result, indent=4))

    # 成功した job の結果を保存
    job_result = results.get_valid_summaries()
    job_failed = results.get_errors()
    print("total jobs: ", total_num_jobs)
    print("success jobs: ", len(job_result))
    print("error jobs: ", len(job_failed))
    print("Jobs not yet started: ", total_num_jobs - len(job_result) - len(job_failed))

    if len(job_result) > 0:
        report_by_json(output / output_file(input, start_datetime), job_result)

    # 失敗した job の例外内容を保存
    if len(job_failed) != 0:
        report_by_json(output / error_logfile(input, start_datetime), job_failed)


def _save_result_local(temp_d: str, output: str):
    output_path_obj = Path(output)
    if len(os.listdir(temp_d)) > 0:
        # 出力先ディレクトリを作成
        output_path_obj.mkdir(mode=0o755, parents=True, exist_ok=True)
        shutil.copytree(temp_d, output_path_obj, dirs_exist_ok=True)


def _push_s3(dname: str, s3_url: str, session: boto3.Session):
    print(f"Push results to {s3_url}")
    s3 = session.resource("s3")

    parsed_s3_url = urlparse(s3_url)
    s3bucket = s3.Bucket(parsed_s3_url.netloc)  # type: ignore
    for file in list(Path(dname).iterdir()):
        key = str(Path(parsed_s3_url.path) / file.name).lstrip("/")
        s3bucket.upload_file(str(file), key)


def _get_session(aws_profile: Optional[str] = None) -> boto3.Session:
    dotenv_path = Path().cwd() / ".env"

    if dotenv_path.exists():
        from dotenv import load_dotenv

        load_dotenv(dotenv_path)

    if aws_profile is not None:
        session = boto3.Session(profile_name=aws_profile)
    else:
        # 1. credentials from ENV
        AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY_ID = os.getenv("AWS_SECRET_ACCESS_KEY_ID")
        TARGET_ACCOUNT = os.getenv("TARGET_ACCOUNT")
        ROLE_NAME = os.getenv("ROLE_NAME")
        REGION_NAME = os.getenv("REGION_NAME")

        response = boto3.client(
            "sts",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY_ID,
            region_name=REGION_NAME,
        ).assume_role(
            RoleArn=f"arn:aws:iam::{TARGET_ACCOUNT}:role/{ROLE_NAME}",
            RoleSessionName=f"{ROLE_NAME}In{TARGET_ACCOUNT}",
        )

        session = boto3.Session(
            aws_access_key_id=response["Credentials"]["AccessKeyId"],
            aws_secret_access_key=response["Credentials"]["SecretAccessKey"],
            aws_session_token=response["Credentials"]["SessionToken"],
            region_name=REGION_NAME,
        )

    return session
