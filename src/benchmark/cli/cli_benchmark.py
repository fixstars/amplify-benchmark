import argparse
import datetime
import json
import os
import shutil
import sys
import warnings
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import boto3
from boto3.session import Session
from jsonschema import validate

from ..client_config.base import ClientConfig, get_client_config
from ..problem.base import Problem, gen_problem
from ..result import BenchmarkResult
from ..runner import ParallelRunner, Runner
from ..timer import timer


def parse_args(args):
    start_datetime: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="QUBO Benchmark")
    parser.add_argument(
        "input_json",
        type=str,
        help="Specify the path to the json file describing the benchmark settings.",
    )
    parser.add_argument(
        "-l",
        "--label",
        type=str,
        default=start_datetime,
        help="Specify the label for the benchmark.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help=(
            "Specify the directory where benchmark results are to be saved. "
            "You can also use the S3 protocol URL to save to an S3 bucket."
        ),
    )
    parser.add_argument(
        "--aws-profile",
        type=str,
        default=None,
        help=(
            "Specify the aws profile. This option is referenced "
            "when using the S3 protocol with the `--output` option."
        ),
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        default=1,
        help="Specifies the number of parallel executions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "It even builds the QUBO model based on the input json configuration. " "It does not run on the machine."
        ),
    )
    args = parser.parse_args(args)
    return args


def cli_benchmark_impl(label: str, input_json: Path, output: Path, n_parallel: int, dry_run: bool):
    start_datetime: str = datetime.datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

    job_template_list = _parse_input_json(input_json)

    # dry run の場合は QUBO model を作って終了
    if dry_run:
        [problem.make_model() for problem, _, _ in job_template_list]
        return
    results = _run_benchmark(job_template_list, n_parallel, label)

    _save_result_json(results, input_json, output, start_datetime)


def cli_benchmark():
    args = parse_args(sys.argv[1:])

    label: str = args.label
    input_json: str = args.input_json
    output: str = args.output if args.output is not None else str(Path(input_json).parent)
    n_parallel: int = args.parallel
    dry_run: bool = args.dry_run
    aws_profile: Optional[str] = args.aws_profile

    # json 出力用の一時ディレクトリを作成
    with TemporaryDirectory() as temp_d:
        cli_benchmark_impl(label, Path(input_json), Path(temp_d), n_parallel, dry_run)

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


def _save_result_local(temp_d: str, output: str):
    output_path_obj = Path(output)
    if len(os.listdir(temp_d)) > 0:
        # 出力先ディレクトリを作成
        output_path_obj.mkdir(mode=0o755, parents=True, exist_ok=True)
        shutil.copytree(temp_d, output_path_obj, dirs_exist_ok=True)


def _push_s3(dname: str, s3_url: str, session: Session):
    print(f"Push results to {s3_url}")
    s3 = session.resource("s3")

    parsed_s3_url = urlparse(s3_url)
    s3bucket = s3.Bucket(parsed_s3_url.netloc)
    for file in list(Path(dname).iterdir()):
        key = str(Path(parsed_s3_url.path) / file.name).lstrip("/")
        s3bucket.upload_file(str(file), key)


@timer
def validation(input_json: dict):
    with open(Path(__file__).parent / "schemas" / "benchmark.json") as f:
        json_schema = json.load(f)
    validate(instance=input_json, schema=json_schema)


@timer
def _parse_input_json(filepath: Path) -> List[Tuple[Problem, ClientConfig, int]]:
    j = json.load(filepath.open())
    validation(j)

    client_json = j["client"]
    client_name: str = client_json["name"]
    client_settings: dict = {k: v for (k, v) in client_json.items() if k not in ["name", "parameters"]}
    client_default_parameters = client_json.get("parameters", dict())

    job_template_list: List[Tuple[Problem, ClientConfig, int]] = []

    for benchmark_group in j["jobs"]:
        num_samples = benchmark_group.get("num_samples", 1)

        # problem_parameters, client_parameters 共に array で指定可能. その場合全てのパラメータの組をjobに登録
        problem_json = benchmark_group["problem"]
        if type(problem_json) == dict:
            problem_json_list = [problem_json]
        else:
            problem_json_list = problem_json

        problem_list: List[Problem] = list()
        for p_json in problem_json_list:
            problem_class: str = p_json["class"]
            problem_instance: str = p_json["instance"]
            problem_parameters: dict = p_json.get("parameters", dict())
            if "path" in p_json:
                problem_parameters["path"] = p_json.get("path")
            problem_list.append(gen_problem(problem_class, problem_instance, **problem_parameters))

        job_client_json = benchmark_group.get("client", dict())
        if type(job_client_json) == dict:
            job_client_json_list = [job_client_json]
        else:
            job_client_json_list = job_client_json

        for problem, j_c_json in product(problem_list, job_client_json_list):
            settings = client_settings.copy()
            settings.update(j_c_json.get("settings", dict()))
            parameters = client_default_parameters.copy()
            parameters.update(j_c_json.get("parameters", dict()))
            client_config = get_client_config(settings, parameters, client_name)

            job_template_list.append((problem, client_config, num_samples))

    return job_template_list


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
):
    def outputfile(inputfile: Path, time: str):
        return inputfile.stem + "_" + time + ".json"

    def errorlogfile(inputfile: Path, time: str):
        return inputfile.stem + "_" + time + "_error.json"

    def report_by_json(file: Path, result: List[dict]):
        with open(file, "w") as ofile:
            ofile.write(json.dumps(result, indent=4))

    # 成功した job の結果を保存
    job_result = results.get_valid_summaries()
    job_failed = results.get_errors()
    num_jobs = len(job_result) + len(job_failed)

    print(f"{len(job_result)} (out of {num_jobs}) jobs finished successfully.")
    if len(job_result) > 0:
        file = output / outputfile(input, start_datetime)
        report_by_json(file, job_result)

    # 失敗した job の例外内容を保存
    if len(job_failed) != 0:
        file = output / errorlogfile(input, start_datetime)
        report_by_json(file, job_failed)

    # 成功すれば0、失敗すれば1
    return 0 if len(job_failed) == 0 else 1


def _get_session(aws_profile: Optional[str] = None) -> Optional[dict]:
    dotenv_path = Path().cwd() / ".env"

    if dotenv_path.exists():
        from dotenv import load_dotenv

        load_dotenv(dotenv_path)

    if aws_profile is not None:
        session = Session(profile_name=aws_profile)
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

        session = Session(
            aws_access_key_id=response["Credentials"]["AccessKeyId"],
            aws_secret_access_key=response["Credentials"]["SecretAccessKey"],
            aws_session_token=response["Credentials"]["SessionToken"],
            region_name=REGION_NAME,
        )

    return session
