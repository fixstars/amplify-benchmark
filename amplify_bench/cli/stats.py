# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
from jsonschema import validate
from mypy_boto3_s3 import S3Client, S3ServiceResource
from tqdm import tqdm

from ..timer import timer


def cli_stats(input_jsons: Union[str, list[str]], output: Optional[str], aws_profile: str):
    cli_stats_impl(input_jsons, output, aws_profile)


@timer
def validation(data: list[dict], label: str = "") -> list[dict]:
    with open(Path(__file__).parent / "schemas" / "result.json") as f:
        json_schema = json.load(f)

    try:
        validate(instance=data, schema=json_schema)
        return data
    except Exception:
        warnings.warn(f"Skipped loading because failed to validate: {label}")
        return []


def delete_y(d):
    if "instance" in d:
        del d["instance"]
    return d


def get_logspace(start, end, num_steps):
    stop = num_steps * (np.log10(end) / np.log10(end / start))
    base = pow(10, np.log10(end) / stop)
    return np.logspace(stop - num_steps, stop, num_steps, base=base)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def cli_stats_impl(input_jsons: Union[str, list[str]], output: Optional[str], aws_profile: str):
    input_data = []
    for input_json_path in tqdm(input_jsons):
        data_li = []
        if urlparse(input_json_path).scheme == "":
            data_li = _load_jsons(input_json_path)
        elif urlparse(input_json_path).scheme == "s3":
            data_li = _load_jsons_from_s3(input_json_path, aws_profile)
        input_data.extend(data_li)

    stats_json = format_result_json_to_stats_json(input_data)

    # 1. デフォルト: cwd() / "stats.json"
    # 2. ディレクトリ名: output / "stats.json"
    # 3. ファイル名: output
    output_path = Path()
    if output is None:
        output_path = Path().cwd() / "stats.json"
    else:
        output_path = Path(output)

        if output_path.suffix == "":
            os.makedirs(output_path, exist_ok=True)
            output_path /= "stats.json"
        elif output_path.suffix == ".json":
            os.makedirs(output_path.parent, exist_ok=True)
            pass
        else:
            raise RuntimeError(
                f"{output} is not supported."
                "The `output` argument must be a directory name or a filename with a json suffix."
            )

    with open(output_path, "w") as f:
        json.dump(stats_json, f, cls=MyEncoder)


def _load_jsons(json_path_str: str) -> list:
    """Load data from local json file(s).

    Args:
        json_path_str (str): json file/directory path

    Returns:
        list: data of json file(s)
    """
    json_path = Path(json_path_str)
    if json_path.is_file():
        with open(json_path, mode="rt", encoding="utf-8") as file:
            data = json.load(file)
    elif json_path.is_dir():
        data = []
        for j_p in json_path.rglob("*.json"):
            d = _load_jsons(str(j_p))
            data.extend(d)
    else:
        raise FileNotFoundError(f"{json_path_str} is not found.")
    return validation(data, label=json_path_str)


def _load_jsons_from_s3(s3_url: str, aws_profile: Optional[str]) -> list:
    """Load data from json file(s) on S3.

    Args:
        s3_url (str): URL string of the S3 protocol where the json file is located
        aws_profile (str, optional): Specify the aws profile.

    Returns:
        list: data of json file(s)
    """

    def _get_object(s3_client: S3Client, bucket_name: str, key: str):
        obj = s3_client.get_object(Bucket=bucket_name, Key=key)
        json_txt = obj["Body"].read()
        return json.loads(json_txt)

    def _get_list_objects(s3_client: S3Client, bucket_name: str, prefix: str) -> list:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=key,
        )
        contents = []
        contents.extend(response["Contents"])
        while "NextContinuationToken" in response:
            token = response["NextContinuationToken"]
            response = s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=key,
                ContinuationToken=token,
            )
            contents.extend(response["Contents"])
        return contents

    print(f"Pull results from {s3_url}")
    session: boto3.Session = _get_session(aws_profile)
    parsed_s3_url = urlparse(s3_url)
    bucket_name = parsed_s3_url.netloc
    key = parsed_s3_url.path.lstrip("/")

    s3: S3ServiceResource = session.resource("s3")  # type: ignore
    s3_client = s3.meta.client
    data_li = []
    if Path(key).suffix == "json":
        # path文字列の拡張子が`.json`ならそのまま取得
        object_json = _get_object(s3_client, bucket_name, key)
        data_li.extend(object_json)
    else:
        # それ以外ならprefixとして扱い中身を全て取得
        contents = _get_list_objects(s3_client, bucket_name, key)
        for obj_info in contents:
            if obj_info["Key"].split(".")[-1] == "json":
                object_json = _get_object(s3_client, bucket_name, obj_info["Key"])
                data_li.extend(object_json)

    return data_li


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


def format_result_json_to_stats_json(result_json):
    def problem_add_group_id(df):
        df_tmp = df["problem"].apply(pd.Series)
        df_tmp["benchmarks"] = df["group_id"]
        return df_tmp

    def client_add_group_id(df):
        df_tmp = df["client"].apply(pd.Series)
        df_tmp["benchmarks"] = df["group_id"]
        return df_tmp

    def pre_process(df):
        df_tmp = df["result"].apply(pd.Series)
        df_tmp["group_id"] = df["group_id"]
        df_tmp["job_id"] = df["job_id"]
        df_tmp["label"] = df["label"]
        df_tmp["best_known"] = df["problem"].apply(pd.Series)["best_known"]
        df_tmp["specified_time"] = df["client"].apply(pd.Series)["timeout_like"].apply(pd.Series)["value"]
        df_tmp["client_name"] = df["client"].apply(pd.Series)["name"]
        df_tmp = df_tmp.reset_index(drop=True)
        return df_tmp

    def expand_solutions(df):
        df_tmp = df.explode("solutions").copy()
        df_tmp = pd.concat([df_tmp, df_tmp["solutions"].apply(pd.Series)], axis=1)
        df_tmp = df_tmp.reset_index(drop=True)
        df_tmp["broken"] = df_tmp["constraints"].apply(pd.Series)["broken"]
        return df_tmp

    def filter_best_solutions(df):
        return df.loc[
            df.groupby("job_id").apply(
                lambda x: x[x["is_feasible"]]["target_energy"].idxmin()
                if (x["is_feasible"]).any()
                else x["broken"].idxmin()
            )
        ]

    def agg_solutions(df):
        df["target_energy"] = np.where(df["is_feasible"], df["target_energy"], np.nan)
        return (
            df.groupby(["group_id", "label", "specified_time"])
            .agg(
                {
                    "job_id": "count",
                    "target_energy": list,
                    "is_feasible": list,
                    "broken": list,
                    "sampling_time": list,
                    "best_known": "first",
                }
            )
            .rename({"job_id": "num_samples"}, axis=1)
            .reset_index()
        )

    def sum_label(df):
        df_tmp = (
            df.groupby(["group_id", "specified_time"])
            .agg(
                {
                    "num_samples": sum,
                    "target_energy": sum,
                    "is_feasible": sum,
                    "sampling_time": sum,
                    "broken": sum,
                    "best_known": "first",
                    "label": lambda x: "",
                }
            )
            .reset_index()
        )
        return pd.concat([df, df_tmp]).drop_duplicates(["group_id", "label", "specified_time"])

    def calc_mean_sampling_time(df):
        df["sampling_time_mean"] = df["sampling_time"].apply(lambda x: pd.Series(x).mean())
        return df

    def calc_feasible_rate(df):
        df["feasible_rate"] = df["is_feasible"].apply(lambda x: pd.Series(x).sum()) / df["num_samples"]
        return df

    def calc_reach_best_rate(df):
        df["reach_best_rate"] = (
            df.apply(
                lambda x: sum(
                    [
                        1
                        if x["target_energy"][i] == x["target_energy"][i] and (x["target_energy"][i] <= x["best_known"])
                        else 0
                        for i in range(x["num_samples"])
                    ]
                ),
                axis=1,
            )
            / df["num_samples"]
        )
        return df

    def calc_raw_data(df):
        df["raw_data"] = df.apply(
            lambda x: [
                {
                    "sampling_time": x["sampling_time"][i],
                    "target_energy": x["target_energy"][i],
                    "broken": x["broken"][i],
                }
                for i in range(x["num_samples"])
                if x["target_energy"][i] == x["target_energy"][i]
            ],
            axis=1,
        )
        return df

    def calc_time_to_solution(df):
        def calc_time_to_solution_row(x, target_percentage):
            if x["num_samples"] == 0:
                return None
            num_reach_target = sum(
                [
                    1
                    if x["target_energy"][i] == x["target_energy"][i]
                    and (x["target_energy"][i] <= x["best_known"] + abs(x["best_known"] * target_percentage / 100))
                    else 0
                    for i in range(x["num_samples"])
                ]
            )

            if num_reach_target == 0:
                return None
            if num_reach_target == x["num_samples"]:
                return x["sampling_time_mean"]
            time_to_solution = (
                x["sampling_time_mean"] * np.log(1 - 0.99) / np.log(1 - num_reach_target / x["num_samples"])
            )
            return time_to_solution

        df["time_to_solution"] = df.apply(
            lambda x: {
                "0%": calc_time_to_solution_row(x, 0),
                "1%": calc_time_to_solution_row(x, 1),
                "5%": calc_time_to_solution_row(x, 5),
                "10%": calc_time_to_solution_row(x, 10),
                "20%": calc_time_to_solution_row(x, 20),
                "50%": calc_time_to_solution_row(x, 50),
            },
            axis=1,
        )
        return df

    def make_results(df):
        return (
            df.groupby(["group_id", "label"])
            .apply(
                lambda x: [
                    {
                        "specified_time": x["specified_time"].iloc[i],
                        "num_samples": x["num_samples"].iloc[i],
                        "feasible_rate": x["feasible_rate"].iloc[i],
                        "reach_best_rate": x["reach_best_rate"].iloc[i],
                        "time_to_solution": x["time_to_solution"].iloc[i],
                        "raw_data": x["raw_data"].iloc[i],
                    }
                    for i in range(len(x))
                ]
            )
            .reset_index()
            .groupby("group_id")
            .apply(lambda x: {k: v for k, v in zip(x["label"], x[0])})
        )

    def filter_fixstars_result(df):
        df_tmp = df["result"].apply(pd.Series)
        df_tmp["group_id"] = df["group_id"]
        df_tmp["client_name"] = df["client"].apply(pd.Series)["name"]
        df_tmp["job_id"] = df["job_id"]
        df_tmp["label"] = df["label"]
        df_tmp["specified_time"] = df["client"].apply(pd.Series)["timeout_like"].apply(pd.Series)["value"]
        df_tmp["best_known"] = df["problem"].apply(pd.Series)["best_known"]
        df_tmp = df_tmp[df_tmp["client_name"] == "FixstarsClient"]
        df_tmp = df_tmp.reset_index(drop=True)
        return df_tmp

    def get_logspace(start, end, num_steps):
        if start == end:
            return np.array([start])
        stop = num_steps * (np.log10(end) / np.log10(end / start))
        base = pow(10, np.log10(end) / stop)
        ret = np.logspace(stop - num_steps, stop, num_steps, base=base)
        ret[0] = start
        ret[-1] = end
        return ret

    def add_time_sampling_point(df):
        df_tmp = (
            df.groupby("group_id")
            .agg({"job_id": set, "sampling_time": lambda x: list(get_logspace(x.min(), x.max(), 100))})
            .explode("job_id")
            .explode("sampling_time")
            .reset_index()
        )
        df_tmp["is_sampling_point"] = True
        df_tmp = (
            pd.concat([df, df_tmp])
            .sort_values(["group_id", "job_id", "sampling_time"])
            .fillna({"is_sampling_point": False})
            .groupby(["group_id", "job_id"], group_keys=True)
            .apply(lambda x: x.ffill())
            .query("is_sampling_point")
            .reset_index(drop=True)
        )
        df_tmp["specified_time"] = df_tmp["sampling_time"]
        df_tmp["label"] = df_tmp.groupby("job_id")["label"].ffill().bfill()
        return df_tmp

    def calc_target_energy_describe(df):
        df_tmp = df.copy()
        df_tmp["target_energy_describe"] = (
            df["target_energy"]
            .apply(lambda x: pd.Series(x).describe())
            .apply(
                lambda x: {"min": x["min"], "25%": x["25%"], "50%": x["50%"], "75%": x["75%"], "max": x["max"]}, axis=1
            )
        )
        return df_tmp

    def calc_broken_describe(df):
        df_tmp = df.copy()
        df_tmp["broken_describe"] = (
            df_tmp["broken"]
            .apply(lambda x: pd.Series(x).describe())
            .apply(
                lambda x: {"min": x["min"], "25%": x["25%"], "50%": x["50%"], "75%": x["75%"], "max": x["max"]}, axis=1
            )
        )
        return df_tmp

    def make_history(df):
        return (
            df.groupby(["group_id", "label"])
            .apply(
                lambda x: [
                    {
                        "sampling_time": x["specified_time"].iloc[i],
                        "num_samples": x["num_samples"].iloc[i],
                        "feasible_rate": x["feasible_rate"].iloc[i],
                        "reach_best_rate": x["reach_best_rate"].iloc[i],
                        "time_to_solution": x["time_to_solution"].iloc[i],
                        "target_energy": x["target_energy_describe"].iloc[i],
                        "broken": x["broken_describe"].iloc[i],
                    }
                    for i in range(len(x))
                ]
            )
            .reset_index()
            .groupby("group_id")
            .apply(lambda x: {k: v for k, v in zip(x["label"], x[0])})
        )

    def make_benchmarks(df):
        df_tmp = df[["group_id"]].copy()
        df_tmp["problem_id"] = df["problem"].apply(pd.Series)["id"]
        df_tmp["client_id"] = df["client"].apply(pd.Series)["id"]
        df_tmp = df_tmp.drop_duplicates()
        df_tmp = df_tmp.set_index("group_id")
        df_tmp["results"] = (
            df.pipe(pre_process)
            .pipe(expand_solutions)
            .pipe(filter_best_solutions)
            .pipe(agg_solutions)
            .pipe(sum_label)
            .pipe(calc_mean_sampling_time)
            .pipe(calc_feasible_rate)
            .pipe(calc_reach_best_rate)
            .pipe(calc_raw_data)
            .pipe(calc_time_to_solution)
            .pipe(make_results)
        )
        if (df["client"].apply(pd.Series)["name"] == "FixstarsClient").any():
            df_tmp["history"] = (
                df.pipe(filter_fixstars_result)
                .pipe(expand_solutions)
                .pipe(add_time_sampling_point)
                .pipe(agg_solutions)
                .pipe(sum_label)
                .pipe(calc_mean_sampling_time)
                .pipe(calc_feasible_rate)
                .pipe(calc_reach_best_rate)
                .pipe(calc_time_to_solution)
                .pipe(calc_target_energy_describe)
                .pipe(calc_broken_describe)
                .pipe(make_history)
            )
        else:
            df_tmp["history"] = None
        return df_tmp

    df = pd.DataFrame(result_json)
    problems = json.loads(
        df.pipe(problem_add_group_id)
        .groupby("id")
        .agg(
            {
                "class": "first",
                "instance": "first",
                "parameters": "first",
                "best_known": "first",
                "benchmarks": lambda x: [{"group_id": i} for i in sorted(list(set(x)))],
            }
        )
        .to_json(orient="index")
    )
    clients = json.loads(
        df.pipe(client_add_group_id)
        .groupby("id")
        .agg(
            {
                "settings": "first",
                "parameters": "first",
                "version": "first",
                "name": "first",
                "benchmarks": lambda x: [{"group_id": i} for i in sorted(list(set(x)))],
            }
        )
        .to_json(orient="index")
    )
    df_benchmarks = make_benchmarks(df)
    benchmarks = json.loads(df_benchmarks[df_benchmarks["history"].isna()].dropna(axis=1).to_json(orient="index"))
    benchmarks.update(json.loads(df_benchmarks[~df_benchmarks["history"].isna()].to_json(orient="index")))

    stats_data = {
        "benchmarks": benchmarks,
        "problems": problems,
        "clients": clients,
    }

    return stats_data
