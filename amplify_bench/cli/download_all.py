# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import glob
import gzip
import os
import warnings
from pathlib import Path
from shutil import unpack_archive
from urllib.parse import urlparse

import vrplib
from tqdm import tqdm

from ..downloader import download_file, download_instance_file

data_dir = Path(__file__).parent / ".." / "problem" / "data"
tsp_dir = data_dir / "TSPLIB"
qap_dir = data_dir / "QAPLIB"
maxcut_dir = data_dir / "GSET"
qplib_dir = data_dir / "QPLIB"
cvrp_dir = data_dir / "CVRPLIB"


def cleanup(filepath: str):
    Path(filepath).unlink(missing_ok=True)


def download_tsp_all():
    URL = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz"
    dest = tsp_dir / Path(urlparse(URL).path).name

    download_file(URL, str(dest))
    unpack_archive(filename=str(dest), extract_dir=tsp_dir, format="gztar")

    # 解凍してできた全ての tsp.gz ファイルをunzipする
    tsp_gz_file_list = glob.glob(str(tsp_dir / "*.tsp.gz"))
    for tsp_gz_file in tsp_gz_file_list:
        tsp_file = tsp_gz_file.strip(".gz")
        with gzip.open(tsp_gz_file, "rb") as f_in:
            with open(tsp_file, "wb") as f_out:
                f_out.write(f_in.read())

    gz_file_list = glob.glob(str(tsp_dir / "*.gz"))
    for gz_file in gz_file_list:
        os.remove(gz_file)


def clean_tsp():
    file_list = glob.glob(str(tsp_dir / "*.tsp"))
    list(map(cleanup, file_list))


def download_qap_all():
    # URL = "https://www.opt.math.tugraz.at/qaplib/data.d/qapdata.tar.gz"
    URL = "http://coral.ise.lehigh.edu/wp-content/uploads/2014/07/qapdata.tar.gz"
    gz_file = qap_dir / "qapdata.tar.gz"

    download_file(URL, str(gz_file))
    unpack_archive(filename=str(gz_file), extract_dir=qap_dir, format="gztar")
    os.remove(gz_file)


def clean_qap():
    file_list = glob.glob(str(qap_dir / "*.dat"))
    list(map(cleanup, file_list))


def download_maxcut_all():
    available_instances: list[str] = []
    with open(maxcut_dir / "instance_list.txt", "r") as f:
        available_instances = [line.strip() for line in f.readlines()]

    for instance in tqdm(available_instances):
        instance_file = maxcut_dir / instance
        if not instance_file.exists():
            download_instance_file("maxcut", instance, str(instance_file))


def clean_maxcut():
    available_instances: list[str] = []
    with open(maxcut_dir / "instance_list.txt", "r") as f:
        available_instances = [line.strip() for line in f.readlines()]

    for instance in tqdm(available_instances):
        instance_file = maxcut_dir / instance
        if instance_file.exists():
            cleanup(str(instance_file))


def download_qplib_all():
    available_instances: list[str] = []
    with open(qplib_dir / "instance_list.txt", "r") as f:
        available_instances = [line.strip() for line in f.readlines()]

    for instance in tqdm(available_instances):
        instance_file = qplib_dir / (f"{instance}.qplib")
        if not instance_file.exists():
            download_instance_file("qplib", instance, str(instance_file))


def clean_qplib():
    file_list = glob.glob(str(qplib_dir / "*.qplib"))
    list(map(cleanup, file_list))


def download_cvrp_all():
    available_instances: list[str] = []
    with open(cvrp_dir / "instance_list.txt", "r") as f:
        available_instances = [line.strip() for line in f.readlines()]

    for instance in tqdm(available_instances):
        instance_file = cvrp_dir / (instance + ".vrp")
        if not instance_file.exists():
            vrplib.download_instance(instance, str(instance_file))


def clean_cvrp():
    file_list = glob.glob(str(cvrp_dir / "*.vrp"))
    list(map(cleanup, file_list))


def cli_download_all(name: str) -> None:
    if name.strip().lower() == "tsp":
        download_tsp_all()
    elif name.strip().lower() == "qap":
        download_qap_all()
    elif name.strip().lower() == "maxcut":
        download_maxcut_all()
    elif name.strip().lower() == "qplib":
        download_qplib_all()
    elif name.strip().lower() == "cvrp":
        download_cvrp_all()
    elif name.strip().lower() == "sudoku":
        warnings.warn("There are no downloadable Sudoku instance files.")
    else:
        warnings.warn(f"The specified class name: {name} is not supported")


def cli_download_clean(name: str) -> None:
    if name.strip().lower() == "tsp":
        clean_tsp()
    elif name.strip().lower() == "qap":
        clean_qap()
    elif name.strip().lower() == "maxcut":
        clean_maxcut()
    elif name.strip().lower() == "qplib":
        clean_qplib()
    elif name.strip().lower() == "cvrp":
        clean_cvrp()
    elif name.strip().lower() == "sudoku":
        pass
    else:
        warnings.warn(f"The specified class name: {name} is not supported")
