# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import os
from pathlib import Path
from urllib.parse import urljoin

import requests
import requests.adapters

current_dir = Path(__file__).parent


def download_file(url: str, dest: str) -> None:
    """
    Downloads a file from the specified URL and saves it to the specified file path.

    Args:
        url (str): The URL to download the file from.
        dest (str): The destination path to save the downloaded file to.

    Returns:
        None
    """

    s = requests.Session()
    s.mount("http://", requests.adapters.HTTPAdapter(max_retries=5))
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=5))
    response = s.get(url, timeout=60.0)
    response.raise_for_status()

    with open(dest, "wb") as f_out:
        f_out.write(response.content)


def download_tsp_file(instance: str, dest: str) -> None:
    TSPLIB_URL = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"
    tsp_dir = current_dir / "problem" / "data" / "TSPLIB"
    instance_file = tsp_dir / (instance + ".tsp")

    gz_file = instance_file.parent / (instance + ".tsp.gz")
    download_file(
        url=urljoin(TSPLIB_URL, gz_file.name),
        dest=str(gz_file),
    )

    # Extract the contents from the .gz file and save it to the specified path
    with gzip.open(gz_file, "rb") as f_in:
        with open(dest, "wb") as f_out:
            f_out.write(f_in.read())

    # Remove the downloaded .gz file
    os.remove(gz_file)


def download_qap_file(instance: str, dest: str) -> None:
    QAPLIB_URL = "https://coral.ise.lehigh.edu/wp-content/uploads/2014/07/"
    ins_url = urljoin(QAPLIB_URL, f"data.d/{instance}.dat")
    ins_file = str(Path(dest).parent / (instance + ".dat"))

    sol_url = urljoin(QAPLIB_URL, f"soln.d/{instance}.sln")
    sol_file = str(Path(dest).parent / (instance + ".sln"))

    if not Path(ins_file).exists():
        download_file(ins_url, ins_file)

    if not Path(sol_file).exists():
        download_file(sol_url, sol_file)


def download_maxcut_file(instance: str, dest: str) -> None:
    GSET_URL = "https://web.stanford.edu/~yyye/yyye/Gset/"
    K2000_URL = "https://raw.githubusercontent.com/hariby/SA-complete-graph/master/WK2000_1.rud"

    if instance.startswith("G"):
        ins_url = urljoin(GSET_URL, instance)
        download_file(ins_url, dest)
    elif instance == "K2000":
        download_file(K2000_URL, dest)
    else:
        raise FileNotFoundError(f"instance: {instance} is not found.")


def download_qplib_file(instance: str, dest: str) -> None:
    QPLIB_URL = "https://qplib.zib.de/qplib/"
    ins_url = urljoin(QPLIB_URL, f"{instance}.qplib")
    download_file(ins_url, dest)


def download_instance_file(name: str, instance: str, dest: str) -> None:
    if name.strip().lower() == "tsp":
        download_tsp_file(instance, dest)
    elif name.strip().lower() == "qap":
        download_qap_file(instance, dest)
    elif name.strip().lower() == "maxcut":
        download_maxcut_file(instance, dest)
    elif name.strip().lower() == "qplib":
        download_qplib_file(instance, dest)
