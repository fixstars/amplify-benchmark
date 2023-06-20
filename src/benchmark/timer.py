# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from functools import wraps
from logging import INFO, basicConfig, getLogger

basicConfig(level=INFO, format="{asctime} [pid:{pid}] [{levelname:.4}]: {message}", style="{")
logger = getLogger(__name__)


def print_log(message: str):
    logger.info(message, extra={"pid": os.getpid()})


def timer(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        # 処理開始直前の時間
        start = time.time()

        # 処理実行
        result = func(*args, **kargs)

        # 処理終了直後の時間から処理時間を算出
        elapsed_time = time.time() - start

        # 処理時間を出力
        print_log(f"{elapsed_time * 1000:>9.2f} ms in {func.__module__}.{func.__name__}")
        return result

    return wrapper
