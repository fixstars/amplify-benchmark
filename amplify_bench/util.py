# Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json


def dict_to_hash(key_dict: dict) -> str:
    key = json.dumps(key_dict, sort_keys=True)
    hs = hashlib.md5(key.encode()).hexdigest()
    return hs
