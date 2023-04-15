import hashlib
import json


def dict_to_hash(key_dict: dict) -> str:
    key = json.dumps(key_dict, sort_keys=True)
    hs = hashlib.md5(key.encode()).hexdigest()
    return hs
