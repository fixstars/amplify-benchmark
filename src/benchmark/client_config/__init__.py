import glob
import importlib
from os.path import basename, dirname, isfile, join

modules = [
    importlib.import_module("." + basename(f)[:-3], __name__)
    for f in glob.glob(join(dirname(__file__), "*.py"))
    if isfile(f) and not f.endswith("__init__.py")
]

for m in modules:
    globals().update(vars(m))

del modules
