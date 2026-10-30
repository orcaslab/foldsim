"""The routability fit (foldsim_storage.py fit-route) over the V80 points
file beside the driver: argv[1] is the foldsim driver, beside the modules."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(sys.argv[1])))
import foldsim_storage as s  # noqa: E402

sys.argv = ['foldsim_storage.py', 'fit-route']
s.main()
