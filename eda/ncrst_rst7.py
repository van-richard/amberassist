import os
import sys
from parmed.amber import Rst7


ncrst = Rst7(sys.argv[1])
pre, ext = os.path.splitext(sys.argv[1])
ncrst.write(pre + ".rst7", netcdf=False)
