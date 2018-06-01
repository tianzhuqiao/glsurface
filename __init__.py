import sys
if sys.version_info.major == 2:
    from glsurface import *
else:
    from .glsurface import *
