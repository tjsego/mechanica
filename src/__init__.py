from . import mx_config
from .mechanica import *
from . import lattice
from .particle_type import ClusterType, ParticleType

__all__ = ['forces', 'math', 'lattice']

if system.is_jupyter_notebook():
    from . import jwidget
    __all__.append('jwidget')
