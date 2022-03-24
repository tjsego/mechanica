from . import mx_config
from .mechanica import *
from . import lattice
from . import models
from .particle_type import ClusterType, ParticleType

__all__ = ['forces', 'math', 'lattice']

if system.is_jupyter_notebook():
    from . import jwidget
    __all__.append('jwidget')
    show = jwidget.show

if system.is_terminal_interactive():
    from .mechanica import _onIPythonNotReady

    def _input_hook(context):
        while not context.input_is_ready():
            _onIPythonNotReady()

        return None

    from .mechanica import _setIPythonInputHook
    _setIPythonInputHook(_input_hook)
