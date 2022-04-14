import mechanica as mx
import os
from os.path import abspath, dirname, isfile, join
import threading

this_dir = dirname(abspath(__file__))

mx.Logger.setLevel(mx.Logger.INFORMATION)

mx.init(windowless=True, window_size=[1024, 1024])

mx.system.gl_info()


class NaType(mx.ParticleType):
    radius = 0.4
    style = {"color": "orange"}


class ClType(mx.ParticleType):
    radius = 0.25
    style = {"color": "spablue"}


uc = mx.lattice.bcc(0.9, [NaType.get(), ClType.get()])

mx.lattice.create_lattice(uc, [10, 10, 10])


def threaded_steps(steps):

    mx.system.contextHasCurrent()

    mx.system.contextMakeCurrent()

    mx.step()

    mx.system.screenshot(os.path.join(this_dir, 'threaded.jpg'))

    mx.system.contextRelease()


mx.system.screenshot(os.path.join(this_dir, 'main.jpg'))

mx.system.contextRelease()

thread = threading.Thread(target=threaded_steps, args=(1,))

thread.start()

thread.join()

mx.system.contextHasCurrent()

mx.system.contextMakeCurrent()

mx.step()

mx.system.screenshot(os.path.join(this_dir, 'main2.jpg'), False, [1, 1, 1])

ss_files = [join(this_dir, ssn + '.jpg') for ssn in ['main', 'main2', 'threaded']]
if any([not isfile(ssf) for ssf in ss_files]):
    raise RuntimeError('An image was not exported')
[os.remove(ssf) for ssf in ss_files]


def test_pass():
    pass
