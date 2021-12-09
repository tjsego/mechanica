"""
Derived from cell_sorting.py, demonstrating runtime-control of GPU acceleration with CUDA.

A callback is implemented such that every time the key "S" is pressed on the keyboard,
Mechanica will test and report CPU vs. GPU performance.

Note that this demo will not run for Mechanica installations that do not have GPU support enabled.
"""
import mechanica as mx
import numpy as np
from time import time_ns

# Test for GPU support
if not mx.has_cuda:
    raise EnvironmentError("This installation of Mechanica was not installed with CUDA support.")

# total number of cells
A_count = 5000
B_count = 5000

# potential cutoff distance
cutoff = 3

# dimensions of universe
dim = [20., 20., 20.]

# new simulator
mx.init(dim=dim, cutoff=cutoff)


class AType(mx.ParticleType):
    mass = 40
    radius = 0.4
    dynamics = mx.Overdamped
    style = {'color': 'red'}


A = AType.get()


class BType(mx.ParticleType):
    mass = 40
    radius = 0.4
    dynamics = mx.Overdamped
    style = {'color': 'blue'}


B = BType.get()

# create three potentials, for each kind of particle interaction
pot_aa = mx.Potential.morse(d=3, a=5, max=3)
pot_bb = mx.Potential.morse(d=3, a=5, max=3)
pot_ab = mx.Potential.morse(d=0.3, a=5, max=3)


# bind the potential with the *TYPES* of the particles
mx.bind.types(pot_aa, A, A)
mx.bind.types(pot_bb, B, B)
mx.bind.types(pot_ab, A, B)

# create a random force. In overdamped dynamcis, we neeed a random force to
# enable the objects to move around, otherwise they tend to get trapped
# in a potential
rforce = mx.Force.random(mean=0, std=50)

# bind it just like any other force
mx.bind.force(rforce, A)
mx.bind.force(rforce, B)

# create particle instances, for a total A_count + B_count cells
for p in np.random.random((A_count, 3)) * 15 + 2.5:
    A(p)

for p in np.random.random((B_count, 3)) * 15 + 2.5:
    B(p)


# Configure GPU acceleration
cuda_config_engine: mx.EngineCUDAConfig = mx.Simulator.getCUDAConfig().engine
cuda_config_engine.setThreads(numThreads=int(32))


# Implement callback: when "S" is pressed on the keyboard, run some steps on and off the GPU and compare performance
def benchmark(e: mx.KeyEvent):
    if e.key_name != "s":
        return

    test_time = 100 * mx.Universe.dt

    print('**************')
    print(' Benchmarking ')
    print('**************')
    print('Sending engine to GPU...', cuda_config_engine.toDevice())
    tgpu_i = time_ns()
    mx.step(test_time)
    tgpu_f = time_ns() - tgpu_i
    print('Returning engine from GPU...', cuda_config_engine.fromDevice())
    print('Execution time (GPU):', tgpu_f / 1E6, 'ms')

    tcpu_i = time_ns()
    mx.step(test_time)
    tcpu_f = time_ns() - tcpu_i
    print('Execution time (CPU):', tcpu_f / 1E6, 'ms')

    print('Measured speedup:', tcpu_f / tgpu_f if tgpu_f > 0 else 0)


mx.on_keypress(invoke_method=benchmark)


# run the simulator
mx.Simulator.show()
