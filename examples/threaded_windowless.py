import mechanica as mx
import threading

mx.Logger.setLevel(mx.Logger.INFORMATION)

mx.init(windowless=True, window_size=[1024, 1024])

print(mx.system.gl_info())


class NaType(mx.ParticleType):
    radius = 0.4
    style = {"color": "orange"}


class ClType(mx.ParticleType):
    radius = 0.25
    style = {"color": "spablue"}


uc = mx.lattice.bcc(0.9, [NaType.get(), ClType.get()])

mx.lattice.create_lattice(uc, [10, 10, 10])


def threaded_steps(steps):

    print('thread start')

    print("thread, calling context_has_current()")
    mx.system.contextHasCurrent()

    print("thread, calling context_make_current())")
    mx.system.contextMakeCurrent()

    mx.step()

    mx.system.screenshot('threaded.jpg')

    print("thread calling release")
    mx.system.contextRelease()

    print("thread done")


print("main writing main.jpg")

mx.system.screenshot('main.jpg')

print("main calling context_release()")
mx.system.contextRelease()

thread = threading.Thread(target=threaded_steps, args=(1,))

thread.start()

thread.join()

print("main thread context_has_current: ", mx.system.contextHasCurrent())

mx.system.contextMakeCurrent()

mx.step()

mx.system.screenshot('main2.jpg', decorate=False, bgcolor=[1, 1, 1])

print("all done")
