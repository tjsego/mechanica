import mechanica as m
from ipywidgets import widgets
import threading
import time
from ipyevents import Event
from IPython.display import display

flag = False
downflag = False
shiftflag = False
ctrlflag = False


def init(*args, **kwargs):
    global flag

    w = widgets.Image(value=m.system.image_data(), width=600)
    d = Event(source=w, watched_events=['mousedown', 'mouseup', 'mousemove', 'keyup', 'keydown', 'wheel'])
    no_drag = Event(source=w, watched_events=['dragstart'], prevent_default_action = True)
    d.on_dom_event(listen_mouse)
    run = widgets.ToggleButton(
        value = False,
        description='Run',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='run the simulation',
        icon = 'play'
        )
    pause = widgets.ToggleButton(
        value = False,
        description='Pause',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='pause the simulation',
        icon = 'pause'
        )

    reset = widgets.ToggleButton(
        value = False,
        description='Reset',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='reset the simulation',
        icon = 'stop'
        )

    def onToggleRun(b):
        global flag
        if run.value:
            run.button_style = 'success'
            pause.value = False
            pause.button_style = ''
            reset.value = False
            reset.button_style = ''
            flag = True
        else:
            run.button_style=''
            flag = False

    def onTogglePause(b):
        global flag
        if pause.value:
            pause.button_style = 'success'
            run.value = False
            run.button_style = ''
            reset.value = False
            reset.button_style = ''
            flag = False
        else:
            pause.button_style=''
            flag = True

    def onToggleReset(b):
        global flag
        if reset.value:
            reset.button_style = 'success'
            pause.value = False
            pause.button_style = ''
            run.value = False
            run.button_style = ''
            flag = False
            m.Universe.reset()
        else:
            reset.button_style=''
            #w = create_simulation()

    buttons = widgets.HBox([run, pause, reset])
    run.observe(onToggleRun,'value')
    pause.observe(onTogglePause,'value')
    reset.observe(onToggleReset,'value')

    box = widgets.VBox([w, buttons])
    display(box)

    # the simulator initializes creating the gl context on the creating thread.
    # this function gets called on that same creating thread, so we need to
    # release the context before calling in on the background thread.
    m.system.contextRelease()

    def background_threading():
        global flag
        m.system.contextMakeCurrent()
        while True:
            if flag:
                m.step()
            w.value = m.system.image_data()
            time.sleep(0.01)

        # done with background thead, release the context.
        m.system.contextRelease()


    t = threading.Thread(target=background_threading)
    t.start()



def run(*args, **kwargs):
    global flag

    flag = True

    # return true to tell Mechanica to not run a simulation loop,
    # jwidget runs it's one loop.
    return True


def listen_mouse(event):
    global downflag, shiftflag, ctrlflag
    if event['type'] == "mousedown":
        m.system.cameraInitMouse(m.MxVector2i([event['dataX'], event['dataY']]))
        downflag = True
    if event['type'] == "mouseup":
        downflag = False
    if event['type'] == "mousemove":
        if downflag and not shiftflag:
            m.system.cameraRotateMouse(m.MxVector2i([event['dataX'], event['dataY']]))
        if downflag and shiftflag:
            m.system.cameraTranslateMouse(m.MxVector2i([event['dataX'], event['dataY']]))

    shiftflag = True if event['shiftKey'] else False
    ctrlflag = True if event['ctrlKey'] else False
    if event['type'] == "wheel":
        m.system.cameraZoomBy(event['deltaY'])
    elif event['type'] == "keydown":
        key_code = event['code']
        if key_code == "KeyB":
            if shiftflag:
                m.system.cameraViewBottom()
        elif key_code == "KeyD":
            if ctrlflag:
                m.system.showDiscretization(not m.system.showingDiscretization())
            else:
                m.system.decorateScene(not m.system.decorated())
        elif key_code == "KeyF":
            if shiftflag:
                m.system.cameraViewFront()
        elif key_code == "KeyK":
            if shiftflag:
                m.system.cameraViewBack()
        elif key_code == "KeyL":
            if shiftflag:
                m.system.cameraViewLeft()
        elif key_code == "KeyR":
            if shiftflag:
                m.system.cameraViewRight()
            else:
                m.system.cameraReset()
        elif key_code == "KeyT":
            if shiftflag:
                m.system.cameraViewTop()
        elif key_code == "ArrowDown":
            if ctrlflag:
                if shiftflag:
                    m.system.cameraTranslateBackward()
                else:
                    m.system.cameraZoomOut()
            elif shiftflag:
                m.system.cameraRotateDown()
            else:
                m.system.cameraTranslateDown()
        elif key_code == "ArrowUp":
            if ctrlflag:
                if shiftflag:
                    m.system.cameraTranslateForward()
                else:
                    m.system.cameraZoomIn()
            elif shiftflag:
                m.system.cameraRotateUp()
            else:
                m.system.cameraTranslateUp()
        elif key_code == "ArrowLeft":
            if ctrlflag:
                m.system.cameraRollLeft()
            elif shiftflag:
                m.system.cameraRotateLeft()
            else:
                m.system.cameraTranslateLeft()
        elif key_code == "ArrowRight":
            if ctrlflag:
                m.system.cameraRollRight()
            elif shiftflag:
                m.system.cameraRotateRight()
            else:
                m.system.cameraTranslateRight()
