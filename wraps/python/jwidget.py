import mechanica as m
from ipywidgets import widgets
import threading
import time
from ipyevents import Event
from IPython.display import display
import os

# When rendering docs, the behvaior of ``show`` is altered to only show the rendering window, run for
# a certain number of steps and then be done.
_rendering_docs = os.environ.get('MXDOCS', None) == '1'
docs_steps_term = os.environ.get('MXDOCS_RENDERSTEPS', 1000)

window_width = 600
"""Width of render window"""

flag = False
downflag = False
shiftflag = False
ctrlflag = False

html_controls = """
<table border="1">
    <tr>
        <th colspan = "2"> Mouse Controls</th>
    </tr>
    <tr>
        <th>Button</th>
        <th>Function</th>
    </tr>
    <tr>
        <td>Click</td>
        <td>Rotate camera</td>
    </tr>
    <tr>
        <td>Shift + Click</td>
        <td>Translate camera</td>
    </tr>
    
    <tr>
        <th colspan = "2"> Keyboard Controls</th>
    </tr>
    <tr>
        <th>Button</th>
        <th>Function</th>
    </tr>
    <tr>
        <td>D</td>
        <td>Toggle scene decorations</td>
    </tr>
    <tr>
        <td>L</td>
        <td>Toggle lagging</td>
    </tr>
    <tr>
        <td>R</td>
        <td>Reset camera</td>
    </tr>
    <tr>
        <td>Arrow down</td>
        <td>Translate camera down</td>
    </tr>
    <tr>
        <td>Arrow left</td>
        <td>Translate camera left</td>
    </tr>
    <tr>
        <td>Arrow right</td>
        <td>Translate camera right</td>
    </tr>
    <tr>
        <td>Arrow up</td>
        <td>Translate camera up</td>
    </tr>
    <tr>
        <td>Ctrl + D</td>
        <td>Toggle discretization rendering</td>
    </tr>
    <tr>
        <td>Ctrl + arrow down</td>
        <td>Zoom camera out</td>
    </tr>
    <tr>
        <td>Ctrl + arrow left</td>
        <td>Rotate camera left</td>
    </tr>
    <tr>
        <td>Ctrl + arrow right</td>
        <td>Rotate camera right</td>
    </tr>
    <tr>
        <td>Ctrl + arrow up</td>
        <td>Zoom camera in</td>
    </tr>
    <tr>
        <td>Shift + B</td>
        <td>Bottom view</td>
    </tr>
    <tr>
        <td>Shift + F</td>
        <td>Front view</td>
    </tr>
    <tr>
        <td>Shift + K</td>
        <td>Back view</td>
    </tr>
    <tr>
        <td>Shift + L</td>
        <td>Left view</td>
    </tr>
    <tr>
        <td>Shift + R</td>
        <td>Right view</td>
    </tr>
    <tr>
        <td>Shift + T</td>
        <td>Top view</td>
    </tr>
    <tr>
        <td>Shift + arrow down</td>
        <td>Rotate camera down</td>
    </tr>
    <tr>
        <td>Shift + arrow left</td>
        <td>Rotate camera left</td>
    </tr>
    <tr>
        <td>Shift + arrow right</td>
        <td>Rotate camera right</td>
    </tr>
    <tr>
        <td>Shift + arrow up</td>
        <td>Rotate camera up</td>
    </tr>
    <tr>
        <td>Shift + Ctrl + arrow down</td>
        <td>Translate backward</td>
    </tr>
    <tr>
        <td>Shift + Ctrl + arrow up</td>
        <td>Translate forward</td>
    </tr>
</table>
"""
"""
HTML source for controls display widget
"""


def init(*args, **kwargs):
    pass


def show():
    global flag

    # If building docs, just do stepping, output image and be done
    if _rendering_docs:
        from IPython.display import Image
        for _ in range(docs_steps_term):
            m.step()
        display(Image(m.system.image_data(), width=window_width, format='png'))
        return

    w = widgets.Image(value=m.system.image_data(), width=window_width)
    d = Event(source=w, watched_events=['mousedown', 'mouseup', 'mousemove', 'keyup', 'keydown', 'wheel'])
    no_drag = Event(source=w, watched_events=['dragstart'], prevent_default_action=True)
    d.on_dom_event(listen_mouse)
    tb_run = widgets.ToggleButton(
        value=False,
        description='Run',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='run the simulation',
        icon='play'
        )
    tb_pause = widgets.ToggleButton(
        value=False,
        description='Pause',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='pause the simulation',
        icon='pause'
        )

    def onToggleRun(b):
        global flag
        if tb_run.value:
            tb_run.button_style = 'success'
            tb_pause.value = False
            tb_pause.button_style = ''
            flag = True
        else:
            tb_run.button_style = ''
            tb_pause.value = True
            tb_pause.button_style = 'success'
            flag = False

    def onTogglePause(b):
        global flag
        if tb_pause.value:
            tb_pause.button_style = 'success'
            tb_run.value = False
            tb_run.button_style = ''
            flag = False
        else:
            tb_pause.button_style = ''
            tb_run.value = True
            tb_run.button_style = 'success'
            flag = True

    tb_run.observe(onToggleRun, 'value')
    tb_pause.observe(onTogglePause, 'value')

    buttons_exec = widgets.HBox([tb_run, tb_pause])

    view_front = widgets.Button(
        value=False,
        description='View: Front', 
        disabled=False, 
        button_style='', 
        tooltip='set the camera to the front view'
    )
    view_back = widgets.Button(
        value=False,
        description='View: Back', 
        disabled=False, 
        button_style='', 
        tooltip='set the camera to the back view'
    )
    view_right = widgets.Button(
        value=False,
        description='View: Right', 
        disabled=False, 
        button_style='', 
        tooltip='set the camera to the right view'
    )
    view_left = widgets.Button(
        value=False,
        description='View: Left', 
        disabled=False, 
        button_style='', 
        tooltip='set the camera to the left view'
    )
    view_top = widgets.Button(
        value=False,
        description='View: Top', 
        disabled=False, 
        button_style='', 
        tooltip='set the camera to the top view'
    )
    view_bottom = widgets.Button(
        value=False,
        description='View: Bottom', 
        disabled=False, 
        button_style='', 
        tooltip='set the camera to the bottom view'
    )
    view_reset = widgets.Button(
        value=False, 
        description='View: Reset', 
        disabled=False, 
        button_style='', 
        tooltip='reset the camera'
    )
    
    view_front.on_click(lambda b: m.system.cameraViewFront())
    view_back.on_click(lambda b: m.system.cameraViewBack())
    view_right.on_click(lambda b: m.system.cameraViewRight())
    view_left.on_click(lambda b: m.system.cameraViewLeft())
    view_top.on_click(lambda b: m.system.cameraViewTop())
    view_bottom.on_click(lambda b: m.system.cameraViewBottom())
    view_reset.on_click(lambda b: m.system.cameraReset())
    
    buttons_view = widgets.HBox([view_front, view_back, view_right, view_left, view_top, view_bottom, view_reset])

    controls_content = widgets.HTML(value=html_controls)
    acc_container = widgets.Accordion(children=[buttons_view, controls_content])
    acc_container.selected_index = None
    acc_container.set_title(0, 'Views')
    acc_container.set_title(1, 'Controls')

    box = widgets.VBox([w, buttons_exec, acc_container])
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
