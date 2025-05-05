from sys import _getframe, settrace

 # (we have ~37 lines of helper code here)


def _trace(frame, event, arg):
    pass  # TODO (we have ~13 lines)


# register our tracing function in this frame and all above
settrace(_trace)
frame = _getframe().f_back
while frame:
    frame.f_trace = _trace
    frame = frame.f_back


# ensure that goto/label lines are executable
class _Label:
    def __getattr__(self, name):
        return None


goto = _Label()
label = _Label()
