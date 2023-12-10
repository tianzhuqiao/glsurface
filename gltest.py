import wx
import wx.lib.agw.aui as aui
import wx.py as py
import numpy as np
from glsurface import TrackingSurface
from _simxpm import run_xpm, pause_xpm


class RunApp(wx.App):
    def __init__(self):
        wx.App.__init__(self, redirect=False)

    def OnInit(self):
        frame = MainFrame()
        frame.Show(True)
        self.SetTopWindow(frame)
        self.frame = frame
        return True


class MainFrame(wx.Frame):
    ID_SHOW_CONSOLE = wx.NewIdRef()

    def __init__(self):
        wx.Frame.__init__(self, None, -1, 'glsurface demo', size=(410, 432))
        self._mgr = aui.AuiManager()

        # tell AuiManager to manage this frame
        self._mgr.SetManagedWindow(self)
        self.panel = SurfacePanel(self)
        self._mgr.AddPane(self.panel,
                          aui.AuiPaneInfo().Caption("glsurface").CenterPane())
        ns = {}
        ns['wx'] = wx
        ns['app'] = wx.GetApp()
        ns['frame'] = self
        self.shell = py.shell.Shell(self, -1, locals=ns)
        self._mgr.AddPane(
            self.shell,
            aui.AuiPaneInfo().Caption('console').DestroyOnClose(False).Bottom(
            ).Snappable().Dockable().Layer(1).Position(1).MinimizeButton(
                True).MaximizeButton(True).CloseButton(False))
        self._mgr.ShowPane(self.shell, False)
        self._mgr.Update()

        accel = [(wx.ACCEL_CTRL, ord('T'), self.ID_SHOW_CONSOLE)]
        self.accel = wx.AcceleratorTable(accel)
        self.SetAcceleratorTable(self.accel)
        self.Bind(wx.EVT_TOOL, self.OnProcessTool)

    def OnProcessTool(self, event):
        eid = event.GetId()
        if eid == self.ID_SHOW_CONSOLE:
            self._mgr.ShowPane(self.shell, not self.shell.IsShown())
            self._mgr.Update()
        else:
            event.Skip()


class Surface(TrackingSurface):
    def __init__(self, *args, **kwargs):
        TrackingSurface.__init__(self, *args, **kwargs)

    def Initialize(self):
        super(Surface, self).Initialize()
        self.SetShowStepSurface(False)
        self.SetShowMode(mesh=True)
        self.rotate_matrix = np.array([[0.9625753, -0.21669953, 0.16275978],
                                       [0.26339024, 0.88946027, -0.3734787],
                                       [-0.06383575, 0.40237066, 0.91324866]],
                                      dtype=np.float32)


def BitmapFromXPM(xpm):
    xpm_b = [x.encode('utf-8') for x in xpm]
    if 'phoenix' in wx.version():
        return wx.Bitmap(xpm_b)
    else:
        return wx.BitmapFromXPMData(xpm_b)


class SurfacePanel(wx.Panel):
    ID_RUN = wx.NewIdRef()
    ID_PAUSE = wx.NewIdRef()

    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        self.x = np.linspace(0, 2 * np.pi, 30).reshape((1, 30))
        z = np.cos(self.x).T * np.sin(self.x)
        sizer = wx.BoxSizer(wx.VERTICAL)

        tb = aui.AuiToolBar(self,
                            -1,
                            wx.DefaultPosition,
                            wx.DefaultSize,
                            agwStyle=aui.AUI_TB_OVERFLOW
                            | aui.AUI_TB_PLAIN_BACKGROUND)
        tb.SetToolBitmapSize(wx.Size(16, 16))
        tb.AddSimpleTool(self.ID_RUN, "Run", BitmapFromXPM(run_xpm))
        tb.AddSimpleTool(self.ID_PAUSE, "Pause", BitmapFromXPM(pause_xpm))
        tb.Realize()
        sizer.Add(tb, 0, wx.EXPAND, 0)

        self.canvas = Surface(self, {'z': z})
        sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(sizer)

        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateTool)
        self.Bind(wx.EVT_TOOL, self.OnProcessTool)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.OnTimer, self.timer)
        #self.timer.Start(100)
        self.is_running = False

    def OnUpdateTool(self, event):
        eid = event.GetId()
        if eid == self.ID_RUN:
            event.Enable(not self.is_running)
        elif eid == self.ID_PAUSE:
            event.Enable(self.is_running)
        else:
            event.Skip()

    def OnProcessTool(self, event):
        eid = event.GetId()
        if eid == self.ID_RUN:
            self.is_running = True
            self.timer.Start(50)
        elif eid == self.ID_PAUSE:
            self.is_running = False
            self.timer.Stop()
        else:
            event.Skip()

    def OnTimer(self, event):
        self.x += 0.01 * 2 * np.pi
        self.x %= 2 * np.pi
        z = np.cos(self.x).T * np.sin(self.x)
        self.canvas.NewFrameArrive(z, False)


def main():
    app = RunApp()
    app.MainLoop()


if __name__ == '__main__':
    main()
