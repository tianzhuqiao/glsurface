import wx
import wx.lib.agw.aui as aui
import wx.py as py
import numpy as np
from glsurface import SimpleSurfaceCanvas

class RunApp(wx.App):
    def __init__(self):
        wx.App.__init__(self, redirect=False)

    def OnInit(self):
        frame = mainFrame()
        frame.Show(True)
        self.SetTopWindow(frame)
        self.frame = frame
        return True

class mainFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, -1, 'glsurface demo', size=(410, 432))
        self._mgr = aui.AuiManager()

        # tell AuiManager to manage this frame
        self._mgr.SetManagedWindow(self)
        self.panel = SurfacePanel(self)
        self._mgr.AddPane(self.panel, aui.AuiPaneInfo().Caption("glsurface").
                          CenterPane())
        ns = {}
        ns['wx'] = wx
        ns['app'] = wx.GetApp()
        ns['frame'] = self
        self.shell = py.shell.Shell(self, -1, locals=ns)
        self._mgr.AddPane(self.shell, aui.AuiPaneInfo().Caption('console').
                          DestroyOnClose(False).Bottom().Snappable().
                          Dockable().Layer(1).Position(1).
                          MinimizeButton(True).MaximizeButton(True).
                          CloseButton(False))
        pane = self._mgr.GetPane(self.shell)
        self._mgr.MinimizePane(pane)
        self._mgr.Update()
class SurfacePanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        xy = np.meshgrid(np.linspace(-3, 3, 61), np.linspace(-3, 3, 61))
        x = xy[0]
        y = xy[1]
        z = np.sinc(x)*np.sinc(y)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.canvas = SimpleSurfaceCanvas(self, {'x':x, 'y':y, 'z':z})
        sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(sizer)
        self.Layout()

def main():
    app = RunApp()
    app.MainLoop()

if __name__ == '__main__':
    main()

