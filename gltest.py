import wx
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
        wx.Frame.__init__(self, None, -1, 'Cube', size=(410, 432))
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel = SurfacePanel(self)
        sizer.Add(self.panel, 1, wx.EXPAND | wx.ALL, 0)
        self.SetSizer(sizer)
        self.Layout()

class SurfacePanel(wx.Panel):
    ID_SHOW_TRIANGLE = wx.NewId()
    ID_SHOW_MESH = wx.NewId()
    ID_SHOW_CONTOUR = wx.NewId()
    ID_SHOW_BOX = wx.NewId()
    ID_SHOW_AXIS = wx.NewId()
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
        self.show = {'showtriangle':True, 'showmesh':False,
                     'showcontour':False, 'showbox':False, 'showaxis':False}
        self.canvas.update(self.show)
        self.canvas.update({'hud':'min x:%d max x:%d min y:%d max y: %d min z: %d max z: %d'%(np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z))})
        self.Bind(wx.EVT_MENU, self.OnProcessMenuEvent)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu)
        self.Bind(wx.EVT_CONTEXT_MENU, self.OnContextMenu)

    def OnUpdateMenu(self, event):
        eid = event.GetId()
        if eid == self.ID_SHOW_TRIANGLE:
            event.Check(self.show['showtriangle'])
        elif eid == self.ID_SHOW_MESH:
            event.Check(self.show['showmesh'])
        elif eid == self.ID_SHOW_CONTOUR:
            event.Check(self.show['showcontour'])
        elif eid == self.ID_SHOW_BOX:
            event.Check(self.show['showbox'])
        elif eid == self.ID_SHOW_AXIS:
            event.Check(self.show['showaxis'])

    def OnContextMenu(self, event):
        menu = wx.Menu()
        menu.Append(self.ID_SHOW_TRIANGLE, 'Show Triangle', '', wx.ITEM_CHECK)
        menu.Append(self.ID_SHOW_MESH, 'Show Mesh', '', wx.ITEM_CHECK)
        menu.Append(self.ID_SHOW_CONTOUR, 'Show Contour', '', wx.ITEM_CHECK)
        menu.Append(self.ID_SHOW_BOX, 'Show Box', '', wx.ITEM_CHECK)
        menu.Append(self.ID_SHOW_AXIS, 'Show Axis', '', wx.ITEM_CHECK)
        self.PopupMenu(menu)

    def OnProcessMenuEvent(self, event):
        eid = event.GetId()
        if eid == self.ID_SHOW_TRIANGLE:
            self.show['showtriangle'] = not self.show['showtriangle']
        elif eid == self.ID_SHOW_MESH:
            self.show['showmesh'] = not self.show['showmesh']
        elif eid == self.ID_SHOW_CONTOUR:
            self.show['showcontour'] = not self.show['showcontour']
        elif eid == self.ID_SHOW_BOX:
            self.show['showbox'] = not self.show['showbox']
        elif eid == self.ID_SHOW_AXIS:
            self.show['showaxis'] = not self.show['showaxis']
        self.canvas.update(self.show)
def main():
    app = RunApp()
    app.MainLoop()

if __name__ == '__main__':
    main()

