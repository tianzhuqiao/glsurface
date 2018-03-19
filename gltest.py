#!/usr/bin/env python

import wx
import sys
import math
import numpy as np
from glsurface import SimpleSurfaceCanvas

#----------------------------------------------------------------------
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
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        xy = np.meshgrid(np.linspace(-3, 3, 21), np.linspace(-3, 3, 21))
        x = xy[0]
        y = xy[1]
        z = np.sinc(x)*np.sinc(y)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.canvas = SimpleSurfaceCanvas(self, {'x':x, 'y':y, 'z':z})
        self.canvas.update({'hud':'x:0 y:0 min:0 max: 0'})
        sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(sizer)
        self.Layout()

def main():
    app = RunApp()
    app.MainLoop()

if __name__ == '__main__':
    main()



