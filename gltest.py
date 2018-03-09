#!/usr/bin/env python

import wx
import sys
import math
import numpy as np
from glcanvas import SimpleSurfaceCanvas

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
        wx.Frame.__init__(self, None, -1, 'Cube', size=(400, 400))
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel = SurfacePanel(self)
        sizer.Add(self.panel, 1, wx.EXPAND | wx.ALL, 0)
        self.SetSizer(sizer)
        self.Layout()

class SurfacePanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        x = np.meshgrid(np.linspace(-3, 3, 61), np.linspace(-3, 3, 61))
        points = np.zeros((len(x[0].flatten()), 3))
        points[:, 0] = x[1].flatten()
        points[:, 1] = x[0].flatten()
        points[:, 2] = np.sinc(points[:, 0])*np.sinc(points[:,1])
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.canvas = SimpleSurfaceCanvas(self, points, {'x':61, 'y':61})
        sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(sizer)
        self.Layout()

def main():
    app = RunApp()
    app.MainLoop()

if __name__ == '__main__':
    main()



