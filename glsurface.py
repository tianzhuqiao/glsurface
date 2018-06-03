#!/usr/bin/env python

import wx
import sys
import math
import six
import numpy as np
from wx import glcanvas
import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = True
from OpenGL.GL import *

vtxShaderSource = """
attribute vec3 VertexPosition;
attribute vec4 VertexColor;
uniform mat3 TransformationMatrix;
uniform vec3 VertexResolution;
uniform vec3 VertexScale;
uniform vec3 VertexOffset;
uniform vec3 VertexScaleGL;
varying vec2 vTextureCoord;
varying vec4 vColor;
uniform int VertexOriginal;
void main(void) {
    if(VertexOriginal == 1) {
        gl_Position = vec4(VertexPosition, 1.0);
    } else if (VertexOriginal == 2) {
        gl_Position = vec4((((VertexPosition))*VertexScale+VertexOffset)/VertexResolution*2.0-1.0, 1.0);
        gl_Position = gl_Position*vec4(VertexScaleGL,1);
    } else {
        gl_Position = vec4(((TransformationMatrix*(VertexPosition))*VertexScale+VertexOffset)/VertexResolution*2.0-1.0, 1.0);
        gl_Position = gl_Position*vec4(VertexScaleGL,1);
    }
    vColor = VertexColor;
}
"""

fragShaderSource = """
varying vec4 vColor;
uniform int mesh;
uniform vec4 clr;
uniform sampler2D uSampler;
void main(void) {
    if(mesh == 0) {
        gl_FragColor = vColor;
    } else if(mesh == 1) {
        gl_FragColor = vec4(0, 0, 1, 0.5);
    } else if(mesh == 2) {
        gl_FragColor = clr;
    } else if(mesh == 3) {
        vec4 textureColor = texture2D(uSampler, vec2(vColor[0], vColor[1]));
        gl_FragColor = vec4(textureColor.rgb , textureColor.a);
    }
}
"""

class SimpleBarBuf(object):
    def __init__(self, sz, rng, horz):
        self.resize(sz, rng, horz)

    def resize(self, sz, rng, horz, clr=[0, 0.6, 0, 1]):
        self.range = rng
        self.dSize = sz
        self.horz = horz
        self.vertex = np.zeros((sz*4, 3))
        self.vertex[:, 2].fill(rng['zmax'])
        if horz:
            # for point d[i], draw lines between points:
            # (i, ymax), (i, ymax-d[i]*g), (i+1, ymax-d[i]*g), (i+1, ymax)
            # see update() for detail
            xmax, xmin = self.range['xmax'], self.range['xmin']
            x = np.linspace(xmin, xmax, sz, endpoint=False).reshape((sz, 1))
            delta = (xmax-xmin)/sz
            self.vertex[:, 0] = (np.repeat(x, 4, axis=1) + [0, 0, delta, delta]).flatten()
        else:
            ymax, ymin = self.range['ymax'], self.range['ymin']
            y = np.linspace(ymin, ymax, sz, endpoint=False).reshape((sz, 1))
            delta = (ymax-ymin)/sz
            self.vertex[:, 1] = (np.repeat(y, 4, axis=1) + [0, 0, delta, delta]).flatten()

        self.color = np.repeat(np.array([clr]), sz*4, axis=0)
        self.line = np.arange(sz*4)

    def glObject(self):
        return self.vertex.flatten(), self.color, self.line

    def update(self, d, g):
        if len(d) != self.dSize:
            raise ValueError()
        zmax = self.range['zmax']
        if self.horz:
            ymax, ymin = self.range['ymax'], self.range['ymin']
            self.vertex[:, 1] = ymax-(np.kron(d, [0, 1, 1, 0])).flatten()*g*ymax/zmax
        else:
            xmax, xmin = self.range['xmax'], self.range['xmin']
            self.vertex[:, 0] = xmax-(np.kron(d, [0, 1, 1, 0])).flatten()*g*xmax/zmax


class SimpleSurfaceCanvas(glcanvas.GLCanvas):
    ID_SHOW_2D = wx.NewId()
    ID_SHOW_TRIANGLE = wx.NewId()
    ID_SHOW_MESH = wx.NewId()
    ID_SHOW_CONTOUR = wx.NewId()
    ID_SHOW_BOX = wx.NewId()
    ID_SHOW_AXIS = wx.NewId()
    ID_SHOW_HORZ_BAR = wx.NewId()
    ID_SHOW_VERT_BAR = wx.NewId()
    ID_ROTATE_0 = wx.NewId()
    ID_ROTATE_90 = wx.NewId()
    ID_ROTATE_180 = wx.NewId()
    ID_ROTATE_270 = wx.NewId()

    def __init__(self, parent, points=None):
        # set the depth size, otherwise the depth test may not work correctly.
        attribs = [glcanvas.WX_GL_RGBA, glcanvas.WX_GL_DOUBLEBUFFER,
                   glcanvas.WX_GL_DEPTH_SIZE, 16]
        glcanvas.GLCanvas.__init__(self, parent, -1, attribList=attribs)
        self.context = glcanvas.GLContext(self)

        self.dragStart = {'x':0, 'y':0}
        self.dragEnd = {'x':0, 'y':0}
        self.points = []
        self.pointsClr = []
        self.dimension = {'x':0, 'y':0}
        self.mouseIsDown = False
        self.isDragging = False
        self.rotation = {'x':110, 'y':0, 'z':110}
        self.default_rotate = 0
        sz = self.GetClientSize()
        self.W = sz.x
        self.H = sz.y
        self.range = {'xmin':0, 'ymin':0, 'zmin':0, 'xmax':0, 'ymax':0, 'zmax':0}
        self.zrange = []
        self.scale = {'base':1, 'zoom':1}
        self.rotateTheta = 0.05
        self.margin = {'top':50, 'bottom':10, 'left':1, 'right':1}
        self.offset = {'base':{'x':0, 'y':0}, 'user':{'x':0, 'y':0}}
        self.dataOffset = {'x':0, 'y':0, 'z':0}
        self.selected = {'x':0, 'y':0, 'clr':[1, 0, 1, 1]}
        self.colorScale = []
        self.blocks = []
        self.zAutoScale = 1
        self.zgain = 1.
        # 2d mode is the fast-drawing mode, only shows the 2D image
        self.mode_2d = False
        self.show = {'triangle': True, 'mesh': False, 'contour': False,
                     'box':False, 'axis':False, 'horz_bar': False,
                     'vert_bar': False}
        self.contourLevels = 100
        self.transformMatrix = np.eye(3, dtype=np.float32)
        self.colorMap = np.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1],
                                  [0, 1, 0, 1], [1, 1, 0, 1], [1, 0.5, 0, 1],
                                  [1, 0, 0, 1]], np.float32)
        self.colorRange = []
        self.initialized = False
        self.rawPoints = points
        self.glObject = {}
        self.hudtext = ''
        self._hudtext = ''
        self._hudBuffer = np.zeros((0, 0, 4))
        # buffers to draw bar
        rows, cols = self.rawPoints['z'].shape
        self.horzbar_buf = SimpleBarBuf(cols, self.range, True)
        self.vertbar_buf = SimpleBarBuf(rows, self.range, False)
        self.SetImage(self.rawPoints)

        self.reset()

        accel_tbl = self.GetAccelList()
        self.SetAcceleratorTable(wx.AcceleratorTable(accel_tbl))

        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.Bind(wx.EVT_LEFT_DCLICK, self.OnDoubleClick)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel)
        self.Bind(wx.EVT_CONTEXT_MENU, self.OnContextMenu)
        self.Bind(wx.EVT_MENU, self.OnProcessMenuEvent)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu)

    def GetAccelList(self):
        accel_tbl = [(wx.ACCEL_SHIFT, ord('T'), self.ID_SHOW_TRIANGLE),
                     (wx.ACCEL_SHIFT, ord('H'), self.ID_SHOW_HORZ_BAR),
                     (wx.ACCEL_SHIFT, ord('V'), self.ID_SHOW_VERT_BAR),
                    ]
        return accel_tbl

    def OnEraseBackground(self, event):
        pass # Do nothing, to avoid flashing on MSW.

    def OnContextMenu(self, event):
        menu = self.GetContextMenu()
        if menu:
            self.PopupMenu(menu)

    def GetContextMenu(self):
        menu = wx.Menu()
        elements = wx.Menu()
        elements.Append(self.ID_SHOW_2D, 'Show 2D Mode\tshift+t', '', wx.ITEM_CHECK)
        elements.AppendSeparator()
        elements.Append(self.ID_SHOW_TRIANGLE, 'Show Triangle\tshift+t', '', wx.ITEM_CHECK)
        elements.Append(self.ID_SHOW_MESH, 'Show Mesh', '', wx.ITEM_CHECK)
        elements.Append(self.ID_SHOW_CONTOUR, 'Show Contour', '', wx.ITEM_CHECK)
        elements.Append(self.ID_SHOW_BOX, 'Show Box', '', wx.ITEM_CHECK)
        elements.Append(self.ID_SHOW_AXIS, 'Show Axis', '', wx.ITEM_CHECK)
        elements.AppendSeparator()
        elements.Append(self.ID_SHOW_HORZ_BAR, 'Show Horz Bar\tshift+h', '', wx.ITEM_CHECK)
        elements.Append(self.ID_SHOW_VERT_BAR, 'Show Vert Bar\tshift+v', '', wx.ITEM_CHECK)
        menu.AppendSeparator()
        menu.AppendSubMenu(elements, 'Elements')
        rotate = wx.Menu()
        rotate.Append(self.ID_ROTATE_0, 'Rotate 0 degree', '', wx.ITEM_CHECK)
        rotate.Append(self.ID_ROTATE_90, 'Rotate 90 degree', '', wx.ITEM_CHECK)
        rotate.Append(self.ID_ROTATE_180, 'Rotate 180 degree', '', wx.ITEM_CHECK)
        rotate.Append(self.ID_ROTATE_270, 'Rotate 270 degree', '', wx.ITEM_CHECK)
        menu.AppendSeparator()
        menu.AppendSubMenu(rotate, 'Orientation')
        return menu

    def Set2dMode(self, is2d):
        self.mode_2d = is2d
        if not self.mode_2d:
            self.SetImage(self.rawPoints)

    def SetShowMode(self, **kwargs):
        refresh = False
        genGLobj = False
        for key in six.iterkeys(kwargs):
            if key in self.show and self.show[key] != kwargs[key]:
                self.show[key] = kwargs[key]
                refresh = True
                if key in ['mesh', 'contour']:
                    genGLobj = True
        if genGLobj:
            self.color()
            # the objects will be re-generated before drawing
            self.glObject = None
        if refresh:
            self.Refresh()

    def OnProcessMenuEvent(self, event):
        eid = event.GetId()
        show = {}
        if eid == self.ID_SHOW_2D:
            self.Set2dMode(True)
        elif eid == self.ID_SHOW_TRIANGLE:
            show['triangle'] = not self.show['triangle']
        elif eid == self.ID_SHOW_MESH:
            show['mesh'] = not self.show['mesh']
        elif eid == self.ID_SHOW_CONTOUR:
            show['contour'] = not self.show['contour']
        elif eid == self.ID_SHOW_BOX:
            show['box'] = not self.show['box']
        elif eid == self.ID_SHOW_AXIS:
            show['axis'] = not self.show['axis']
        elif eid == self.ID_SHOW_HORZ_BAR:
            show['horz_bar'] = not self.show['horz_bar']
        elif eid == self.ID_SHOW_VERT_BAR:
            show['vert_bar'] = not self.show['vert_bar']
        else:
            if eid == self.ID_ROTATE_0:
                self.default_rotate = 0
            elif eid == self.ID_ROTATE_90:
                self.default_rotate = 90
            elif eid == self.ID_ROTATE_180:
                self.default_rotate = 180
            elif eid == self.ID_ROTATE_270:
                self.default_rotate = 270
            self.reset()
            self.resize()
            self.Refresh()
            return
        if show:
            self.SetShowMode(**show)

    def OnUpdateMenu(self, event):
        eid = event.GetId()
        if eid == self.ID_SHOW_2D:
            event.Check(self.mode_2d)
        elif eid == self.ID_SHOW_TRIANGLE:
            event.Check(self.show['triangle'])
        elif eid == self.ID_SHOW_MESH:
            event.Check(self.show['mesh'])
            event.Enable(not self.mode_2d)
        elif eid == self.ID_SHOW_CONTOUR:
            event.Check(self.show['contour'])
            event.Enable(not self.mode_2d)
        elif eid == self.ID_SHOW_BOX:
            event.Check(self.show['box'])
            event.Enable(not self.mode_2d)
        elif eid == self.ID_SHOW_AXIS:
            event.Check(self.show['axis'])
            event.Enable(not self.mode_2d)
        elif eid == self.ID_SHOW_HORZ_BAR:
            event.Check(self.show['horz_bar'])
        elif eid == self.ID_SHOW_VERT_BAR:
            event.Check(self.show['vert_bar'])
        elif eid == self.ID_ROTATE_0:
            event.Check(self.default_rotate == 0)
        elif eid == self.ID_ROTATE_90:
            event.Check(self.default_rotate == 90)
        elif eid == self.ID_ROTATE_180:
            event.Check(self.default_rotate == 180)
        elif eid == self.ID_ROTATE_270:
            event.Check(self.default_rotate == 270)

    def resize(self):
        sz = self.GetClientSize()
        self.W = sz.x
        self.H = sz.y
        glViewport(0, 0, self.W, self.H)
        xmax, xmin = self.range['xmax'], self.range['xmin']
        ymax, ymin = self.range['ymax'], self.range['ymin']
        if self.default_rotate in [90, 270]:
            xmax, xmin, ymax, ymin = ymax, ymin, xmax, xmin
        if xmax-xmin > 0 and ymax-ymin >= 0:
            t = self.margin['top']
            b = self.margin['bottom']
            l = self.margin['left']
            r = self.margin['right']
            self.scale['base'] = min((self.W-l-r)/(xmax-xmin), (self.H-t-b)/(ymax-ymin))
            self.offset['base'] = {'x': self.W/2 + (l-r)/2, 'y': self.H/2 + (t-b)/2}

    def OnSize(self, event):
        # update the size
        self.resize()
        self.Refresh()
        event.Skip()

    def OnPaint(self, event):
        #dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.initialized:
            self.InitGL()
            if self.rawPoints is not None:
                self.SetImage(self.rawPoints)
            self.initialized = True
        self.draw()
        self.SwapBuffers()

    def InitGL(self):
        self.SetCurrent(self.context)

        glClearColor(1.0, 1.0, 1.0, 1.0) # this is the color
        glEnable(GL_DEPTH_TEST) # Enable Depth Testing
        glDepthFunc(GL_LEQUAL) # Set Perspective View
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

        FShader = fragShaderSource
        VShader = vtxShaderSource
        #Load and Compile Fragment Shader
        Code = FShader
        FShader = glCreateShader(GL_FRAGMENT_SHADER)
        glShaderSource(FShader, Code)
        glCompileShader(FShader)

        #Load and Compile Vertex Shader
        Code = VShader
        VShader = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(VShader, Code)
        glCompileShader(VShader)
        #print('cmp', glGetShaderiv(VShader, GL_COMPILE_STATUS))
        #print(glGetProgramInfoLog(FShader))
        #Create The Shader Program
        self.ShaderProgram = glCreateProgram()
        glAttachShader(self.ShaderProgram, FShader)
        glAttachShader(self.ShaderProgram, VShader)
        glLinkProgram(self.ShaderProgram)
        glUseProgram(self.ShaderProgram)
        #Link Vertex Position Attribute from Shader
        self.VertexPosition = glGetAttribLocation(self.ShaderProgram, "VertexPosition")
        glEnableVertexAttribArray(self.VertexPosition)
        self.VertexColor = glGetAttribLocation(self.ShaderProgram, "VertexColor")
        glEnableVertexAttribArray(self.VertexColor)
        glViewport(0, 0, self.W, self.H)
        # generate the buffer
        self.glBufs = glGenBuffers(3)
        self.glTextures = glGenTextures(3) # axis, hud, 2d image
        self.vtxResolution = glGetUniformLocation(self.ShaderProgram, "VertexResolution")
        self.vtxOffset = glGetUniformLocation(self.ShaderProgram, "VertexOffset")
        self.vtxMatrix = glGetUniformLocation(self.ShaderProgram, "TransformationMatrix")
        self.vtxScaleGL = glGetUniformLocation(self.ShaderProgram, "VertexScaleGL")
        self.vtxScale = glGetUniformLocation(self.ShaderProgram, "VertexScale")
        self.fraguSampler = glGetUniformLocation(self.ShaderProgram, "uSampler")

    def OnKeyDown(self, event):
        if event.ShiftDown():
            if self.mode_2d:
                event.Skip()
                return
            keycode = event.GetKeyCode()
            x, y, z = 0, 0, 0
            if keycode == wx.WXK_UP:
                x = 1
            elif keycode == wx.WXK_DOWN:
                x = -1
            elif keycode == wx.WXK_LEFT:
                if event.CmdDown():
                    z = -1
                else:
                    y = -1
            elif keycode == wx.WXK_RIGHT:
                if event.CmdDown():
                    z = 1
                else:
                    y = 1
            else:
                event.Skip()
            self.rotate(x, y, z)
        else:
            if self.dimension['x'] <= 0 or self.dimension['y'] <= 0:
                return
            keycode = event.GetKeyCode()
            x = self.selected['x']
            y = self.selected['y']
            mx = self.dimension['x'] - 1
            my = self.dimension['y'] - 1
            delta = 1
            dx, dy = 0, 0
            if event.CmdDown():
                delta = 5
            if keycode == wx.WXK_UP:
                dy = -delta
            elif keycode == wx.WXK_DOWN:
                dy = delta
            elif keycode == wx.WXK_LEFT:
                dx = -delta
            elif keycode == wx.WXK_RIGHT:
                dx = delta
            else:
                event.Skip()
                return
            # adjust the movement based on the current orientation
            if self.default_rotate == 90:
                dx, dy = dy, -dx
            elif self.default_rotate == 180:
                dx, dy = -dx, -dy
            elif self.default_rotate == 270:
                dx, dy = -dy, dx
            y = (y+dy)%my
            x = (x+dx)%mx
            self.SetSelected({'x': x, 'y':y})

    def OnMouseDown(self, event):
        bRect = self.GetClientRect()
        pos = event.GetPosition()
        self.dragStart['x'] = (pos.x - bRect.left)*(self.W/bRect.width)
        self.dragStart['y'] = (pos.y - bRect.top)*(self.H/bRect.height)
        self.mouseIsDown = True

    def OnMouseMotion(self, event):
        bRect = self.GetClientRect()
        pos = event.GetPosition()
        self.dragEnd['x'] = (pos.x - bRect.left)*((self.W)/bRect.width)
        self.dragEnd['y'] = (pos.y - bRect.top)*((self.H)/bRect.height)
        if self.mouseIsDown:
            if event.CmdDown():
                self.offset['user']['x'] += self.dragEnd['x'] - self.dragStart['x']
                self.offset['user']['y'] += self.dragEnd['y'] - self.dragStart['y']
                self.dragStart['x'] = self.dragEnd['x']
                self.dragStart['y'] = self.dragEnd['y']
                self.Refresh()
            elif not self.mode_2d:
                # only all rotate in 3d mode
                if abs(self.dragStart['x']-self.dragEnd['x']) > 1 or \
                   abs(self.dragStart['y']-self.dragEnd['y']) > 1:
                    self.isDragging = True
                if self.isDragging:
                    self.rotate(-(self.dragEnd['y']-self.dragStart['y']),
                                (self.dragEnd['x']-self.dragStart['x']), 0)
                    self.dragStart['x'] = self.dragEnd['x']
                    self.dragStart['y'] = self.dragEnd['y']

    def OnMouseUp(self, event):
        self.mouseIsDown = False
        self.isDragging = False
        self.Refresh()

    def OnDoubleClick(self, event):
        self.reset()
        self.Refresh()

    def OnMouseWheel(self, event):
        delta = (event.GetWheelRotation() > 0)*2-1
        bRect = self.GetClientRect()
        pos = event.GetPosition()
        x = (pos.x-bRect.left)*((self.W)/bRect.width)
        y = (pos.y-bRect.top)*((self.H)/bRect.height)
        if delta:
            zoom = 1.1**delta
            self.scale['zoom'] *= zoom
            base = self.offset['base']
            user = self.offset['user']
            user['x'] = x - (x-base['x']-user['x'])*zoom - base['x']
            user['y'] = y - (y-base['y']-user['y'])*zoom - base['y']
            self.Refresh()

    def reset(self):
        # reset rotation, zoom, offset
        self.rotation = {'x':0, 'y':0, 'z':0}
        self.transformMatrix = np.eye(3, dtype=np.float32)
        self.scale['zoom'] = 1
        self.offset['user'] = {'x':0, 'y':0}
        # set the default orientation
        self.rotate(0, 0, self.default_rotate/180*np.pi/self.rotateTheta)

    def getColorByZ(self, z):
        zMin = self.range['zmin']
        zMax = self.range['zmax']
        scale = self.zAutoScale
        if zMax-zMin > 0:
            scale = 1/(zMax-zMin)
        offset = 0
        if len(self.colorScale) == 2:
            scale = self.colorScale[0]/self.zAutoScale
            offset = self.colorScale[1]

        C = self.colorMap
        norm_p = (z - zMin)*scale + offset
        norm_p = norm_p*len(self.colorMap)
        norm_p = np.clip(norm_p, 0, len(self.colorMap)-1)
        norm_p_l = np.floor(norm_p).astype(int)
        norm_p_r = np.ceil(norm_p).astype(int)
        norm_p_d = norm_p - norm_p_l
        return C[norm_p_l, :] + (((C[norm_p_r, :] - C[norm_p_l, :]).T)*norm_p_d).T

    def color(self):
        self.pointsClr = self.getColorByZ(self.points[:, 2])
        sel = self.selected['x'] * self.dimension['y'] + self.selected['y']
        if sel >= 0 and sel < self.points.shape[0]:
            self.pointsClr[sel, :] = self.selected['clr']

    def setContourLevels(self, level):
        if level == self.contourLevels:
            return
        self.contourLevels = level

        # the objects will be re-generated before drawing
        self.glObject = None
        self.Refresh()

    def SetRangeZ(self, zrange):
        # Set the range of the image values; otherwise it is calculated from
        # the data itself. It will impact the color of each pixel. Instead of
        # fully utilizing all the colors in color map, we may need to fix the
        # color code for each value (e.g., 0 is always blue)
        self.zrange = []
        if len(zrange) == 2:
            self.zrange = zrange
            zmin, zmax = self.zrange
        elif self.rawPoints:
            z = self.rawPoints['z']
            zmin, zmax = np.min(z), np.max(z)
        else:
            zmin, zmax = 0, 1
        self.dataOffset['z'] = o = (zmax+zmin)/2
        ymax, ymin = self.range['ymax'], self.range['ymin']
        zg = 1
        if zmax-zmin > 0:
            zg = float(ymax-ymin)/(zmax-zmin)
        if zg == 0:
            zg = 1
        self.zAutoScale = zg
        zmin = (zmin-o)*zg
        zmax = (zmax-o)*zg
        self.range.update({'zmin':zmin, 'zmax':zmax})
        zm = max(abs(zmax), abs(zmin))
        self.zgain = -1
        if zm > 0:
            self.zgain = -0.1/zm

    def SetHudText(self, txt):
        if txt != self.hudtext:
            self.hudtext = txt
            self.Refresh()

    def GetHudText(self):
        return self.hudtext

    def UpdateHudText(self):
        y, x = self.selected['x'], self.selected['y']
        r, c = self.rawPoints['z'].shape
        if x >= 0 and y >= 0 and x < c and y < r:
            self.SetHudText('(%d, %d) %.4f'%(x, y, self.rawPoints['z'][y, x]))

    def SetSelected(self, sel):
        self.selected.update(sel)
        self.color()
        self.glObject = None
        self.UpdateHudText()
        self.Refresh()

    def GetSelected(self):
        return self.selected

    def SetColorMap(self, clrmap):
        self.colorMap = clrmap
        self.color()
        self.glObject = None
        self.Refresh()

    def GetColorMap(self):
        return self.colorMap

    def SetColorScale(self, scale):
        self.colorScale = scale

    def GetColorScale(self, scale):
        return self.colorScale

    def UpdateImage(self, points):
        # light-weighted function to update the image value as fast as possible,
        #assume its dimension does not change

        self.rawPoints['z'] = points
        self.UpdateHudText()
        rows, cols = points.shape
        self.Pz[0:rows, 0:cols] = points
        self.Pz[-1, :] = self.Pz[-2, :]
        self.Pz[:, -1] = self.Pz[:, -2]
        zg = self.zAutoScale
        if self.show['triangle']:
            self.points[:, 2] = ((self.Pz - self.dataOffset['z'])*zg).T.flatten()
        else:
            self.points[:, 2] = -self.dataOffset['z']*zg
        self.color()
        self.glObject = None
        self.Refresh()

    def SetDimension(self, rows, cols):
        # 'x', 'y' is used in 3D mode, where the image is expanded 1 pixel on
        # each dimension
        self.dimension = {'y': rows+1, 'x': cols+1, 'rows': rows, 'cols': cols}
        # resize the horz/vert bar buffer
        self.horzbar_buf.resize(cols, self.range, True)
        self.vertbar_buf.resize(rows, self.range, False)
        # resize the image buffer
        self.Pz = np.zeros((rows+1, cols+1))
        self.points = np.zeros((self.Pz.size, 3))

    def GetDimension(self):
        return self.dimension

    def SetImage(self, points):
        # points is a dictionary with keys ['x', 'y', 'z'], and each should
        # be a 2D matrix with same dimension. 'z' is mandatory.
        # This function is slow, since it will allocate the memory and
        # regenerate x/y data if necessary
        self.rawPoints = points
        self.points = []
        rows, cols = points['z'].shape
        self.SetDimension(rows, cols)
        xy = np.meshgrid(np.arange(0, cols+1), np.arange(0, rows+1))
        # if 'x' data is not defined, assume it is [0:cols]
        if 'x' not in points:
            Px = xy[0]
        else:
            Px = np.zeros((rows+1, cols+1))
            Px[0:rows, 0:cols] = points['x']
            Px[-1, :] = Px[-2, :]
            if cols > 1:
                Px[:, -1] = Px[:, -2]*2 - Px[:, -3]
            else:
                Px[:, -1] = Px[:, -2] + 1

        # if 'x' data is not defined, assume it is [0:cols]
        if 'y' not in points:
            Py = xy[1]
        else:
            Py = np.zeros((rows+1, cols+1))
            Py[0:rows, 0:cols] = points['y']
            Py[:, -1] = Py[:, -2]
            if rows > 1:
                Py[-1, :] = Py[-2, :]*2 - Py[-3, :]
            else:
                Py[-1, :] = Py[-2, :] + 1

        self.Pz[0:rows, 0:cols] = points['z']
        self.Pz[-1, :] = self.Pz[-2, :]
        self.Pz[:, -1] = self.Pz[:, -2]
        xmin, ymin, zmin = np.min(Px), np.min(Py), np.min(self.Pz)
        xmax, ymax, zmax = np.max(Px), np.max(Py), np.max(self.Pz)
        if self.zrange:
            zmin, zmax = self.zrange
        self.dataOffset['x'] = (xmax+xmin)/2
        self.dataOffset['y'] = (ymax+ymin)/2
        self.dataOffset['z'] = (zmax+zmin)/2
        o = self.dataOffset
        zg = 1
        if zmax-zmin > 0:
            zg = (ymax-ymin)/(zmax-zmin)
        self.zAutoScale = zg
        self.points = np.zeros((self.Pz.size, 3))
        self.points[:, 0] = (Px - o['x']).T.flatten()
        self.points[:, 1] = (Py - o['y']).T.flatten()
        self.points[:, 2] = ((self.Pz - o['z'])*zg).T.flatten()
        xmin -= o['x']
        xmax -= o['x']
        ymin -= o['y']
        ymax -= o['y']
        zmin = (zmin-o['z'])*zg
        zmax = (zmax-o['z'])*zg
        self.range = {'xmin':xmin, 'ymin':ymin, 'zmin':zmin,
                      'xmax':xmax, 'ymax':ymax, 'zmax':zmax}
        self.SetRangeZ([])
        self.resize()

        self.UpdateHudText()
        self.color()
        # the objects will be re-generated before drawing
        self.glObject = None
        self.Refresh()

    def calc_contour(self, v, level):
        # calculate the contour of level within a triangle defined by v
        # sort by Z
        v = v[v[:, 2].argsort()]

        p1 = None
        p2 = None
        valid = False
        if v[2][2] < level or v[0][2] > level:
            # do nothing
            pass
        elif v[0][2] < level and v[1][2] > level:
            p1 = (v[0]-v[1])/(v[0][2]-v[1][2])*(level-v[0][2])+v[0]
            p2 = (v[0]-v[2])/(v[0][2]-v[2][2])*(level-v[0][2])+v[0]
            valid = True
        elif v[1][2] < level and v[2][2] > level:
            p1 = (v[2]-v[0])/(v[2][2]-v[0][2])*(level-v[2][2])+v[2]
            p2 = (v[2]-v[1])/(v[2][2]-v[1][2])*(level-v[2][2])+v[2]
            valid = True
        elif v[0][2] == level and v[1][2] == level:
            p1 = v[0]
            p2 = v[1]
            valid = True
        elif v[1][2] == level and v[2][2] == level:
            p1 = v[1]
            p2 = v[2]
            valid = True

        return {'valid':valid, 'p1':p1, 'p2':p2}

    def prepareContour(self, L, R):
        xmin, xmax = self.range['xmin'], self.range['xmax']
        ymin, ymax = self.range['ymin'], self.range['ymax']
        zmin, zmax = self.range['zmin'], self.range['zmax']
        # ugly, offset the contour to make it 'align' with mesh and quad
        ox = (xmax-xmin)/(self.dimension['x']-1)/2
        oy = (ymax-ymin)/(self.dimension['y']-1)/2
        oz = 0
        # calculate the contour levels uniformly, ignore endpoints (zmin, zmax)
        levels = np.linspace(zmin, zmax, self.contourLevels+1, endpoint=False)[1:]
        my = self.dimension['y']
        P = self.points
        vertexAll = []
        meshAll = []
        colourAll = []

        vertex = []
        mesh = []
        colour = []
        minmax = []
        for i in range(L, R):
            if i%my == my-1 or i+my+1 >= len(P):
                minmax.append([0, 0])
                continue
            t1 = min(P[i][2], P[i+1][2], P[i+my][2], P[i+my+1][2])
            t2 = max(P[i][2], P[i+1][2], P[i+my][2], P[i+my+1][2])
            minmax.append([t1, t2])

        for level in levels:
            clr = self.getColorByZ(level)
            clr = [clr[0], clr[1], clr[2], 1]
            for i in range(L, R):
                if i%my == my-1 or i+my+1 >= len(P):
                    # the contour is calculate with for points
                    # i   i+my
                    # i+1 i+my+1
                    # so ignore the bottom and right edges
                    continue
                if minmax[i-L][0] > level or minmax[i-L][1] < level:
                    # the current level is not cross ith point and its 3
                    # neighbours, do nothing
                    continue
                if len(vertex) > 2**16-10:
                    # too many vertex, add to separate block to avoid overflow
                    # when rendering
                    vertex = (np.array(vertex)+[ox, oy, oz]).flatten()
                    colour = np.array(colour).flatten()
                    vertexAll.append(vertex)
                    meshAll.append(mesh)
                    colourAll.append(colour)
                    vertex = []
                    mesh = []
                    colour = []
                v = np.array([P[i], P[i+1], P[i+my+1]])
                p = self.calc_contour(v, level)
                if p['valid']:
                    vertex.append(p['p1'])
                    mesh.append(len(vertex)-1)
                    vertex.append(p['p2'])
                    mesh.append(len(vertex)-1)
                    colour.append(clr)
                    colour.append(clr)

                v = np.array([P[i], P[i+my], P[i+my+1]])
                p = self.calc_contour(v, level)
                if p['valid']:
                    vertex.append(p['p1'])
                    mesh.append(len(vertex)-1)
                    vertex.append(p['p2'])
                    mesh.append(len(vertex)-1)
                    colour.append(clr)
                    colour.append(clr)
        if vertex:
            vertex = (np.array(vertex)+[ox, oy, oz]).flatten()
            colour = np.array(colour).flatten()
            mesh = np.array(mesh).flatten()
            vertexAll.append(vertex)
            meshAll.append(mesh)
            colourAll.append(colour)
        return {'Vertices': vertexAll, 'Mesh': meshAll, 'Color': colourAll}

    def prepareGL(self):
        colorAll = []
        vertexAll = []
        triangleAll = []
        meshAll = []
        contourAll = []
        block = int(len(self.points)*4/(2**16)+1)
        my = self.dimension['y']
        mx = self.dimension['x']
        # the index of each block
        blocks = np.linspace(0, len(self.points), block+1, dtype=int)
        self.blocks = blocks
        r = blocks[0]
        l = 0
        P = self.points
        for k in range(1, len(blocks)):
            l, r = r, blocks[k]
            acolor = np.repeat(self.pointsClr[np.arange(l, min(r+my+1, len(P)))], 4, axis=0)
            # for each point (x[i], y[i], z[i]), construct a quad with
            # (x[i],   y[i],   z[i])
            # (x[i+1], y[i],   z[i])
            # (x[i+1], y[i+1], z[i])
            # (x[i],   y[i+1], z[i])
            avertex = np.zeros(((min(r+my+1, len(P)) - l)*4, 3))
            # the index
            idx = np.arange(l, min(r+my+1, len(P)))
            idx = np.matrix(idx).T
            idx = np.repeat(idx, 4, axis=1)
            # the corresponding row & col
            cc, rr = np.divmod(idx, my)
            #cc = (idx/my).astype(int)
            # the row for each 4 points in the same quad
            rr = rr + [0, 1, 1, 0]
            rr = rr.flatten().clip(0, my-1)
            # the col for each 4 points in the same quad
            cc = cc + [0, 0, 1, 1]
            cc = cc.flatten().clip(0, mx-1)
            # update the vertex of each quad
            avertex[:, 0] = P[(cc*my + rr), 0]
            avertex[:, 1] = P[(cc*my + rr), 1]
            # the quad use the same z value of (i, j)
            avertex[:, 2] = P[idx.flatten(), 2]

            # index for quad and mesh
            idx = np.arange(0, r-l)
            idx = np.matrix(idx).T
            idx = np.repeat(idx, 4, axis=1)
            COLS, ROWS = np.divmod(idx, my)
            idx = idx*4

            # quad
            triangle = np.zeros(((r-l)*3, 4))
            # the current quad
            # triangle.append([4*i, 4*i+1, 4*i+2, 4*i+3])
            triangle[0:r-l, :] = idx + [0, 1, 2, 3]

            # connect to the quad right
            # triangle.append([4*i+3, 4*i+2, 4*(i+my)+1, 4*(i+my)])
            rr = ROWS #+ [0, 0, 0, 0]
            #rr = rr.clip(0, my-1)
            cc = COLS + [0, 0, 1, 1]
            cc = cc.clip(0, mx-1)
            triangle[r-l:2*(r-l), :] = (cc*my + rr)*4 + [3, 2, 1, 0]

            # connect to the quad below
            # triangle.append([4*i+1, 4*(i+1), 4*(i+1)+3, 4*(i)+2])
            rr = ROWS + [0, 1, 1, 0]
            rr = rr.clip(0, my-1)
            cc = COLS #+ [0, 0, 0, 0]
            #cc = cc.clip(0, mx-1)
            triangle[2*(r-l):3*(r-l), :] = (cc*my + rr)*4 + [1, 0, 3, 2]

            # mesh
            mesh = None
            if self.show['mesh']:
                mesh = np.zeros(((r-l)*4, 4))
                # mesh in the same quad
                # mesh.append([4*i, 4*i+1, 4*i+1, 4*i+2, 4*i+2, 4*i+3, 4*i+3, 4*i])
                mesh[0*(r-l):1*r-l] = idx + [0, 1, 1, 2]
                mesh[1*(r-l):2*r-l] = idx + [2, 3, 3, 0]

                # connect to the quad right
                # mesh.append([4*i+3, 4*(i+my), 4*i+2, 4*(i+my)+1])
                rr = ROWS #+ [0, 0, 0, 0]
                #rr = rr.clip(0, my-1)
                cc = COLS + [0, 1, 0, 1]
                cc = cc.clip(0, mx-1)
                mesh[2*(r-l):3*(r-l), :] = (cc*my + rr)*4 + [3, 0, 2, 1]

                # connect to the quad below
                # mesh.append([4*i+1, 4*(i+1), 4*i+2, 4*(i+1)+3])
                rr = ROWS + [0, 1, 0, 1]
                rr = rr.clip(0, my-1)
                cc = COLS #+ [0, 0, 0, 0]
                #cc = cc.clip(0, mx-1)
                mesh[3*(r-l):4*(r-l), :] = (cc*my + rr)*4 + [1, 0, 2, 3]

                mesh = mesh.flatten()
            contour = None
            if self.show['contour']:
                contour = self.prepareContour(l, r)
            avertex = avertex.flatten()
            acolor = acolor.flatten()
            triangle = triangle.flatten()
            vertexAll.append(avertex)
            colorAll.append(acolor)
            triangleAll.append(triangle)
            meshAll.append(mesh)
            contourAll.append(contour)
        return {'block':block, 'Color':colorAll, 'Vertices':vertexAll,
                'Trinagles':triangleAll, 'Mesh':meshAll, 'Contour':contourAll}

    def setGLBuffer(self, v, c):
        # vertex buffer
        # Bind it as The Current Buffer'
        glBindBuffer(GL_ARRAY_BUFFER, self.glBufs[0])
        # Fill it With the Data
        glBufferData(GL_ARRAY_BUFFER, v.astype(np.float32), GL_STATIC_DRAW)
        glVertexAttribPointer(self.VertexPosition, 3, GL_FLOAT, GL_FALSE, 0, None)

        # color buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.glBufs[1])
        glBufferData(GL_ARRAY_BUFFER, c.astype(np.float32), GL_STATIC_DRAW)
        glVertexAttribPointer(self.VertexColor, 4, GL_FLOAT, GL_FALSE, 0, None)

    def drawElementGL(self, t, v, m, o=0):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.glBufs[2])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, v.astype(np.uint16), GL_STATIC_DRAW)
        glUniform1i(glGetUniformLocation(self.ShaderProgram, "mesh"), m)
        glUniform1i(glGetUniformLocation(self.ShaderProgram, "VertexOriginal"), o)
        glDrawElements(t, v.size, GL_UNSIGNED_SHORT, None)

    def drawAxisGL(self):
        r = self.range
        gx = 32/self.scale['base']
        p = [r['xmax'], r['ymax'], r['zmax']]
        vertex = np.array([p[0]-gx,   p[1],   p[2],
                           p[0]-gx,   p[1]-gx,      p[2],
                           p[0],      p[1]-gx,      p[2],
                           p[0],      p[1],   p[2],
                           p[0]-gx,   p[1], p[2],
                           p[0],      p[1], p[2],
                           p[0]-gx/3, p[1]+gx/5,   p[2],
                           p[0]-gx/3, p[1]-gx/5, p[2]])
        self.drawAxisGL_help('x', vertex)
        p = [r['xmin'], r['ymax'], r['zmax']]
        vertex = np.array([p[0],    p[1],    p[2],
                           p[0],    p[1]-gx, p[2],
                           p[0]+gx, p[1]-gx, p[2],
                           p[0]+gx, p[1],    p[2],
                           p[0], p[1]-gx,     p[2],
                           p[0], p[1],    p[2],
                           p[0]+gx/5,p[1]-gx/3,   p[2],
                           p[0]-gx/5,p[1]-gx/3, p[2]])
        self.drawAxisGL_help('y', vertex)
        p = [r['xmin'], r['ymin'], r['zmax']]
        vertex = np.array([p[0], p[1]+gx, p[2]-gx,
                           p[0], p[1],    p[2]-gx,
                           p[0], p[1],    p[2],
                           p[0], p[1]+gx, p[2],
                           p[0], p[1],  p[2]-gx,
                           p[0], p[1],  p[2],
                           p[0], p[1]+gx/5, p[2]-gx/3,
                           p[0], p[1]-gx/5, p[2]-gx/3])
        self.drawAxisGL_help('z', vertex)

    def drawAxisGL_help(self, letter, vertex):
        W, H = 32, 32
        bitmap = wx.Bitmap(W, H, 32)
        tdc = wx.MemoryDC()
        tdc.SelectObject(bitmap)
        gc = wx.GraphicsContext.Create(tdc)
        font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        gc.SetFont(font, wx.WHITE)
        tw, th = gc.GetTextExtent(letter)
        gc.DrawText(letter, (W-tw)/2, (H-th)/2)
        tdc.SelectObject(wx.NullBitmap)

        texture = self.glTextures[0]
        glPixelStorei(GL_UNPACK_ALIGNMENT, GL_TRUE)
        glBindTexture(GL_TEXTURE_2D, texture)
        mybuffer = np.zeros((W, H, 4), np.uint8)
        bitmap.CopyToBuffer(mybuffer, wx.BitmapBufferFormat_RGBA)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, mybuffer.flatten())
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST)
        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)

        gx = 2*32/self.W*self.scale['base']
        color = np.array([0, 1, 0, 1,
                          0, 0, 0, 1,
                          1, 0, 0, 1,
                          1, 1, 0, 1,
                          1, 1, 1, 1,
                          1, 1, 1, 1,
                          1, 1, 1, 1,
                          1, 1, 1, 1])
        triangle = np.array([0, 1, 2, 0, 2, 3])
        mesh = np.array([4, 5, 5, 6, 5, 7])
        glUniform1i(glGetUniformLocation(self.ShaderProgram, "uSampler"), 0)
        self.setGLBuffer(vertex, color)
        self.drawElementGL(GL_TRIANGLES, triangle, 3)
        self.drawElementGL(GL_LINES, mesh, 0)

    def drawGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #Setup shader
        vertexResolution = glGetUniformLocation(self.ShaderProgram, "VertexResolution")
        glUniform3f(vertexResolution, self.W, self.H, self.W)

        vertexOffset = glGetUniformLocation(self.ShaderProgram, "VertexOffset")
        ox = self.offset['base']['x']+self.offset['user']['x']
        oy = self.offset['base']['y']+self.offset['user']['y']
        glUniform3f(vertexOffset, ox, oy, 0)

        #Set Transformation Matrices
        TransformMatrix = self.transformMatrix.flatten('F')
        tmatrix = glGetUniformLocation(self.ShaderProgram, "TransformationMatrix")
        glUniformMatrix3fv(tmatrix, 1, GL_FALSE, TransformMatrix)

        vertexScaleGL = glGetUniformLocation(self.ShaderProgram, "VertexScaleGL")
        zgain = max(abs(self.range['zmax']), abs(self.range['zmin']))
        if zgain == 0:
            zgain = 1
        glUniform3f(vertexScaleGL, 1, -1, -0.1/zgain)

        vertexScale = glGetUniformLocation(self.ShaderProgram, "VertexScale")
        scaleAll = self.scale['base']*self.scale['zoom']
        glUniform3f(vertexScale, scaleAll, scaleAll, 1)

        if self.show['triangle'] or self.show['mesh'] or self.show['contour']:
            if not self.glObject:
                self.glObject = self.prepareGL()
            obj = self.glObject
            block = obj.get('block', 0)
            for b in range(block):
                self.setGLBuffer(obj['Vertices'][b], obj['Color'][b])
                clr_mesh = 0
                if self.show['triangle']:
                    clr_mesh = 1
                    self.drawElementGL(GL_QUADS, obj['Trinagles'][b], 0)
                if self.show['mesh']:
                    self.drawElementGL(GL_LINES, obj['Mesh'][b], clr_mesh)

                if self.show['contour']:
                    ctr = obj['Contour'][b]
                    for c in range(len(ctr['Vertices'])):
                        self.setGLBuffer(ctr['Vertices'][c], ctr['Color'][c])
                        self.drawElementGL(GL_LINES, ctr['Mesh'][c], 0)

    def draw2dImg(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Setup shader
        # it looks like allocating 'vertexResolution' as class member does
        # not apparently improve the speed
        vertexResolution = glGetUniformLocation(self.ShaderProgram, "VertexResolution")
        glUniform3f(vertexResolution, self.W, self.H, self.W)

        vertexOffset = glGetUniformLocation(self.ShaderProgram, "VertexOffset")
        ox = self.offset['base']['x']+self.offset['user']['x']
        oy = self.offset['base']['y']+self.offset['user']['y']
        glUniform3f(vertexOffset, ox, oy, 0)

        #Set Transformation Matrices
        TransformMatrix = self.transformMatrix.flatten('F')
        tmatrix = glGetUniformLocation(self.ShaderProgram, "TransformationMatrix")
        glUniformMatrix3fv(tmatrix, 1, GL_FALSE, TransformMatrix)

        vertexScaleGL = glGetUniformLocation(self.ShaderProgram, "VertexScaleGL")
        glUniform3f(vertexScaleGL, 1, -1, self.zgain)

        vertexScale = glGetUniformLocation(self.ShaderProgram, "VertexScale")
        scaleAll = self.scale['base']*self.scale['zoom']
        glUniform3f(vertexScale, scaleAll, scaleAll, 1)

        W = self.dimension['x']
        H = self.dimension['y']
        #if self.img_texture is None:
        glPixelStorei(GL_UNPACK_ALIGNMENT, GL_TRUE)
        glBindTexture(GL_TEXTURE_2D, self.glTextures[2])
        # the pointsClr is col-wise (i.e., 1st col, 2nd col...)
        mybuffer = self.pointsClr.astype(np.float32).flatten()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, H, W, 0, GL_RGBA,
                     GL_FLOAT, mybuffer)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.glTextures[2])

        xmax, xmin = self.range['xmax'], self.range['xmin']
        ymax, ymin = self.range['ymax'], self.range['ymin']
        zmax, zmin = self.range['zmax'], self.range['zmin']
        vertex = np.array([xmin, ymax, zmin,
                           xmin, ymin, zmin,
                           xmax, ymin, zmin,
                           xmax, ymax, zmin])
        # texture coordinate
        color = np.array([1-1./H, 0, 0, 1,
                          0, 0, 0, 1,
                          0, 1-1./W, 0, 1,
                          1-1./H, 1-1./W, 0, 1])
        triangle = np.array([0, 1, 2, 3])
        glUniform1i(glGetUniformLocation(self.ShaderProgram, "uSampler"), 0)
        self.setGLBuffer(vertex, color)
        self.drawElementGL(GL_QUADS, triangle, 3, 0)

    def drawBox(self):
        if self.show['box']:
            r = self.range
            vertex = np.array([r['xmin'], r['ymin'], r['zmin'],
                               r['xmin'], r['ymax'], r['zmin'],
                               r['xmax'], r['ymax'], r['zmin'],
                               r['xmax'], r['ymin'], r['zmin'],
                               r['xmin'], r['ymin'], r['zmax'],
                               r['xmin'], r['ymax'], r['zmax'],
                               r['xmax'], r['ymax'], r['zmax'],
                               r['xmax'], r['ymin'], r['zmax']])
            color = np.array([1, 1, 1, 1,
                              1, 1, 1, 1,
                              1, 1, 1, 1,
                              1, 1, 1, 1,
                              1, 1, 1, 1,
                              1, 1, 1, 1,
                              1, 1, 1, 1,
                              1, 1, 1, 1])
            axis = np.array([0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6,
                             6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7])
            self.setGLBuffer(vertex, color)
            self.drawElementGL(GL_LINES, axis, 0)

        if self.show['axis']:
            self.drawAxisGL()

    def drawBar(self, horz=True):
        if horz:
            y = self.selected['y']
            if y < 0 or y >= self.rawPoints['z'].shape[0]:
                # out of range
                return
            d = self.rawPoints['z'][y, :]
            self.horzbar_buf.update(d, self.zAutoScale)
            v, c, l = self.horzbar_buf.glObject()
        else:
            x = self.selected['x']
            if x < 0 or x >= self.rawPoints['z'].shape[1]:
                # out of range
                return
            d = self.rawPoints['z'][:, x]
            self.vertbar_buf.update(d, self.zAutoScale)
            v, c, l = self.vertbar_buf.glObject()

        self.setGLBuffer(v, c)
        self.drawElementGL(GL_LINE_STRIP, l, 0)

    def draw(self):
        if self.mode_2d:
            self.draw2dImg()
        else:
            self.drawGL()
            self.drawBox()
        glLineWidth(2.0)
        if self.show['horz_bar']:
            self.drawBar()
        if self.show['vert_bar']:
            self.drawBar(False)
        self.drawHud()

    def drawHud(self):
        if not self._hudtext and not self.hudtext:
            return
        W, H = int(self.W), int(self.H)
        HudW, HudH = self._hudBuffer.shape[0:2]
        if self._hudtext != self.hudtext or HudW < W:
            self._hudtext = self.hudtext
            letter = self.hudtext
            tdc = wx.MemoryDC()
            HudW, HudH = W, 38
            bitmap = wx.Bitmap(HudW, HudH, depth=32)
            tdc.SelectObject(bitmap)
            gc = wx.GraphicsContext.Create(tdc)
            font = wx.Font(14, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
            gc.SetFont(font, wx.WHITE)
            gc.DrawText(letter, 5, 5)
            tdc.SelectObject(wx.NullBitmap)
            self._hudBuffer = np.zeros((HudW, HudH, 4), np.uint8)
            bitmap.CopyToBuffer(self._hudBuffer, wx.BitmapBufferFormat_RGBA)
        texture = self.glTextures[1]
        glPixelStorei(GL_UNPACK_ALIGNMENT, GL_TRUE)
        glBindTexture(GL_TEXTURE_2D, texture)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, HudW, HudH, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, self._hudBuffer)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)

        vertex = np.array([-1, 1, -1,
                           -1, 1-2.*HudH/H, -1,
                           -1+2.*HudW/W, 1-2.*HudH/H, -1,
                           -1+2.*HudW/W, 1, -1])
        # texture coordinate
        color = np.array([0, 0, 0, 1,
                          0, 1, 0, 1,
                          1, 1, 0, 1,
                          1, 0, 0, 1])
        triangle = np.array([0, 1, 2, 3])
        glUniform1i(self.fraguSampler, 0)
        self.setGLBuffer(vertex, color)
        self.drawElementGL(GL_QUADS, triangle, 3, 1)

    def rotate(self, x, y, z):
        if x == 0 and y == 0 and z == 0:
            return
        R = self.transformMatrix
        if x != 0:
            Rx = self.xRotateMatrix(x)
            R = np.matmul(Rx, R)
        if y != 0:
            Ry = self.yRotateMatrix(y)
            R = np.matmul(Ry, R)
        if z != 0:
            Rz = self.zRotateMatrix(z)
            R = np.matmul(Rz, R)
        self.rotation = {'x':0, 'y':0, 'z':0}
        self.transformMatrix = R
        self.Refresh()

    def xRotateMatrix(self, sign):
        Rx = np.zeros((3, 3), np.float32)
        Rx[0][0] = 1
        Rx[1][1] = math.cos(sign*self.rotateTheta)
        Rx[1][2] = -math.sin(sign*self.rotateTheta)
        Rx[2][1] = math.sin(sign*self.rotateTheta)
        Rx[2][2] = math.cos(sign*self.rotateTheta)
        return Rx

    def yRotateMatrix(self, sign):
        Ry = np.zeros((3, 3), np.float32)
        Ry[0][0] = math.cos(sign*self.rotateTheta)
        Ry[0][2] = math.sin(sign*self.rotateTheta)
        Ry[1][1] = 1
        Ry[2][0] = -math.sin(sign*self.rotateTheta)
        Ry[2][2] = math.cos(sign*self.rotateTheta)
        return Ry

    def zRotateMatrix(self, sign):
        Rz = np.zeros((3, 3), np.float32)
        Rz[0][0] = math.cos(sign*self.rotateTheta)
        Rz[0][1] = -math.sin(sign*self.rotateTheta)
        Rz[1][0] = math.sin(sign*self.rotateTheta)
        Rz[1][1] = math.cos(sign*self.rotateTheta)
        Rz[2][2] = 1
        return Rz

