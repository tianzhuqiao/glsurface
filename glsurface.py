#!/usr/bin/env python

import wx
import sys
import math
import numpy as np
from wx import glcanvas
import OpenGL
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False
OpenGL.ERROR_ON_COPY = True
from OpenGL.GL import *

vtxShaderSource = (
    "attribute vec3 VertexPosition;\n"
    "attribute vec4 VertexColor;\n"
    "uniform mat3 TransformationMatrix;\n"
    "uniform mat4 PerspectiveMatrix;\n"
    "uniform vec3 VertexResolution;\n"
    "uniform vec3 VertexScale;\n"
    "uniform vec3 VertexOffset;\n"
    "uniform vec3 VertexOffsetView;\n"
    "uniform vec3 VertexScaleGL;\n"
    "uniform vec3 VertexOffsetGL;\n"
    "varying vec2 vTextureCoord;\n"
    "varying vec4 vColor;\n"
    "uniform int VertexOriginal;\n"
    "void main(void) {\n"
    "    if(VertexOriginal == 1) {\n"
    "        gl_Position = vec4(VertexPosition, 1.0);\n"
    "    } else if (VertexOriginal == 2) {\n"
    "        gl_Position = vec4((((VertexPosition-VertexOffsetView))*VertexScale+VertexOffset)/VertexResolution*2.0-1.0, 1.0);\n"
    "        gl_Position = gl_Position*vec4(VertexScaleGL,1)+vec4(VertexOffsetGL,0);\n"
    "    } else {\n"
    "        gl_Position = vec4(((TransformationMatrix*(VertexPosition-VertexOffsetView))*VertexScale+VertexOffset)/VertexResolution*2.0-1.0, 1.0);\n"
    "        gl_Position = gl_Position*vec4(VertexScaleGL,1)+vec4(VertexOffsetGL,0);\n"
    "    }\n"
    "    vColor = VertexColor;\n"
    "}\n")

fragShaderSource = (
    "varying vec4 vColor;\n"
    "uniform int mesh;\n"
    "uniform vec4 clr;\n"
    "uniform sampler2D uSampler;\n"
    "void main(void) {\n"
    "    if(mesh == 0) {\n"
    "        gl_FragColor = vColor;\n"
    "    } else if(mesh == 1) {\n"
    "        gl_FragColor = vec4(0, 0, 1, 0.5);\n"
    "    } else if(mesh == 2) {\n"
    "        gl_FragColor = clr;\n"
    "    } else if(mesh == 3) {\n"
    "        vec4 textureColor = texture2D(uSampler, vec2(vColor[0], vColor[1]));\n"
    "        gl_FragColor = vec4(textureColor.rgb , textureColor.a);\n"
    "    }\n"
    "}\n")

class SimpleSurfaceCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, points=None):
        # set the depth size, otherwise the depth test may not work correctly.
        attribs = [glcanvas.WX_GL_RGBA, glcanvas.WX_GL_DOUBLEBUFFER,
                   glcanvas.WX_GL_DEPTH_SIZE, 32]
        glcanvas.GLCanvas.__init__(self, parent, -1, attribList=attribs)
        self.init = False
        self.context = glcanvas.GLContext(self)

        self.dragStart = {'x':0, 'y':0}
        self.dragEnd = {'x':0, 'y':0}
        self.points = []
        self.pointsClr = []
        self.dimension = {'x':0, 'y':0}
        self.mouseIsDown = False
        self.isDragging = False
        self.rotation = {'x':110, 'y':0, 'z':110}
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
        self.selected = {'x':-1, 'y':-1, 'clr':[1, 0, 1, 1]}
        self.colorScale = []
        self.blocks = []
        self.zAutoScale = 1
        self.show = {'triangle': True, 'mesh': False, 'contour': False,
                     'box':False, 'axis':False}
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
        self._hudBuffer = None
        self.reset()

        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.Bind(wx.EVT_LEFT_DCLICK, self.OnDoubleClick)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel)

    def OnEraseBackground(self, event):
        pass # Do nothing, to avoid flashing on MSW.

    def resize(self):
        sz = self.GetClientSize()
        self.W = sz.x
        self.H = sz.y
        glViewport(0, 0, self.W, self.H)
        xmax, xmin = self.range['xmax'], self.range['xmin']
        ymax, ymin = self.range['ymax'], self.range['ymin']
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
        self.redraw()
        event.Skip()

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.initialized:
            self.InitGL()
            if self.rawPoints is not None:
                self.update({'points':self.rawPoints})
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
        self.glTextures = glGenTextures(2)

    def MakePerspective(self, FOV, AspectRatio, Closest, Farest):
        YLimit = Closest * math.tan(FOV * math.pi / 360)
        A = -(Farest + Closest)/(Farest - Closest)
        B = -2 * Farest * Closest / (Farest - Closest)
        C = (2 * Closest)/((YLimit * AspectRatio) * 2)
        D = (2 * Closest)/(YLimit * 2)
        return [C, 0, 0, 0,
                0, D, 0, 0,
                0, 0, A, -1,
                0, 0, B, 0]

    def OnKeyDown(self, event):
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
                self.redraw()
            else:
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
        self.redraw()

    def OnDoubleClick(self, event):
        self.reset()
        self.redraw()

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
            self.redraw()

    def reset(self):
        self.rotation = {'x':0, 'y':0, 'z':0}
        self.transformMatrix = np.eye(3, dtype=np.float32)
        self.scale['zoom'] = 1
        self.offset['user'] = {'x':0, 'y':0}

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

    def update(self, param):
        genGLobj = False
        refresh = False
        if "colorscale" in param:
            self.colorScale = param['colorscale']
        if "showtriangle" in param and self.show['triangle'] != param['showtriangle']:
            self.show['triangle'] = param['showtriangle']
            refresh = True
        if "showmesh" in param and self.show['mesh'] != param['showmesh']:
            self.show['mesh'] = param['showmesh']
            refresh = True
            genGLobj = True
        if "showcontour" in param and self.show['contour'] != param['showcontour']:
            self.show['contour'] = param['showcontour']
            refresh = True
            genGLobj = True
        if "showbox" in param and self.show['box'] != param['showbox']:
            self.show['box'] = param['showbox']
            refresh = True
        if "showaxis" in param and self.show['axis'] != param['showaxis']:
            self.show['axis'] = param['showaxis']
            refresh = True
        if "contourlevel" in param and self.contourLevels != param['contourlevel']:
            self.contourLevels = param['contourlevel']
            refresh = True
            genGLobj = True
        if 'zrange' in param:
            self.zrange = param['zrange']
        if 'hud' in param:
            self.hudtext = param['hud']
        if 'selected' in param:
            self.selected.update(param['selected'])
            genGLobj = True
            refresh = True
        if "colormap" in param:
            self.colorMap = param['colormap']
            updateClr = True
            genGLobj = True
            refresh = True

        if 'points' in param:
            #self.reset()
            self.points = []
            points = param['points']
            self.rawPoints = points
            zdim = points['z'].shape
            xy = np.meshgrid(np.arange(0, zdim[1]+1), np.arange(0, zdim[0]+1))
            if 'x' not in points:
                Px = xy[0]
            else:
                Px = np.zeros((zdim[0]+1, zdim[1]+1))
                Px[0:zdim[0], 0:zdim[1]] = points['x']
                Px[-1, 0:zdim[1]] = points['x'][-1, :]
                Px[:, -1] = Px[:, -2]*2 - Px[:, -3]
            if 'y' not in points:
                Py = xy[1]
            else:
                Py = np.zeros((zdim[0]+1, zdim[1]+1))
                Py[0:zdim[0], 0:zdim[1]] = points['y']
                Py[0:zdim[0], -1] = points['y'][:, -1]
                Py[-1, :] = Py[-2, :]*2 - Py[-3, :]

            Pz = np.zeros((zdim[0]+1, zdim[1]+1))
            Pz[0:zdim[0], 0:zdim[1]] = points['z']
            Pz[-1, 0:zdim[1]] = points['z'][-1, :]
            Pz[:, -1] = Pz[:, -2]
            self.dimension = {'x': zdim[1]+1, 'y':zdim[0]+1}
            xmin, ymin, zmin = np.min(Px), np.min(Py), np.min(Pz)
            xmax, ymax, zmax = np.max(Px), np.max(Py), np.max(Pz)
            self.dataOffset['x'] = (xmax+xmin)/2
            self.dataOffset['y'] = (ymax+ymin)/2
            self.dataOffset['z'] = (zmax+zmin)/2
            if self.zrange:
                zmin, zmax = self.zrange
            o = self.dataOffset
            zg = 1
            if zmax-zmin > 0:
                zg = (ymax-ymin)/(zmax-zmin)
            self.zAutoScale = zg
            self.points = np.zeros((Pz.size, 3))
            self.points[:, 0] = (Px - o['x']).T.flatten()
            self.points[:, 1] = (Py - o['y']).T.flatten()
            self.points[:, 2] = ((Pz - o['z'])*zg).T.flatten()
            xmin -= o['x']
            xmax -= o['x']
            ymin -= o['y']
            ymax -= o['y']
            zmin = (zmin-o['z'])*zg
            zmax = (zmax-o['z'])*zg
            self.range = {'xmin':xmin, 'ymin':ymin, 'zmin':zmin,
                          'xmax':xmax, 'ymax':ymax, 'zmax':zmax}
            self.resize()
            refresh = True
            genGLobj = True
        if genGLobj:
            self.color()
            # the objects will be re-generated before drawing
            self.glObject = None
        if refresh:
            self.redraw()

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

        vertexOffsetView = glGetUniformLocation(self.ShaderProgram, "VertexOffsetView")
        glUniform3f(vertexOffsetView, 0, 0, 0)

        vertexOffset = glGetUniformLocation(self.ShaderProgram, "VertexOffset")
        offsetAll = {}
        offsetAll['x'] = self.offset['base']['x']+self.offset['user']['x']
        offsetAll['y'] = self.offset['base']['y']+self.offset['user']['y']
        offsetAll['z'] = 0
        glUniform3f(vertexOffset, offsetAll['x'], offsetAll['y'], offsetAll['z'])
        #Generate The Perspective Matrix
        #var PerspectiveMatrix = this.MakePerspective(45, this.AspectRatio, 1, 10000.0)
        #var pmatrix = this.GL.getUniformLocation(this.ShaderProgram, "PerspectiveMatrix")
        #this.GL.uniformMatrix4fv(pmatrix, false, new Float32Array(PerspectiveMatrix))

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

        vertexOffsetGL = glGetUniformLocation(self.ShaderProgram, "VertexOffsetGL")
        glUniform3f(vertexOffsetGL, 0, 0, 0)

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

        # draw the box without zooming
        scaleAll = self.scale['base']
        glUniform3f(vertexScale, scaleAll, scaleAll, 1)
        offsetAll['x'] = self.offset['base']['x']
        offsetAll['y'] = self.offset['base']['y']
        glUniform3f(vertexOffset, offsetAll['x'], offsetAll['y'], offsetAll['z'])
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

    def draw(self):
        self.doRotate()
        self.drawGL()
        self.drawHud()

    def drawHud(self):
        if not self._hudtext and not self.hudtext:
            return
        W = int(self.W)
        H = int(self.H)
        if self._hudtext != self.hudtext:
            self._hudtext = self.hudtext
            letter = self.hudtext
            tdc = wx.MemoryDC()
            HudW, HudH = self.W, 32
            bitmap = wx.Bitmap(HudW, HudH, depth=32)
            tdc.SelectObject(bitmap)
            gc = wx.GraphicsContext.Create(tdc)
            font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
            gc.SetFont(font, wx.WHITE)
            tw, th = gc.GetTextExtent(letter)
            gc.DrawText(letter, 5, th/2)
            tdc.SelectObject(wx.NullBitmap)
            self._hudBuffer = np.zeros((HudW, HudH, 4), np.uint8)
            bitmap.CopyToBuffer(self._hudBuffer, wx.BitmapBufferFormat_RGBA)

        texture = self.glTextures[1]
        glPixelStorei(GL_UNPACK_ALIGNMENT, GL_TRUE)
        glBindTexture(GL_TEXTURE_2D, texture)

        HudW, HudH = self._hudBuffer.shape[0:2]
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, HudW, HudH, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, self._hudBuffer)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)

        vertex = np.array([-1, 1, 1,
                           -1, 1-2.*HudH/H, 1,
                           -1+2.*HudW/W, 1-2.*HudH/H, 1,
                           -1+2.*HudW/W, 1, 1])
        # texture coordinate
        color = np.array([0, 0, 0, 1,
                          0, 1, 0, 1,
                          1, 1, 0, 1,
                          1, 0, 0, 1])
        triangle = np.array([0, 1, 2, 3])
        glUniform1i(glGetUniformLocation(self.ShaderProgram, "uSampler"), 0)
        self.setGLBuffer(vertex, color)
        self.drawElementGL(GL_QUADS, triangle, 3, 1)

    def redraw(self):
        self.Refresh()

    def rotate(self, x, y, z):
        if x != 0 or y != 0 or z != 0:
            self.rotation['x'] += x
            self.rotation['y'] += y
            self.rotation['z'] += z
            self.redraw()

    def doRotate(self):
        x = self.rotation['x']
        y = self.rotation['y']
        z = self.rotation['z']
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

