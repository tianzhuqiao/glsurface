#!/usr/bin/env python

import wx
import sys
import math
import numpy as np
from functools import cmp_to_key
from wx import glcanvas
from OpenGL.GL import *
#from OpenGL.GLUT import *

#----------------------------------------------------------------------
class SimpleSurfaceCanvas(glcanvas.GLCanvas):
    def __init__(self, parent, points, dim):
        glcanvas.GLCanvas.__init__(self, parent, -1)
        self.init = False
        self.context = glcanvas.GLContext(self)

        self.dragStart = {'x':0, 'y':0}
        self.dragEnd = {'x':0, 'y':0}
        self.points = []
        self.rawPoints = []
        self.pointsClr = []
        self.dimension = {'x':0, 'y':0}
        self.mouseIsDown = False
        self.isDragging = False
        self.rotation = {'x':110, 'y':0, 'z':110}
        sz = self.GetClientSize()
        self.W = sz.x
        self.H = sz.y
        self.range = {'xmin':0, 'ymin':0, 'zmin':0, 'xmax':0, 'ymax':0, 'zmax':0}
        self.scale = {'base':1, 'zoom':1}
        self.invalidate = False
        self.rotateTheta = 0.05
        self.margin = {'x':25, 'y':25}
        self.offset = {'base':{'x':0, 'y':0}, 'user':{'x':0, 'y':0}}
        self.dataOffset = {'x':0, 'y':0, 'z':0}
        self.iSelect = -1
        self.colorScale = []
        self.blocks = []
        self.zAutoScale = 1
        self.show = {'triangle': True, 'mesh': False, 'contour': False,
                     'box':True, 'axis':True}
        self.contourLevels = 100
        self.transformMatrix = np.eye(3, dtype=np.float32)
        self.colorMap = [[0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0],
                         [255, 0, 0]]
        self.colorRange = []
        self.initialized = False
        self.points = points
        self.dim = dim
        #self.InitGL()
        #self.update({'points':points, 'dimension':dim})

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

    def OnSize(self, event):
        sz = self.GetClientSize()
        # update the size
        self.W = sz.x
        self.H = sz.y
        glViewport(0, 0, self.W, self.H)
        xmax, xmin = self.range['xmax'], self.range['xmin']
        ymax, ymin = self.range['ymax'], self.range['ymin']
        if xmax-xmin > 0 and ymax-ymin >= 0:
            self.scale['base'] = min((self.W-self.margin['x']*2)/(xmax-xmin), (self.H-self.margin['y']*2)/(ymax-ymin))
            self.offset['base'] = {'x': self.W/2 - (xmax+xmin)*self.scale['base']/2,
                                   'y': self.H/2 - (ymax+ymin)*self.scale['base']/2}
        event.Skip()

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.initialized:
            self.InitGL()
            self.update({'points':self.points, 'dimension':self.dim})
            self.initialized = True
        self.draw()
        self.SwapBuffers()

    def InitGL(self):
        self.SetCurrent(self.context)
        vtxShaderSource = ("attribute  vec3 VertexPosition;\n"
                           "attribute  vec4 VertexColor;\n"
                           "uniform  mat3 TransformationMatrix;\n"
                           "uniform  mat4 PerspectiveMatrix;\n"
                           "uniform  vec3 VertexResolution;\n"
                           "uniform  vec3 VertexScale;\n"
                           "uniform  vec3 VertexOffset;\n"
                           "uniform  vec3 VertexOffsetView;\n"
                           "uniform  vec3 VertexScaleGL;\n"
                           "uniform  vec3 VertexOffsetGL;\n"
                           "varying  vec2 vTextureCoord;\n"
                           "varying vec4 vColor;\n"
                           "void main(void) {\n"
                           "    gl_Position = vec4(((TransformationMatrix*(VertexPosition-VertexOffsetView))*VertexScale+VertexOffset)/VertexResolution*2.0-1.0, 1.0);\n"
                           "    gl_Position = gl_Position*vec4(VertexScaleGL,1)+vec4(VertexOffsetGL,0);\n"
                           "    vColor = VertexColor;\n"
                           "}\n")
        fragShaderSource = ("varying vec4 vColor;\n"
                            "uniform int mesh;\n"
                            "uniform vec4 clr;\n"
                            "uniform sampler2D uSampler;\n"
                            "void main(void) {\n"
                            "    if(mesh==0) {\n"
                            "        gl_FragColor = vColor;\n"
                            "    } else if(mesh==1) {\n"
                            "        gl_FragColor = vec4(0,0,1,0.5);\n"
                            "    } else if(mesh==2) {\n"
                            "        gl_FragColor = clr;\n"
                            "    } else if(mesh==3) {\n"
                            "        vec4 textureColor = texture2D(uSampler, vec2(vColor[0], vColor[1]));\n"
                            "        gl_FragColor = vec4(textureColor.rgb , textureColor.a);\n"
                            "    }\n"
                            "}\n")

        glClearColor(1.0, 1.0, 1.0, 1.0) # this is the color
        glEnable(GL_DEPTH_TEST) # Enable Depth Testing
        glDepthFunc(GL_LEQUAL) # Set Perspective View
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)
        # move this onsize
        #this.AspectRatio = this.canvas.width / this.canvas.height

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
        if keycode == 38: # upArrow:
            x = 1
        elif keycode == 40: # downArrow:
            x = -1
        elif keycode == 37: # leftArrow:
            if event.CmdDown():
                z = -1
            else:
                y = -1
        elif keycode == 39: # rightArrow:
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
                    rotatex, rotatey = 0, 0
                    if self.dragEnd['x']-self.dragStart['x'] > 10:
                        rotatex = 1
                    if self.dragEnd['x']-self.dragStart['x'] < -10:
                        rotatex = -1
                    if self.dragEnd['y']-self.dragStart['y'] > 10:
                        rotatey = 1
                    if self.dragEnd['y']-self.dragStart['y'] < -10:
                        rotatey = -1
                    self.rotate(-(self.dragEnd['y']-self.dragStart['y']), (self.dragEnd['x']-self.dragStart['x']), 0)
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
        delta = event.GetWheelRotation()
        bRect = self.GetClientRect()
        pos = event.GetPosition()
        x = (pos.x - bRect.left)*((self.W)/bRect.width)
        y = (pos.y - bRect.top)*((self.H)/bRect.height)
        if delta:
            zoom = 1.1**delta
            self.scale['zoom'] *= zoom
            self.offset['user']['x'] = x - (x-self.offset['base']['x']-self.offset['user']['x'])*zoom - self.offset['base']['x']
            self.offset['user']['y'] = y - (y-self.offset['base']['y']-self.offset['user']['y'])*zoom - self.offset['base']['y']
            self.redraw()

    def reset(self):
        self.rotation = {'x':0, 'y':0, 'z':0}
        self.transformMatrix = np.eye(3, dtype=np.float32)
        self.scale['zoom'] = 1
        self.offset['user'] = {'x':0, 'y':0}
        self.iSelect = -1
        self.invalidate = False

    def getColorByZ(self, z):
        zMin = self.range['zmin']
        zMax = self.range['zmax']
        cMin = 0
        cMax = 1
        if len(self.colorRange) == 2:
            cMin = self.colorRange[0]
            cMax = self.colorRange[1]
        scale = 1
        if zMax-zMin > 0:
            scale = 1/(zMax-zMin)

        zStep = abs(zMax - zMin) / (len(self.colorMap)-1)
        P = self.rawPoints
        C = self.colorMap
        CL = len(self.colorMap)-1
        z = (z-zMin)*scale
        if z <= cMin:
            r, g, b = C[0][0], C[0][1], C[0][2]
        elif z >= cMax:
            r, g, b = C[CL][0], C[CL][1], C[CL][2]
        else:
            z = min(z, cMax)
            zi = math.floor(z*CL)
            delta = z*CL - zi
            r = C[zi][0] + (C[zi+1][0]-C[zi][0])*(delta)
            g = C[zi][1] + (C[zi+1][1]-C[zi][1])*(delta)
            b = C[zi][2] + (C[zi+1][2]-C[zi][2])*(delta)
        clr = [r, g, b]
        #clr.str = "#%x"%((1 << 24) + (r << 16) + (g << 8) + b)
        return clr

    def color(self):
        zMin = self.range['zmin']
        zMax = self.range['zmax']
        cMin = 0
        cMax = 1
        if len(self.colorRange) == 2:
            cMin = self.colorRange[0]
            cMax = self.colorRange[1]
        scale = self.zAutoScale
        if zMax-zMin > 0:
            scale = 1/(zMax-zMin)
        offset = -self.dataOffset['z']*scale
        if len(self.colorScale) == 2:
            scale = self.colorScale[0]
            offset = self.colorScale[1]

        zStep = abs(zMax - zMin) / (len(self.colorMap)-1)
        P = self.rawPoints
        C = self.colorMap
        CL = len(self.colorMap)-1
        self.pointsClr = []
        for i in range(len(P)):
            #z = (P[i][2]/ self.zAutoScale + self.dataOffset['z'])*scale + offset
            z = (P[i][2] - zMin)*scale
            if z <= cMin:
                r, g, b = C[0][0], C[0][1], C[0][2]
            elif z >= cMax:
                r, g, b = C[CL][0], C[CL][1], C[CL][2]
            else:
                z = min(z, cMax)
                zi = math.floor(z*CL)
                delta = z*CL - zi
                r = round(C[zi][0] + (C[zi+1][0]-C[zi][0])*(delta))
                g = round(C[zi][1] + (C[zi+1][1]-C[zi][1])*(delta))
                b = round(C[zi][2] + (C[zi+1][2]-C[zi][2])*(delta))
            self.pointsClr.append([r, g, b])
            #P[i].color.str = "#%x"%((1 << 24) + (r << 16) + (g << 8) + b)
        #print(self.pointsClr)
    def sortByZIndex(self, A, B):
        return A[2] - B[2]

    def update(self, param):
        genGLobj = False
        refresh = False

        if "dimension" in param:
            self.dimension = param['dimension']
            genGLobj = True
        if "colorscale" in param:
            self.colorScale = param.colorscale
        if "showtriangle" in param and self.show['triangle'] != param['showtriangle']:
            self.show['triangle'] = param['showtriangle']
            refresh = True
        if "showmesh" in param and self.show['mesh'] != param['showmesh']:
            self.show['mesh'] = param['showmesh']
            refresh = True
        if "showcontour" in param and self.show['contour'] != param['showcontour']:
            self.show['contour'] = param['showcontour']
            refresh = True
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

        if 'points' in param:
            self.reset()
            self.rawPoints = []
            self.points = []
            xmin = sys.maxsize
            xmax = -sys.maxsize
            ymin = sys.maxsize
            ymax = -sys.maxsize
            zmin = sys.maxsize
            zmax = -sys.maxsize
            points = param['points']
            for p in points:
                if xmin > p[0]:
                    xmin = p[0]
                if xmax < p[0]:
                    xmax = p[0]
                if ymin > p[1]:
                    ymin = p[1]
                if ymax < p[1]:
                    ymax = p[1]
                if zmin > p[2]:
                    zmin = p[2]
                if zmax < p[2]:
                    zmax = p[2]
            xmin, ymin, zmin = np.amin(points, 0)
            xmax, ymax, zmax = np.amax(points, 0)
            self.dataOffset['x'] = (xmax+xmin)/2
            self.dataOffset['y'] = (ymax+ymin)/2
            self.dataOffset['z'] = (zmax+zmin)/2
            o = self.dataOffset
            zg = 1
            if zmax-zmin > 0:
                zg = (ymax-ymin)/(zmax-zmin)
            self.zAutoScale = zg
            for p in points:
                t = [p[0]-o['x'], p[1]-o['y'], (p[2]-o['z'])*zg]
                self.rawPoints.append(list(t))
                self.points.append(list(t))
            xmin -= o['x']
            xmax -= o['x']
            ymin -= o['y']
            ymax -= o['y']
            zmin = (zmin-o['z'])*zg
            zmax = (zmax-o['z'])*zg
            self.range = {'xmin':xmin, 'ymin':ymin, 'zmin':zmin,
                          'xmax':xmax, 'ymax':ymax, 'zmax':zmax}
            self.scale['base'] = min((self.W-self.margin['x']*2)/(xmax-xmin), (self.H-self.margin['y']*2)/(ymax-ymin))
            # align the default graph around the center
            # W/2-(xmax-xmin)*scale/2 = offset + xmin*scale
            self.offset['base'] = {'x': self.W/2 - (xmax+xmin)*self.scale['base']/2,
                                   'y': self.H/2 - (ymax+ymin)*self.scale['base']/2}
            self.color()
            refresh = True
            genGLobj = True

        if "colormap" in param:
            self.colorMap = param['colormap']
            self.color()
            genGLobj = True
            refresh = True

        if genGLobj:
            self.glObject = self.prepareGL()
        if refresh:
            self.redraw()
    def sortByZValue(self, A, B):
        return A[2] - B[2]

    def calc_contour(self, v, level):
        v.sort(key=cmp_to_key(self.sortByZValue))
        p1 = [0, 0, level]
        p2 = [0, 0, level]
        valid = False
        if v[2][2] < level or v[0][2] > level:
            # do nothing
            pass
        elif v[0][2] < level and v[1][2] > level:
            p1[0] = (v[0][0] - v[1][0])/(v[0][2] - v[1][2])*(level-v[0][2])+v[0][0]
            p1[1] = (v[0][1] - v[1][1])/(v[0][2] - v[1][2])*(level-v[0][2])+v[0][1]
            p2[0] = (v[0][0] - v[2][0])/(v[0][2] - v[2][2])*(level-v[0][2])+v[0][0]
            p2[1] = (v[0][1] - v[2][1])/(v[0][2] - v[2][2])*(level-v[0][2])+v[0][1]
            valid = True
        elif v[1][2] < level and v[2][2] > level:
            p1[0] = (v[2][0] - v[0][0])/(v[2][2] - v[0][2])*(level-v[2][2])+v[2][0]
            p1[1] = (v[2][1] - v[0][1])/(v[2][2] - v[0][2])*(level-v[2][2])+v[2][1]
            p2[0] = (v[2][0] - v[1][0])/(v[2][2] - v[1][2])*(level-v[2][2])+v[2][0]
            p2[1] = (v[2][1] - v[1][1])/(v[2][2] - v[1][2])*(level-v[2][2])+v[2][1]
            valid = True
        elif v[0][2] == level and v[1][2] == level:
            p1[0] = v[0][0]
            p1[1] = v[0][1]
            p2[0] = v[1][0]
            p2[1] = v[1][1]
            valid = True
        elif v[1][2] == level and v[2][2] == level:
            p1[0] = v[1][0]
            p1[1] = v[1][1]
            p2[0] = v[2][0]
            p2[1] = v[2][1]
            valid = True

        return {'valid':valid, 'p1':p1, 'p2':p2}

    def prepareContour(self, L, R):
        # buffer canvas
        level = []
        zmin = self.range['zmin']
        zmax = self.range['zmax']
        for i in range(self.contourLevels):
            level.append((zmax-zmin)/self.contourLevels*i+zmin)
        my = self.dimension['y']
        P = self.rawPoints
        vertexAll = []
        meshAll = []
        colourAll = []

        vertex = []
        mesh = []
        colour = []
        minmax = []
        for i in range(L, R):
            if i%my == my-1 or i+my+1 >= len(self.points):
                minmax.append([0, 0])
                continue
            t1 = min(P[i][2], P[i+1][2], P[i+my][2], P[i+my+1][2])
            t2 = max(P[i][2], P[i+1][2], P[i+my][2], P[i+my+1][2])
            minmax.append([t1, t2])

        for lev in level:
            clr = self.getColorByZ(lev)
            clr = [clr[0]/255, clr[1]/255, clr[2]/255, 1]
            for i in range(L, R):
                if i%my == my-1 or i+my+1 >= len(self.points):
                    continue
                if minmax[i-L][0] > lev or minmax[i-L][1] < lev:
                    continue
                if len(vertex) > 2**16-10:
                    vertex = [j for i in vertex for j in i]
                    mesh = [j for i in mesh for j in i]
                    colour = [j for i in colour for j in i]
                    vertexAll.append(vertex)
                    meshAll.append(mesh)
                    colourAll.append(colour)
                    vertex = []
                    mesh = []
                    colour = []
                v = []
                v.append(list(P[i]))
                v.append(list(P[i+1]))
                v.append(list(P[i+my+1]))
                p = self.calc_contour(v, lev)
                if p['valid']:
                    vertex.append(list(p['p1']))
                    mesh.append(len(vertex)-1)
                    vertex.append(list(p['p2']))
                    mesh.append(len(vertex)-1)
                    colour.append(clr)
                    colour.append(clr)

                v = []
                v.append(list(P[i]))
                v.append(list(P[i+my]))
                v.append(list(P[i+my+1]))
                p = self.calc_contour(v, lev)
                if p['valid']:
                    vertex.append(p['p1'])
                    mesh.append(len(vertex)-1)
                    vertex.append(p['p2'])
                    mesh.append(len(vertex)-1)
                    colour.append(clr)
                    colour.append(clr)
        vertex = [j for i in vertex for j in i]
        #mesh = [j for i in mesh]
        colour = [j for i in colour for j in i]
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
        clr = {'r':255, 'g':255, 'b':255}
        block = math.ceil(len(self.points)*4/(2**16)+1)
        blocks = []
        step = math.floor(len(self.points)/block)
        my = self.dimension['y']
        for i in range(0, len(self.points), step):
            blocks.append(i)
        blocks.append(len(self.points))
        self.blocks = blocks
        r = blocks[0]
        l = 0
        for k in range(1, len(blocks)):
            l = r
            r = blocks[k]
            acolor = []
            avertex = []
            P = self.rawPoints
            LEN = len(self.points)
            for i in range(l, min(r+my+1, len(P))):
                clr = self.pointsClr[i]
                clr = [clr[0]/255.0, clr[1]/255.0, clr[2]/255.0, 1.0]
                acolor.append(list(clr))
                acolor.append(list(clr))
                acolor.append(list(clr))
                acolor.append(list(clr))
                pos = list(P[i])
                avertex.append(list(pos))
                if i+1 < LEN:
                    pos[0] = P[i+1][0]
                    pos[1] = P[i+1][1]

                avertex.append(list(pos))
                if i+1+my < LEN:
                    pos[0] = P[i+my+1][0]
                    pos[1] = P[i+my+1][1]

                avertex.append(list(pos))
                if i+my < LEN:
                    pos[0] = P[i+my][0]
                    pos[1] = P[i+my][1]
                avertex.append(list(pos))

            triangle = []
            mesh = []
            for i in range(0, r-l):
                if (i+l)%my == my-1:
                    continue
                if i+l+my+1 < len(self.points):
                    triangle.append([4*i, 4*i+1, 4*i+2, 4*i, 4*i+2, 4*i+3])
                    mesh.append([4*i, 4*i+1, 4*i+1, 4*i+2, 4*i+2, 4*i+3, 4*i+3, 4*i])
                    triangle.append([4*i+3, 4*i+2, 4*(i+my)+1, 4*i+3, 4*(i+my)+1, 4*(i+my)])
                    mesh.append([4*i+3, 4*(i+my), 4*i+2, 4*(i+my)+1])
                    triangle.append([4*i+1, 4*(i+1), 4*(i+1)+3, 4*i+1, 4*(i+1)+3, 4*(i)+2])
                    mesh.append([4*i+1, 4*(i+1), 4*i+2, 4*(i+1)+3])

            contour = self.prepareContour(l, r)
            avertex = [j for i in avertex for j in i]
            acolor = [j for i in acolor for j in i]
            triangle = [j for i in triangle for j in i]
            mesh = [j for i in mesh for j in i]
            vertexAll.append(avertex)
            colorAll.append(acolor)
            triangleAll.append(triangle)
            meshAll.append(mesh)
            contourAll.append(contour)
        return {'Color':colorAll, 'Vertices':vertexAll,
                'Trinagles':triangleAll, 'Mesh':meshAll, 'Contour':contourAll}

    def setGLBuffer(self, v, c):
        VertexBuffer = glGenBuffers(1) # Create a New Buffer
        #Bind it as The Current Buffer
        glBindBuffer(GL_ARRAY_BUFFER, VertexBuffer)
        # Fill it With the Data
        glBufferData(GL_ARRAY_BUFFER, np.array(v).astype(np.float32), GL_STATIC_DRAW)

        glVertexAttribPointer(self.VertexPosition, 3, GL_FLOAT, GL_FALSE, 0, None)
        cubeVerticesColorBuffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, cubeVerticesColorBuffer)
        glBufferData(GL_ARRAY_BUFFER, np.array(c).astype(np.float32), GL_STATIC_DRAW)
        glVertexAttribPointer(self.VertexColor, 4, GL_FLOAT, GL_FALSE, 0, None)

    def drawElementGL(self, t, v, m):
        TriangleBuffer = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, TriangleBuffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.array(v).astype(np.uint16), GL_STATIC_DRAW)
        glUniform1i(glGetUniformLocation(self.ShaderProgram, "mesh"), m)
        glDrawElements(t, len(v), GL_UNSIGNED_SHORT, None)

    def drawAxisGL(self):
        # var gx = 2*32/this.W*this.scale.base
        r = self.range
        p = [r['xmax'], r['ymin'], r['zmin']]
        gx = 32/self.scale['base']
        vertex = [p[0]-gx,   p[1]+gx,   p[2],
                  p[0]-gx,   p[1],      p[2],
                  p[0],      p[1],      p[2],
                  p[0],      p[1]+gx,   p[2],
                  p[0]-gx,   p[1]+gx/4, p[2],
                  p[0],      p[1]+gx/4, p[2],
                  p[0]-gx/3, p[1]+gx/10,   p[2],
                  p[0]-gx/3, p[1]+4*gx/10, p[2]]
        self.drawAxisGL_help('x', vertex)
        p = [r['xmin'], r['ymax'], r['zmin']]
        vertex = [p[0],    p[1],    p[2],
                  p[0],    p[1]-gx, p[2],
                  p[0]+gx, p[1]-gx, p[2],
                  p[0]+gx, p[1],    p[2],
                  p[0]+gx/4, p[1]-gx,     p[2],
                  p[0]+gx/4, p[1],    p[2],
                  p[0]+gx/10,p[1]-gx/3,   p[2],
                  p[0]+4*gx/10,p[1]-gx/3, p[2]]
        self.drawAxisGL_help('y', vertex)
        p = [r['xmin'], r['ymin'], r['zmax']]
        vertex = [p[0], p[1]+gx, p[2]-gx,
                  p[0], p[1],    p[2]-gx,
                  p[0], p[1],    p[2],
                  p[0], p[1]+gx, p[2],
                  p[0], p[1]+gx/4,  p[2]-gx,
                  p[0], p[1]+gx/4,  p[2],
                  p[0], p[1]+gx/10,  p[2]-gx/3,
                  p[0], p[1]+4*gx/10, p[2]-gx/3]
        self.drawAxisGL_help('z', vertex)

    def drawAxisGL_help(self, letter, vertex):
        bitmap = wx.Bitmap(32, 32)
        tdc = wx.MemoryDC()
        tdc.SelectObject(bitmap)
        tdc.SetBackground(wx.Brush('black',wx.BRUSHSTYLE_TRANSPARENT))
        tdc.Clear()
        tdc.SetPen(wx.Pen('blue'))
        tdc.SetTextForeground(wx.RED)
        tdc.SetFont(wx.Font(wx.FontInfo(16).Bold()))
        tw, th = tdc.GetTextExtent(letter)
        tdc.DrawText(letter, (32-tw)/2, (32-th)/2)
        tdc.SelectObject(wx.NullBitmap)

        texture = glGenTextures(1);
        glPixelStorei(GL_UNPACK_ALIGNMENT, GL_TRUE);
        glBindTexture(GL_TEXTURE_2D, texture);
        mybuffer = np.zeros((32, 32,4), np.uint8)
        bitmap.CopyToBuffer(mybuffer, wx.BitmapBufferFormat_RGBA)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 32,32,0,GL_RGBA, GL_UNSIGNED_BYTE, mybuffer);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
        glGenerateMipmap(GL_TEXTURE_2D);

        glBindTexture(GL_TEXTURE_2D, 0);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);

        gx = 2*32/self.W*self.scale['base']
        color = [0, 0, 0, 1,
                 0, 1, 0, 1,
                 1, 1, 0, 1,
                 1, 0, 0, 1,
                 0, 0, 0, 1,
                 0, 0, 0, 1,
                 0, 0, 0, 1,
                 0, 0, 0, 1]
        triangle = [0, 1, 2, 0, 2, 3]
        mesh = [4, 5, 5, 6, 5, 7]
        glUniform1i(glGetUniformLocation(self.ShaderProgram, "uSampler"), 0)
        self.setGLBuffer(vertex, color)
        self.drawElementGL(GL_TRIANGLES, triangle, 3)
        glUniform4f(glGetUniformLocation(self.ShaderProgram, "clr"), 1, 0, 0, 1)
        self.drawElementGL(GL_LINES, mesh, 2)

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
        glUniform3f(vertexScaleGL, 1, -1, -0.1/zgain)

        vertexScale = glGetUniformLocation(self.ShaderProgram, "VertexScale")
        scaleAll = self.scale['base']*self.scale['zoom']
        glUniform3f(vertexScale, scaleAll, scaleAll, 1)

        vertexOffsetGL = glGetUniformLocation(self.ShaderProgram, "VertexOffsetGL")
        glUniform3f(vertexOffsetGL, 0, 0, 0)

        obj = self.glObject
        for b in range(len(self.blocks)-1):
            self.setGLBuffer(obj['Vertices'][b], obj['Color'][b])
            if self.show['triangle']:
                self.drawElementGL(GL_TRIANGLES, obj['Trinagles'][b], 0)
            if self.show['mesh']:
                self.drawElementGL(GL_LINES, obj['Mesh'][b], 1)

            if self.show['contour']:
                ctr = obj['Contour'][b]
                for c in range(len(ctr['Vertices'])):
                    self.setGLBuffer(ctr['Vertices'][c], ctr['Color'][c])
                    self.drawElementGL(GL_LINES, ctr['Mesh'][c], 0)

        scaleAll = self.scale['base']
        glUniform3f(vertexScale, scaleAll, scaleAll, 1)

        offsetAll['x'] = self.offset['base']['x']
        offsetAll['y'] = self.offset['base']['y']

        glUniform3f(vertexOffset, offsetAll['x'], offsetAll['y'], offsetAll['z'])
        if self.show['box']:
            glUniform1i(glGetUniformLocation(self.ShaderProgram, "mesh"), 0)
            r = self.range
            vertex = [r['xmin'], r['ymin'], r['zmin'],
                      r['xmin'], r['ymax'], r['zmin'],
                      r['xmax'], r['ymax'], r['zmin'],
                      r['xmax'], r['ymin'], r['zmin'],
                      r['xmin'], r['ymin'], r['zmax'],
                      r['xmin'], r['ymax'], r['zmax'],
                      r['xmax'], r['ymax'], r['zmax'],
                      r['xmax'], r['ymin'], r['zmax']]
            color = [0, 0, 0, 1,
                     0, 0, 0, 1,
                     0, 0, 0, 1,
                     0, 0, 0, 1,
                     0, 0, 0, 1,
                     0, 0, 0, 1,
                     0, 0, 0, 1,
                     0, 0, 0, 1]
            axis = [0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6,
                    6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7]
            #vertex = [ v / 10 for v in vertex]
            self.setGLBuffer(vertex, color)
            self.drawElementGL(GL_LINES, axis, 1)

        if self.show['axis']:
            self.drawAxisGL()

    def draw(self):
        #if not self.invalidate:
        #    return
        #self.invalidate = False
        self.doRotate()
        self.drawGL()

    def redraw(self):
        self.Refresh()
        #self.invalidate = True
        #self.draw()
        #window.requestAnimationFrame(function() { self.draw(); } )

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
        Rx = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], np.float32)
        Rx[0][0] = 1
        Rx[0][1] = 0
        Rx[0][2] = 0
        Rx[1][0] = 0
        Rx[1][1] = math.cos(sign*self.rotateTheta)
        Rx[1][2] = -math.sin(sign*self.rotateTheta)
        Rx[2][0] = 0
        Rx[2][1] = math.sin(sign*self.rotateTheta)
        Rx[2][2] = math.cos(sign*self.rotateTheta)
        return Rx

    def yRotateMatrix(self, sign):
        Ry = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], np.float32)
        Ry[0][0] = math.cos(sign*self.rotateTheta)
        Ry[0][1] = 0
        Ry[0][2] = math.sin(sign*self.rotateTheta)
        Ry[1][0] = 0
        Ry[1][1] = 1
        Ry[1][2] = 0
        Ry[2][0] = -math.sin(sign*self.rotateTheta)
        Ry[2][1] = 0
        Ry[2][2] = math.cos(sign*self.rotateTheta)
        return Ry

    def zRotateMatrix(self, sign):
        Rz = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], np.float32)
        Rz[0][0] = math.cos(sign*self.rotateTheta)
        Rz[0][1] = -math.sin(sign*self.rotateTheta)
        Rz[0][2] = 0
        Rz[1][0] = math.sin(sign*self.rotateTheta)
        Rz[1][1] = math.cos(sign*self.rotateTheta)
        Rz[1][2] = 0
        Rz[2][0] = 0
        Rz[2][1] = 0
        Rz[2][2] = 1
        return Rz
