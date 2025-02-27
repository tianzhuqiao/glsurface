import math
import six
import numpy as np
import wx
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
uniform vec3 VertexScaleData;
uniform vec3 VertexOffset;
uniform vec3 VertexOffsetData;
uniform vec3 VertexScaleGL;
varying vec2 vTextureCoord;
varying vec4 vColor;
uniform int VertexOriginal;
void main(void) {
    if(VertexOriginal == 1) {
        gl_Position = vec4(VertexPosition, 1.0);
    } else if (VertexOriginal == 2) {
        gl_Position = vec4((((VertexPosition-VertexOffsetData)*VertexScaleData)*VertexScale+VertexOffset)/VertexResolution*2.0-1.0, 1.0);
        gl_Position = gl_Position*vec4(VertexScaleGL,1);
    } else if (VertexOriginal == 3) {
        gl_Position = vec4(((TransformationMatrix*((VertexPosition-VertexOffsetData)*VertexScaleData)*VertexScale)+VertexOffset)/VertexResolution*2.0-1.0, 1.0);
        gl_Position = gl_Position*vec4(VertexScaleGL,1);
        gl_Position = gl_Position + abs(gl_Position)*vec4(0, 0, -0.03, 0);
    } else {
        gl_Position = vec4(((TransformationMatrix*((VertexPosition-VertexOffsetData)*VertexScaleData)*VertexScale)+VertexOffset)/VertexResolution*2.0-1.0, 1.0);
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
    def __init__(self, sz, rng, horz, clr=(0, 0.6, 0, 1)):
        self.horz = horz
        self.range = dict(rng)
        self.bar_color = clr[:]
        self.buf_size = -1
        self.Resize(sz)
        self.bar_color = None
        self.color = []
        self.SetColor(clr)

    def SetRange(self, rng):
        if rng == self.range:
            return
        self.range.update(rng)
        self.Resize(self.buf_size)

    def SetColor(self, clr):
        if clr == self.bar_color:
            return
        self.bar_color = clr[:]
        if self.buf_size > 0:
            self.color = np.repeat(np.array([clr]), self.buf_size * 4,
                                   axis=0).flatten()

    def Resize(self, sz, forced=True):
        if sz == self.buf_size and not forced:
            return
        self.buf_size = sz
        zmax = self.range['zmax']
        self.vertex = np.zeros((sz * 4, 3))
        self.vertex[:, 2].fill(zmax)
        if self.horz:
            # for point d[i], draw lines between points:
            # (i, ymax), (i, ymax-d[i]*g), (i+1, ymax-d[i]*g), (i+1, ymax)
            # see update() for detail
            xmax, xmin = self.range['xmax'], self.range['xmin']
            x = np.linspace(xmin, xmax, sz, endpoint=False).reshape((sz, 1))
            delta = (xmax - xmin) / sz
            self.vertex[:, 0] = (np.repeat(x, 4, axis=1) +
                                 [0, 0, delta, delta]).flatten()
        else:
            ymax, ymin = self.range['ymax'], self.range['ymin']
            y = np.linspace(ymin, ymax, sz, endpoint=False).reshape((sz, 1))
            delta = (ymax - ymin) / sz
            self.vertex[:, 1] = (np.repeat(y, 4, axis=1) +
                                 [0, 0, delta, delta]).flatten()

        self.color = np.repeat(np.array([self.bar_color]),
                               self.buf_size * 4,
                               axis=0).flatten()
        self.line = np.arange(sz * 4)

    def GetGLObject(self):
        return self.vertex.flatten(), self.color, self.line

    def SetData(self, d):
        if len(d) != self.buf_size:
            raise ValueError()
        if self.horz:
            ymax = self.range['ymax']
            self.vertex[:, 1] = ymax - (np.kron(d, [0, 1, 1, 0])).flatten()
        else:
            xmax = self.range['xmax']
            self.vertex[:, 0] = xmax - (np.kron(d, [0, 1, 1, 0])).flatten()


class SurfaceBase(glcanvas.GLCanvas):
    ID_SHOW_2D = wx.NewIdRef()
    ID_SHOW_STEP_SURFACE = wx.NewIdRef()
    ID_SHOW_TRIANGLE = wx.NewIdRef()
    ID_SHOW_MESH = wx.NewIdRef()
    ID_SHOW_CONTOUR = wx.NewIdRef()
    ID_SHOW_BOX = wx.NewIdRef()
    ID_SHOW_AXIS = wx.NewIdRef()
    ID_SHOW_HORZ_BAR = wx.NewIdRef()
    ID_SHOW_VERT_BAR = wx.NewIdRef()
    ID_SHOW_HUDTEXT = wx.NewIdRef()
    ID_ROTATE_0 = wx.NewIdRef()
    ID_ROTATE_90 = wx.NewIdRef()
    ID_ROTATE_180 = wx.NewIdRef()
    ID_ROTATE_270 = wx.NewIdRef()

    def __init__(self, parent, points=None, dispAttrs=None):
        if dispAttrs is None:
            dispAttrs = glcanvas.GLAttributes()
            # set the depth size, otherwise the depth test may not work correctly.
            dispAttrs.PlatformDefaults().RGBA().DoubleBuffer().Depth(16).EndList()

        glcanvas.GLCanvas.__init__(self, parent, dispAttrs=dispAttrs)
        self.context = glcanvas.GLContext(self)

        self.drag_start = wx.Point(0, 0)
        self.drag_end = wx.Point(0, 0)
        self.points = []
        self.pointsClr = []
        self.dimension = {'x': 0, 'y': 0}
        self.is_mouse_down = False
        self.is_dragging = False
        self.rotation = {'x': 110, 'y': 0, 'z': 110}
        self.default_rotate = 0
        sz = self.GetClientSize()
        self.W = sz.x
        self.H = sz.y
        self.range = {
            'xmin': 0,
            'ymin': 0,
            'zmin': 0,
            'xmax': 0,
            'ymax': 0,
            'zmax': 1
        }
        self.zrange = []
        self.scale = {'base': 1, 'zoom_x': 1, 'zoom_y': 1}
        self.rotate_delta = 0.05
        self.margin = {'top': 10, 'bottom': 10, 'left': 1, 'right': 1}
        self.offset = {'base': {'x': 0, 'y': 0}, 'user': {'x': 0, 'y': 0}}
        self.data_offset = {'x': 0, 'y': 0, 'z': 0}
        self.selected = {'x': 0, 'y': 0, 'clr': [1, 0, 1, 1]}
        self.color_scale = []
        self.blocks = []
        self.data_zscale = 1
        self.data_zscale_bar = 1 # the scale for the horz/vert bar
        # 2d mode is the fast-drawing mode, only shows the 2D image
        self.mode_2d = False
        self.show = {
            'surface': True,
            'mesh': False,
            'contour': False,
            'box': False,
            'axis': False,
            'horz_bar': False,
            'vert_bar': False,
            'hudtext': True
        }
        self.show_step_surface = True
        self.contour_levels = 100
        self.rotate_matrix = np.eye(3, dtype=np.float32)
        self.color_map = np.array(
            [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [0, 1, 0, 1],
             [1, 1, 0, 1], [1, 0.5, 0, 1], [1, 0, 0, 1]], np.float32)
        self.initialized = False
        self.raw_points = points
        self.globjects = {}
        self.hudtext = ''
        self._hudtext = ''
        self._hudBuffer = np.zeros((0, 0, 4))
        self.hudtext_clr = wx.WHITE
        self.hudtext_font = wx.Font(14, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.axis_clr = (1, 1, 1, 1)
        self.axistext_clr = wx.WHITE
        self.axistext_font = wx.Font(14, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.box_clr = (1, 1, 1, 1)

        # buffers to draw bar
        if self.raw_points is not None and 'z' in self.raw_points:
            rows, cols = self.raw_points['z'].shape
        else:
            rows, cols = 16, 16
        self.horzbar_buf = SimpleBarBuf(cols, self.range, True)
        self.vertbar_buf = SimpleBarBuf(rows, self.range, False)

        self.pixels = None
        self.pixels_size = (0, 0)

        self.background_color = [0, 0, 0, 1]

        # Ugly. do not call SetImage or similar methods here, since it may be
        # overridden or may call some overridden methods. Instead, it should be
        # put in Initialize() method
        #self.SetImage(self.raw_points)

        accel_tbl = self.GetAccelList()
        self.SetAcceleratorTable(wx.AcceleratorTable(accel_tbl))

        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)
        self.Bind(wx.EVT_CHAR_HOOK, self.OnKeyDown)
        self.Bind(wx.EVT_LEFT_DCLICK, self.OnDoubleClick)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel)
        self.Bind(wx.EVT_CONTEXT_MENU, self.OnContextMenu)
        self.Bind(wx.EVT_MENU, self.OnProcessMenuEvent)
        self.Bind(wx.EVT_UPDATE_UI, self.OnUpdateMenu)

    def Clear(self):
        self.raw_points = None
        self.globjects = None
        self.Refresh()

    def GetAccelList(self):
        accel_tbl = [
            (wx.ACCEL_SHIFT, ord('S'), self.ID_SHOW_TRIANGLE),
            (wx.ACCEL_SHIFT, ord('H'), self.ID_SHOW_HORZ_BAR),
            (wx.ACCEL_SHIFT, ord('V'), self.ID_SHOW_VERT_BAR),
        ]
        return accel_tbl

    def OnEraseBackground(self, event):
        pass  # Do nothing, to avoid flashing on MSW.

    def OnContextMenu(self, event):
        menu = self.GetContextMenu()
        if menu:
            self.PopupMenu(menu)

    def BuildElementMenu(self):
        menu = wx.Menu()
        menu.Append(self.ID_SHOW_2D, 'Show 2D Mode', '', wx.ITEM_CHECK)
        menu.AppendSeparator()
        menu.Append(self.ID_SHOW_STEP_SURFACE, 'Step Surface', '',
                wx.ITEM_CHECK)
        menu.Append(self.ID_SHOW_TRIANGLE, 'Show Surface\tshift+s', '',
                wx.ITEM_CHECK)
        menu.Append(self.ID_SHOW_MESH, 'Show Mesh', '', wx.ITEM_CHECK)
        menu.AppendSeparator()
        menu.Append(self.ID_SHOW_CONTOUR, 'Show Contour', '',
                wx.ITEM_CHECK)
        menu.Append(self.ID_SHOW_BOX, 'Show Box', '', wx.ITEM_CHECK)
        menu.Append(self.ID_SHOW_AXIS, 'Show Axis', '', wx.ITEM_CHECK)
        menu.AppendSeparator()
        menu.Append(self.ID_SHOW_HUDTEXT, 'Show Hud Text', '', wx.ITEM_CHECK)
        menu.AppendSeparator()
        menu.Append(self.ID_SHOW_HORZ_BAR, 'Show Horz Bar\tshift+h', '',
                wx.ITEM_CHECK)
        menu.Append(self.ID_SHOW_VERT_BAR, 'Show Vert Bar\tshift+v', '',
                        wx.ITEM_CHECK)

        return menu

    def GetContextMenu(self):
        menu = wx.Menu()
        elements = self.BuildElementMenu()
        menu.AppendSubMenu(elements, 'Elements')
        rotate = wx.Menu()
        rotate.Append(self.ID_ROTATE_0, 'Rotate 0 degree', '', wx.ITEM_CHECK)
        rotate.Append(self.ID_ROTATE_90, 'Rotate 90 degree', '', wx.ITEM_CHECK)
        rotate.Append(self.ID_ROTATE_180, 'Rotate 180 degree', '',
                      wx.ITEM_CHECK)
        rotate.Append(self.ID_ROTATE_270, 'Rotate 270 degree', '',
                      wx.ITEM_CHECK)
        menu.AppendSeparator()
        menu.AppendSubMenu(rotate, 'Orientation')
        return menu

    def SetHudTextClr(self, clr):
        self.hudtext_clr = clr
        self._hudtext = ''
        self.Refresh()

    def SetHudTextFont(self, font):
        self.hudtext_font = font
        self._hudtext = ''
        self.Refresh()

    def SetBoxClr(self, clr):
        self.box_clr = clr
        self.Refresh()

    def SetAxisClr(self, clr, clr_text):
        self.axis_clr = clr
        self.axistext_clr = clr_text
        self.Refresh()

    def SetAxisTextFont(self, font):
        self.axistext_font = font
        self.Refresh()

    def SetSurfaceBackground(self, clr):
        self.background_color = clr
        self.Refresh()

    def GetSurfaceBackground(self):
        return self.background_color

    def Set2dMode(self, is2d):
        self.mode_2d = is2d
        if not self.mode_2d:
            self.SetImage(self.raw_points)

    def Get2dMode(self):
        return self.mode_2d

    def SetShowStepSurface(self, step):
        self.show_step_surface = step
        self.Invalidate()

    def GetShowStepSurface(self):
        return self.show_step_surface

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
            self.Invalidate()
        if refresh:
            self.Refresh()

    def OnProcessMenuEvent(self, event):
        eid = event.GetId()
        show = {}
        if eid == self.ID_SHOW_2D:
            self.Set2dMode(not self.mode_2d)
            self.Refresh()
        elif eid == self.ID_SHOW_STEP_SURFACE:
            self.SetShowStepSurface(not self.GetShowStepSurface())
        elif eid == self.ID_SHOW_TRIANGLE:
            show['surface'] = not self.show['surface']
        elif eid == self.ID_SHOW_MESH:
            show['mesh'] = not self.show['mesh']
        elif eid == self.ID_SHOW_CONTOUR:
            show['contour'] = not self.show['contour']
        elif eid == self.ID_SHOW_BOX:
            show['box'] = not self.show['box']
        elif eid == self.ID_SHOW_AXIS:
            show['axis'] = not self.show['axis']
        elif eid == self.ID_SHOW_HUDTEXT:
            show['hudtext'] = not self.show['hudtext']
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
            self.ResetRotate()
            self.Resize()
            self.Refresh()
            return
        if show:
            self.SetShowMode(**show)
            if 'hudtext' in show:
                self.Resize()

    def OnUpdateMenu(self, event):
        eid = event.GetId()
        if eid == self.ID_SHOW_2D:
            event.Check(self.mode_2d)
        elif eid == self.ID_SHOW_STEP_SURFACE:
            event.Check(self.GetShowStepSurface())
            event.Enable(not self.mode_2d)
        elif eid == self.ID_SHOW_TRIANGLE:
            event.Check(self.show['surface'])
            event.Enable(not self.mode_2d)
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
        elif eid == self.ID_SHOW_HUDTEXT:
            event.Check(self.show['hudtext'])
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

    def SetMargin(self, left=None, right=None, top=None, bottom=None):
        for m in zip([left, right, top, bottom],
                     ['left', 'right', 'top', 'bottom']):
            if m[0] is not None:
                self.margin[m[1]] = m[0]

    def GetMargin(self):
        return self.margin

    def Resize(self):
        self.SetCurrent(self.context)
        sz = self.GetClientSize()
        self.W = sz.x
        self.H = sz.y
        scale = self.GetContentScaleFactor()
        glViewport(0, 0, int(self.W*scale), int(self.H*scale))
        xmax, xmin = self.range['xmax'], self.range['xmin']
        ymax, ymin = self.range['ymax'], self.range['ymin']
        if self.default_rotate in [90, 270]:
            xmax, xmin, ymax, ymin = ymax, ymin, xmax, xmin
        if xmax - xmin > 0 and ymax - ymin > 0:
            t, b = self.margin['top'], self.margin['bottom']
            l, r = self.margin['left'], self.margin['right']
            if self.show['hudtext']:
                t += 40
            self.scale['base'] = min((self.W - l - r) / (xmax - xmin),
                                     (self.H - t - b) / (ymax - ymin))
            self.offset['base'] = {
                'x': self.W / 2 + (l - r) / 2,
                'y': self.H / 2 + (t - b) / 2
            }

    def OnSize(self, event):
        # update the size
        self.Resize()
        self.Refresh()
        event.Skip()

    def _has_anything_to_draw(self):
        return self.raw_points is not None or (self.globjects is not None and bool(self.globjects))

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self._has_anything_to_draw():
            glClearColor(*self.background_color)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        else:
            if not self.initialized:
                self.initialized = True
                self.Initialize()
            self.Draw()

        # save the buffer
        view = glGetIntegerv(GL_VIEWPORT)
        self.pixels = glReadPixels(view[0], view[1], view[2], view[3], GL_RGBA, GL_UNSIGNED_BYTE)
        self.pixels_size = (view[2], view[3])

        # actual drawing
        self.SwapBuffers()

    def Initialize(self):
        self.InitGL()
        if self.raw_points is not None:
            self.SetImage(self.raw_points)
            self.ResetRotate()

    def InitGL(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)  # this is the color
        glEnable(GL_DEPTH_TEST)  # Enable Depth Testing
        glDepthFunc(GL_LEQUAL)  # Set Perspective View
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

        glEnable(GL_LINE_SMOOTH)

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
        self.VertexPosition = glGetAttribLocation(self.ShaderProgram,
                                                  "VertexPosition")
        glEnableVertexAttribArray(self.VertexPosition)
        self.VertexColor = glGetAttribLocation(self.ShaderProgram,
                                               "VertexColor")
        glEnableVertexAttribArray(self.VertexColor)
        glViewport(0, 0, self.W, self.H)
        # generate the buffer
        self.glBufs = glGenBuffers(3)
        self.glTextures = glGenTextures(3)  # axis, hud, 2d image

    def GetBitmap(self):
        if self.pixels is not None:
            bitmap = wx.Bitmap.FromBufferRGBA(int(self.pixels_size[0]),
                                              int(self.pixels_size[1]),
                                              self.pixels)
            scale_factor = self.GetContentScaleFactor()
            bitmap.SetScaleFactor(scale_factor)

            img = bitmap.ConvertToImage()
            img = img.Mirror(False)
            bitmap = img.ConvertToBitmap()
            bitmap.SetScaleFactor(scale_factor)
            return bitmap
        return None

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
            self.Rotate(x, y, z)
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
            y = (y + dy) % my
            x = (x + dx) % mx
            self.SetSelected({'x': x, 'y': y})

    def OnMouseDown(self, event):
        bRect = self.GetClientRect()
        pos = event.GetPosition()
        self.drag_start.x = (pos.x - bRect.left)  #*(self.W/bRect.width)
        self.drag_start.y = (pos.y - bRect.top)  #*(self.H/bRect.height)
        self.is_mouse_down = True
        event.Skip()

    def OnMouseMotion(self, event):
        event.Skip()
        bRect = self.GetClientRect()
        pos = event.GetPosition()
        self.drag_end.x = (pos.x - bRect.left)  #*((self.W)/bRect.width)
        self.drag_end.y = (pos.y - bRect.top)  #*((self.H)/bRect.height)
        if self.is_mouse_down:
            if event.CmdDown():
                # when ctrl is down, move the image with mouse
                self.offset['user']['x'] += self.drag_end.x - self.drag_start.x
                self.offset['user']['y'] += self.drag_end.y - self.drag_start.y
                self.drag_start.x = self.drag_end.x
                self.drag_start.y = self.drag_end.y
                self.Refresh()
            elif not self.mode_2d:
                # rotate in 3d mode
                if abs(self.drag_start.x-self.drag_end.x) > 1 or \
                   abs(self.drag_start.y-self.drag_end.y) > 1:
                    self.is_dragging = True
                if self.is_dragging:
                    self.Rotate(-(self.drag_end.y - self.drag_start.y),
                                (self.drag_end.x - self.drag_start.x), 0)
                self.drag_start.x = self.drag_end.x
                self.drag_start.y = self.drag_end.y

    def OnMouseUp(self, event):
        self.is_mouse_down = False
        self.is_dragging = False
        self.Refresh()
        event.Skip()

    def OnDoubleClick(self, event):
        self.ResetRotate()
        self.Refresh()
        event.Skip()

    def OnMouseWheel(self, event):
        delta = (event.GetWheelRotation() > 0) * 2 - 1
        bRect = self.GetClientRect()
        if bRect.width <= 0 or bRect.height <= 0:
            return
        pos = event.GetPosition()
        x = (pos.x - bRect.left) * ((self.W) / bRect.width)
        y = (pos.y - bRect.top) * ((self.H) / bRect.height)
        if delta:
            zoom_x = 1.1**delta
            zoom_y = 1.1**delta
            if event.ShiftDown():
                # shift down, only y-axis
                zoom_x = 1
            if event.CmdDown():
                # cmd down, only x-axis
                zoom_y = 1
            self.scale['zoom_x'] *= zoom_x
            self.scale['zoom_y'] *= zoom_y
            base = self.offset['base']
            user = self.offset['user']
            user['x'] = x - (x - base['x'] - user['x']) * zoom_x - base['x']
            user['y'] = y - (y - base['y'] - user['y']) * zoom_y - base['y']
            self.Refresh()

        event.Skip()

    def ResetRotate(self):
        # reset rotation, zoom, offset
        self.rotation = {'x': 0, 'y': 0, 'z': 0}
        self.rotate_matrix = np.eye(3, dtype=np.float32)
        self.scale['zoom_x'] = 1
        self.scale['zoom_y'] = 1
        self.offset['user'] = {'x': 0, 'y': 0}
        # set the default orientation
        self.Rotate(0, 0,
                    self.default_rotate / 180. * np.pi / self.rotate_delta)

    def GetColorByZ(self, z, zmin=None, zmax=None, clr_map=None):
        zMin = zmin
        if zMin is None:
            zMin = self.range['zmin']
        zMax = zmax
        if zMax is None:
            zMax = self.range['zmax']
        scale = 1
        if zMax - zMin > 0:
            scale = 1 / (zMax - zMin)
        offset = 0
        if len(self.color_scale) == 2:
            scale = self.color_scale[0]
            offset = self.color_scale[1]
        C = clr_map
        if C is None:
            C = self.color_map
        norm_p = (z - zMin) * scale + offset
        norm_p = norm_p * len(self.color_map)
        norm_p = np.clip(norm_p, 0, len(self.color_map) - 1)
        norm_p_l = np.floor(norm_p).astype(int)
        norm_p_r = np.ceil(norm_p).astype(int)
        norm_p_d = norm_p - norm_p_l
        return C[norm_p_l, :] + ((
            (C[norm_p_r, :] - C[norm_p_l, :]).T) * norm_p_d).T

    def Colorize(self):
        if self.points is None or len(self.points) <= 0:
            return
        self.pointsClr = self.GetColorByZ(self.points[:, 2])
        sel = self.selected['x'] * self.dimension['y'] + self.selected['y']
        if sel >= 0 and sel < self.points.shape[0]:
            self.pointsClr[sel, :] = self.selected['clr']

    def SetContourLevels(self, level):
        if level == self.contour_levels:
            return
        self.contour_levels = level

        # the objects will be re-generated before drawing
        self.Invalidate()

    def GetContourLevels(self):
        return self.contour_levels

    def SetDataScaleZ(self, scale):
        # add additional scale to z value, so that it may look better when
        # rotate the image (to project z value on x/y axis); otherwise, the
        # project may be too small or large. For example, x/y: [0: 100],
        # z [0: 0.01]
        self.data_zscale = scale

    def SetRange(self, rng):
        self.range.update(rng)

        xmin, xmax = self.range['xmin'], self.range['xmax']
        ymin, ymax = self.range['ymin'], self.range['ymax']
        zmin, zmax = self.range['zmin'], self.range['zmax']
        self.data_offset['x'] = (xmax + xmin) / 2
        self.data_offset['y'] = (ymax + ymin) / 2
        self.data_offset['z'] = (zmax + zmin) / 2

        self.horzbar_buf.SetRange(self.range)
        self.vertbar_buf.SetRange(self.range)
        self.Resize()
        self._UpdateDataSacleZ()
        self.Invalidate()

    def GetRange(self):
        return self.range

    def _UpdateDataSacleZ(self):
        zmax, zmin = self.range['zmax'], self.range['zmin']
        xmax, xmin = self.range['xmax'], self.range['xmin']
        ymax, ymin = self.range['ymax'], self.range['ymin']
        rng = max(ymax - ymin, xmax - xmin)
        rng_min = min(ymax - ymin, xmax - xmin)
        # scale the z data, such that it has similar span as x or y axis.
        # otherwise, when rotate the image, the projection of z on x or y axis
        # will not look good (e.g., too small or too large).
        zscale = 1
        zscale_bar = 1
        if zmax - zmin > 0 and rng > 0:
            zscale = float(rng) / (zmax - zmin)
            zscale_bar = float(rng_min) / (zmax - zmin)
        self.data_zscale_bar = zscale_bar
        self.SetDataScaleZ(zscale)

    def SetRangeZ(self, zrange):
        # Set the range of the image values; otherwise it is calculated from
        # the data itself. It will impact the color of each pixel. Instead of
        # fully utilizing all the colors in color map, we may need to fix the
        # color code for each value (e.g., 0 is always blue)
        self.zrange = []
        if len(zrange) == 2:
            self.zrange = zrange
            self.SetRange({'zmin': zrange[0], 'zmax': zrange[1]})

    def SetHudText(self, txt):
        if txt != self.hudtext:
            self.hudtext = txt
            self.Refresh()

    def GetHudText(self):
        return self.hudtext

    def UpdateHudText(self):
        x, y = self.selected['x'], self.selected['y']
        r, c = self.raw_points['z'].shape
        if x >= 0 and y >= 0 and x < c and y < r:
            self.SetHudText('(%d, %d) %.4f' %
                            (x, y, self.raw_points['z'][y, x]))

    def Invalidate(self, globject=True, color=True):
        if not self.initialized:
            # haven't been initialized yet, no need to do any processing. It
            # will be done during the initialization.
            return

        if color:
            self.Colorize()
        if globject:
            # the objects will be re-generated before drawing
            self.globjects = None
        self.Refresh()

    def SetSelected(self, sel):
        self.selected.update(sel)
        if self.frames is not None:
            r, c = self.selected['y'], self.selected['x']
            _, rows, cols = self.frames.shape
            if r >= 0 and c >= 0 and r < rows and c < cols:
                data = self.frames[:, r, c]
                self.selected_buf.buf = data

        self.UpdateHudText()
        self.Invalidate()

    def GetSelected(self):
        return self.selected

    def SetColorMap(self, clrmap):
        self.color_map = clrmap
        self.Invalidate()

    def GetColorMap(self):
        return self.color_map

    def SetColorScale(self, scale):
        self.color_scale = scale

    def GetColorScale(self):
        return self.color_scale

    def UpdateImage(self, points):
        # light-weighted function to update the image value as fast as possible,
        #assume its dimension does not change

        self.raw_points['z'] = points
        self.UpdateHudText()
        rows, cols = points.shape
        self.horzbar_buf.Resize(cols, forced=False)
        self.vertbar_buf.Resize(rows, forced=False)

        self.Pz[0:rows, 0:cols] = points
        self.Pz[-1, :] = self.Pz[-2, :]
        self.Pz[:, -1] = self.Pz[:, -2]
        if self.mode_2d and not self.show['surface']:
            # in 2d mode, point will only affect surface
            self.points[:, 2].fill(0)
        else:
            self.points[:, 2] = self.Pz.T.flatten()

        self.Invalidate()

    def SetDimension(self, rows, cols):
        # 'x', 'y' is used in 3D mode, where the image is expanded 1 pixel on
        # each dimension
        self.dimension = {
            'y': rows + 1,
            'x': cols + 1,
            'rows': rows,
            'cols': cols
        }
        # resize the horz/vert bar buffer
        self.horzbar_buf.Resize(cols)
        self.vertbar_buf.Resize(rows)
        # resize the image buffer
        self.Pz = np.zeros((rows + 1, cols + 1))
        self.points = np.zeros((self.Pz.size, 3))

    def GetDimension(self):
        return self.dimension

    def SetImage(self, points):
        # points is a dictionary with keys ['x', 'y', 'z'], and each should
        # be a 2D matrix with same dimension. 'z' is mandatory.
        # This function is slow, since it will allocate the memory and
        # regenerate x/y data if necessary
        self.raw_points = points
        rows, cols = points['z'].shape
        self.SetDimension(rows, cols)
        # expand the image by 1 pixel on each dimension, since for pixel (i, j)
        # the surface is draw as a quad defined by
        # (x(i) y(i), z(i)), (x(i+1) y(i), z(i))
        # (x(i) y(i+1), z(i)), (x(i+1) y(i+1), z(i))
        xy = np.meshgrid(np.arange(0, cols + 1), np.arange(0, rows + 1))
        # if 'x' data is not defined, assume it is [0:cols]
        if 'x' not in points:
            Px = xy[0]
        else:
            Px = np.zeros((rows + 1, cols + 1))
            Px[0:rows, 0:cols] = points['x']
            Px[-1, :] = Px[-2, :]
            if cols > 1:
                Px[:, -1] = Px[:, -2] * 2 - Px[:, -3]
            else:
                Px[:, -1] = Px[:, -2] + 1

        # if 'x' data is not defined, assume it is [0:cols]
        if 'y' not in points:
            Py = xy[1]
        else:
            Py = np.zeros((rows + 1, cols + 1))
            Py[0:rows, 0:cols] = points['y']
            Py[:, -1] = Py[:, -2]
            if rows > 1:
                Py[-1, :] = Py[-2, :] * 2 - Py[-3, :]
            else:
                Py[-1, :] = Py[-2, :] + 1

        self.Pz[0:rows, 0:cols] = points['z']
        self.Pz[-1, :] = self.Pz[-2, :]
        self.Pz[:, -1] = self.Pz[:, -2]
        xmin, ymin, zmin = np.min(Px), np.min(Py), np.min(self.Pz)
        xmax, ymax, zmax = np.max(Px), np.max(Py), np.max(self.Pz)
        if self.zrange:
            zmin, zmax = self.zrange

        self.points = np.zeros((self.Pz.size, 3))
        self.points[:, 0] = Px.T.flatten()
        self.points[:, 1] = Py.T.flatten()
        self.points[:, 2] = self.Pz.T.flatten()
        self.SetRange({
            'xmin': xmin,
            'ymin': ymin,
            'zmin': zmin,
            'xmax': xmax,
            'ymax': ymax,
            'zmax': zmax
        })

        self.UpdateHudText()
        self.Invalidate()

    def CalcContour(self, v, level):
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
            p1 = (v[0] - v[1]) / (v[0][2] - v[1][2]) * (level - v[0][2]) + v[0]
            p2 = (v[0] - v[2]) / (v[0][2] - v[2][2]) * (level - v[0][2]) + v[0]
            valid = True
        elif v[1][2] < level and v[2][2] > level:
            p1 = (v[2] - v[0]) / (v[2][2] - v[0][2]) * (level - v[2][2]) + v[2]
            p2 = (v[2] - v[1]) / (v[2][2] - v[1][2]) * (level - v[2][2]) + v[2]
            valid = True
        elif v[0][2] == level and v[1][2] == level:
            p1 = v[0]
            p2 = v[1]
            valid = True
        elif v[1][2] == level and v[2][2] == level:
            p1 = v[1]
            p2 = v[2]
            valid = True

        return {'valid': valid, 'p1': p1, 'p2': p2}

    def PrepareContour(self, L, R):
        xmin, xmax = self.range['xmin'], self.range['xmax']
        ymin, ymax = self.range['ymin'], self.range['ymax']
        zmin, zmax = self.range['zmin'], self.range['zmax']
        # ugly, offset the contour to make it 'align' with mesh and quad
        ox = (xmax - xmin) / (self.dimension['x'] - 1) / 2
        oy = (ymax - ymin) / (self.dimension['y'] - 1) / 2
        oz = 0
        # calculate the contour levels uniformly, ignore endpoints (zmin, zmax)
        levels = np.linspace(zmin,
                             zmax,
                             self.contour_levels + 1,
                             endpoint=False)[1:]
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
            if i % my == my - 1 or i + my + 1 >= len(P):
                minmax.append([0, 0])
                continue
            t1 = min(P[i][2], P[i + 1][2], P[i + my][2], P[i + my + 1][2])
            t2 = max(P[i][2], P[i + 1][2], P[i + my][2], P[i + my + 1][2])
            minmax.append([t1, t2])

        for level in levels:
            clr = self.GetColorByZ(level)
            clr = [clr[0], clr[1], clr[2], 1]
            for i in range(L, R):
                if i % my == my - 1 or i + my + 1 >= len(P):
                    # the contour is calculate with for points
                    # i   i+my
                    # i+1 i+my+1
                    # so ignore the bottom and right edges
                    continue
                if minmax[i - L][0] > level or minmax[i - L][1] < level:
                    # the current level is not cross ith point and its 3
                    # neighbours, do nothing
                    continue
                if len(vertex) > 2**16 - 10:
                    # too many vertex, add to separate block to avoid overflow
                    # when rendering
                    vertex = (np.array(vertex) + [ox, oy, oz]).flatten()
                    colour = np.array(colour).flatten()
                    vertexAll.append(vertex)
                    mesh = np.array(mesh).flatten()
                    meshAll.append(mesh)
                    colourAll.append(colour)
                    vertex = []
                    mesh = []
                    colour = []
                v = np.array([P[i], P[i + 1], P[i + my + 1]])
                p = self.CalcContour(v, level)
                if p['valid']:
                    vertex.append(p['p1'])
                    mesh.append(len(vertex) - 1)
                    vertex.append(p['p2'])
                    mesh.append(len(vertex) - 1)
                    colour.append(clr)
                    colour.append(clr)

                v = np.array([P[i], P[i + my], P[i + my + 1]])
                p = self.CalcContour(v, level)
                if p['valid']:
                    vertex.append(p['p1'])
                    mesh.append(len(vertex) - 1)
                    vertex.append(p['p2'])
                    mesh.append(len(vertex) - 1)
                    colour.append(clr)
                    colour.append(clr)
        if vertex:
            vertex = (np.array(vertex) + [ox, oy, oz]).flatten()
            colour = np.array(colour).flatten()
            mesh = np.array(mesh).flatten()
            vertexAll.append(vertex)
            meshAll.append(mesh)
            colourAll.append(colour)
        return {'Vertices': vertexAll, 'Mesh': meshAll, 'Color': colourAll}

    def SetGLObjects(self, globjects):
        # set the globjects directly, see `PrepareGLObjects` for details about
        # accepted keys
        self.globjects = globjects
        # clear the raw_points, so it will not conflict with globjects
        self.raw_points = None
        self.Refresh()

    def PrepareGLObjects(self):
        colorAll = []
        vertexAll = []
        triangleAll = []
        meshAll = []
        contourAll = []
        my = self.dimension['y']
        mx = self.dimension['x']
        # initial guess of the number of blocks
        # subtract 'my', as last column in self.points is artificial (guard),
        # it is same as the 2nd last column (the last column of the actual
        # data). The last column is to simplify the following process, so no need
        # to worry about the boundary condition of the last column
        block = int(np.ceil((len(self.points) - my) * 4 / (2**16 - (my+1)*4)))
        # the index of each block, and make sure the size of each block is < 2**16
        while True:
            blocks = np.linspace(0, len(self.points)-my, block + 1, dtype=int)
            if np.any((blocks[1:] - blocks[:-1] + my + 1)*4 > 2**16):
                block +=1
                continue
            break

        self.blocks = blocks
        r = blocks[0]
        l = 0
        P = self.points
        for k in range(1, len(blocks)):
            l, r = r, blocks[k]
            acolor = np.repeat(self.pointsClr[np.arange(
                               l, min(r + my + 1, len(P)))],
                               4, axis=0)
            # for each point (x[i], y[j], z[i,j]), construct a quad
            # when show_step_surface is true
            # (x[i],   y[i],   z[i,j])
            # (x[i+1], y[i],   z[i,j])
            # (x[i+1], y[i+1], z[i,j])
            # (x[i],   y[i+1], z[i,j])
            # otherwise
            # (x[i],   y[i],   z[i,j])
            # (x[i+1], y[i],   z[i,j])
            # (x[i+1], y[i+1], z[i,j])
            # (x[i],   y[i+1], z[i,j])

            avertex = np.zeros(((min(r + my + 1, len(P)) - l) * 4, 3))
            # the index
            idx = np.arange(l, min(r + my + 1, len(P)))
            idx = np.matrix(idx).T
            idx = np.repeat(idx, 4, axis=1)
            # the corresponding row & col
            cc, rr = np.divmod(idx, my)
            #cc = (idx/my).astype(int)
            # the row for each 4 points in the same quad
            rr = rr + [0, 1, 1, 0]
            rr = rr.flatten().clip(0, my - 1)
            # the col for each 4 points in the same quad
            cc = cc + [0, 0, 1, 1]
            cc = cc.flatten().clip(0, mx - 1)
            # update the vertex of each quad
            avertex[:, 0] = P[(cc * my + rr), 0]
            avertex[:, 1] = P[(cc * my + rr), 1]
            if self.show_step_surface:
                # x, y
                avertex[:, 0:-1] = P[(cc * my + rr), 0:-1]
                # the quad use the same z value of (i, j)
                avertex[:, 2] = P[idx.flatten(), 2]
            else:
                avertex[:, :] = P[(cc * my + rr), :]

            # index for quad and mesh
            idx = np.arange(0, r - l)
            idx = np.matrix(idx).T
            idx = np.repeat(idx, 4, axis=1)
            COLS, ROWS = np.divmod(idx, my)
            idx = idx * 4

            # quad
            triangle = np.zeros(((r - l) * 3, 4))
            # the current quad
            # triangle.append([4*i, 4*i+1, 4*i+2, 4*i+3])
            triangle[0:(r - l), :] = idx + [0, 1, 2, 3]

            # connect to the quad right
            # triangle.append([4*i+3, 4*i+2, 4*(i+my)+1, 4*(i+my)])
            rr = ROWS  #+ [0, 0, 0, 0]
            #rr = rr.clip(0, my-1)
            cc = COLS + [0, 0, 1, 1]
            cc = cc.clip(0, mx - 1)
            triangle[(r - l):2 * (r - l), :] = (cc * my + rr) * 4 + [3, 2, 1, 0]

            # connect to the quad below
            # triangle.append([4*i+1, 4*(i+1), 4*(i+1)+3, 4*(i)+2])
            rr = ROWS + [0, 1, 1, 0]
            rr = rr.clip(0, my - 1)
            cc = COLS  #+ [0, 0, 0, 0]
            #cc = cc.clip(0, mx-1)
            triangle[2 * (r - l):3 * (r - l), :] = (cc * my + rr) * 4 + [1, 0, 3, 2]

            # mesh
            mesh = None
            if self.show['mesh']:
                mesh = np.zeros(((r - l) * 4, 4))
                # mesh in the same quad
                # mesh.append([4*i, 4*i+1, 4*i+1, 4*i+2, 4*i+2, 4*i+3, 4*i+3, 4*i])
                mesh[0 * (r - l):1 * (r - l)] = idx + [0, 1, 1, 2]
                mesh[1 * (r - l):2 * (r - l)] = idx + [2, 3, 3, 0]

                # connect to the quad right
                # mesh.append([4*i+3, 4*(i+my), 4*i+2, 4*(i+my)+1])
                rr = ROWS  #+ [0, 0, 0, 0]
                #rr = rr.clip(0, my-1)
                cc = COLS + [0, 1, 0, 1]
                cc = cc.clip(0, mx - 1)
                mesh[2 * (r - l):3 * (r - l), :] = (cc * my + rr) * 4 + [3, 0, 2, 1]

                # connect to the quad below
                # mesh.append([4*i+1, 4*(i+1), 4*i+2, 4*(i+1)+3])
                rr = ROWS + [0, 1, 0, 1]
                rr = rr.clip(0, my - 1)
                cc = COLS  #+ [0, 0, 0, 0]
                #cc = cc.clip(0, mx-1)
                mesh[3 * (r - l):4 * (r - l), :] = (cc * my + rr) * 4 + [1, 0, 2, 3]

                mesh = mesh.flatten()
            contour = None
            if self.show['contour']:
                contour = self.PrepareContour(l, r)
            avertex = avertex.flatten()
            acolor = acolor.flatten()
            triangle = triangle.flatten()
            vertexAll.append(avertex)
            colorAll.append(acolor)
            triangleAll.append(triangle)
            meshAll.append(mesh)
            contourAll.append(contour)
        return {
            'block': block,
            'Color': colorAll,
            'Vertices': vertexAll,
            'Triangles': triangleAll,
            'Mesh': meshAll,
            'Contour': contourAll
        }

    def SetGLBuffer(self, v, c):
        # vertex buffer
        # Bind it as The Current Buffer'
        glBindBuffer(GL_ARRAY_BUFFER, self.glBufs[0])
        # Fill it With the Data
        glBufferData(GL_ARRAY_BUFFER, v.astype(np.float32), GL_STATIC_DRAW)
        glVertexAttribPointer(self.VertexPosition, 3, GL_FLOAT, GL_FALSE, 0,
                              None)

        # color buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.glBufs[1])
        glBufferData(GL_ARRAY_BUFFER, c.astype(np.float32), GL_STATIC_DRAW)
        glVertexAttribPointer(self.VertexColor, 4, GL_FLOAT, GL_FALSE, 0, None)

    def DrawElement(self, t, v, m, o=0):
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.glBufs[2])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, v.astype(np.uint16),
                     GL_STATIC_DRAW)
        glUniform1i(glGetUniformLocation(self.ShaderProgram, "mesh"), m)
        glUniform1i(glGetUniformLocation(self.ShaderProgram, "VertexOriginal"),
                    o)
        glDrawElements(t, v.size, GL_UNSIGNED_SHORT, None)

    def DrawAxis(self):
        r = self.range
        g = 32 / self.scale['base']
        rng = max(r['xmax'] - r['xmin'], r['ymax'] - r['ymin'])
        rng_min = min(r['xmax'] - r['xmin'], r['ymax'] - r['ymin'])
        if rng <= 0:
            return
        gy = g / rng * rng_min
        gx = gy
        gz = gx / rng * (r['zmax'] - r['zmin'])

        p = [r['xmax'], r['ymax'], r['zmax']]
        vertex = np.array([
            p[0] - gx, p[1], p[2], p[0] - gx, p[1] - gy, p[2], p[0], p[1] - gy,
            p[2], p[0], p[1], p[2], p[0] - gx, p[1], p[2], p[0], p[1], p[2],
            p[0] - gx / 3, p[1] + gy / 5, p[2], p[0] - gx / 3, p[1] - gy / 5,
            p[2]
        ])
        self.DrawAxisHelp('x', vertex)

        p = [r['xmin'], r['ymax'], r['zmax']]
        vertex = np.array([
            p[0], p[1], p[2], p[0], p[1] - gy, p[2], p[0] + gx, p[1] - gy,
            p[2], p[0] + gx, p[1], p[2], p[0], p[1] - gy, p[2], p[0], p[1],
            p[2], p[0] + gx / 5, p[1] - gy / 3, p[2], p[0] - gx / 5,
            p[1] - gy / 3, p[2]
        ])
        self.DrawAxisHelp('y', vertex)

        p = [r['xmin'], r['ymin'], r['zmax']]
        vertex = np.array([
            p[0], p[1] + gy, p[2] - gz, p[0], p[1], p[2] - gz, p[0], p[1],
            p[2], p[0], p[1] + gy, p[2], p[0], p[1], p[2] - gz, p[0], p[1],
            p[2], p[0], p[1] + gy / 5, p[2] - gz / 3, p[0], p[1] - gy / 5,
            p[2] - gz / 3
        ])
        self.DrawAxisHelp('z', vertex)

    def DrawAxisHelp(self, letter, vertex):
        scale = self.GetContentScaleFactor()*self.scale['base']
        W, H = int(32*scale), int(32*scale)

        mybuffer = self.DrawTextToBuffer(letter, (W, H), self.axistext_font,
                                         self.axistext_clr, align_center=True,
                                         scale=2*self.scale['base'])
        texture = self.glTextures[0]
        glPixelStorei(GL_UNPACK_ALIGNMENT, GL_TRUE)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, W, H, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, mybuffer.flatten())
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                        GL_LINEAR_MIPMAP_NEAREST)
        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture)

        color = np.array([
            [0, 1, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            self.axis_clr,
            self.axis_clr,
            self.axis_clr,
            self.axis_clr,
        ]).flatten()

        triangle = np.array([0, 1, 2, 3])
        mesh = np.array([4, 5, 5, 6, 5, 7])
        glUniform1i(glGetUniformLocation(self.ShaderProgram, "uSampler"), 0)
        self.SetGLBuffer(vertex, color)
        self.DrawElement(GL_QUADS, triangle, 3)
        self.DrawElement(GL_LINES, mesh, 0)

    def Draw3dImg(self):
        if self.show['surface'] or self.show['mesh'] or self.show['contour']:
            if not self.globjects:
                self.globjects = self.PrepareGLObjects()
            obj = self.globjects
            block = obj.get('block', 0)
            glLineWidth(0.5)
            for b in range(block):
                self.SetGLBuffer(obj['Vertices'][b], obj['Color'][b])
                clr_mesh = 0
                if self.show['surface'] and 'Triangles' in obj:
                    # if surface is on, show the mesh in blue
                    clr_mesh = 1
                    self.DrawElement(GL_QUADS, obj['Triangles'][b], 0)

                if self.show['mesh'] and 'Mesh' in obj:
                    self.DrawElement(GL_LINES, obj['Mesh'][b], clr_mesh, 3)

                if self.show['contour'] and 'Contour' in obj:
                    ctr = obj['Contour'][b]
                    for c in range(len(ctr['Vertices'])):
                        self.SetGLBuffer(ctr['Vertices'][c], ctr['Color'][c])
                        self.DrawElement(GL_LINES, ctr['Mesh'][c], 0)

    def Draw2dImg(self):
        # Set Transformation Matrices, no rotation allowed in 2d mode
        TransformMatrix = np.eye(3, dtype=np.float32).flatten('F')
        tmatrix = glGetUniformLocation(self.ShaderProgram,
                                       "TransformationMatrix")
        glUniformMatrix3fv(tmatrix, 1, GL_FALSE, TransformMatrix)

        W = self.dimension['x']
        H = self.dimension['y']
        if W <= 0 or H <= 0:
            return
        #if self.img_texture is None:
        glPixelStorei(GL_UNPACK_ALIGNMENT, GL_TRUE)
        glBindTexture(GL_TEXTURE_2D, self.glTextures[2])
        # the pointsClr is col-wise (i.e., 1st col, 2nd col...)
        mybuffer = self.pointsClr.astype(np.float32).flatten()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, H, W, 0, GL_RGBA, GL_FLOAT,
                     mybuffer)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glGenerateMipmap(GL_TEXTURE_2D)

        glBindTexture(GL_TEXTURE_2D, 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.glTextures[2])

        xmax, xmin = self.range['xmax'], self.range['xmin']
        ymax, ymin = self.range['ymax'], self.range['ymin']
        zmin = self.range['zmin']
        vertex = np.array([
            xmin, ymax, zmin, xmin, ymin, zmin, xmax, ymin, zmin, xmax, ymax,
            zmin
        ])
        # texture coordinate
        color = np.array([
            1 - 1. / H, 0, 0, 1, 0, 0, 0, 1, 0, 1 - 1. / W, 0, 1, 1 - 1. / H,
            1 - 1. / W, 0, 1
        ])
        triangle = np.array([0, 1, 2, 3])
        glUniform1i(glGetUniformLocation(self.ShaderProgram, "uSampler"), 0)
        self.SetGLBuffer(vertex, color)
        self.DrawElement(GL_QUADS, triangle, 3)

    def DrawBox(self):
        glLineWidth(2)
        if self.show['box']:
            r = self.range
            vertex = np.array([
                r['xmin'], r['ymin'], r['zmin'], r['xmin'], r['ymax'],
                r['zmin'], r['xmax'], r['ymax'], r['zmin'], r['xmax'],
                r['ymin'], r['zmin'], r['xmin'], r['ymin'], r['zmax'],
                r['xmin'], r['ymax'], r['zmax'], r['xmax'], r['ymax'],
                r['zmax'], r['xmax'], r['ymin'], r['zmax']
            ])
            color = np.tile(self.box_clr, 8)
            axis = np.array([
                0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2,
                6, 3, 7
            ])
            self.SetGLBuffer(vertex, color)
            self.DrawElement(GL_LINES, axis, 0)

        if self.show['axis']:
            self.DrawAxis()

    def DrawBar(self):
        glLineWidth(2.0)
        if self.show['horz_bar']:
            y = self.selected['y']
            if y < 0 or y >= self.raw_points['z'].shape[0]:
                # out of range
                return
            d = (self.raw_points['z'][y, :] - self.range['zmin']) * self.data_zscale_bar
            self.horzbar_buf.SetData(d)
            v, c, l = self.horzbar_buf.GetGLObject()
            self.SetGLBuffer(v, c)
            self.DrawElement(GL_LINE_STRIP, l, 0)

        if self.show['vert_bar']:
            x = self.selected['x']
            if x < 0 or x >= self.raw_points['z'].shape[1]:
                # out of range
                return
            d = (self.raw_points['z'][:, x] - self.range['zmin']) * self.data_zscale_bar
            self.vertbar_buf.SetData(d)
            v, c, l = self.vertbar_buf.GetGLObject()

            self.SetGLBuffer(v, c)
            self.DrawElement(GL_LINE_STRIP, l, 0)

    def Draw(self):
        glClearColor(*self.background_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #Setup shader
        vertexResolution = glGetUniformLocation(self.ShaderProgram,
                                                "VertexResolution")
        glUniform3f(vertexResolution, self.W, self.H, self.W)

        vertexOffset = glGetUniformLocation(self.ShaderProgram, "VertexOffset")
        ox = self.offset['base']['x'] + self.offset['user']['x']
        oy = self.offset['base']['y'] + self.offset['user']['y']
        glUniform3f(vertexOffset, ox, oy, self.offset['base']['x'])

        vertexOffsetData = glGetUniformLocation(self.ShaderProgram,
                                                "VertexOffsetData")
        glUniform3f(vertexOffsetData, self.data_offset['x'],
                    self.data_offset['y'], self.data_offset['z'])

        #Set Transformation Matrices
        TransformMatrix = self.rotate_matrix.flatten('F')
        tmatrix = glGetUniformLocation(self.ShaderProgram,
                                       "TransformationMatrix")
        glUniformMatrix3fv(tmatrix, 1, GL_FALSE, TransformMatrix)

        vertexScaleGL = glGetUniformLocation(self.ShaderProgram,
                                             "VertexScaleGL")
        glUniform3f(vertexScaleGL, 1, -1, -1)
        vertexScaleData = glGetUniformLocation(self.ShaderProgram,
                                               "VertexScaleData")
        glUniform3f(vertexScaleData, 1, 1, self.data_zscale)

        vertexScale = glGetUniformLocation(self.ShaderProgram, "VertexScale")
        scaleX = self.scale['base'] * self.scale['zoom_x']
        scaleY = self.scale['base'] * self.scale['zoom_y']
        scaleZ = self.scale['base']
        glUniform3f(vertexScale, scaleX, scaleY, scaleZ)

        if self.mode_2d:
            self.Draw2dImg()
        else:
            self.Draw3dImg()
            self.DrawBox()
        self.DrawBar()
        self.DrawHud()

    def DrawTextToBuffer(self, text, sz, font, clr, align_center=False, pos=(5,5), scale=1):

        tdc = wx.MemoryDC()
        bitmap = wx.Bitmap(sz[0], sz[1], depth=32)
        scale = self.GetContentScaleFactor() * scale
        bitmap.SetScaleFactor(scale)
        tdc.SelectObject(bitmap)
        gc = wx.GraphicsContext.Create(tdc)
        if wx.Platform == '__WXMSW__':
            gc.SetBrush(wx.Brush(np.array(self.background_color)*255))
            gc.DrawRectangle(0, 0, sz[0], sz[1])
        gc.SetFont(font, clr)

        if align_center:
            tw, th = gc.GetTextExtent(text)
            if wx.Platform == '__WXMSW__':
                # in windows, tw, th is after scale
                gc.DrawText(text, (sz[0]-tw)//2, (sz[1]-th)//2)
            else:
                gc.DrawText(text, int(sz[0]/scale-tw)//2, int(sz[1]/scale-th)//2)
        else:
            gc.DrawText(text, pos[0], pos[0])

        tdc.SelectObject(wx.NullBitmap)
        buffer = np.zeros((sz[0], sz[1], 4), np.uint8)
        bitmap.CopyToBuffer(buffer, wx.BitmapBufferFormat_RGBA)
        if wx.Platform == '__WXMSW__':
            # remove black background in windows by setting the corresponding alpha channel to 0
            bk = ((buffer[:, :, 0] == self.background_color[0]*255) &
                  (buffer[:, :, 1] == self.background_color[1]*255) &
                  (buffer[:, :, 2] == self.background_color[2]*255) &
                  (buffer[:, :, 3] == self.background_color[3]*255))
            buffer[bk, -1] = 0

        return buffer

    def DrawHud(self):
        if not self.show['hudtext'] or (not self._hudtext and not self.hudtext):
            return
        scale = self.GetContentScaleFactor()
        W, H = int(self.W*scale), int(self.H*scale)
        if W <= 0 or H <= 0:
            return
        HudW, HudH = self._hudBuffer.shape[0:2]
        if self._hudtext != self.hudtext or HudW < W:
            self._hudtext = self.hudtext
            letter = self.hudtext
            HudW, HudH = W, int(38*scale)
            self._hudBuffer = self.DrawTextToBuffer(letter, (HudW, HudH),
                                                    self.hudtext_font, self.hudtext_clr)

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

        vertex = np.array([
            -1, 1, -1, -1, 1 - 2. * HudH / H, -1, -1 + 2. * HudW / W,
            1 - 2. * HudH / H, -1, -1 + 2. * HudW / W, 1, -1
        ])
        # texture coordinate
        color = np.array([0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1])
        triangle = np.array([0, 1, 2, 3])
        glUniform1i(glGetUniformLocation(self.ShaderProgram, "uSampler"), 0)
        self.SetGLBuffer(vertex, color)
        self.DrawElement(GL_QUADS, triangle, 3, 1)

    def Rotate(self, x, y, z):
        if x == 0 and y == 0 and z == 0:
            return
        R = self.rotate_matrix
        if x != 0:
            Rx = self.GetRotateMatrixX(x)
            R = np.matmul(Rx, R)
        if y != 0:
            Ry = self.GetRotateMatrixY(y)
            R = np.matmul(Ry, R)
        if z != 0:
            Rz = self.GetRotateMatrixZ(z)
            R = np.matmul(Rz, R)
        self.rotation = {'x': 0, 'y': 0, 'z': 0}
        self.rotate_matrix = R
        self.Refresh()

    def SetRotateDelta(self, delta):
        self.rotate_delta = delta

    def GetRotateDelta(self):
        return self.rotate_delta

    def GetRotateMatrixX(self, sign):
        Rx = np.zeros((3, 3), np.float32)
        theta = sign * self.rotate_delta
        Rx[0][0] = 1
        Rx[1][1] = math.cos(theta)
        Rx[1][2] = -math.sin(theta)
        Rx[2][1] = math.sin(theta)
        Rx[2][2] = math.cos(theta)
        return Rx

    def GetRotateMatrixY(self, sign):
        Ry = np.zeros((3, 3), np.float32)
        theta = sign * self.rotate_delta
        Ry[0][0] = math.cos(theta)
        Ry[0][2] = math.sin(theta)
        Ry[1][1] = 1
        Ry[2][0] = -math.sin(theta)
        Ry[2][2] = math.cos(theta)
        return Ry

    def GetRotateMatrixZ(self, sign):
        Rz = np.zeros((3, 3), np.float32)
        theta = sign * self.rotate_delta
        Rz[0][0] = math.cos(theta)
        Rz[0][1] = -math.sin(theta)
        Rz[1][0] = math.sin(theta)
        Rz[1][1] = math.cos(theta)
        Rz[2][2] = 1
        return Rz


class SelectedPixelBuf(object):
    def __init__(self, sz, rng):
        self.idx = 0
        self.range = dict(rng)
        self.line_color = [1., 1., 1., 1.]
        self.sel_color = [1, 0, 0, 1]
        self.sel_pos = 0
        self.buf_size = -1
        self.color = None
        self.vertex = None
        self.Resize(sz)

    def SetRange(self, rng):
        if self.range == rng:
            return
        self.range.update(rng)
        self.InitVertex()

    def InitVertex(self):
        # make z value larger than zmax, so it shows on top of the surface.
        self.vertex = np.ones((self.buf_size, 3)) * self.range['zmax']

    def SetColor(self, clr, clr_sel=None):
        if self.line_color == clr and self.sel_color == clr_sel:
            return
        self.line_color = clr[:]
        self.sel_color = clr_sel[:]
        if self.buf_size > 0:
            self.color = np.repeat(np.array([clr]), self.buf_size,
                                   axis=0).flatten()
            self.SetSelectedPos(self.sel_pos)

    def Resize(self, sz):
        self.buf_size = sz
        self.buf = np.zeros(sz)
        self.InitVertex()
        self.color = np.repeat(np.array([self.line_color]), sz,
                               axis=0).flatten()
        self.line = np.arange(sz)
        self.idx = 0

    def GetGLObject(self, l, r, t, b, o, g):
        self.vertex[:, 0] = np.linspace(l, r, self.buf.size, endpoint=False)
        self.vertex[:, 1] = b - (np.roll(self.buf, -(self.idx + 1)) - o) * g
        return self.vertex.flatten(), self.color, self.line

    def SetSelectedPos(self, pos):
        # pos is within [-self.buf.size+1, 0]
        def _set_pos_clr(p, clr):
            p = (self.buf.size - 1 + p) % self.buf.size
            if p >=0 and p < self.buf_size:
                self.color[p * 4:(p + 1) * 4] = clr

        if pos != self.sel_pos:
            _set_pos_clr(self.sel_pos, self.line_color)
        self.sel_pos = pos

        _set_pos_clr(pos, self.sel_color)

    def SetData(self, d, idx):
        self.idx = idx
        self.buf[idx] = d


class TrackingSurface(SurfaceBase):
    DISPLAY_ORIGINAL = 0
    DISPLAY_MAX = 1
    DISPLAY_MIN = 2
    DISPLAY_MINMAX = 3
    ID_SIM_CLEAR = wx.NewIdRef()
    ID_DISP_ORIGINAL = wx.NewIdRef()
    ID_DISP_MAX = wx.NewIdRef()
    ID_DISP_MIN = wx.NewIdRef()
    ID_DISP_MINMAX = wx.NewIdRef()
    ID_AUTO_SCALE = wx.NewIdRef()
    ID_SHOW_SELECTED_BUF = wx.NewIdRef()

    def __init__(self, parent, points=None, buf_len=256):
        super().__init__(parent, points)
        self.buf_len = buf_len
        self.frames = None
        self.frames_idx = 0
        self.display_mode = self.DISPLAY_ORIGINAL
        self.frame_display = None
        self.selected_buf = None
        self.SetBufLen(self.buf_len)

        self.show['selected_buf'] = True

    def GetAccelList(self):
        accel_tbl = super().GetAccelList()
        accel_tbl += [(wx.ACCEL_CTRL, ord('K'), self.ID_SIM_CLEAR),
                      (wx.ACCEL_SHIFT, ord('M'), self.ID_DISP_MAX),
                      (wx.ACCEL_SHIFT, ord('O'), self.ID_DISP_ORIGINAL),
                      (wx.ACCEL_SHIFT, ord('N'), self.ID_DISP_MIN),
                      (wx.ACCEL_SHIFT, ord('R'), self.ID_DISP_MINMAX),
                      (wx.ACCEL_SHIFT|wx.ACCEL_CTRL, ord('A'), self.ID_AUTO_SCALE),
                      ]
        return accel_tbl

    def SetBufLen(self, sz):
        self.buf_len = sz
        self.frames = None
        self.frame_display = None
        self.frames_idx = 0
        self.selected_buf = SelectedPixelBuf(self.buf_len, self.range)

    def GetBufLen(self):
        return self.buf_len

    def SetRange(self, rng):
        super().SetRange(rng)
        if self.selected_buf:
            self.selected_buf.SetRange(rng)

    def UpdateHudText(self):
        super().UpdateHudText()
        if self.frames is None or not self.frames.size:
            return
        _, rows, cols = self.frames.shape
        r = self.selected['y']
        c = self.selected['x']
        if r >= 0 and c >= 0 and r < rows and c < cols:
            d = self.frames[:, r, c]
            m = np.min(d)
            M = np.max(d)
            mn = np.mean(d)
            std = np.std(d)
            d = self.raw_points['z'][r, c]
            self.SetHudText(
                '(%d, %d) %f min: %f max: %f mean: %.2f std: %.2f' %
                (c, r, d, m, M, mn, std))

    def GetFrame(self, num):
        # get the num th frame,
        #   0: the latest frame
        #  -1: the 2nd latest frame
        #  ...
        if self.frames is None:
            return None
        idx = (self.frames_idx - 1 + num) % self.buf_len
        return self.frames[idx, :, :]

    def SetCurrentFrame(self, num):
        # show the num th frame,
        #   0: the latest frame
        #  -1: the 2nd latest frame
        #  ...
        if self.frames is None:
            return
        self.UpdateImage(self.GetFrame(num))
        self.selected_buf.SetSelectedPos(num)

    def SetFrames(self, frames, reset_buf_len=False, time_axis=0, silent=True):
        self.Clear()
        if len(frames.shape) == 2:
            if reset_buf_len:
                self.SetBufLen(1)
            self.NewFrameArrive(frames, silent=silent)
        elif len(frames.shape) == 3:
            frames = np.moveaxis(frames, time_axis, 0)
            shape = list(frames.shape)

            if reset_buf_len:
                self.SetBufLen(shape[0])
            for f in range(shape[0]):
                self.NewFrameArrive(frames[f, :, :], silent=silent)

    def NewFrameArrive(self, frame, silent=True):

        rows, cols = frame.shape
        if self.frames is None or rows != self.frames.shape[1] \
                or cols != self.frames.shape[2]:
            self.frames = np.zeros((self.buf_len, rows, cols))
            self.frames_idx = 0
            if self.raw_points and 'z' in self.raw_points and self.raw_points['z'].shape == frame.shape:
                self.frames[self.frames_idx, :, :] = self.raw_points['z']
                self.frames_idx += 1
                self.frames_idx %= self.buf_len

            self.SetImage({'z': frame})


        # all the following code is time sensitive, needs to finish as soon
        # as possible
        if self.frame_display is not None:
            if self.display_mode == self.DISPLAY_MAX:
                frame = np.maximum(frame, self.frame_display)
            elif self.display_mode == self.DISPLAY_MIN:
                frame = np.minimum(frame, self.frame_display)
            elif self.display_mode == self.DISPLAY_MINMAX:
                frame = np.where(
                    np.absolute(frame) > np.absolute(self.frame_display),
                    frame, self.frame_display)
        self.frame_display = frame

        if self.frames.size:
            self.frames[self.frames_idx, :, :] = frame
            r, c = self.selected['y'], self.selected['x']
            if r >= 0 and c >= 0 and r < rows and c < cols:
                self.selected_buf.SetData(frame[r, c], self.frames_idx)
            else:
                self.SetSelected({'x': 0, 'y': 0})

        if not silent:
            # update the data
            self.UpdateImage(frame)

        self.frames_idx += 1
        self.frames_idx %= self.buf_len

    def Clear(self):
        self.frame_display = None
        self.frames = None
        super().Clear()

    def Draw(self):
        super().Draw()
        if not self.frames is None:
            self.DrawSelectedBuf()

    def DrawSelectedBuf(self):
        if self.frames is None or not self.show.get('selected_buf', False):
            return
        xmax, xmin = self.range['xmax'], self.range['xmin']
        ymax, ymin = self.range['ymax'], self.range['ymin']
        zmax, zmin = self.range['zmax'], self.range['zmin']
        zrange = zmax - zmin

        if self.default_rotate in [90, 270]:
            o = self.data_offset['y'] - self.data_offset['x']
            xmax, xmin, ymax, ymin = ymax - o, ymin - o, xmax, xmin
        if zrange == 0:
            g = 1
        else:
            g = (ymax-ymin)/(zmax-zmin)
        v, c, l = self.selected_buf.GetGLObject(
            xmin, xmax, ymin, ymax,
            zmin,
            g)
        if len(l) >= 2**16 - 10:
            # if the buffer length is larger, draw line with multiple sections
            for s in range(0, len(l), 2**16 - 10):
                e = min(s + 2**16 - 10 + 1, len(l))
                self.SetGLBuffer(v[s * 3:e * 3], c[s * 4:e * 4])
                self.DrawElement(GL_LINE_STRIP, l[s:e] - l[s], 0, 2)
        else:
            self.SetGLBuffer(v, c)
            self.DrawElement(GL_LINE_STRIP, l, 0, 2)

    def BuildElementMenu(self):
        menu = super().BuildElementMenu()
        menu.AppendSeparator()
        menu.Append(self.ID_SHOW_SELECTED_BUF, 'Show line of selected pixel', '', wx.ITEM_CHECK)

        return menu

    def GetContextMenu(self):
        menu = super().GetContextMenu()
        if not menu:
            menu = wx.Menu()
            menu.Append(self.ID_AUTO_SCALE, 'Auto scale\tshift+ctrl+A')
            menu.AppendSeparator()
        else:
            menu.Insert(0, self.ID_AUTO_SCALE, 'Auto scale\tshift+ctrl+A')
            menu.InsertSeparator(1)
            menu.AppendSeparator()
        display = wx.Menu()
        display.Append(self.ID_DISP_ORIGINAL, 'Original\tshift+o', '',
                       wx.ITEM_CHECK)
        display.Append(self.ID_DISP_MAX, 'Max\tshift+m', '', wx.ITEM_CHECK)
        display.Append(self.ID_DISP_MIN, 'Min\tshift+n', '', wx.ITEM_CHECK)
        display.Append(self.ID_DISP_MINMAX, 'Min-Max\tshift+r', '',
                       wx.ITEM_CHECK)
        menu.AppendSubMenu(display, 'Display')
        return menu

    def OnUpdateMenu(self, event):
        eid = event.GetId()
        if eid == self.ID_AUTO_SCALE:
            event.Enable(self.raw_points is not None)
        elif eid == self.ID_DISP_ORIGINAL:
            event.Check(self.display_mode == self.DISPLAY_ORIGINAL)
        elif eid == self.ID_DISP_MAX:
            event.Check(self.display_mode == self.DISPLAY_MAX)
        elif eid == self.ID_DISP_MIN:
            event.Check(self.display_mode == self.DISPLAY_MIN)
        elif eid == self.ID_DISP_MINMAX:
            event.Check(self.display_mode == self.DISPLAY_MINMAX)
        elif eid == self.ID_SHOW_SELECTED_BUF:
            event.Check(self.show['selected_buf'])
        else:
            super().OnUpdateMenu(event)

    def OnProcessMenuEvent(self, event):
        eid = event.GetId()
        if eid == self.ID_SIM_CLEAR:
            self.Clear()
        elif eid == self.ID_DISP_ORIGINAL:
            self.display_mode = self.DISPLAY_ORIGINAL
        elif eid == self.ID_DISP_MAX:
            self.display_mode = self.DISPLAY_MAX
        elif eid == self.ID_DISP_MIN:
            self.display_mode = self.DISPLAY_MIN
        elif eid == self.ID_DISP_MINMAX:
            self.display_mode = self.DISPLAY_MINMAX
        elif eid == self.ID_AUTO_SCALE:
            self.SetImage(self.raw_points)
        elif eid == self.ID_SHOW_SELECTED_BUF:
            self.SetShowMode(selected_buf=not self.show['selected_buf'])
        else:
            super().OnProcessMenuEvent(event)
