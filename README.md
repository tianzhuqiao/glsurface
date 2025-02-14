
# glsurface
**glsurface** is a python class to show 2d/3d data with [wxPython](https://wxpython.org/) and [PyOpenGL](https://pyopengl.sourceforge.net/).

## Installation
```shell
$ pip install glsurface
```

## Usage
1. Create a class from `TrackingSurface`
    ```python
    class Surface(TrackingSurface):
        def __init__(self, *args, **kwargs):
        TrackingSurface.__init__(self, *args, **kwargs)
        ...
    ```
2. Create an instance, and set data (optional)
    ```python
    class SurfacePanel(wx.Panel):

        def __init__(self, parent):
            wx.Panel.__init__(self, parent, -1)
            ...
            self.x = np.linspace(0, 2 * np.pi, 30).reshape((1, 30))
            z = np.cos(self.x).T * np.sin(self.x)
            self.canvas = Surface(self, {'z': z})
            ...

    ```
3. Update the data
    1. Add new frame data to the current frame buffer
        ```python
        self.canvas.NewFrameArrive(z, silent=False)
        ```
    2. Or reset the frame buffer with new data
        ```python
        self.canvas.SetFrames(points, reset_buf_len=True, silent=False)
        ```
Check `gltest.py` for details.

## Known issues
1. On Linux, it may show the following error.
    ```bash
    OpenGL.error.Error: Attempt to retrieve context when no valid context
    ```
    In this case, try different OpenGL platform, e.g.,
    ```bash
    $ export PYOPENGL_PLATFORM='egl'
    ```
    All supported platforms can be found here `/path/to/python/site-packages/OpenGL/__init__.py`

2. For EGL Surface, you may need to call `CreateSurface`, as mentioned [here](https://docs.wxpython.org/wx.glcanvas.GLCanvas.html#wx.glcanvas.GLCanvas.CreateSurface)

<img src="https://github.com/tianzhuqiao/glsurface/blob/master/images/demo.gif?raw=true" width="400"></img>
