# Render

`modules/render` holds the layers that draw to the screen. An app builds its picture from them.

## What it holds

Three categories, each a folder. A layer is found by what it draws, not by its technical kind.

| Category   | Holds                                                               | Packages                                                     |
|------------|---------------------------------------------------------------------|--------------------------------------------------------------|
| `modules/` | the layers that draw a module's data                                | `pose/`, `camera/`, `tracker/`, `inference/`, `flow/`, `board/` |
| `general/` | the layers that draw no module's data and are useful to any app     | `composite/`, `grade/`                                       |
| `shared/`  | the layers that are not general and that more than one app uses     | none yet: every layer two apps use draws a module's data or is general |

A layer that is not general and that one app uses sits in that app, in `apps/<app>/render/layers/`. It moves to
`shared/` when a second app takes it.

Beside the categories, the package's root holds what every layer uses: `LayerBase` and `HasCaptureTime`
(`layer_base.py`), `DataCache`, `ColorSettings`, and the window's layout (`subdivision.py`).

## Packages

A category holds packages. A package is named after its subject: a module, a kind of work, a shared thing. It holds
its layers and, in `shaders/`, the shaders of those layers, so a shader sits with the layer that uses it.

```
modules/render/
  modules/   pose/  camera/  tracker/ (with panoramic/)  inference/  flow/  board/
  general/   composite/  grade/
```

A shader useful to any app (a mask operation, a blend, a tint) sits in `modules/gl/shaders/` ([GL.md](../gl/GL.md));
an app's own shader sits in that app.

`modules/render` is the package: apps import it (`from modules import render`) and use its `__init__.py`'s names
(`render.PoseCompositor`); the categories order it inside. No other module imports it (`tests/test_layering.py`).

## Kinds of layer

| Kind         | Owns              | Draws                     |
|--------------|-------------------|---------------------------|
| `Layer`      | its FBO or texture | into its own             |
| `Renderer`   | no texture        | into a target it is given |
| `Compositor` | an FBO            | other layers into it      |

A layer hands out its Fbo or SwapFbo, not the texture inside it.

## Names

- A layer's class: what it draws, then its kind (`TrackletRenderer`, `PoseCompositor`).
- A shader: the operation it does, without a `Shader` ending; the `shaders` package in front says what it is
  (`shaders.PosePointLines`).
