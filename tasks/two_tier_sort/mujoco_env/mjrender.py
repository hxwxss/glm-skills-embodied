# -*- coding: utf-8 -*-
"""
mjrender.py -- shared offscreen-render helper
==============================================

All demo/observation rendering goes through mujoco.Renderer bound to the
LIVE env model (robosuite's own offscreen renderer keeps a stale model
after hard_reset -- see pitfalls). Collision hulls (geom group 0) are
hidden, mirroring what robosuite's viewer does.
"""

import mujoco


def make_renderer(env, height=480, width=640):
    model = env.sim.model._model
    # robosuite bakes a 640x480 offscreen framebuffer into the arena XML;
    # grow it to fit the requested render size (larger of the two)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    return mujoco.Renderer(model, height=height, width=width)


def camera_fixed(env, name):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(env.sim.model._model,
                                       mujoco.mjtObj.mjOBJ_CAMERA, name)
    return cam


def render_snapshot(renderer, env, cam):
    """Render one RGB frame from a prepared camera, collision hulls hidden."""
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 0          # group 0 = robosuite collision hulls
    renderer.update_scene(data=env.sim.data._data, camera=cam,
                          scene_option=opt)
    return renderer.render()      # RGB
