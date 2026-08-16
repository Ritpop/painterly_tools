bl_info = {
    "name": "Painterly PBR Generator",
    "author": "ritpop",
    "version": (1, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D / Image Editor > Sidebar > Painterly PBR",
    "description": "GPU-accelerated painterly texture baking engine. Generates directional brush stroke diffuse, impasto normal maps, and cavity/crevice maps with Kuwahara filtering.",
    "category": "Material",
}

import bpy
from . import properties, operators, ui

modules = [
    properties,
    operators,
    ui,
]

def register():
    for mod in modules:
        mod.register()

def unregister():
    for mod in reversed(modules):
        mod.unregister()

if __name__ == "__main__":
    register()