import bpy
from .operators import PAINTERLY_OT_BakePainterlyPBR

def draw_main(layout, context):
    settings = context.scene.painterly_settings

    box = layout.box()
    box.label(text="Input Textures:", icon='IMAGE_DATA')
    col = box.column(align=True)
    col.label(text="Albedo (Base Color):")
    col.template_ID(settings, "target_albedo", open="image.open", new="image.new")
    col.label(text="Normal Map (Optional):")
    col.template_ID(settings, "target_normal", open="image.open", new="image.new")

    layout.separator()
    row = layout.row()
    row.scale_y = 1.5
    row.operator(PAINTERLY_OT_BakePainterlyPBR.bl_idname, icon='MATERIAL')


def draw_brush(layout, context):
    s = context.scene.painterly_settings
    col = layout.column()
    col.prop(s, "stroke_scale")
    col.prop(s, "stroke_density")
    col.prop(s, "color_jitter")
    col.prop(s, "stroke_length")
    col.prop(s, "stroke_curvature")
    col.prop(s, "flow_smoothing")
    col.prop(s, "edge_influence")
    col.prop(s, "edge_blur")
    col.prop(s, "edge_threshold")
    col.prop(s, "edge_contrast")
    col.prop(s, "flow_scale_blend")
    col.prop(s, "cell_jitter")
    col.prop(s, "random_seed")


def draw_underpaint(layout, context):
    s = context.scene.painterly_settings
    col = layout.column()
    col.prop(s, "use_kuwahara_underpaint")
    if s.use_kuwahara_underpaint:
        col.prop(s, "kuw_radius")


def draw_normal(layout, context):
    s = context.scene.painterly_settings
    col = layout.column()
    col.prop(s, "stroke_tilt")
    col.prop(s, "normal_strength")
    col.prop(s, "facet_hardness")
    col.prop(s, "bristle_detail")


def draw_detail(layout, context):
    s = context.scene.painterly_settings
    col = layout.column()
    col.prop(s, "generate_detail_layer")
    if s.generate_detail_layer:
        col.prop(s, "detail_scale")
        col.prop(s, "detail_opacity")
        col.prop(s, "detail_edge_influence")
        col.prop(s, "detail_density_from_edges")
        if s.detail_density_from_edges:
            col.prop(s, "detail_edge_density_threshold")
        else:
            col.prop(s, "detail_density")


def draw_export(layout, context):
    s = context.scene.painterly_settings
    col = layout.column()
    col.prop(s, "assign_material")
    col.prop(s, "generate_diffuse")
    col.prop(s, "debug_export_textures")


# --- View3D Panels ---
class PAINTERLY_PT_View3D(bpy.types.Panel):
    bl_label = "Painterly PBR"
    bl_idname = "PAINTERLY_PT_view3d"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Painterly PBR'

    def draw(self, context):
        draw_main(self.layout, context)


class PAINTERLY_PT_View3DBrush(bpy.types.Panel):
    bl_label = "Brush Stroke Settings"
    bl_parent_id = "PAINTERLY_PT_view3d"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Painterly PBR'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        draw_brush(self.layout, context)


class PAINTERLY_PT_View3DUnderpaint(bpy.types.Panel):
    bl_label = "Underpaint Settings"
    bl_parent_id = "PAINTERLY_PT_view3d"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Painterly PBR'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        draw_underpaint(self.layout, context)


class PAINTERLY_PT_View3DNormal(bpy.types.Panel):
    bl_label = "Per-Stroke Normal Settings"
    bl_parent_id = "PAINTERLY_PT_view3d"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Painterly PBR'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        draw_normal(self.layout, context)


class PAINTERLY_PT_View3DDetail(bpy.types.Panel):
    bl_label = "Detail / Lineart Layer"
    bl_parent_id = "PAINTERLY_PT_view3d"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Painterly PBR'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        draw_detail(self.layout, context)


class PAINTERLY_PT_View3DExport(bpy.types.Panel):
    bl_label = "Output & Material Setup"
    bl_parent_id = "PAINTERLY_PT_view3d"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Painterly PBR'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        draw_export(self.layout, context)


classes = (
    PAINTERLY_PT_View3D,
    PAINTERLY_PT_View3DBrush,
    PAINTERLY_PT_View3DUnderpaint,
    PAINTERLY_PT_View3DNormal,
    PAINTERLY_PT_View3DDetail,
    PAINTERLY_PT_View3DExport,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)