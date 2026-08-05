import bpy
from .image_utils import get_pixels_array, create_blender_image
from .engine import execute_painterly_pipeline, apply_toon_material

class PAINTERLY_OT_BakePainterlyPBR(bpy.types.Operator):
    """Executes GPU-accelerated Directional Brush Stroke and Normal Map Generator"""
    bl_idname = "image.generate_painterly_normal"
    bl_label = "Generate Painterly PBR (GPU)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.painterly_settings.target_albedo is not None

    def execute(self, context):
        settings = context.scene.painterly_settings
        src_img = settings.target_albedo

        if not src_img or src_img.size[0] == 0:
            self.report({'ERROR'}, "Select a valid Albedo image.")
            return {'CANCELLED'}

        w, h = src_img.size[0], src_img.size[1]
        img_rgba = get_pixels_array(src_img)

        input_normal = get_pixels_array(settings.target_normal, target_size=(w, h)) if settings.target_normal else None

        wm = context.window_manager
        wm.progress_begin(0, 100)

        def progress_cb(fract):
            wm.progress_update(int(fract * 100))

        try:
            diff_out, norm_out, detail_out, flow_out, edge_out = execute_painterly_pipeline(
                img_rgba, settings, input_normal=input_normal, progress_cb=progress_cb
            )
        except Exception as exc:
            wm.progress_end()
            self.report({'ERROR'}, f"Bake failed: {exc}")
            return {'CANCELLED'}

        wm.progress_end()

        norm_name = f"{src_img.name}_PainterlyNorm"
        norm_img = create_blender_image(norm_name, w, h, norm_out, is_normal=True)

        diff_img = None
        if settings.generate_diffuse:
            diff_name = f"{src_img.name}_PainterlyDiff"
            diff_img = create_blender_image(diff_name, w, h, diff_out, is_normal=False)

        if settings.generate_detail_layer and detail_out is not None:
            detail_name = f"{src_img.name}_DetailLayer"
            create_blender_image(detail_name, w, h, detail_out, is_normal=False, straight_alpha=True)

        if settings.debug_export_textures:
            create_blender_image(f"{src_img.name}_FlowMapDebug", w, h, flow_out)
            create_blender_image(f"{src_img.name}_EdgeMapDebug", w, h, edge_out)

        if settings.assign_material and context.active_object:
            apply_toon_material(context.active_object, diff_img, norm_img, scene=context.scene)

        self.report({'INFO'}, f"Generated Painterly PBR textures for '{src_img.name}'")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(PAINTERLY_OT_BakePainterlyPBR)

def unregister():
    bpy.utils.unregister_class(PAINTERLY_OT_BakePainterlyPBR)