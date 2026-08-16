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
        print("\n==================================================")
        print("[Painterly Console Debug] Starting Bake Process")
        settings = context.scene.painterly_settings
        src_img = settings.target_albedo

        if not src_img or src_img.size[0] == 0:
            self.report({'ERROR'}, "Select a valid Albedo image.")
            return {'CANCELLED'}

        w, h = src_img.size[0], src_img.size[1]
        print(f"[Painterly Debug] Source Image: Name='{src_img.name}', Dimensions=({w}x{h})")

        img_rgba = get_pixels_array(src_img)
        print(f"[Painterly Debug] Loaded Input RGBA Array: shape={img_rgba.shape}, min={img_rgba.min():.4f}, max={img_rgba.max():.4f}, mean={img_rgba.mean():.4f}")

        input_normal = None
        if settings.target_normal:
            input_normal = get_pixels_array(settings.target_normal, target_size=(w, h))
            print(f"[Painterly Debug] Loaded Target Normal Array: shape={input_normal.shape}, min={input_normal.min():.4f}, max={input_normal.max():.4f}, mean={input_normal.mean():.4f}")

        act_obj = context.active_object
        if act_obj:
            print(f"[Painterly Debug] Active Mesh Object: '{act_obj.name}' (Type: {act_obj.type})")
        else:
            print("[Painterly Debug] Active Mesh Object: None")

        wm = context.window_manager
        wm.progress_begin(0, 100)

        def progress_cb(fract):
            wm.progress_update(int(fract * 100))

        try:
            diff_out, norm_out, detail_out, cavity_out, flow_out, edge_out = execute_painterly_pipeline(
                img_rgba, settings, active_obj=act_obj, input_normal=input_normal, progress_cb=progress_cb
            )
        except Exception as exc:
            wm.progress_end()
            print(f"[Painterly Debug ERROR] Execution exception: {exc}")
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Bake failed: {exc}")
            return {'CANCELLED'}

        wm.progress_end()

        print("\n--- [Painterly Console Debug] Writing Output Textures ---")

        norm_name = f"{src_img.name}_PainterlyNorm"
        norm_img = create_blender_image(norm_name, w, h, norm_out, is_normal=True)
        print(f"[Painterly Debug] Normal Image Output: '{norm_name}', Pixel Stats: min={norm_out.min():.4f}, max={norm_out.max():.4f}, mean={norm_out.mean():.4f}")

        diff_img = None
        if settings.generate_diffuse:
            diff_name = f"{src_img.name}_PainterlyDiff"
            diff_img = create_blender_image(diff_name, w, h, diff_out, is_normal=False)
            print(f"[Painterly Debug] Diffuse Image Output: '{diff_name}', Pixel Stats: min={diff_out.min():.4f}, max={diff_out.max():.4f}, mean={diff_out.mean():.4f}")

        cavity_img = None
        if settings.generate_cavity and cavity_out is not None:
            cavity_name = f"{src_img.name}_CavityMap"
            cavity_img = create_blender_image(cavity_name, w, h, cavity_out, is_normal=True)
            print(f"[Painterly Debug] Cavity Image Output: '{cavity_name}', Pixel Stats: min={cavity_out.min():.4f}, max={cavity_out.max():.4f}, mean={cavity_out.mean():.4f}")

        if settings.generate_detail_layer and detail_out is not None:
            detail_name = f"{src_img.name}_DetailLayer"
            create_blender_image(detail_name, w, h, detail_out, is_normal=False, straight_alpha=True)
            print(f"[Painterly Debug] Detail Layer Output: '{detail_name}'")

        if settings.debug_export_textures:
            create_blender_image(f"{src_img.name}_FlowMapDebug", w, h, flow_out)
            create_blender_image(f"{src_img.name}_EdgeMapDebug", w, h, edge_out)
            print(f"[Painterly Debug] Exported Debug Flow and Edge Maps.")

        if settings.assign_material and act_obj:
            apply_toon_material(act_obj, diff_img, norm_img, cavity_img=cavity_img, cavity_mode=settings.cavity_mode, scene=context.scene)
            print(f"[Painterly Debug] Assigned Toon Material to '{act_obj.name}' (Cavity Mode: {settings.cavity_mode}).")

        print("==================================================\n")
        self.report({'INFO'}, f"Generated Painterly PBR textures for '{src_img.name}'")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(PAINTERLY_OT_BakePainterlyPBR)

def unregister():
    bpy.utils.unregister_class(PAINTERLY_OT_BakePainterlyPBR)