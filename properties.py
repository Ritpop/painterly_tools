import bpy

class PainterlySettings(bpy.types.PropertyGroup):
    target_albedo: bpy.props.PointerProperty(
        name="Albedo / Color",
        type=bpy.types.Image,
        description="Source albedo image to generate painterly textures from"
    )
    target_normal: bpy.props.PointerProperty(
        name="Normal Map (Optional)",
        type=bpy.types.Image,
        description="Existing normal map to blend brush stroke directions onto"
    )

    stroke_scale: bpy.props.FloatProperty(
        name="Stroke Size", default=0.06, min=0.005, max=0.2,
        description="Base brush size relative to image dimensions"
    )
    stroke_density: bpy.props.FloatProperty(
        name="Stroke Density", default=3.0, min=0.1, max=3.0,
        description="Stroke placement density across the canvas"
    )
    stroke_tilt: bpy.props.FloatProperty(
        name="Stroke Tilt Angle", default=0.35, min=0.05, max=1.2,
        description="Controls light reflection angle across stroke edges"
    )
    bristle_detail: bpy.props.FloatProperty(
        name="Bristle Detail", default=0.4, min=0.0, max=1.0,
        description="Internal bristle line intensity"
    )
    normal_strength: bpy.props.FloatProperty(
        name="Normal Strength", default=0.9, min=0.0, max=3.0,
        description="Bump height of generated brush stroke directions"
    )
    facet_hardness: bpy.props.FloatProperty(
        name="Facet Hardness", default=0.1, min=0.0, max=1.0,
        description="Sharpness of stroke boundaries (0 = hard impasto facets, 1 = soft blend)"
    )
    color_jitter: bpy.props.FloatProperty(
        name="Color Jitter", default=0.01, min=0.0, max=0.2,
        description="Random per-stroke hue and saturation offset"
    )
    stroke_length: bpy.props.FloatProperty(
        name="Stroke Length", default=3.0, min=1.0, max=6.0,
        description="Elongates round marks into directional brush marks"
    )
    stroke_curvature: bpy.props.FloatProperty(
        name="Stroke Curvature", default=0.15, min=0.0, max=1.5,
        description="Bending along stroke trajectories"
    )
    flow_smoothing: bpy.props.IntProperty(
        name="Flow Blur", default=1, min=1, max=30,
        description="Smoothness of vector field orientation"
    )
    use_uv_flow: bpy.props.BoolProperty(
        name="Use Mesh UV Orientation", default=False,
        description="Align brush stroke flow direction using active object UV layout directions"
    )
    uv_flow_direction: bpy.props.EnumProperty(
        name="UV Direction Axis",
        items=[
            ('U', "U Axis", "Align brush strokes along UV horizontal axis"),
            ('V', "V Axis", "Align brush strokes along UV vertical axis"),
            ('TANGENT', "UV Tangent Field", "Align flow field relative to active UV islands")
        ],
        default='U'
    )
    uv_flow_mix: bpy.props.FloatProperty(
        name="UV Flow Blend", default=0.7, min=0.0, max=1.0,
        description="Blend ratio between UV directions and image visual edge gradients"
    )
    edge_influence: bpy.props.FloatProperty(
        name="Edge Alignment", default=0.5, min=0.0, max=1.0,
        description="Snapping force toward detected image contours"
    )
    edge_blur: bpy.props.IntProperty(
        name="Edge Map Blur", default=1, min=0, max=10,
        description="Pre-filter blur for edge detection"
    )
    edge_threshold: bpy.props.FloatProperty(
        name="Edge Threshold", default=0.15, min=0.01, max=1.0,
        description="Cutoff threshold for edge detection mask"
    )
    edge_contrast: bpy.props.FloatProperty(
        name="Edge Contrast", default=1.5, min=0.1, max=5.0,
        description="Contrast curve applied to edge mask"
    )
    flow_scale_blend: bpy.props.FloatProperty(
        name="Multi-Scale Blend", default=0.3, min=0.0, max=1.0,
        description="Blend ratio between local and broad directional fields"
    )
    flow_scale_multiplier: bpy.props.FloatProperty(
        name="Coarse Scale Multiplier", default=3.0, min=1.5, max=8.0,
        description="Scale difference for secondary flow pass"
    )
    cell_jitter: bpy.props.FloatProperty(
        name="Stroke Jitter", default=0.5, min=0.0, max=1.0,
        description="Random stroke center displacement from grid"
    )

    use_kuwahara_underpaint: bpy.props.BoolProperty(
        name="Kuwahara Underpaint", default=True,
        description="Applies edge-preserving Kuwahara filter to the base underpaint layer"
    )
    kuw_radius: bpy.props.IntProperty(
        name="Kuwahara Radius", default=4, min=1, max=8,
        description="Neighborhood size for the Kuwahara pass"
    )
    random_seed: bpy.props.IntProperty(
        name="Random Seed", default=0, min=0,
        description="Seed value for procedural placement"
    )

    generate_diffuse: bpy.props.BoolProperty(
        name="Export Painterly Color", default=True,
        description="Bake painterly albedo output texture"
    )
    generate_cavity: bpy.props.BoolProperty(
        name="Bake Cavity / Wear Map", default=True,
        description="Extract curvature crevices and high edge wear"
    )
    cavity_source: bpy.props.EnumProperty(
        name="Cavity Source",
        items=[
            ('MESH', "Mesh Geometry", "Calculate 3D curvature directly from active mesh object"),
            ('TEXTURE', "Texture Bump", "Calculate 2D curvature from baked normal map details")
        ],
        default='MESH',
        description="Source used to compute cavity crevices and edge wear"
    )
    cavity_mode: bpy.props.EnumProperty(
        name="Cavity Target",
        items=[
            ('COMBINED', "Combined (Cavity + Edge)", "Dark crevices (<0.5), neutral flat (0.5), bright edges (>0.5)"),
            ('CAVITY', "Cavity Only (Crevices)", "Dark crevices on clean white background"),
            ('EDGE', "Edge Wear Only (Ridges)", "Bright highlights on sharp corners and bevels"),
            ('SPLIT_RG', "RGBA Packed (R: Edge, G: Cavity)", "Red channel = Edge Wear, Green channel = Cavity Occlusion")
        ],
        default='COMBINED',
        description="Type of curvature details to isolate in the output map"
    )
    cavity_strength: bpy.props.FloatProperty(
        name="Cavity Strength", default=1.8, min=0.1, max=5.0,
        description="Contrast/Depth intensity of extracted cavities and edges"
    )
    cavity_blur: bpy.props.IntProperty(
        name="Cavity Smoothness", default=1, min=0, max=5,
        description="Smoothing filter radius for cavity details"
    )
    assign_material: bpy.props.BoolProperty(
        name="Apply Material to Active Mesh", default=True,
        description="Automatically assigns a Toon material node setup to active object"
    )
    debug_export_textures: bpy.props.BoolProperty(
        name="Export Debug Maps", default=False,
        description="Output internal vector flow and edge strength maps"
    )
    generate_detail_layer: bpy.props.BoolProperty(
        name="Generate Lineart Pass", default=False,
        description="Bakes a transparent overlay pass targeting crisp edges"
    )
    detail_scale: bpy.props.FloatProperty(
        name="Detail Size", default=0.012, min=0.002, max=0.1,
        description="Stroke size for lineart overlay"
    )
    detail_opacity: bpy.props.FloatProperty(
        name="Detail Opacity", default=1.0, min=0.0, max=1.0,
        description="Opacity of fine lineart pass"
    )
    detail_edge_influence: bpy.props.FloatProperty(
        name="Detail Edge Snap", default=0.9, min=0.0, max=1.0,
        description="Lineart snapping tightness to image edges"
    )
    detail_density: bpy.props.FloatProperty(
        name="Detail Density", default=1.5, min=0.1, max=4.0,
        description="Stroke density for lineart pass"
    )
    detail_density_from_edges: bpy.props.BoolProperty(
        name="Mask to Edges Only", default=True,
        description="Restrict lineart strokes exclusively to detected edge masks"
    )
    detail_edge_density_threshold: bpy.props.FloatProperty(
        name="Edge Threshold", default=0.3, min=0.0, max=1.0,
        description="Edge strength cutoff for detail placement"
    )


def register():
    bpy.utils.register_class(PainterlySettings)
    bpy.types.Scene.painterly_settings = bpy.props.PointerProperty(type=PainterlySettings)


def unregister():
    del bpy.types.Scene.painterly_settings
    bpy.utils.unregister_class(PainterlySettings)