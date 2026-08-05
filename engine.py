import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from .shaders import get_painterly_shader
from .image_utils import build_flow_texture_rgba, build_edge_texture_rgba

LAYER_PRESETS = {
    "scale": [2.6, 2.0, 1.2, 0.8, 0.4],
    "opacity": [1.00, 0.65, 0.80, 0.90, 1.00],
    "elongate": [0.0, 1.0, 0.8, 0.4, 0.0],
    "paints_normal": [False, True, True, True, True],
    "taper": [0.0, 0.3, 0.5, 0.7, 0.9],
    "dry_brush": [0.0, 0.1, 0.2, 0.4, 0.7],
    "wet_blend": [0.6, 0.4, 0.3, 0.1, 0.0]
}

def _make_gpu_texture(rgba_array, width, height):
    flat = np.ascontiguousarray(rgba_array, dtype=np.float32).ravel()
    expected_size = width * height * 4
    
    if flat.shape[0] != expected_size:
        flat = np.resize(flat, expected_size)
        
    buf = gpu.types.Buffer('FLOAT', expected_size, flat.tolist())
    return gpu.types.GPUTexture((width, height), format='RGBA32F', data=buf)

def _run_pass(shader, batch, src_tex, flow_tex, edge_tex, prev_tex, width, height,
              cell_size, opacity, elongate, density, bristle_detail,
              color_jitter, stroke_tilt, stroke_curvature, normal_strength,
              seed, edge_influence, cell_jitter, paints_normal, output_mode,
              edge_gate_density=False, edge_density_threshold=0.3,
              stroke_taper=0.7, dry_brush=0.3, wet_blend=0.2, 
              facet_hardness=0.1, kuw_radius=2):

    offscreen = gpu.types.GPUOffScreen(width, height, format='RGBA32F')
    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.0, 0.0, 0.0, 0.0) if output_mode == 'DETAIL' else (0.0, 0.0, 0.0, 1.0))
        shader.bind()
        
        shader.uniform_sampler("srcTex", src_tex)
        shader.uniform_sampler("flowTex", flow_tex)
        shader.uniform_sampler("edgeTex", edge_tex)
        shader.uniform_sampler("prevTex", prev_tex)
        shader.uniform_float("edgeInfluence", float(edge_influence))
        shader.uniform_float("resolution", (float(width), float(height)))
        shader.uniform_float("cellSize", float(cell_size))
        shader.uniform_float("opacity", float(opacity))
        shader.uniform_float("elongate", float(elongate))
        shader.uniform_float("density", float(density))
        shader.uniform_float("bristleDetail", float(bristle_detail))
        shader.uniform_float("colorJitter", float(color_jitter))
        shader.uniform_float("strokeTilt", float(stroke_tilt))
        shader.uniform_float("strokeCurvature", float(stroke_curvature))
        shader.uniform_float("normalStrength", float(normal_strength))
        shader.uniform_float("seed", float(seed))
        shader.uniform_float("cellJitter", float(cell_jitter))
        shader.uniform_float("strokeTaper", float(stroke_taper))
        shader.uniform_float("dryBrush", float(dry_brush))
        shader.uniform_float("wetBlend", float(wet_blend))
        shader.uniform_float("facetHardness", float(facet_hardness))
        shader.uniform_int("kuwRadius", int(kuw_radius))
        shader.uniform_int("edgeGateDensity", 1 if edge_gate_density else 0)
        shader.uniform_float("edgeDensityThreshold", float(edge_density_threshold))
        shader.uniform_int("paintsNormal", 1 if paints_normal else 0)
        
        mode_idx = {'DIFFUSE': 0, 'NORMAL': 1, 'DETAIL': 2, 'KUWAHARA': 3}.get(output_mode, 0)
        shader.uniform_int("outputMode", mode_idx)
        
        batch.draw(shader)
        buf = fb.read_color(0, 0, width, height, 4, 0, 'FLOAT')
        buf.dimensions = width * height * 4
        out = np.array(buf, dtype=np.float32).reshape(height, width, 4)

    offscreen.free()
    return out

def execute_painterly_pipeline(img_rgba, settings, input_normal=None, progress_cb=None):
    H, W = img_rgba.shape[0], img_rgba.shape[1]

    flow_rgba = build_flow_texture_rgba(
        img_rgba, blur_radius=settings.flow_smoothing,
        scale_blend=settings.flow_scale_blend, coarse_multiplier=settings.flow_scale_multiplier,
    )
    edge_rgba = build_edge_texture_rgba(
        img_rgba, edge_blur=settings.edge_blur,
        edge_threshold=settings.edge_threshold, edge_contrast=settings.edge_contrast,
    )

    shader = get_painterly_shader()
    batch = batch_for_shader(shader, 'TRIS', {"pos": [(-1.0, -1.0), (3.0, -1.0), (-1.0, 3.0)]})

    src_tex = _make_gpu_texture(img_rgba, W, H)
    flow_tex = _make_gpu_texture(flow_rgba, W, H)
    edge_tex = _make_gpu_texture(edge_rgba, W, H)

    num_layers = len(LAYER_PRESETS["scale"])
    diff_canvas = img_rgba.copy()

    if input_normal is not None:
        norm_canvas = input_normal.copy()
    else:
        norm_canvas = np.zeros((H, W, 4), dtype=np.float32)
        norm_canvas[..., 0:3] = [0.5, 0.5, 1.0]
        norm_canvas[..., 3] = 1.0

    total_steps = num_layers * ((1 if settings.generate_diffuse else 0) + 1) + (1 if settings.generate_detail_layer else 0)
    step = 0

    for i in range(num_layers):
        cell_size = max(2.0, min(H, W) * settings.stroke_scale * LAYER_PRESETS["scale"][i])
        elong = 1.0 + LAYER_PRESETS["elongate"][i] * (settings.stroke_length - 1.0)
        paints_norm = LAYER_PRESETS["paints_normal"][i]
        
        l_taper = LAYER_PRESETS["taper"][i]
        l_dry = LAYER_PRESETS["dry_brush"][i]
        l_wet = LAYER_PRESETS["wet_blend"][i]

        if settings.generate_diffuse:
            prev_tex = _make_gpu_texture(diff_canvas, W, H)
            if i == 0 and settings.use_kuwahara_underpaint:
                diff_canvas = _run_pass(
                    shader, batch, src_tex, flow_tex, edge_tex, prev_tex, W, H,
                    0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    settings.random_seed, 0.0, 0.0, False, 'KUWAHARA',
                    kuw_radius=settings.kuw_radius
                )
            else:
                diff_canvas = _run_pass(
                    shader, batch, src_tex, flow_tex, edge_tex, prev_tex, W, H,
                    cell_size, LAYER_PRESETS["opacity"][i], elong, settings.stroke_density,
                    settings.bristle_detail, settings.color_jitter, settings.stroke_tilt,
                    settings.stroke_curvature, settings.normal_strength,
                    settings.random_seed + i * 97, settings.edge_influence,
                    settings.cell_jitter, paints_norm, 'DIFFUSE',
                    stroke_taper=l_taper, dry_brush=l_dry, wet_blend=l_wet,
                    facet_hardness=settings.facet_hardness,
                )
            step += 1
            if progress_cb: progress_cb(step / total_steps)

        if not (i == 0 and settings.use_kuwahara_underpaint):
            prev_norm_tex = _make_gpu_texture(norm_canvas, W, H)
            norm_canvas = _run_pass(
                shader, batch, src_tex, flow_tex, edge_tex, prev_norm_tex, W, H,
                cell_size, LAYER_PRESETS["opacity"][i], elong, settings.stroke_density,
                settings.bristle_detail, settings.color_jitter, settings.stroke_tilt,
                settings.stroke_curvature, settings.normal_strength,
                settings.random_seed + i * 97, settings.edge_influence,
                settings.cell_jitter, paints_norm, 'NORMAL',
                stroke_taper=l_taper, dry_brush=l_dry, wet_blend=l_wet,
                facet_hardness=settings.facet_hardness,
            )
            step += 1
            if progress_cb: progress_cb(step / total_steps)

    detail_rgba = None
    if settings.generate_detail_layer:
        detail_cell_size = max(2.0, min(H, W) * settings.detail_scale)
        transparent_tex = _make_gpu_texture(np.zeros((H, W, 4), dtype=np.float32), W, H)
        detail_rgba = _run_pass(
            shader, batch, src_tex, flow_tex, edge_tex, transparent_tex, W, H,
            detail_cell_size, settings.detail_opacity, 1.0, settings.detail_density,
            settings.bristle_detail, settings.color_jitter, settings.stroke_tilt,
            settings.stroke_curvature, settings.normal_strength,
            settings.random_seed + 9973, settings.detail_edge_influence,
            settings.cell_jitter, True, 'DETAIL',
            edge_gate_density=settings.detail_density_from_edges,
            edge_density_threshold=settings.detail_edge_density_threshold,
            stroke_taper=1.0, dry_brush=0.8, wet_blend=0.0,
            facet_hardness=settings.facet_hardness,
        )
        step += 1
        if progress_cb: progress_cb(step / total_steps)

    nx, ny, nz = norm_canvas[..., 0] * 2.0 - 1.0, norm_canvas[..., 1] * 2.0 - 1.0, norm_canvas[..., 2] * 2.0 - 1.0
    length = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    length = np.where(length < 1e-6, 1.0, length)
    norm_canvas[..., 0] = (nx / length) * 0.5 + 0.5
    norm_canvas[..., 1] = (ny / length) * 0.5 + 0.5
    norm_canvas[..., 2] = (nz / length) * 0.5 + 0.5
    norm_canvas[..., 3] = 1.0

    diff_canvas[..., 3] = img_rgba[..., 3]

    return diff_canvas, norm_canvas, detail_rgba, flow_rgba, edge_rgba

def build_staircase_node_group():
    grp_name = "Smooth Staircase Quantizer"
    if grp_name in bpy.data.node_groups:
        return bpy.data.node_groups[grp_name]

    grp = bpy.data.node_groups.new(name=grp_name, type='ShaderNodeTree')

    if hasattr(grp, "interface"):
        grp.interface.new_socket(name="Value", in_out='INPUT', socket_type='NodeSocketFloat')
        steps = grp.interface.new_socket(name="Steps", in_out='INPUT', socket_type='NodeSocketFloat')
        steps.default_value = 3.0
        grp.interface.new_socket(name="Stepped Value", in_out='OUTPUT', socket_type='NodeSocketFloat')
    else:
        grp.inputs.new('NodeSocketFloat', 'Value')
        steps = grp.inputs.new('NodeSocketFloat', 'Steps')
        steps.default_value = 3.0
        grp.outputs.new('NodeSocketFloat', 'Stepped Value')

    nodes = grp.nodes
    links = grp.links

    inp = nodes.new('NodeGroupInput')
    out = nodes.new('NodeGroupOutput')
    inp.location = (-400, 0)
    out.location = (600, 0)

    m_pi = nodes.new('ShaderNodeMath')
    m_pi.operation = 'MULTIPLY'
    m_pi.inputs[1].default_value = 6.2831853

    m_val = nodes.new('ShaderNodeMath')
    m_val.operation = 'MULTIPLY'

    s_node = nodes.new('ShaderNodeMath')
    s_node.operation = 'SINE'

    d_node = nodes.new('ShaderNodeMath')
    d_node.operation = 'DIVIDE'

    sub_node = nodes.new('ShaderNodeMath')
    sub_node.operation = 'SUBTRACT'

    links.new(inp.outputs[1], m_pi.inputs[0])
    links.new(inp.outputs[0], m_val.inputs[0])
    links.new(m_pi.outputs[0], m_val.inputs[1])
    links.new(m_val.outputs[0], s_node.inputs[0])
    links.new(s_node.outputs[0], d_node.inputs[0])
    links.new(m_pi.outputs[0], d_node.inputs[1])
    links.new(inp.outputs[0], sub_node.inputs[0])
    links.new(d_node.outputs[0], sub_node.inputs[1])
    links.new(sub_node.outputs[0], out.inputs[0])

    return grp

def _create_color_mix_node(nodes, blend_type='MULTIPLY', factor=1.0):
    if hasattr(bpy.types, "ShaderNodeMix"):
        mix = nodes.new('ShaderNodeMix')
        mix.data_type = 'RGBA'
        mix.blend_type = blend_type
        mix.inputs['Factor'].default_value = factor
        return mix, mix.inputs['A'], mix.inputs['B'], mix.outputs['Result']
    else:
        mix = nodes.new('ShaderNodeMixRGB')
        mix.blend_type = blend_type
        mix.inputs['Fac'].default_value = factor
        return mix, mix.inputs['Color1'], mix.inputs['Color2'], mix.outputs['Color']

def apply_toon_material(obj, diff_img, norm_img, scene=None):
    if not obj or obj.type != 'MESH':
        return

    is_cycles = (scene and scene.render.engine == 'CYCLES')
    mat_name = f"{obj.name}_PainterlyMat"
    mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(name=mat_name)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new('ShaderNodeOutputMaterial')
    out_node.location = (1400, 0)

    tex_norm = nodes.new('ShaderNodeTexImage')
    tex_norm.image = norm_img
    tex_norm.image.colorspace_settings.name = 'Non-Color'
    tex_norm.location = (-1000, -100)

    norm_map = nodes.new('ShaderNodeNormalMap')
    norm_map.location = (-800, -100)
    links.new(tex_norm.outputs[0], norm_map.inputs[1])

    tex_diff = nodes.new('ShaderNodeTexImage')
    tex_diff.image = diff_img
    tex_diff.location = (400, 300)

    if is_cycles:
        toon = nodes.new('ShaderNodeBsdfToon')
        toon.component = 'DIFFUSE'
        toon.inputs[1].default_value = 0.15
        toon.inputs[2].default_value = 0.40
        toon.location = (1000, 0)

        links.new(tex_diff.outputs[0], toon.inputs[0])
        links.new(norm_map.outputs[0], toon.inputs[3])
        links.new(toon.outputs[0], out_node.inputs[0])
    else:
        bsdf = nodes.new('ShaderNodeBsdfDiffuse')
        bsdf.location = (-600, -100)
        links.new(norm_map.outputs[0], bsdf.inputs[2])

        s2rgb = nodes.new('ShaderNodeShaderToRGB')
        s2rgb.location = (-400, -100)
        links.new(bsdf.outputs[0], s2rgb.inputs[0])

        sep_hsv = nodes.new('ShaderNodeSeparateColor')
        sep_hsv.mode = 'HSV'
        sep_hsv.location = (-200, -100)
        links.new(s2rgb.outputs[0], sep_hsv.inputs[0])

        stair = nodes.new('ShaderNodeGroup')
        stair.node_tree = build_staircase_node_group()
        stair.location = (0, -100)
        links.new(sep_hsv.outputs[2], stair.inputs[0])

        comb_hsv = nodes.new('ShaderNodeCombineColor')
        comb_hsv.mode = 'HSV'
        comb_hsv.location = (200, -100)
        links.new(sep_hsv.outputs[0], comb_hsv.inputs[0])
        links.new(sep_hsv.outputs[1], comb_hsv.inputs[1])
        links.new(stair.outputs[0], comb_hsv.inputs[2])

        mix_mult, mult_in_a, mult_in_b, mult_out = _create_color_mix_node(nodes, blend_type='MULTIPLY', factor=1.0)
        mix_mult.location = (600, 0)

        links.new(tex_diff.outputs[0], mult_in_a)
        links.new(comb_hsv.outputs[0], mult_in_b)

        mix_over, over_in_a, over_in_b, over_out = _create_color_mix_node(nodes, blend_type='OVERLAY', factor=0.3)
        mix_over.location = (800, 0)

        links.new(mult_out, over_in_a)
        links.new(tex_diff.outputs[0], over_in_b)

        emit = nodes.new('ShaderNodeEmission')
        emit.location = (1100, 0)
        links.new(over_out, emit.inputs[0])
        links.new(emit.outputs[0], out_node.inputs[0])

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)