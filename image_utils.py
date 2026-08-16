import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from .shaders import get_mesh_cavity_shader

def _clamped_shift(arr, dy, dx):
    H, W = arr.shape[:2]
    y_idx = np.clip(np.arange(H) - dy, 0, H - 1)
    x_idx = np.clip(np.arange(W) - dx, 0, W - 1)
    return arr[y_idx][:, x_idx]

def _resize_nearest(arr, out_h, out_w):
    in_h, in_w = arr.shape[0], arr.shape[1]
    y_idx = np.clip((np.arange(out_h) * in_h / out_h).astype(np.int32), 0, in_h - 1)
    x_idx = np.clip((np.arange(out_w) * in_w / out_w).astype(np.int32), 0, in_w - 1)
    return arr[y_idx][:, x_idx]

def get_pixels_array(image, target_size=None):
    w, h = image.size[0], image.size[1]
    raw = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(raw)
    data = raw.reshape((h, w, 4))
    
    if target_size is not None:
        tw, th = target_size
        if w != tw or h != th:
            data = _resize_nearest(data, th, tw)
            
    return np.ascontiguousarray(data, dtype=np.float32)

def create_blender_image(name, width, height, rgba_array, is_normal=False, straight_alpha=False):
    rgba_clean = np.ascontiguousarray(rgba_array, dtype=np.float32)
    if rgba_clean.shape[2] == 4 and not straight_alpha:
        rgba_clean[:, :, 3] = 1.0
    elif rgba_clean.shape[2] == 4:
        rgba_clean[:, :, 3] = np.clip(rgba_clean[:, :, 3], 0.0, 1.0)

    if name in bpy.data.images:
        final_img = bpy.data.images[name]
        if final_img.size[0] != width or final_img.size[1] != height:
            final_img.scale(width, height)
    else:
        final_img = bpy.data.images.new(
            name=name,
            width=width,
            height=height,
            alpha=True,
            float_buffer=True
        )

    final_img.use_fake_user = True

    if is_normal:
        try:
            final_img.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass

    final_img.pixels.foreach_set(rgba_clean.ravel())
    
    try:
        final_img.pack()
    except Exception as e:
        print(f"[Painterly Debug Warning] Could not pack image '{name}': {e}")

    final_img.update()

    if bpy.context and bpy.context.screen:
        for area in bpy.context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                area.tag_redraw()

    return final_img

def _box_blur_1d(arr, r, axis):
    if r < 1:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (r, r)
    padded = np.pad(arr, pad_width, mode='edge')
    
    shape = list(padded.shape)
    shape[axis] = 1
    zeros = np.zeros(shape, dtype=arr.dtype)
    c = np.concatenate([zeros, np.cumsum(padded, axis=axis)], axis=axis)
    
    N = arr.shape[axis]
    slc1 = [slice(None)] * arr.ndim
    slc2 = [slice(None)] * arr.ndim
    slc1[axis] = slice(2 * r + 1, N + 2 * r + 1)
    slc2[axis] = slice(0, N)
    
    return (c[tuple(slc1)] - c[tuple(slc2)]) / (2 * r + 1)

def _box_blur(arr, r):
    if r < 1:
        return arr
    return _box_blur_1d(_box_blur_1d(arr, r, axis=0), r, axis=1)

def calculate_flow_angles(img_rgba, blur_radius=6):
    rgb = img_rgba[..., :3]
    lum = (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722).astype(np.float32)

    gy, gx = np.gradient(lum)
    Jxx, Jxy, Jyy = gx * gx, gx * gy, gy * gy
    Jxx = _box_blur(Jxx, blur_radius)
    Jxy = _box_blur(Jxy, blur_radius)
    Jyy = _box_blur(Jyy, blur_radius)

    return (0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy) + (np.pi / 2.0)).astype(np.float32)

def generate_uv_flow_angles(obj, width, height, axis_mode='U'):
    angles = np.zeros((height, width), dtype=np.float32)
    if not obj or obj.type != 'MESH' or not obj.data.uv_layers.active:
        return angles

    uv_layer = obj.data.uv_layers.active.data
    mesh = obj.data

    u_dir = 0.0 if axis_mode == 'U' else (np.pi / 2.0 if axis_mode == 'V' else 0.0)

    for poly in mesh.polygons:
        uvs = [uv_layer[loop_idx].uv for loop_idx in poly.loop_indices]
        if len(uvs) < 3:
            continue
        
        if axis_mode == 'TANGENT':
            du1 = uvs[1].x - uvs[0].x
            dv1 = uvs[1].y - uvs[0].y
            ang = np.arctan2(dv1, du1)
        else:
            ang = u_dir

        min_u = max(0, int(min(u.x for u in uvs) * width))
        max_u = min(width - 1, int(max(u.x for u in uvs) * width))
        min_v = max(0, int(min(u.y for u in uvs) * height))
        max_v = min(height - 1, int(max(u.y for u in uvs) * height))

        if max_u > min_u and max_v > min_v:
            angles[min_v:max_v, min_u:max_u] = ang

    return angles

def build_edge_texture_rgba(img_rgba, edge_blur=1, edge_threshold=0.15, edge_contrast=1.5):
    rgb = img_rgba[..., :3]
    lum = (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722).astype(np.float32)

    smoothed = _box_blur(lum, max(1, edge_blur))
    gy, gx = np.gradient(smoothed)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = mag / (mag.max() + 1e-6)
    angle = np.arctan2(gy, gx)

    mask = np.clip(mag * edge_contrast, 0.0, 1.0)
    mask = np.where(mask > edge_threshold, mask, 0.0)

    if img_rgba.shape[2] == 4:
        mask *= np.clip(img_rgba[..., 3], 0.0, 1.0)

    tangent_angle = angle + (np.pi / 2.0)
    r = np.cos(2.0 * tangent_angle) * 0.5 + 0.5
    g = np.sin(2.0 * tangent_angle) * 0.5 + 0.5
    b = np.clip(mask, 0.0, 1.0)
    a = np.ones_like(r)
    return np.ascontiguousarray(np.stack([r, g, b, a], axis=-1), dtype=np.float32)

def _flow_direction_vector(img_rgba, blur_radius):
    angles = calculate_flow_angles(img_rgba, blur_radius=blur_radius)
    return np.cos(2.0 * angles), np.sin(2.0 * angles)

def build_flow_texture_rgba(img_rgba, blur_radius, scale_blend=0.0, coarse_multiplier=3.0,
                            uv_angles=None, uv_mix=0.0):
    r, g = _flow_direction_vector(img_rgba, blur_radius)

    if scale_blend > 0.0:
        coarse_radius = max(1, int(round(blur_radius * coarse_multiplier)))
        r2, g2 = _flow_direction_vector(img_rgba, coarse_radius)
        r = r * (1.0 - scale_blend) + r2 * scale_blend

    if uv_angles is not None and uv_mix > 0.0:
        uv_r = np.cos(2.0 * uv_angles)
        uv_g = np.sin(2.0 * uv_angles)
        r = r * (1.0 - uv_mix) + uv_r * uv_mix
        g = g * (1.0 - uv_mix) + uv_g * uv_mix

    norm = np.sqrt(r ** 2 + g ** 2) + 1e-6
    r, g = r / norm, g / norm

    r_enc = r * 0.5 + 0.5
    g_enc = g * 0.5 + 0.5
    b = np.zeros_like(r_enc)
    a = np.ones_like(r_enc)
    return np.ascontiguousarray(np.stack([r_enc, g_enc, b, a], axis=-1), dtype=np.float32)

def build_cavity_texture_rgba(norm_rgba, strength=1.8, blur_radius=1):
    nx = norm_rgba[..., 0] * 2.0 - 1.0
    ny = norm_rgba[..., 1] * 2.0 - 1.0

    ddx_x = np.gradient(nx, axis=1)
    ddy_y = np.gradient(ny, axis=0)

    curvature = -(ddx_x + ddy_y)
    curvature = _box_blur(curvature, max(0, blur_radius))

    cavity = np.clip(0.5 + curvature * strength, 0.0, 1.0)
    
    out = np.ones_like(norm_rgba)
    out[..., 0] = cavity
    out[..., 1] = cavity
    out[..., 2] = cavity
    out[..., 3] = 1.0
    return np.ascontiguousarray(out, dtype=np.float32)

def _postprocess_cavity_channels(raw_canvas, mode='COMBINED'):
    c = raw_canvas[..., 0]
    out = np.ones_like(raw_canvas)

    if mode == 'CAVITY':
        cav_only = np.where(c < 0.5, c * 2.0, 1.0)
        out[..., 0] = cav_only
        out[..., 1] = cav_only
        out[..., 2] = cav_only
    elif mode == 'EDGE':
        edge_only = np.where(c > 0.5, (c - 0.5) * 2.0, 0.0)
        out[..., 0] = edge_only
        out[..., 1] = edge_only
        out[..., 2] = edge_only
    elif mode == 'SPLIT_RG':
        edge_chan = np.where(c > 0.5, (c - 0.5) * 2.0, 0.0)
        cav_chan = np.where(c < 0.5, (0.5 - c) * 2.0, 0.0)
        out[..., 0] = edge_chan
        out[..., 1] = cav_chan
        out[..., 2] = 0.0
    else:
        out[..., 0] = c
        out[..., 1] = c
        out[..., 2] = c

    out[..., 3] = 1.0
    return np.ascontiguousarray(out, dtype=np.float32)

def bake_mesh_geometric_cavity(obj, width, height, strength=1.8, blur_radius=1, mode='COMBINED'):
    if not obj or obj.type != 'MESH' or not obj.data.uv_layers.active:
        return None

    mesh = obj.data
    mesh.calc_loop_triangles()

    num_polys = len(mesh.polygons)
    num_edges = len(mesh.edges)
    num_tris = len(mesh.loop_triangles)

    if num_polys == 0 or num_tris == 0:
        return None

    poly_normals = np.empty((num_polys, 3), dtype=np.float32)
    poly_centers = np.empty((num_polys, 3), dtype=np.float32)
    mesh.polygons.foreach_get('normal', poly_normals.ravel())
    mesh.polygons.foreach_get('center', poly_centers.ravel())

    edge_verts = np.empty((num_edges, 2), dtype=np.int32)
    mesh.edges.foreach_get('vertices', edge_verts.ravel())

    edge_face_map = {}
    for p in mesh.polygons:
        p_edges = p.edge_keys
        for e in p_edges:
            e_key = tuple(sorted(e))
            edge_face_map.setdefault(e_key, []).append(p.index)

    edge_cavity_dict = {}
    for e_idx, (v0, v1) in enumerate(edge_verts):
        e_key = tuple(sorted((v0, v1)))
        f_list = edge_face_map.get(e_key, [])
        if len(f_list) == 2:
            f0, f1 = f_list[0], f_list[1]
            n0, n1 = poly_normals[f0], poly_normals[f1]
            dot_n = np.clip(np.dot(n0, n1), -1.0, 1.0)
            
            if dot_n > 0.98:
                edge_cavity_dict[e_key] = 0.5
            else:
                sharpness = 1.0 - dot_n
                dir_vec = poly_centers[f1] - poly_centers[f0]
                sign = 1.0 if np.dot(n0 + n1, dir_vec) >= 0.0 else -1.0
                cav_val = np.clip(0.5 + sign * sharpness * strength * 0.5, 0.0, 1.0)
                edge_cavity_dict[e_key] = cav_val
        else:
            edge_cavity_dict[e_key] = 0.5

    uv_layer = mesh.uv_layers.active.data
    total_verts = num_tris * 3

    gpu_bary = np.zeros((total_verts, 3), dtype=np.float32)
    gpu_edge_cavity = np.full((total_verts, 3), 0.5, dtype=np.float32)
    gpu_uvs = np.zeros((total_verts, 2), dtype=np.float32)

    for i, tri in enumerate(mesh.loop_triangles):
        l_idxs = getattr(tri, "loops", getattr(tri, "loop_indices", None))
        v_idxs = tri.vertices
        
        idx0, idx1, idx2 = i * 3, i * 3 + 1, i * 3 + 2

        gpu_bary[idx0] = (1.0, 0.0, 0.0)
        gpu_bary[idx1] = (0.0, 1.0, 0.0)
        gpu_bary[idx2] = (0.0, 0.0, 1.0)

        gpu_uvs[idx0] = uv_layer[l_idxs[0]].uv
        gpu_uvs[idx1] = uv_layer[l_idxs[1]].uv
        gpu_uvs[idx2] = uv_layer[l_idxs[2]].uv

        e01 = edge_cavity_dict.get(tuple(sorted((v_idxs[0], v_idxs[1]))), 0.5)
        e12 = edge_cavity_dict.get(tuple(sorted((v_idxs[1], v_idxs[2]))), 0.5)
        e20 = edge_cavity_dict.get(tuple(sorted((v_idxs[2], v_idxs[0]))), 0.5)

        cav_vec = (e01, e12, e20)
        gpu_edge_cavity[idx0] = cav_vec
        gpu_edge_cavity[idx1] = cav_vec
        gpu_edge_cavity[idx2] = cav_vec

    shader = get_mesh_cavity_shader()
    batch = batch_for_shader(
        shader, 'TRIS',
        {"bary": gpu_bary, "edge_cavity": gpu_edge_cavity, "uv": gpu_uvs}
    )

    offscreen = gpu.types.GPUOffScreen(width, height, format='RGBA32F')
    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        fb.clear(color=(0.5, 0.5, 0.5, 1.0))
        shader.bind()
        shader.uniform_float("strength", float(strength))
        batch.draw(shader)
        
        buf = fb.read_color(0, 0, width, height, 4, 0, 'FLOAT')
        buf.dimensions = width * height * 4
        out = np.array(buf, dtype=np.float32).reshape(height, width, 4)

    offscreen.free()

    if blur_radius > 0:
        cav_chan = _box_blur(out[..., 0], blur_radius)
        out[..., 0] = cav_chan
        out[..., 1] = cav_chan
        out[..., 2] = cav_chan

    return _postprocess_cavity_channels(out, mode=mode)