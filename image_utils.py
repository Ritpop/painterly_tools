import bpy
import numpy as np
import os
import tempfile

def _clamped_shift(arr, dy, dx):
    H, W = arr.shape
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

    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, f"{name}.png")

    temp_img = bpy.data.images.new(
        name="___temp_write___",
        width=width,
        height=height,
        alpha=True,
        float_buffer=is_normal
    )
    temp_img.pixels.foreach_set(rgba_clean.ravel())
    temp_img.filepath_raw = file_path
    temp_img.file_format = 'PNG'
    temp_img.save()
    bpy.data.images.remove(temp_img)

    if name in bpy.data.images:
        final_img = bpy.data.images[name]
        final_img.filepath = file_path
        final_img.reload()
    else:
        final_img = bpy.data.images.load(file_path)
        final_img.name = name

    if is_normal:
        try:
            final_img.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass

    final_img.pack()

    try:
        os.remove(file_path)
    except OSError:
        pass

    return final_img

def _box_blur(arr, r):
    if r < 1:
        return arr
    H, W = arr.shape
    padded = np.pad(arr, r, mode='edge')
    csum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)), mode='constant')
    sz = 2 * r + 1
    total = (csum[sz:sz + H, sz:sz + W]
             - csum[0:H, sz:sz + W]
             - csum[sz:sz + H, 0:W]
             + csum[0:H, 0:W])
    return total / (sz * sz)

def calculate_flow_angles(img_rgba, blur_radius=6):
    rgb = img_rgba[..., :3]
    lum = (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722).astype(np.float32)

    gy, gx = np.gradient(lum)

    Jxx, Jxy, Jyy = gx * gx, gx * gy, gy * gy
    for _ in range(3):
        Jxx = _box_blur(Jxx, blur_radius)
        Jxy = _box_blur(Jxy, blur_radius)
        Jyy = _box_blur(Jyy, blur_radius)

    return (0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy) + (np.pi / 2.0)).astype(np.float32)

def _non_max_suppress(mag, angle):
    angle_deg = np.degrees(angle) % 180.0

    left, right = _clamped_shift(mag, 0, 1), _clamped_shift(mag, 0, -1)
    up, down = _clamped_shift(mag, 1, 0), _clamped_shift(mag, -1, 0)
    diag_a1, diag_a2 = _clamped_shift(mag, 1, -1), _clamped_shift(mag, -1, 1)
    diag_b1, diag_b2 = _clamped_shift(mag, 1, 1), _clamped_shift(mag, -1, -1)

    sector0 = (angle_deg < 22.5) | (angle_deg >= 157.5)
    sector45 = (angle_deg >= 22.5) & (angle_deg < 67.5)
    sector90 = (angle_deg >= 67.5) & (angle_deg < 112.5)
    sector135 = (angle_deg >= 112.5) & (angle_deg < 157.5)

    is_max = np.zeros_like(mag, dtype=bool)
    is_max |= sector0 & (mag >= left) & (mag >= right)
    is_max |= sector90 & (mag >= up) & (mag >= down)
    is_max |= sector45 & (mag >= diag_a1) & (mag >= diag_a2)
    is_max |= sector135 & (mag >= diag_b1) & (mag >= diag_b2)

    return np.where(is_max, mag, 0.0)

def _hysteresis_link(strong, weak, max_iters=24):
    final = strong.copy()
    for _ in range(max_iters):
        dilated = final.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                dilated |= _clamped_shift(final, dy, dx)
        new_final = final | (dilated & weak)
        if np.array_equal(new_final, final):
            return new_final
        final = new_final
    return final

def build_edge_texture_rgba(img_rgba, edge_blur=1, edge_threshold=0.15, edge_contrast=1.5,
                            canny_max_dim=768, canny_low_ratio=0.4, canny_max_iters=24):
    rgb = img_rgba[..., :3]
    lum = (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722).astype(np.float32)
    H, W = lum.shape

    scale = min(1.0, canny_max_dim / max(H, W))
    if scale < 1.0:
        small_h = max(8, int(round(H * scale)))
        small_w = max(8, int(round(W * scale)))
        small_lum = _resize_nearest(lum, small_h, small_w)
    else:
        small_h, small_w = H, W
        small_lum = lum

    smoothed = small_lum
    for _ in range(2):
        smoothed = _box_blur(smoothed, max(1, edge_blur))

    gy, gx = np.gradient(smoothed)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = mag / (mag.max() + 1e-6)
    angle = np.arctan2(gy, gx)

    nms = _non_max_suppress(mag, angle)

    high = edge_threshold
    low = edge_threshold * canny_low_ratio
    strong = nms >= high
    weak = (nms >= low) & (nms < high)
    linked = _hysteresis_link(strong, weak, max_iters=canny_max_iters)

    mask_small = linked.astype(np.float32)
    mask_small = _box_blur(mask_small, max(1, edge_blur))
    mask_small = np.clip(mask_small * 1.6, 0.0, 1.0)
    mask_small = np.power(mask_small, max(edge_contrast, 1e-3))

    if scale < 1.0:
        mask = _resize_nearest(mask_small, H, W)
        angle_full = _resize_nearest(angle, H, W)
    else:
        mask = mask_small
        angle_full = angle

    tangent_angle = angle_full + (np.pi / 2.0)
    r = np.cos(2.0 * tangent_angle) * 0.5 + 0.5
    g = np.sin(2.0 * tangent_angle) * 0.5 + 0.5
    b = np.clip(mask, 0.0, 1.0)
    a = np.ones_like(r)
    return np.ascontiguousarray(np.stack([r, g, b, a], axis=-1), dtype=np.float32)

def _flow_direction_vector(img_rgba, blur_radius):
    angles = calculate_flow_angles(img_rgba, blur_radius=blur_radius)
    return np.cos(2.0 * angles), np.sin(2.0 * angles)

def build_flow_texture_rgba(img_rgba, blur_radius, scale_blend=0.0, coarse_multiplier=3.0):
    r, g = _flow_direction_vector(img_rgba, blur_radius)

    if scale_blend > 0.0:
        coarse_radius = max(1, int(round(blur_radius * coarse_multiplier)))
        r2, g2 = _flow_direction_vector(img_rgba, coarse_radius)
        r = r * (1.0 - scale_blend) + r2 * scale_blend
        g = g * (1.0 - scale_blend) + g2 * scale_blend
        norm = np.sqrt(r ** 2 + g ** 2) + 1e-6
        r, g = r / norm, g / norm

    r_enc = r * 0.5 + 0.5
    g_enc = g * 0.5 + 0.5
    b = np.zeros_like(r_enc)
    a = np.ones_like(r_enc)
    return np.ascontiguousarray(np.stack([r_enc, g_enc, b, a], axis=-1), dtype=np.float32)