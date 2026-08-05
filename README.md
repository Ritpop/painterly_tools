# Painterly Brush & Directional Normal Generator (Blender Addon)

A GPU-accelerated texture baking engine for Blender 4.0+. Converts flat or procedural albedo images into painterly brush stroke textures and directional impasto normal maps.

## Features
- **GPU Shader Baking:** Uses offscreen rendering buffers to process high-resolution textures in seconds.
- **Kuwahara Filtering:** Edge-preserving underpaint pass guarantees seamless base layer coverage.
- **Edge Tangent Flow:** Stroke orientation follows image contours and Sobel edge vectors.
- **Toon Shader Setup:** Automatic material creation supporting EEVEE and Cycles.

## Installation
1. Download the repo as a `.zip`.
2. Open Blender > `Edit` > `Preferences` > `Add-ons`.
3. Click the drop down menu on the right and then on `Install from disk...` and select the zip file.
4. Enable **Painterly Brush & Normal Map Generator**.