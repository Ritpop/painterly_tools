import gpu

VERT_SHADER = """
void main()
{
    v_uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

FRAG_SHADER = """
vec2 hash22(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p + seed) * 43758.5453123);
}

float hash21(vec2 p) {
    return fract(sin(dot(p, vec2(41.3, 289.1)) + seed) * 43758.5453123);
}

vec2 decodeFlow(vec2 enc) {
    return vec2(enc.x * 2.0 - 1.0, enc.y * 2.0 - 1.0);
}

void main() {
    vec2 fragPixel = v_uv * resolution;
    float cell = max(2.0, cellSize / max(density, 0.05));

    vec3 baseColor = texture(prevTex, v_uv).rgb;
    vec3 baseNormal = texture(prevTex, v_uv).rgb;

    if (outputMode == 3) {
        vec2 texel = 1.0 / resolution;
        vec3 m[4];
        vec3 s[4];
        float counts[4];
        
        for (int k = 0; k < 4; ++k) {
            m[k] = vec3(0.0);
            s[k] = vec3(0.0);
            counts[k] = 0.0;
        }
        
        for (int x = -8; x <= 8; x++) {
            for (int y = -8; y <= 8; y++) {
                if (abs(x) > kuwRadius || abs(y) > kuwRadius) continue;
                
                vec3 c = texture(srcTex, v_uv + vec2(float(x), float(y)) * texel).rgb;
                int i = 0;
                if (x > 0 && y <= 0) i = 1;
                else if (x <= 0 && y > 0) i = 2;
                else if (x > 0 && y > 0) i = 3;
                
                m[i] += c;
                s[i] += c * c;
                counts[i] += 1.0;
            }
        }
        
        float min_var = 1e10;
        vec3 bestCol = texture(srcTex, v_uv).rgb;
        for (int i = 0; i < 4; ++i) {
            if (counts[i] > 0.0) {
                vec3 mean = m[i] / counts[i];
                vec3 variance = abs(s[i] / counts[i] - mean * mean);
                float var_sum = variance.r + variance.g + variance.b;
                if (var_sum < min_var) {
                    min_var = var_sum;
                    bestCol = mean;
                }
            }
        }
        fragColor = vec4(bestCol, 1.0);
        return;
    }

    float bestMask = 0.0;
    vec2 bestCenterUV = v_uv;
    vec2 bestBump = vec2(0.0);
    float bestPressure = 1.0;
    float bestDry = 0.0;

    vec2 cellCoord = fragPixel / cell;
    vec2 baseCell = floor(cellCoord);

    for (int oy = -1; oy <= 1; oy++) {
        for (int ox = -1; ox <= 1; ox++) {
            vec2 cid = baseCell + vec2(float(ox), float(oy));

            vec2 j = hash22(cid) * 2.0 - 1.0;
            vec2 center = (cid + 0.5 + j * cellJitter) * cell;
            vec2 centerUV = center / resolution;
            if (centerUV.x < 0.0 || centerUV.x > 1.0 || centerUV.y < 0.0 || centerUV.y > 1.0)
                continue;

            vec2 flowEnc = texture(flowTex, centerUV).rg;
            vec3 edgeSample = texture(edgeTex, centerUV).rgb;

            if (edgeGateDensity == 1 && edgeSample.b < edgeDensityThreshold)
                continue;

            vec2 flowVec = decodeFlow(flowEnc);
            vec2 edgeVec = decodeFlow(edgeSample.rg);
            float edgeW = clamp(edgeSample.b * edgeInfluence, 0.0, 1.0);
            vec2 dirVec = mix(flowVec, edgeVec, edgeW);
            float dirLen = length(dirVec);
            if (dirLen > 1e-5) dirVec /= dirLen;
            float angle = 0.5 * atan(dirVec.y, dirVec.x);
            float ca = cos(angle), sa = sin(angle);

            vec2 d = fragPixel - center;
            vec2 local = vec2(d.x * ca + d.y * sa, -d.x * sa + d.y * ca);

            float rWidth = 0.6 + 0.8 * hash21(cid + 11.1);
            float rLength = 0.5 + 1.0 * hash21(cid + 12.2);

            float halfLen = cell * 0.5 * elongate * rLength;
            float halfWid = cell * 0.32 * rWidth;

            float bendDir = hash21(cid + 5.21) * 2.0 - 1.0;
            float bend = strokeCurvature * bendDir * (local.x * local.x) / max(halfLen * halfLen, 1e-4);
            float bentV = local.y + bend * halfWid;

            float u = local.x / max(halfLen, 1e-4);
            float v = bentV / max(halfWid, 1e-4);

            float pressOffset = (hash21(cid + 1.7) - 0.5) * 0.3;
            float uNorm = clamp(u * 0.5 + 0.5 + pressOffset, 0.0, 1.0);
            float pressure = sin(uNorm * 3.14159265);
            pressure = clamp(pressure, 0.0, 1.0);

            float rTaper = 0.2 + 1.0 * hash21(cid + 13.3);
            float currentTaper = clamp(strokeTaper * rTaper, 0.0, 1.0);

            float taperFactor = mix(1.0, mix(0.06, 1.0, pressure), currentTaper);
            float vTapered = v / max(taperFactor, 1e-4);

            float widthMask  = 1.0 - smoothstep(0.45, 1.0, abs(vTapered));
            float lengthMask = 1.0 - smoothstep(0.80, 1.0, abs(u));
            float mask = widthMask * lengthMask;
            if (mask <= 0.001) continue;

            float rDry = 0.2 + 1.2 * hash21(cid + 14.4);
            float currentDry = clamp(dryBrush * rDry, 0.0, 1.0);

            float dryPhase = u * 14.0 + hash21(cid + 2.3) * 6.28318;
            float dry1 = sin(dryPhase) * 0.5 + 0.5;
            float dry2 = sin(dryPhase * 2.7 + 1.3) * 0.5 + 0.5;
            float dryPattern = dry1 * dry2;
            dryPattern = smoothstep(0.10, 0.70, dryPattern);
            mask *= mix(1.0, dryPattern, currentDry);

            float freq = 40.0 * (0.35 + bristleDetail);
            float bristlePhase = local.y / cell * freq;
            float bristle = 0.5 + 0.5 * sin(bristlePhase);
            mask *= mix(1.0, bristle, bristleDetail);

            mask *= mix(0.25, 1.0, pressure);

            if (mask > bestMask) {
                bestMask = mask;
                bestCenterUV = centerUV;
                bestPressure = pressure;
                bestDry = currentDry;

                float randAngle = hash21(cid + 9.1) * 6.28318;
                float tiltMag = sin(strokeTilt);
                vec2 tiltLocal = vec2(tiltMag * cos(randAngle), tiltMag * sin(randAngle));
                
                bestBump = vec2(tiltLocal.x * ca - tiltLocal.y * sa,
                                tiltLocal.x * sa + tiltLocal.y * ca);
            }
        }
    }

    float finalOpacity = opacity * bestMask;

    if (outputMode == 0) {
        vec3 srcCol = texture(srcTex, bestCenterUV).rgb;
        vec3 canvasCol = texture(prevTex, bestCenterUV).rgb;
        vec3 col = mix(srcCol, canvasCol, wetBlend);

        vec2 jitter = (hash22(bestCenterUV * resolution * 0.37) * 2.0 - 1.0) * colorJitter;
        col = clamp(col + vec3(jitter, jitter.x * 0.5), 0.0, 1.0);
        fragColor = vec4(mix(baseColor, col, finalOpacity), 1.0);

    } else if (outputMode == 1) {
        if (paintsNormal == 0) {
            fragColor = vec4(baseNormal, 1.0);
        } else {
            vec3 n = vec3(bestBump * normalStrength, 1.0);
            n = normalize(n);

            vec3 nBase = baseNormal * 2.0 - 1.0;
            nBase.z = max(nBase.z, 0.1); 
            
            vec3 t = nBase + vec3(0.0, 0.0, 1.0);
            vec3 u_rnm = n * vec3(-1.0, -1.0, 1.0);
            vec3 r = t * dot(t, u_rnm) / max(t.z, 1e-4) - u_rnm;
            r = normalize(r);
            vec3 rRemap = r * 0.5 + 0.5;

            float hardMask = smoothstep(0.5 - facetHardness * 0.5, 0.5 + facetHardness * 0.5, bestMask);
            float facetOpacity = opacity * hardMask;

            vec3 blended = mix(baseNormal, rRemap, facetOpacity);
            fragColor = vec4(blended, 1.0);
        }
    } else {
        vec3 col = texture(srcTex, bestCenterUV).rgb;
        vec2 jitter = (hash22(bestCenterUV * resolution * 0.37) * 2.0 - 1.0) * colorJitter;
        col = clamp(col + vec3(jitter, jitter.x * 0.5), 0.0, 1.0);
        fragColor = vec4(col, clamp(finalOpacity, 0.0, 1.0));
    }
}
"""

_shader_cache = None

def get_painterly_shader():
    global _shader_cache
    if _shader_cache is not None:
        return _shader_cache

    vert_out = gpu.types.GPUStageInterfaceInfo("painterly_iface")
    vert_out.smooth('VEC2', "v_uv")

    info = gpu.types.GPUShaderCreateInfo()
    info.sampler(0, 'FLOAT_2D', "srcTex")
    info.sampler(1, 'FLOAT_2D', "flowTex")
    info.sampler(2, 'FLOAT_2D', "prevTex")
    info.sampler(3, 'FLOAT_2D', "edgeTex")
    info.push_constant('FLOAT', "edgeInfluence")
    info.push_constant('VEC2', "resolution")
    info.push_constant('FLOAT', "cellSize")
    info.push_constant('FLOAT', "opacity")
    info.push_constant('FLOAT', "elongate")
    info.push_constant('FLOAT', "density")
    info.push_constant('FLOAT', "bristleDetail")
    info.push_constant('FLOAT', "colorJitter")
    info.push_constant('FLOAT', "strokeTilt")
    info.push_constant('FLOAT', "strokeCurvature")
    info.push_constant('FLOAT', "normalStrength")
    info.push_constant('FLOAT', "seed")
    info.push_constant('FLOAT', "cellJitter")
    info.push_constant('FLOAT', "strokeTaper")
    info.push_constant('FLOAT', "dryBrush")
    info.push_constant('FLOAT', "wetBlend")
    info.push_constant('FLOAT', "facetHardness")
    info.push_constant('INT', "kuwRadius")
    info.push_constant('INT', "edgeGateDensity")
    info.push_constant('FLOAT', "edgeDensityThreshold")
    info.push_constant('INT', "paintsNormal")
    info.push_constant('INT', "outputMode")
    info.vertex_in(0, 'VEC2', "pos")
    info.vertex_out(vert_out)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(VERT_SHADER)
    info.fragment_source(FRAG_SHADER)

    _shader_cache = gpu.shader.create_from_info(info)
    return _shader_cache