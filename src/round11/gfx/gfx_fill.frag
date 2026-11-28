#version 450
#extension GL_GOOGLE_include_directive : require
#include "gfx_common.glsl"
// MODE 0: one constant colour (compressible). MODE 1: a per-pixel pseudorandom colour (not compressible): the four bytes of px_hash / 255,
// which an 8-bit UNORM target stores back as exactly those bytes.
layout(constant_id = 0) const int MODE = 0;
layout(push_constant) uniform PC { vec4 color; float z; uint seed; uint u2; uint u3; } pc;
layout(location = 0) out vec4 o;
void main() {
    if (MODE == 0) {
        o = pc.color;
    } else {
        const uvec2 p = uvec2(gl_FragCoord.xy);
        const uint h = px_hash(p.x, p.y, pc.seed);
        o = vec4(uvec4(h & 255u, (h >> 8) & 255u, (h >> 16) & 255u, h >> 24)) * (1.0 / 255.0);
    }
}
