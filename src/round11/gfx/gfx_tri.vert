#version 450
// A grid of right triangles, vertex 0 at (cell * stride + 77/256) pixels, legs `leg` pixels: no coverage ties (see r11g_tri_cover).
// 2/width and 2/height are powers of two for the 2048 target, and every pixel coordinate has at most 19 significant bits, so the
// clip-space values are exact in float and the vertex lands on the 1/256 grid the rasteriser snaps to.
layout(push_constant) uniform PC { float leg; float stride; float invw; float invh; uint gridw; float z; uint u6; uint u7; } pc;
void main() {
    const uint tri = uint(gl_VertexIndex) / 3u, k = uint(gl_VertexIndex) % 3u;
    const vec2 o = vec2(float(tri % pc.gridw), float(tri / pc.gridw)) * pc.stride + vec2(0.30078125);
    const vec2 p = o + (k == 1u ? vec2(pc.leg, 0.0) : (k == 2u ? vec2(0.0, pc.leg) : vec2(0.0)));
    gl_Position = vec4(p * vec2(pc.invw, pc.invh) - 1.0, pc.z, 1.0);
}
