#version 450
// P2: a pixel shader of adjustable ALU length; the output depends on the pixel (a hash), so it does not compress to a
// constant. Two dependent FMA chains per pixel, bounded values.
layout(push_constant) uniform PC { uint iters; } pc;
layout(location = 0) out vec4 o;
void main() {
    vec2 p = gl_FragCoord.xy;
    float a = fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
    float b = a * 0.5 + 0.25;
    for (uint i = 0u; i < pc.iters; i++) {
        a = fma(a, b, 0.25);
        b = fma(b, a, 0.1);
        a = a * 0.5;
        b = b * 0.5;
    }
    o = vec4(a, b, a * b, 1.0);
}
