// Shared pieces of the round 11 ray-query shaders (item 1). The hash and the ray lattice are mirrored in r11_rt_ref.h; the
// CPU test checks the two agree, so the expected hit counts of the rate cells are exact, not statistical.
#extension GL_EXT_ray_query : require
#extension GL_ARB_shader_clock : require
#extension GL_EXT_shader_realtime_clock : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(set = 0, binding = 0) uniform accelerationStructureEXT tlas;

// A 32-bit integer hash (lowbias32); the C copy in r11_rt_ref.h must stay identical.
uint hash32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
