// Shared pieces of the round 11 item 3 shaders. The hash and the per-pixel colour are mirrored in r11_gfx_ref.h (r11g_hash32, r11g_px_hash);
// the runner's --cpu-checks greps both files for the constants, so the incompressible pattern's read-back check is exact.
uint hash32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
// x, y below 16384
uint px_hash(uint x, uint y, uint seed) { return hash32(((y << 14) | x) ^ seed); }
