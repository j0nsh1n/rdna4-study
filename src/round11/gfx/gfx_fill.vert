#version 450
// Full-target quad (two triangles over the whole viewport): every pixel centre is inside, the shared diagonal is drawn once, so an
// occlusion query over K instances counts exactly K x W x H x samples. K is the instance count of one draw.
layout(push_constant) uniform PC { vec4 color; float z; uint seed; uint u2; uint u3; } pc;
void main() {
    const vec2 c[6] = vec2[6](vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(-1, 1), vec2(1, -1), vec2(1, 1));
    gl_Position = vec4(c[gl_VertexIndex], pc.z, 1.0);
}
