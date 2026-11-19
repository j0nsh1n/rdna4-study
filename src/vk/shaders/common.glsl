// Shared declarations for the lane 9 compute shaders.
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_ARB_shader_clock : require
#extension GL_EXT_shader_realtime_clock : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout(local_size_x_id = 0) in;

// Per subgroup: chunks done, shader-clock cycles (lo, hi), realtime ticks (100 MHz) over the timed part.
layout(set = 0, binding = 1) buffer Info {
    uint counter;
    uint pad0, pad1, pad2;
    uvec4 info[];
};

uint global_subgroup() { return gl_WorkGroupID.x * gl_NumSubgroups + gl_SubgroupID; }

// Work stealing keeps every SIMD busy to the end whatever the driver's placement of subgroups.
uint next_chunk() {
    uint c = 0;
    if (subgroupElect()) c = atomicAdd(counter, 1u);
    return subgroupBroadcastFirst(c);
}

void write_info(uint done, uint64_t c0, uint64_t r0) {
    uint64_t c1 = clockARB();
    uint64_t r1 = clockRealtimeEXT();
    if (subgroupElect()) {
        uint64_t dc = c1 - c0;
        info[global_subgroup()] = uvec4(done, uint(dc), uint(dc >> 32), uint(r1 - r0));
    }
}
