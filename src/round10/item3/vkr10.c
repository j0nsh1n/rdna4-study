// Round 10, item 3: why does Vulkan read untouched memory more slowly than HIP? The Vulkan half of the two-way bisect.
// Built on the same harness as round 9's vkr9 (src/vk/vkgame.c is included, so the device set-up, the 5 ms read-only
// sysfs sampler, buffers and pipelines are the ones every earlier probe used; nothing in src/vk or src/round9 is edited).
// The one addition to the device: buffer device addresses (bufferDeviceAddress) are switched on by wrapping vkCreateDevice,
// because the 64-bit pointer-chase form needs them.
//
//   info                    memory heaps and types, the type chosen for every kind of placement, then every pipeline of the item is
//                           built and its machine code written to $VKG_OUT/asm (no dispatch; under a second of device use)
//   bis <spec>...           one cell per spec (the cold chase of round 9 item C/C2 with one factor changed)
//   reach <spec>            the translation-reach test: N 4 KiB pages walked twice (second pass warm)
//
// bis keys (defaults are round 9's cold cell: the Vulkan side of vkr9 cbar order=cold):
//   name=  q=c|g (queue, default c)  form=u32|p64|p64v  walk=random|page  order=cold|fused|noevict|fresh  reps=20  steps=1024
//   alloc=sep|one|sub|ded   sep: one buffer and one memory per region (round 9); one: both regions in ONE 8 MiB buffer (HIP's
//        hipMalloc); sub: buffers bound at 2 MiB-aligned offsets inside one 512 MiB allocation; ded: dedicated allocations
//   mem=vram|bar|gtt        vram: device-local, not host-visible (round 9); bar: device-local and host-visible; gtt: host-cached system
//   evict=256|512|0         MiB read by the evict dispatch every repetition (0: none)
//   sg=32|64                the required subgroup size of the chase pipeline (round 9: 32)
//   coh=0|1                 (form p64/p64v) the pointer block is `coherent`
//   stride=<bytes>          (form p64/p64v) one cyclic chain of `steps` nodes, one node per stride (page order), instead of the 4 MiB walk
//   loop=<slots>            the walk wraps inside this many 128-byte lines (power of two): the footprint control (default 32768)
//   warm=150                ms of r9_spin before the first repetition (round 9 default; 0 = none)
// order: cold = the region was written once before the loop and every repetition's evict pushes it out (round 9);
//   fused = one kernel writes and chases (cache-hit control); noevict = cold without any eviction (the deliberate break:
//   must read at a cache level); fresh = each repetition chases a region never read before (one evict before the first).
// form: u32 = round 9's chase (4-byte loads, mask, stamp sum, index arithmetic; r9_chase.comp unchanged);
//   p64 = 64-bit absolute addresses, nothing else in the loop (HIP's form); p64v = p64 with `lane & zero` added to the address.
// Output (stdout): INFO/CELL/ROW/RESULT/VERDICT lines, one ROW per repetition, with the memory clock read before and after.
#define _GNU_SOURCE
#include <vulkan/vulkan.h>

static VkResult r10_create_device(VkPhysicalDevice pd, const VkDeviceCreateInfo *ci, const VkAllocationCallbacks *a,
                                  VkDevice *d) {
    for (const VkBaseInStructure *s = (const VkBaseInStructure *)ci->pNext; s; s = s->pNext)
        if (s->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
            ((VkPhysicalDeviceVulkan12Features *)(void *)s)->bufferDeviceAddress = VK_TRUE;
    return vkCreateDevice(pd, ci, a, d);
}
#define vkCreateDevice r10_create_device
#define main vkgame_original_main
#include "../../vk/vkgame.c"
#undef main
#undef vkCreateDevice

#define NSLOT 32768u
#define REG_BYTES (NSLOT * 128u)
#define LCG_A 1664525u
#define MAXREC 1024u
#define MIB (1ull << 20)

enum { F_U32, F_P64, F_P64V };
enum { O_COLD, O_FUSED, O_NOEVICT, O_FRESH };
enum { A_SEP, A_ONE, A_SUB, A_DED };
enum { M_VRAM, M_BAR, M_GTT };

typedef struct {
    char name[64];
    int q, form, order, page, reps, alloc, mem, ev_mib, sg, coh;
    uint32_t steps, stride, loop;
    double warm;
} Cfg;

static int uclk_now(void) {
    char p[600];
    snprintf(p, sizeof p, "%s/gpu_metrics", g_drm);
    FILE *f = fopen(p, "rb");
    if (!f) return -1;
    unsigned char b[64];
    size_t n = fread(b, 1, sizeof b, f);
    fclose(f);
    return n >= 60 ? (int)(b[58] | (b[59] << 8)) : -1;
}

static int cmp_dbl(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static double med(const double *v, int n) {
    if (n <= 0) return NAN;
    double *c = malloc((size_t)n * sizeof *c);
    memcpy(c, v, (size_t)n * sizeof *c);
    qsort(c, (size_t)n, sizeof *c, cmp_dbl);
    double m = n % 2 ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
    free(c);
    return m;
}

// ---------------------------------------------------------------------------------------------------------------
// memory: explicit control of the type, of dedicated against suballocated, and of the buffer's device address

typedef struct {
    VkBuffer b;
    VkDeviceMemory m;
    VkDeviceSize off, size;
    VkDeviceAddress addr;
    void *map;
    int own_buf, own_mem;
} Reg;

static uint32_t pick_type(uint32_t bits, int mem) {
    static const struct {
        VkMemoryPropertyFlags want, avoid;
    } pref[3][2] = {
        {{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT}, {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0}},
        {{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0}, {0, UINT32_MAX}},
        {{VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT},
         {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT}}};
    for (int k = 0; k < 2; k++) {
        if (pref[mem][k].avoid == UINT32_MAX) continue;
        for (uint32_t i = 0; i < g_mp.memoryTypeCount; i++) {
            VkMemoryPropertyFlags f = g_mp.memoryTypes[i].propertyFlags;
            if ((bits & (1u << i)) && (f & pref[mem][k].want) == pref[mem][k].want && !(f & pref[mem][k].avoid)) return i;
        }
    }
    return UINT32_MAX;
}

static VkBuffer new_buffer(VkDeviceSize size) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBuffer b;
    VK_CHECK(vkCreateBuffer(g_dev, &bi, NULL, &b));
    return b;
}

static VkDeviceMemory new_memory(VkDeviceSize size, uint32_t type, VkBuffer ded) {
    VkMemoryAllocateFlagsInfo fi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryDedicatedAllocateInfo di = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    di.buffer = ded;
    if (ded) fi.pNext = &di;
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &fi};
    ai.allocationSize = size;
    ai.memoryTypeIndex = type;
    VkDeviceMemory m;
    VK_CHECK(vkAllocateMemory(g_dev, &ai, NULL, &m));
    return m;
}

static VkDeviceAddress addr_of(VkBuffer b) {
    VkBufferDeviceAddressInfo ai = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    ai.buffer = b;
    return vkGetBufferDeviceAddress(g_dev, &ai);
}

// One buffer with its own memory (dedicated: the memory is tied to the buffer by VkMemoryDedicatedAllocateInfo).
static int reg_alloc(Reg *r, VkDeviceSize size, int mem, int ded) {
    memset(r, 0, sizeof *r);
    r->b = new_buffer(size);
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, r->b, &mr);
    const uint32_t t = pick_type(mr.memoryTypeBits, mem);
    if (t == UINT32_MAX) {
        vkDestroyBuffer(g_dev, r->b, NULL);
        memset(r, 0, sizeof *r);
        return 0;
    }
    r->m = new_memory(mr.size, t, ded ? r->b : VK_NULL_HANDLE);
    VK_CHECK(vkBindBufferMemory(g_dev, r->b, r->m, 0));
    if (mem != M_VRAM) VK_CHECK(vkMapMemory(g_dev, r->m, 0, VK_WHOLE_SIZE, 0, &r->map));
    r->size = size;
    r->addr = addr_of(r->b);
    r->own_buf = r->own_mem = 1;
    return 1;
}

static void reg_free(Reg *r) {
    if (r->map) vkUnmapMemory(g_dev, r->m);
    if (r->own_buf && r->b) vkDestroyBuffer(g_dev, r->b, NULL);
    if (r->own_mem && r->m) vkFreeMemory(g_dev, r->m, NULL);
    memset(r, 0, sizeof *r);
}

// n slots of `sz` bytes (slot 0 is round 9's `ra`, the cold chases use slots 1..). Returns 0 if the memory type does not exist.
static int make_regions(const Cfg *c, Reg *R, int n, VkDeviceSize sz) {
    memset(R, 0, (size_t)n * sizeof *R);
    if (c->alloc == A_SEP || c->alloc == A_DED) {
        for (int i = 0; i < n; i++)
            if (!reg_alloc(&R[i], sz, c->mem, c->alloc == A_DED)) return 0;
        return 1;
    }
    if (c->alloc == A_ONE) {
        Reg big;
        if (!reg_alloc(&big, sz * (VkDeviceSize)n, c->mem, 0)) return 0;
        for (int i = 0; i < n; i++) {
            R[i] = big;
            R[i].off = sz * (VkDeviceSize)i;
            R[i].addr = big.addr + R[i].off;
            R[i].own_buf = R[i].own_mem = (i == 0);
            if (i) R[i].map = big.map ? (char *)big.map + R[i].off : NULL;
        }
        R[0].size = sz;
        return 1;
    }
    // sub: buffers bound at 2 MiB-aligned offsets inside one allocation of at least 512 MiB (like HIP's one big hipMalloc)
    const VkDeviceSize step = (sz + 2 * MIB - 1) / (2 * MIB) * (2 * MIB), first = 128 * MIB;
    VkDeviceSize total = first + step * (VkDeviceSize)n + 128 * MIB;
    if (total < 512 * MIB) total = 512 * MIB;
    uint32_t bits = ~0u;
    for (int i = 0; i < n; i++) {
        R[i].b = new_buffer(sz);
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(g_dev, R[i].b, &mr);
        bits &= mr.memoryTypeBits;
        if (mr.alignment > 2 * MIB) return 0;
    }
    const uint32_t t = pick_type(bits, c->mem);
    if (t == UINT32_MAX) return 0;
    VkDeviceMemory m = new_memory(total, t, VK_NULL_HANDLE);
    void *map = NULL;
    if (c->mem != M_VRAM) VK_CHECK(vkMapMemory(g_dev, m, 0, VK_WHOLE_SIZE, 0, &map));
    for (int i = 0; i < n; i++) {
        VK_CHECK(vkBindBufferMemory(g_dev, R[i].b, m, first + step * (VkDeviceSize)i));
        R[i].m = m;
        R[i].size = sz;
        R[i].addr = addr_of(R[i].b);
        R[i].own_buf = 1;
        R[i].own_mem = (i == 0);
        R[i].map = i == 0 ? map : NULL;
    }
    return 1;
}

static void free_regions(Reg *R, int n) {
    for (int i = n - 1; i >= 0; i--) {
        if (R[i].own_buf == 0 && R[i].b) {  // a view into the first region's buffer or memory: nothing of its own to destroy
            memset(&R[i], 0, sizeof R[i]);
            continue;
        }
        reg_free(&R[i]);
    }
}

// A descriptor set with b0 at (off0, range0) and the ordinary round 9 buffers on bindings 1 and 2.
static VkDescriptorSet mkset_o(VkBuffer b0, VkDeviceSize off0, VkDeviceSize range0, Buf *b1, Buf *b2) {
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = g_dpool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &g_dsl;
    VkDescriptorSet ds;
    VK_CHECK(vkAllocateDescriptorSets(g_dev, &dai, &ds));
    VkDescriptorBufferInfo bi[3];
    VkWriteDescriptorSet w[3];
    int n = 0;
    const struct {
        VkBuffer b;
        VkDeviceSize off, rng;
        uint32_t bind;
    } s[3] = {{b0, off0, range0, 0}, {b1 ? b1->b : VK_NULL_HANDLE, 0, VK_WHOLE_SIZE, 1}, {b2 ? b2->b : VK_NULL_HANDLE, 0, VK_WHOLE_SIZE, 2}};
    for (int i = 0; i < 3; i++) {
        if (!s[i].b) continue;
        bi[n] = (VkDescriptorBufferInfo){s[i].b, s[i].off, s[i].rng};
        w[n] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[n].dstSet = ds;
        w[n].dstBinding = s[i].bind;
        w[n].descriptorCount = 1;
        w[n].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[n].pBufferInfo = &bi[n];
        n++;
    }
    vkUpdateDescriptorSets(g_dev, (uint32_t)n, w, 0, NULL);
    return ds;
}

static void dispatch(VkCommandBuffer cb, const Pipe *p, VkDescriptorSet ds, const void *pc, uint32_t npc, uint32_t groups) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p->p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_cpl, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(cb, g_cpl, VK_SHADER_STAGE_COMPUTE_BIT, 0, npc, pc);
    vkCmdDispatch(cb, groups, 1, 1);
}

static void b2_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cb, &di);
}

// ---------------------------------------------------------------------------------------------------------------
// pipelines (named by variant so every machine-code listing has its own file)

static Pipe mk(const char *spv, const char *label, uint32_t local, uint32_t sg) { return mkcomp(spv, label, local, sg); }

static Pipe chase_pipe(const Cfg *c) {
    char lab[96];
    const uint32_t local = (uint32_t)c->sg;
    if (c->form == F_U32) {
        snprintf(lab, sizeof lab, "r10_chase32_sg%d", c->sg);
        return mk("r9_chase.spv", lab, local, local);
    }
    const char *spv = c->coh ? "r10_chase64c.spv" : c->form == F_P64V ? "r10_chase64v.spv" : "r10_chase64.spv";
    snprintf(lab, sizeof lab, "r10_chase64_%s%s_sg%d", c->form == F_P64V ? "v" : "n", c->coh ? "_coh" : "", c->sg);
    if (c->coh && c->form == F_P64V) {
        printf("VERDICT vkr10 spec: FAIL coh=1 exists only for form=p64 (r10_chase64c.spv has no LANEV)\n");
        g_fail++;
        exit(2);
    }
    return mk(spv, lab, local, local);
}

static Pipe warm_pipe;
static int warm_made;
// round 9 vkr9's warm_up: r9_spin for `ms`, so the core clock has left idle when the first repetition starts (records unused)
static void warm_up(Qu *q, double ms) {
    if (ms <= 0) return;
    const uint32_t groups = 256, local = 64, wpg = local / 32, waves = groups * wpg;
    if (!warm_made) warm_pipe = mk("r9_spin.spv", "r9_spin", local, 32), warm_made = 1;
    Buf res = mkbuf((size_t)groups * local * 4, 0), rec = mkbuf(16u * (MAXREC + 2), 2), grp = mkbuf((size_t)waves * 32, 2);
    VkDescriptorSet ds = mkset(&res, &rec, &grp, NULL);
    VkCommandBuffer c = cb_alloc(q);
    VkFence f = mkfence();
    const uint64_t lim = (uint64_t)llround(ms * 1e5);
    const uint32_t pc[8] = {(uint32_t)lim, (uint32_t)(lim >> 32), MAXREC, 0u, 0u, wpg, 4096u, 0u};
    cb_begin(c);
    dispatch(c, &warm_pipe, ds, pc, sizeof pc, groups);
    VK_CHECK(vkEndCommandBuffer(c));
    submit(q, c, f);
    fwait(f);
    vkDestroyFence(g_dev, f, NULL);
    vkFreeCommandBuffers(g_dev, q->pool, 1, &c);
    vkFreeDescriptorSets(g_dev, g_dpool, 1, &ds);
    rmbuf(&res), rmbuf(&rec), rmbuf(&grp);
}

static uint32_t stamp_of(uint32_t salt) { return (salt >> 15) | 1u; }
static uint32_t lcg_final(uint32_t a, uint32_t start, uint32_t steps, uint32_t c, uint32_t mask) {
    uint32_t p = start;
    for (uint32_t i = 0; i < steps; i++) p = (a * p + c) & mask;
    return p;
}

static const char *const FORM_N[] = {"u32", "p64", "p64v"}, *const ORD_N[] = {"cold", "fused", "noevict", "fresh"},
                  *const ALLOC_N[] = {"sep", "one", "sub", "ded"}, *const MEM_N[] = {"vram", "bar", "gtt"};

static int idx_of(const char *s, const char *const *names, int n) {
    for (int i = 0; i < n; i++)
        if (!strcmp(s, names[i])) return i;
    return -1;
}

static int wait_cool(const char *name) {
    for (int w = 0; hotspot() >= 58.0; w++) {
        if (g_hot_stop || w > 240) {
            printf("NOT RUN %s: hotspot %.0f C (a cell starts below 58 C)\n", name, hotspot());
            return 0;
        }
        until_ns(mono_ns() + 500e6);
    }
    return 1;
}

// ---------------------------------------------------------------------------------------------------------------
// bis: one cell

static int run_bis(const char *spec) {
    Cfg c = {0};
    char s[64], walk[16];
    kvs(spec, "name", "bis", c.name, sizeof c.name);
    kvs(spec, "q", "c", s, sizeof s);
    c.q = !strcmp(s, "c");
    kvs(spec, "form", "u32", s, sizeof s);
    c.form = idx_of(s, FORM_N, 3);
    kvs(spec, "order", "cold", s, sizeof s);
    c.order = idx_of(s, ORD_N, 4);
    kvs(spec, "walk", "random", walk, sizeof walk);
    c.page = !strcmp(walk, "page");
    kvs(spec, "alloc", "sep", s, sizeof s);
    c.alloc = idx_of(s, ALLOC_N, 4);
    kvs(spec, "mem", "vram", s, sizeof s);
    c.mem = idx_of(s, MEM_N, 3);
    c.reps = (int)kvd(spec, "reps", 20);
    c.steps = (uint32_t)kvd(spec, "steps", 1024);
    c.ev_mib = (int)kvd(spec, "evict", 256);
    c.sg = (int)kvd(spec, "sg", 32);
    c.coh = (int)kvd(spec, "coh", 0);
    c.stride = (uint32_t)kvd(spec, "stride", 0);
    c.loop = (uint32_t)kvd(spec, "loop", NSLOT);
    c.warm = kvd(spec, "warm", 150);
    Qu *q = &QU[c.q];
    const int p64 = c.form != F_U32;
    if (!q->valid || c.form < 0 || c.order < 0 || c.alloc < 0 || c.mem < 0 || (strcmp(walk, "random") && !c.page) ||
        c.reps < 2 || c.reps > 400 || c.steps < 16 || c.steps > NSLOT || (c.ev_mib != 0 && c.ev_mib != 256 && c.ev_mib != 512) ||
        (c.sg != 32 && c.sg != 64) || c.loop < 8 || c.loop > NSLOT || (c.loop & (c.loop - 1)) || c.warm < 0 || c.warm > 500 ||
        (c.stride && (!p64 || c.stride % 128 || c.stride > 2u * MIB || c.order == O_FUSED)) ||
        (c.order == O_FUSED && c.sg != 32) || (c.coh && !p64) || (c.order == O_NOEVICT && c.ev_mib)) {
        printf("VERDICT vkr10 bis_spec: FAIL bad cell %s\n", spec);
        g_fail++;
        return 1;
    }
    if (g_hot_stop || !wait_cool(c.name)) return 1;
    const int fused = c.order == O_FUSED, fresh = c.order == O_FRESH;
    const int nreg = fresh ? c.reps : 1;  // cold regions (slots 1..nreg); slot 0 is round 9's ra, used by the fused cell
    const uint32_t mask = c.loop - 1;
    const uint32_t count = c.stride ? c.steps : 0;
    const VkDeviceSize regsize = c.stride ? (VkDeviceSize)c.stride * count > REG_BYTES ? (VkDeviceSize)c.stride * count : REG_BYTES : REG_BYTES;
    Reg R[402];
    if (!make_regions(&c, R, nreg + 1, regsize)) {
        printf("NOT RUN %s: no memory type or alignment for mem=%s alloc=%s\n", c.name, MEM_N[c.mem], ALLOC_N[c.alloc]);
        free_regions(R, nreg + 1);
        return 1;
    }
    Buf ev = {0}, sink = mkbuf(4096, 0);
    if (c.ev_mib) ev = mkbuf((VkDeviceSize)c.ev_mib * MIB, 0);
    Buf out = mkbuf(32u * (size_t)c.reps, 2);
    // the descriptor pool of the harness holds 16 sets: a region's set exists only while a command buffer uses it
    const VkDeviceSize rng = REG_BYTES < regsize ? regsize : REG_BYTES;
    VkDescriptorSet dse = mkset(c.ev_mib ? &ev : NULL, NULL, &sink, NULL);
    VkCommandBuffer cb = cb_alloc(q);
    VkFence fence = mkfence();
    uint32_t *o = out.map;
    const Pipe pev = mk("r9_evict.spv", "r9_evict", 256, 32), pch = chase_pipe(&c);
    Pipe ppr = {0}, pfu = {0}, pcn = {0};
    if (c.stride) pcn = mk("r10_chain.spv", "r10_chain", 256, 32);
    else if (p64) ppr = mk("r10_prod64.spv", "r10_prod64", 256, 32), pfu = mk("r10_fused64.spv", "r10_fused64", 256, 32);
    else ppr = mk("r9_prod.spv", "r9_prod", 256, 32), pfu = mk("r9_fused.spv", "r9_fused", 256, 32);
    const uint32_t cold_a = c.page ? 1u : LCG_A, cold_c = c.page ? 1u : 1013904223u, cold_salt = 0x1234567u;
    // setup: the chains of the cold regions, written once (the fused cell writes its own)
    if (!fused) {
        for (int i = 1; i <= nreg; i++) {
            VkDescriptorSet ds = mkset_o(R[i].b, R[i].off, rng, &out, NULL);
            cb_begin(cb);
            if (c.stride) {
                const struct { uint64_t base; uint32_t stride, count; } pc = {R[i].addr, c.stride, count};
                dispatch(cb, &pcn, ds, &pc, sizeof pc, (count + 255) / 256);
            } else if (p64) {
                const struct { uint64_t base; uint32_t a, c, mask, salt; } pc = {R[i].addr, cold_a, cold_c, mask, cold_salt};
                dispatch(cb, &ppr, ds, &pc, sizeof pc, NSLOT / 8);
            } else {
                const uint32_t pc[4] = {cold_a, cold_c, mask, cold_salt};
                dispatch(cb, &ppr, ds, pc, sizeof pc, NSLOT / 8);
            }
            VK_CHECK(vkEndCommandBuffer(cb));
            submit(q, cb, fence);
            fwait(fence);
            vkFreeDescriptorSets(g_dev, g_dpool, 1, &ds);
        }
        if (fresh && c.ev_mib) {  // one evict after the pool is written, none between repetitions
            const uint32_t pe[2] = {(uint32_t)((uint64_t)c.ev_mib * MIB / 16), 512u * 256u};
            cb_begin(cb);
            dispatch(cb, &pev, dse, pe, sizeof pe, 512);
            VK_CHECK(vkEndCommandBuffer(cb));
            submit(q, cb, fence);
            fwait(fence);
        }
    }
    warm_up(q, c.warm);
    const int uc0 = uclk_now();
    printf("CELL %s api=vulkan q=%s form=%s order=%s walk=%s alloc=%s mem=%s evict=%d sg=%d coh=%d stride=%u loop=%u reps=%d steps=%u "
           "warm_ms=%.0f uclk_start=%d hotspot=%.0f\n",
           c.name, q->name, FORM_N[c.form], ORD_N[c.order], c.page ? "page" : "random", ALLOC_N[c.alloc], MEM_N[c.mem], c.ev_mib,
           c.sg, c.coh, c.stride, c.loop, c.reps, c.steps, c.warm, uc0, hotspot());
    double *cps = calloc((size_t)c.reps, sizeof *cps), *nps = calloc((size_t)c.reps, sizeof *nps), *clk = calloc((size_t)c.reps, sizeof *clk);
    int okn = 0, ncps = 0;
    for (int r = 0; r < c.reps && !g_hot_stop; r++) {
        const int ri = fresh ? r + 1 : 1;
        const Reg *rg = fused ? &R[0] : &R[ri];
        VkDescriptorSet ds = mkset_o(rg->b, rg->off, rng, &out, NULL);
        const uint32_t salt = 0x5bd1e995u * (uint32_t)(r + 1);
        const uint32_t a = c.page ? 1u : LCG_A, cc = c.page ? 1u : 1013904223u + 2u * (uint32_t)r;
        const uint32_t start = c.stride ? (uint32_t)((uint64_t)r * 37u % count) : ((uint32_t)r * 7919u + 1u) & mask;
        const uint32_t fin_slot = c.stride ? (start + c.steps) % count : lcg_final(fused ? a : cold_a, start, c.steps, fused ? cc : cold_c, mask);
        const uint64_t unit = c.stride ? c.stride : 128u;
        const uint64_t start_adr = rg->addr + start * unit, exp_adr = rg->addr + fin_slot * unit;
        for (int k = 0; k < 8; k++) o[8 * r + k] = 0xdeadbeefu;
        const int uc_a = uclk_now();
        cb_begin(cb);
        if (c.ev_mib && !fresh) {
            const uint32_t pe[2] = {(uint32_t)((uint64_t)c.ev_mib * MIB / 16), 512u * 256u};
            dispatch(cb, &pev, dse, pe, sizeof pe, 512);
            b2_barrier(cb);
        }
        if (fused && !p64) {
            const uint32_t pf[8] = {a, cc, mask, salt, start, c.steps, 2u * (uint32_t)r, 0u};
            dispatch(cb, &pfu, ds, pf, sizeof pf, 1);
        } else if (fused) {
            const struct { uint64_t base; uint32_t a, c, mask, salt; uint64_t start; uint32_t steps, slot, zero; } pf = {
                rg->addr, a, cc, mask, salt, start_adr, c.steps, (uint32_t)r, 0u};
            dispatch(cb, &pfu, ds, &pf, sizeof pf, 1);
        } else if (!p64) {
            const uint32_t pk[8] = {start, c.steps, cold_a, cold_c, mask, 1u, 2u * (uint32_t)r, 0u};
            dispatch(cb, &pch, ds, pk, sizeof pk, 1);
        } else {
            const struct { uint64_t start; uint32_t steps, slot, zero; } pk = {start_adr, c.steps, (uint32_t)r, 0u};
            dispatch(cb, &pch, ds, &pk, sizeof pk, 1);
        }
        VK_CHECK(vkEndCommandBuffer(cb));
        submit(q, cb, fence);
        fwait(fence);
        vkFreeDescriptorSets(g_dev, g_dpool, 1, &ds);
        const int uc_b = uclk_now();
        const uint64_t cyc = o[8 * r];
        const uint32_t rt = o[8 * r + 2];
        const int written = o[8 * r] != 0xdeadbeefu;
        int ok;
        uint64_t fin;
        if (p64) {
            fin = o[8 * r + 4] | (uint64_t)o[8 * r + 5] << 32;
            ok = written && fin == exp_adr;
        } else {
            fin = o[8 * r + 3];
            const uint32_t acc = o[8 * r + 1], expacc = c.steps * stamp_of(fused ? salt : cold_salt);
            ok = written && fin == fin_slot && acc == expacc;
        }
        const double mhz = rt ? (double)cyc / rt * 100.0 : 0.0;
        printf("ROW r10 api=vulkan cell=%s q=%s form=%s order=%s walk=%s alloc=%s mem=%s evict=%d sg=%d coh=%d stride=%u loop=%u rep=%d "
               "cycles=%llu rt=%u ok=%d steps=%u clk_mhz=%.0f uclk0=%d uclk1=%d\n",
               c.name, q->name, FORM_N[c.form], ORD_N[c.order], c.page ? "page" : "random", ALLOC_N[c.alloc], MEM_N[c.mem], c.ev_mib,
               c.sg, c.coh, c.stride, c.loop, r, (unsigned long long)cyc, rt, ok, c.steps, mhz, uc_a, uc_b);
        if (r > 0 && written) {  // repetition 0 loads the code and builds the evict state: kept in the file, not counted
            cps[ncps] = (double)cyc / c.steps;
            nps[ncps] = (double)rt * 10.0 / c.steps;
            clk[ncps] = mhz;
            ncps++;
            okn += ok;
        }
    }
    const int uc1 = uclk_now();
    double cmin = 1e30, cmax = 0;
    for (int i = 0; i < ncps; i++) cmin = fmin(cmin, clk[i]), cmax = fmax(cmax, clk[i]);
    const double cmed = med(clk, ncps);
    printf("RESULT r10 api=vulkan cell=%s q=%s form=%s order=%s walk=%s alloc=%s mem=%s evict=%d sg=%d coh=%d stride=%u loop=%u reps=%d "
           "counted=%d cyc_per_step_med=%.1f ns_per_step_med=%.1f pointer_ok=%d/%d clk_mhz_med=%.0f min=%.0f max=%.0f uclk_start=%d "
           "uclk_end=%d\n",
           c.name, q->name, FORM_N[c.form], ORD_N[c.order], c.page ? "page" : "random", ALLOC_N[c.alloc], MEM_N[c.mem], c.ev_mib,
           c.sg, c.coh, c.stride, c.loop, c.reps, ncps, med(cps, ncps), med(nps, ncps), okn, ncps, cmed, ncps ? cmin : 0.0, cmax,
           uc0, uc1);
    verdict("cell_clock", ncps > 0 && (cmax - cmin) <= 0.10 * cmed,
            "%s: core clock of the counted repetitions %.0f-%.0f MHz, median %.0f (the spread must stay within 10 %% of the median)",
            c.name, ncps ? cmin : 0.0, cmax, cmed);
    free(cps), free(nps), free(clk);
    vkDestroyFence(g_dev, fence, NULL);
    vkFreeCommandBuffers(g_dev, q->pool, 1, &cb);
    vkFreeDescriptorSets(g_dev, g_dpool, 1, &dse);
    free_regions(R, nreg + 1);
    rmbuf(&ev), rmbuf(&sink), rmbuf(&out);
    return 0;
}

// ---------------------------------------------------------------------------------------------------------------
// reach: N distinct 4 KiB pages (one 128-byte node each, cyclic), walked twice; the second pass is warm in the caches, so
// what it costs beyond the L0/L2 level is address translation

static int run_reach(const char *spec) {
    Cfg c = {0};
    char list[256], s[64];
    kvs(spec, "name", "reach", c.name, sizeof c.name);
    kvs(spec, "q", "c", s, sizeof s);
    c.q = !strcmp(s, "c");
    kvs(spec, "n", "64,256,1024,4096,8192", list, sizeof list);
    c.reps = (int)kvd(spec, "reps", 6);
    kvs(spec, "alloc", "sep", s, sizeof s);
    c.alloc = idx_of(s, ALLOC_N, 4);
    kvs(spec, "mem", "vram", s, sizeof s);
    c.mem = idx_of(s, MEM_N, 3);
    c.ev_mib = (int)kvd(spec, "evict", 256);
    c.warm = kvd(spec, "warm", 150);
    Qu *qu = &QU[c.q];
    if (!qu->valid || c.alloc < 0 || c.mem < 0 || c.reps < 2 || c.reps > 100 || (c.ev_mib != 0 && c.ev_mib != 256 && c.ev_mib != 512)) {
        printf("VERDICT vkr10 reach_spec: FAIL bad spec %s\n", spec);
        g_fail++;
        return 1;
    }
    if (g_hot_stop || !wait_cool(c.name)) return 1;
    c.sg = 32;
    c.form = F_P64V;
    const Pipe pev = mk("r9_evict.spv", "r9_evict", 256, 32), pch = mk("r10_reach.spv", "r10_reach_sg32", 32, 32),
               pcn = mk("r10_chain.spv", "r10_chain", 256, 32);
    Buf ev = {0}, sink = mkbuf(4096, 0), out = mkbuf(32u * (size_t)c.reps, 2);
    if (c.ev_mib) ev = mkbuf((VkDeviceSize)c.ev_mib * MIB, 0);
    VkDescriptorSet dse = mkset(c.ev_mib ? &ev : NULL, NULL, &sink, NULL);
    VkCommandBuffer cb = cb_alloc(qu);
    VkFence fence = mkfence();
    uint32_t *o = out.map;
    int rc = 0;
    for (const char *p = list; *p && !g_hot_stop;) {
        const uint32_t n = (uint32_t)atoi(p);
        const char *cm = strchr(p, ',');
        p = cm ? cm + 1 : p + strlen(p);
        if (n < 16 || n > 8192) {
            printf("VERDICT vkr10 reach_spec: FAIL n=%u outside 16..8192\n", n);
            g_fail++;
            rc = 1;
            continue;
        }
        Reg R[2];
        if (!make_regions(&c, R, 1, 32 * MIB)) {
            printf("NOT RUN %s: no memory type\n", c.name);
            return 1;
        }
        VkDescriptorSet ds = mkset_o(R[0].b, R[0].off, 4 * MIB, &out, NULL);
        const struct { uint64_t base; uint32_t stride, count; } pcn_pc = {R[0].addr, 4096u, n};
        cb_begin(cb);
        dispatch(cb, &pcn, ds, &pcn_pc, sizeof pcn_pc, (n + 255) / 256);
        VK_CHECK(vkEndCommandBuffer(cb));
        submit(qu, cb, fence);
        fwait(fence);
        warm_up(qu, c.warm);
        printf("CELL %s api=vulkan q=%s n=%u reps=%d alloc=%s mem=%s evict=%d uclk_start=%d hotspot=%.0f\n", c.name, qu->name, n, c.reps,
               ALLOC_N[c.alloc], MEM_N[c.mem], c.ev_mib, uclk_now(), hotspot());
        for (int r = 0; r < c.reps && !g_hot_stop; r++) {
            const uint32_t s0 = (uint32_t)((uint64_t)r * 37u % n);
            for (int k = 0; k < 8; k++) o[8 * r + k] = 0xdeadbeefu;
            const int ua = uclk_now();
            cb_begin(cb);
            if (c.ev_mib) {
                const uint32_t pe[2] = {(uint32_t)((uint64_t)c.ev_mib * MIB / 16), 512u * 256u};
                dispatch(cb, &pev, dse, pe, sizeof pe, 512);
                b2_barrier(cb);
            }
            const struct { uint64_t start; uint32_t steps, slot, zero; } pk = {R[0].addr + (uint64_t)s0 * 4096u, n, (uint32_t)r, 0u};
            dispatch(cb, &pch, ds, &pk, sizeof pk, 1);
            VK_CHECK(vkEndCommandBuffer(cb));
            submit(qu, cb, fence);
            fwait(fence);
            const uint64_t fin = o[8 * r + 4] | (uint64_t)o[8 * r + 5] << 32;
            const uint64_t exp = R[0].addr + (uint64_t)((s0 + 2ull * n) % n) * 4096u;
            const int ok = o[8 * r] != 0xdeadbeefu && fin == exp;
            printf("ROW r10 api=vulkan cell=%s mode=reach q=%s n=%u rep=%d cycles1=%u cycles2=%u rt=%u ok=%d steps=%u uclk0=%d uclk1=%d\n", c.name,
                   qu->name, n, r, o[8 * r], o[8 * r + 1], o[8 * r + 2], ok, n, ua, uclk_now());
        }
        vkFreeDescriptorSets(g_dev, g_dpool, 1, &ds);
        free_regions(R, 1);
    }
    vkDestroyFence(g_dev, fence, NULL);
    vkFreeCommandBuffers(g_dev, qu->pool, 1, &cb);
    vkFreeDescriptorSets(g_dev, g_dpool, 1, &dse);
    rmbuf(&ev), rmbuf(&sink), rmbuf(&out);
    return rc;
}

// ---------------------------------------------------------------------------------------------------------------
// info: what the driver offers and which pipelines of the item look like on the card (nothing is dispatched)

static int run_info(void) {
    for (uint32_t i = 0; i < g_mp.memoryHeapCount; i++)
        printf("INFO heap %u size_mib=%llu flags=0x%x%s\n", i, (unsigned long long)(g_mp.memoryHeaps[i].size >> 20), g_mp.memoryHeaps[i].flags,
               g_mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? " DEVICE_LOCAL" : "");
    for (uint32_t i = 0; i < g_mp.memoryTypeCount; i++) {
        const VkMemoryPropertyFlags f = g_mp.memoryTypes[i].propertyFlags;
        printf("INFO type %u heap=%u flags=0x%x%s%s%s%s\n", i, g_mp.memoryTypes[i].heapIndex, f,
               f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? " DEVICE_LOCAL" : "", f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? " HOST_VISIBLE" : "",
               f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? " HOST_COHERENT" : "", f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? " HOST_CACHED" : "");
    }
    VkBuffer b = new_buffer(REG_BYTES);
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_dev, b, &mr);
    for (int m = 0; m < 3; m++) {
        const uint32_t t = pick_type(mr.memoryTypeBits, m);
        printf("INFO chosen mem=%s type=%d (round 9's ra, rb and ev were created with kind 0 = vram)\n", MEM_N[m], t == UINT32_MAX ? -1 : (int)t);
    }
    printf("INFO region buffer requirements: size=%llu alignment=%llu typebits=0x%x\n", (unsigned long long)mr.size,
           (unsigned long long)mr.alignment, mr.memoryTypeBits);
    vkDestroyBuffer(g_dev, b, NULL);
    // every pipeline of the item (capture() writes <out>/asm/<label>.s and prints one PIPE line per executable)
    Cfg c = {0};
    for (int sg = 32; sg <= 64; sg += 32) {
        c.sg = sg;
        c.form = F_U32;
        chase_pipe(&c);
        c.form = F_P64;
        chase_pipe(&c);
        c.form = F_P64V;
        chase_pipe(&c);
    }
    c.sg = 32;
    c.form = F_P64;
    c.coh = 1;
    chase_pipe(&c);
    mk("r9_prod.spv", "r9_prod", 256, 32);
    mk("r9_fused.spv", "r9_fused", 256, 32);
    mk("r9_evict.spv", "r9_evict", 256, 32);
    mk("r10_prod64.spv", "r10_prod64", 256, 32);
    mk("r10_fused64.spv", "r10_fused64", 256, 32);
    mk("r10_chain.spv", "r10_chain", 256, 32);
    mk("r10_reach.spv", "r10_reach_sg32", 32, 32);
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: vkr10 info | bis <spec>... | reach <spec>\n");
        return 2;
    }
    const char *o = getenv("VKG_OUT"), *s = getenv("VKG_SPV"), *t = getenv("VKG_TAG");
    snprintf(g_out, sizeof g_out, "%s", o ? o : "");
    snprintf(g_spv, sizeof g_spv, "%s", s ? s : "spv");
    snprintf(g_tag, sizeof g_tag, "%s", t ? t : argv[1]);
    dev_init(!strcmp(argv[1], "info"));
    sampler_start();
    const double t0 = mono_ns();
    printf("START hotspot %.0f C\n", hotspot());
    int rc = 2;
    if (!strcmp(argv[1], "info")) rc = run_info();
    else if (!strcmp(argv[1], "bis")) {
        rc = 0;
        for (int i = 2; i < argc && !g_hot_stop; i++) rc |= run_bis(argv[i]);
    } else if (!strcmp(argv[1], "reach")) {
        rc = 0;
        for (int i = 2; i < argc && !g_hot_stop; i++) rc |= run_reach(argv[i]);
    } else fprintf(stderr, "unknown mode %s\n", argv[1]);
    vkDeviceWaitIdle(g_dev);
    const double t1 = mono_ns();
    sampler_stop_write();
    state_line("whole-job", t0, t1);
    printf("THERMAL highest %.0f C (5 ms samples); end %.0f C; hot_stop=%d\n", g_hottest, hotspot(), g_hot_stop);
    if (g_hot_stop) verdict("thermal", 0, "hotspot reached %.0f C; load stopped", (double)g_hot_stop_c);
    if (rc) return rc;
    return g_fail ? 1 : 0;
}
