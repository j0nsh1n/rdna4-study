; Leftovers C, lane A, MS-7: does the !amdgpu.last.use metadata give a last-use load (th:TH_LOAD_LU) on gfx1201?
; Compiled by scripts/leftc_lastuse.py with clang -x ir; three loads from global memory: plain, marked last use, and
; marked non-temporal (the control: !nontemporal is known to give th:TH_LOAD_NT, memory-system item 8).
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @lastuse(ptr addrspace(1) %in, ptr addrspace(1) %out) {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %i = zext i32 %id to i64
  %p0 = getelementptr inbounds i32, ptr addrspace(1) %in, i64 %i
  %a = load i32, ptr addrspace(1) %p0, align 4
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 4096
  %b = load i32, ptr addrspace(1) %p1, align 4, !amdgpu.last.use !0
  %p2 = getelementptr inbounds i32, ptr addrspace(1) %p0, i64 8192
  %c = load i32, ptr addrspace(1) %p2, align 4, !nontemporal !1
  %s0 = add i32 %a, %b
  %s1 = add i32 %s0, %c
  %q = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %s1, ptr addrspace(1) %q, align 4
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

!0 = !{}
!1 = !{i32 1}
