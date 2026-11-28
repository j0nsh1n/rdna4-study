; v_permlane16 / v_permlanex16 with fetch-inactive: the source value is computed inside the divergent region
; (gfx12): minimal IR reproducer, in the style of llvm/test/CodeGen/AMDGPU.
;
; The IR computes %v (a load and an arithmetic op) in the entry block, before the lane-varying branch, so every
; lane defines it. The permute inside the branch has fetch-inactive = true: it reads %v from lanes that are
; switched off. The backend sinks the load and the arithmetic into the region guarded by s_and_saveexec (it is
; the only user), so a switched-off lane never defines the register the permute reads from it.
;
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 -O2 < %s | FileCheck %s --check-prefix=SUNK
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 -O2 < %s | FileCheck %s --check-prefix=PINNED
;
; SUNK documents today's output of @bad_sunk: the source is produced after the mask write (SUNK, then the
; load, the multiply-add, then the permute, with no function end in between). These lines pass today. A fix
; that keeps the definition in front of the branch would make them fail.
;
; SUNK-LABEL: bad_sunk:
; SUNK:       {{s_and_saveexec_b32|v_cmpx_[a-z0-9_]+}}
; SUNK-NOT:   s_endpgm
; SUNK:       global_load_b32
; SUNK-NOT:   s_endpgm
; SUNK:       v_permlane16_b32
;
; PINNED is the control (@good_pinned, source pinned by an empty inline asm): the load is in front of the
; mask write.
;
; PINNED-LABEL: good_pinned:
; PINNED:       global_load_b32
; PINNED-NOT:   s_endpgm
; PINNED:       {{s_and_saveexec_b32|v_cmpx_[a-z0-9_]+}}
; PINNED-NOT:   s_endpgm
; PINNED:       v_permlane16_b32

target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.permlane16.i32(i32, i32, i32, i32, i1 immarg, i1 immarg)
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @bad_sunk(ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %n) {
entry:
  %l = call i32 @llvm.amdgcn.workitem.id.x()
  %i = zext i32 %l to i64
  %gx = getelementptr i32, ptr addrspace(1) %x, i64 %i
  %gy = getelementptr i32, ptr addrspace(1) %y, i64 %i
  %xv = load i32, ptr addrspace(1) %gx
  %v3 = mul i32 %xv, 3
  %v = add i32 %v3, 1
  %old = or i32 %l, -559087616
  %valid = icmp slt i32 %l, %n
  br i1 %valid, label %then, label %end

then:
  %p = call i32 @llvm.amdgcn.permlane16.i32(i32 %old, i32 %v, i32 -1985229329, i32 19088743, i1 true, i1 false)
  br label %end

end:
  %r = phi i32 [ %old, %entry ], [ %p, %then ]
  store i32 %r, ptr addrspace(1) %gy
  ret void
}

define amdgpu_kernel void @good_pinned(ptr addrspace(1) %x, ptr addrspace(1) %y, i32 %n) {
entry:
  %l = call i32 @llvm.amdgcn.workitem.id.x()
  %i = zext i32 %l to i64
  %gx = getelementptr i32, ptr addrspace(1) %x, i64 %i
  %gy = getelementptr i32, ptr addrspace(1) %y, i64 %i
  %xv = load i32, ptr addrspace(1) %gx
  %v3 = mul i32 %xv, 3
  %v0 = add i32 %v3, 1
  %v = call i32 asm sideeffect "", "=v,0"(i32 %v0)
  %old = or i32 %l, -559087616
  %valid = icmp slt i32 %l, %n
  br i1 %valid, label %then, label %end

then:
  %p = call i32 @llvm.amdgcn.permlane16.i32(i32 %old, i32 %v, i32 -1985229329, i32 19088743, i1 true, i1 false)
  br label %end

end:
  %r = phi i32 [ %old, %entry ], [ %p, %then ]
  store i32 %r, ptr addrspace(1) %gy
  ret void
}
