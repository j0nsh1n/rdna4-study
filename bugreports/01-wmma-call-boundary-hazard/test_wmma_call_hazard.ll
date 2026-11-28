; WMMA hazard across a call boundary (gfx12): minimal IR reproducer, in the style of llvm/test/CodeGen/AMDGPU.
;
; A non-inlined callee ends with a WMMA and returns the result; the caller's next instruction after the
; call is a WMMA that reads that result as A. The WMMA-after-WMMA hazard needs a V_NOP (or any VALU
; instruction) in between; the generated code has none on either side of the call.
;
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 -O2 < %s | FileCheck %s --check-prefix=CURRENT
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -O2 < %s | FileCheck %s --check-prefix=CURRENT
;
; CURRENT documents today's output: no V_NOP before the return and none after the call. These lines
; pass today and are expected to FAIL once the hazard is handled across calls.
;
; CURRENT-LABEL: first_mm:
; CURRENT:       v_wmma_f16_16x16x16_f16
; CURRENT-NEXT:  s_setpc_b64
;
; CURRENT-LABEL: caller:
; CURRENT:       s_swappc_b64
; CURRENT-NEXT:  v_wmma_f32_16x16x16_f16
;
; FIXED is one acceptable post-fix shape (a V_NOP before the return). A V_NOP or any other VALU
; instruction after the call would close the hazard as well. These lines FAIL today.
;
; FIXED-LABEL: first_mm:
; FIXED:       v_wmma_f16_16x16x16_f16
; FIXED-NEXT:  v_nop
; FIXED-NEXT:  s_setpc_b64

target triple = "amdgcn-amd-amdhsa"

declare <8 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.v8f16.v8f16(<8 x half>, <8 x half>, <8 x half>, i1 immarg)
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half>, <8 x half>, <8 x float>)
declare i32 @llvm.amdgcn.workitem.id.x()

define fastcc <8 x half> @first_mm(<8 x half> %c, <8 x half> %a, <8 x half> %b) noinline {
  %d = call <8 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.v8f16.v8f16(<8 x half> %a, <8 x half> %b, <8 x half> %c, i1 false)
  ret <8 x half> %d
}

define amdgpu_kernel void @caller(ptr addrspace(1) %pa, ptr addrspace(1) %pb, ptr addrspace(1) %pc, ptr addrspace(1) %pb2, ptr addrspace(1) %pc2, ptr addrspace(1) %out) {
  %l = call i32 @llvm.amdgcn.workitem.id.x()
  %i = zext i32 %l to i64
  %ga = getelementptr <8 x half>, ptr addrspace(1) %pa, i64 %i
  %gb = getelementptr <8 x half>, ptr addrspace(1) %pb, i64 %i
  %gc = getelementptr <8 x half>, ptr addrspace(1) %pc, i64 %i
  %gb2 = getelementptr <8 x half>, ptr addrspace(1) %pb2, i64 %i
  %gc2 = getelementptr <8 x float>, ptr addrspace(1) %pc2, i64 %i
  %go = getelementptr <8 x float>, ptr addrspace(1) %out, i64 %i
  %b2 = load <8 x half>, ptr addrspace(1) %gb2
  %c2 = load <8 x float>, ptr addrspace(1) %gc2
  %a = load <8 x half>, ptr addrspace(1) %ga
  %b = load <8 x half>, ptr addrspace(1) %gb
  %c = load <8 x half>, ptr addrspace(1) %gc
  %d1 = call fastcc <8 x half> @first_mm(<8 x half> %c, <8 x half> %a, <8 x half> %b)
  %d2 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %d1, <8 x half> %b2, <8 x float> %c2)
  store <8 x float> %d2, ptr addrspace(1) %go
  ret void
}
