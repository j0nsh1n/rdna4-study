; WMMA emitted under a partial EXEC mask (gfx12): minimal IR reproducer, in the style of llvm/test/CodeGen/AMDGPU.
;
; The WMMA is control-dependent on a lane-varying condition (workitem id < n). The hardware instruction
; ignores EXEC (reads and writes all lanes), but the generated code places it inside the s_and_saveexec /
; v_cmpx region, and no diagnostic is produced. The second function pads with a select instead and keeps the
; WMMA under full EXEC.
;
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 -O2 < %s | FileCheck %s --check-prefix=BAD
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 -O2 < %s | FileCheck %s --check-prefix=GOOD
;
; BAD documents today's output: the WMMA follows the mask write and precedes the store of the masked lanes. These lines pass
; today. A fix that diagnoses the pattern or runs the instruction outside the partial region would make them fail.
;
; BAD-LABEL: bad_if:
; BAD:       {{s_and_saveexec_b32|v_cmpx_[a-z0-9_]+}}
; BAD-NOT:   s_endpgm
; BAD:       v_wmma_f32_16x16x16_f16
; BAD:       global_store_b128
;
; GOOD is the control: no mask write before the WMMA.
;
; GOOD-LABEL: good_select:
; GOOD-NOT:   s_and_saveexec
; GOOD-NOT:   v_cmpx
; GOOD:       v_wmma_f32_16x16x16_f16

target triple = "amdgcn-amd-amdhsa"

declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half>, <8 x half>, <8 x float>)
declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @bad_if(ptr addrspace(1) %pa, ptr addrspace(1) %pb, ptr addrspace(1) %pc, ptr addrspace(1) %out, i32 %n) {
entry:
  %l = call i32 @llvm.amdgcn.workitem.id.x()
  %valid = icmp slt i32 %l, %n
  br i1 %valid, label %then, label %end

then:
  %i = zext i32 %l to i64
  %ga = getelementptr <8 x half>, ptr addrspace(1) %pa, i64 %i
  %gb = getelementptr <8 x half>, ptr addrspace(1) %pb, i64 %i
  %gc = getelementptr <8 x float>, ptr addrspace(1) %pc, i64 %i
  %go = getelementptr <8 x float>, ptr addrspace(1) %out, i64 %i
  %a = load <8 x half>, ptr addrspace(1) %ga
  %b = load <8 x half>, ptr addrspace(1) %gb
  %c = load <8 x float>, ptr addrspace(1) %gc
  %d = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %a, <8 x half> %b, <8 x float> %c)
  store <8 x float> %d, ptr addrspace(1) %go
  br label %end

end:
  ret void
}

define amdgpu_kernel void @good_select(ptr addrspace(1) %pa, ptr addrspace(1) %pb, ptr addrspace(1) %pc, ptr addrspace(1) %out, i32 %n) {
  %l = call i32 @llvm.amdgcn.workitem.id.x()
  %lo = and i32 %l, 15
  %valid = icmp slt i32 %lo, %n
  %i = zext i32 %l to i64
  %ga = getelementptr <8 x half>, ptr addrspace(1) %pa, i64 %i
  %gb = getelementptr <8 x half>, ptr addrspace(1) %pb, i64 %i
  %gc = getelementptr <8 x float>, ptr addrspace(1) %pc, i64 %i
  %go = getelementptr <8 x float>, ptr addrspace(1) %out, i64 %i
  %a0 = load <8 x half>, ptr addrspace(1) %ga
  %a = select i1 %valid, <8 x half> %a0, <8 x half> zeroinitializer
  %b = load <8 x half>, ptr addrspace(1) %gb
  %c = load <8 x float>, ptr addrspace(1) %gc
  %d = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half> %a, <8 x half> %b, <8 x float> %c)
  store <8 x float> %d, ptr addrspace(1) %go
  ret void
}
