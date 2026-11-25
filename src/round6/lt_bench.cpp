// hipBLASLt GEMM speed: D[N x M] = W^T * X, the layout of a linear layer y = x W^T (weight stored [N,K] row-major).
// Usage: lt_bench fp8|bf16 M N K   -> prints heuristic kernel name and TFLOPS of 3 bursts of ~0.2 s.
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <unistd.h>
#define CK(x) do{auto e=(x); if(e){printf("ERR %s line %d: %d\n",#x,__LINE__,(int)e); return 1;}}while(0)
int main(int argc,char**argv){
  if(argc<5) return 2;
  bool fp8=std::string(argv[1])=="fp8"; int M=atoi(argv[2]),N=atoi(argv[3]),K=atoi(argv[4]);
  hipblasLtHandle_t h; CK(hipblasLtCreate(&h));
  hipDataType in=fp8?HIP_R_8F_E4M3:HIP_R_16BF, out=HIP_R_16BF; size_t es=fp8?1:2;
  hipblasLtMatmulDesc_t d; CK(hipblasLtMatmulDescCreate(&d,HIPBLAS_COMPUTE_32F,HIP_R_32F));
  hipblasOperation_t tA=HIPBLAS_OP_T,tB=HIPBLAS_OP_N;
  CK(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSA,&tA,sizeof tA));
  CK(hipblasLtMatmulDescSetAttribute(d,HIPBLASLT_MATMUL_DESC_TRANSB,&tB,sizeof tB));
  hipblasLtMatrixLayout_t lA,lB,lD;
  CK(hipblasLtMatrixLayoutCreate(&lA,in,K,N,K)); CK(hipblasLtMatrixLayoutCreate(&lB,in,K,M,K)); CK(hipblasLtMatrixLayoutCreate(&lD,out,N,M,N));
  void *A,*B,*D,*W; size_t ws=64<<20;
  CK(hipMalloc(&A,es*K*N)); CK(hipMalloc(&B,es*K*M)); CK(hipMalloc(&D,2*N*M)); CK(hipMalloc(&W,ws));
  auto fill=[&](void*p,size_t n){ std::vector<unsigned char> v(n*es); unsigned s=12345; for(size_t i=0;i<v.size();i++){ s=s*1664525u+1013904223u; unsigned r=(s>>24);
      if(fp8) v[i]=(r&0x8f)|0x30 ? ((r&0x80)|(0x30+(r&0x0f))) : 0x38; else v[i]=(i&1)?((r&0x80)|0x3c):(r); } if(!fp8){ for(size_t i=1;i<v.size();i+=2){ v[i]=(v[i]&0x80)|0x3c|(v[i-1]&1);} } hipMemcpy(p,v.data(),v.size(),hipMemcpyHostToDevice); };
  fill(A,(size_t)K*N); fill(B,(size_t)K*M); hipMemset(D,0,2*N*M);
  hipblasLtMatmulPreference_t pref; CK(hipblasLtMatmulPreferenceCreate(&pref));
  CK(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof ws));
  hipblasLtMatmulHeuristicResult_t r[1]; int cnt=0;
  CK(hipblasLtMatmulAlgoGetHeuristic(h,d,lA,lB,lD,lD,pref,1,r,&cnt));
  if(!cnt){printf("NOALGO\n");return 3;}
  printf("kernel %s\n",hipblaslt_ext::getKernelNameFromAlgo(h,r[0].algo).c_str());
  printf("solution %s\n",hipblaslt_ext::getSolutionNameFromAlgo(h,r[0].algo).c_str());
  float al=1,be=0; hipStream_t st; hipStreamCreate(&st);
  auto run=[&](){ return hipblasLtMatmul(h,d,&al,A,lA,B,lB,&be,D,lD,D,lD,&r[0].algo,W,ws,st); };
  for(int i=0;i<20;i++) CK(run()); hipStreamSynchronize(st);
  hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1);
  hipEventRecord(e0,st); for(int i=0;i<20;i++) run(); hipEventRecord(e1,st); hipEventSynchronize(e1);
  float ms; hipEventElapsedTime(&ms,e0,e1); int n=(int)(200.0f/(ms/20)); if(n<20)n=20;
  double fl=2.0*M*N*K;
  for(int rep=0;rep<3;rep++){
    hipEventRecord(e0,st); for(int i=0;i<n;i++) run(); hipEventRecord(e1,st); hipEventSynchronize(e1);
    hipEventElapsedTime(&ms,e0,e1);
    printf("burst %d iters %d ms_total %.2f us_per %.2f TFLOPS %.2f\n",rep,n,ms,ms*1000/n,fl*n/(ms*1e-3)/1e12);
    sleep(2);
  }
  return 0;
}
