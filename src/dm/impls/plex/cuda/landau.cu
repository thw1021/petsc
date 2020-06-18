/*
   Implements the Landau kernal
*/
#include <petscconf.h>
#include <petsc/private/dmpleximpl.h>   /*I   "petscdmplex.h"   I*/
#include <../src/mat/impls/aij/seq/aij.h>  /* put CUDA SeqAIJ */
#include <petsc/private/kernels/petscaxpy.h>
#include <omp.h>

#define PETSC_DEVICE_SYNC __syncthreads()
#define PETSC_DEVICE_FUNC_DECL __device__
#define PETSC_DEVICE_DATA_DECL __constant__
#include "fp_kernels.h"

// Macro to catch CUDA errors in CUDA runtime calls
#define CUDA_SAFE_CALL(call)                                          \
do {                                                                  \
    cudaError_t err = call;                                           \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)
// Macro to catch CUDA errors in kernel launches
#define CHECK_LAUNCH_ERROR()                                          \
do {                                                                  \
    /* Check synchronous errors, i.e. pre-launch */                   \
    cudaError_t err = cudaGetLastError();                             \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString(err) );       \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
    /* Check asynchronous errors, i.e. kernel failed (ULF) */         \
    err = cudaDeviceSynchronize();                                    \
    if (cudaSuccess != err) {                                         \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.\n",\
                 __FILE__, __LINE__, cudaGetErrorString( err) );      \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

// #define FP_USE_SHARED_GPU_MEM
//j
// The GPU Landau kernel
//
__global__
void land_kernel(const PetscInt nip, const PetscInt dim, const PetscInt totDim, const PetscInt Nf, const PetscInt Nb, const PetscReal invJj[],
		 const PetscReal nu_alpha[], const PetscReal nu_beta[], const PetscReal invMass[], const PetscReal Eq_m[],
		 const PetscReal * const a_TabBD, const PetscReal * const IPDataGlobal, const PetscReal wiGlobal[],
#if !defined(FP_USE_SHARED_GPU_MEM)
		 PetscReal *g2arr, PetscReal *g3arr,
#endif
		 PetscBool quarter3DDomain, PetscScalar elemMats_out[])
{
  const PetscInt  Nq = blockDim.x, myelem = blockIdx.x;
#if defined(FP_USE_SHARED_GPU_MEM)
  extern __shared__ PetscReal g2_g3_qi[]; // Nq * { [NSubBlocks][Nf][dim] ; [NSubBlocks][Nf][dim][dim] }
  PetscReal       (*g2)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM]         = (PetscReal (*)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM])         &g2_g3_qi[0];
  PetscReal       (*g3)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM][FP_DIM] = (PetscReal (*)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM][FP_DIM]) &g2_g3_qi[FP_MAX_SUB_THREAD_BLOCKS*FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM];
#else
  PetscReal       (*g2)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM]         = (PetscReal (*)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM])         &g2arr[myelem*FP_MAX_SUB_THREAD_BLOCKS*FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM       ];
  PetscReal       (*g3)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM][FP_DIM] = (PetscReal (*)[FP_MAX_NQ][FP_MAX_SUB_THREAD_BLOCKS][FP_MAX_SPECIES][FP_DIM][FP_DIM]) &g3arr[myelem*FP_MAX_SUB_THREAD_BLOCKS*FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM*FP_DIM];
#endif
  const PetscInt  myqi = threadIdx.x, mySubBlk = threadIdx.y, nSubBlocks = blockDim.y;
  const PetscInt  jpidx = myqi + myelem * Nq;
  const PetscInt  subblocksz = nip/nSubBlocks + !!(nip%nSubBlocks), ip_start = mySubBlk*subblocksz, ip_end = (mySubBlk+1)*subblocksz > nip ? nip : (mySubBlk+1)*subblocksz; /* this could be wrong with very few global IPs */

  landau_inner_integral(myqi, mySubBlk, nSubBlocks, ip_start, ip_end, jpidx, Nf, dim, IPDataGlobal, wiGlobal, &invJj[jpidx*dim*dim], nu_alpha, nu_beta, invMass, Eq_m, quarter3DDomain, *g2, *g3);

  /* FE matrix construction */
  __syncthreads();   // Synchronize (ensure all the data is available) and sum IP matrices
  {
    const PetscReal *iTab,*TabBD[FP_MAX_SPECIES][2];
    int              fieldA,d,f,qj,d2,g;
    PetscScalar     *elemMat  = &elemMats_out[myelem*totDim*totDim]; /* my output */
    for (iTab = a_TabBD, fieldA = 0 ; fieldA < Nf ; fieldA++, iTab += Nq*Nb*(1+dim)) { // get pointers for convenience
      TabBD[fieldA][0] = iTab;
      TabBD[fieldA][1] = &iTab[Nq*Nb];
    }
    /* assemble - on the diagonal (I,I) */
    for (fieldA = threadIdx.y; fieldA < Nf ; fieldA += blockDim.y) {
      for (f = threadIdx.x; f < Nb ; f += blockDim.x) {
	const PetscInt i = fieldA*Nb + f; /* Element matrix row */
	for (g = 0; g < Nb; ++g) {
	  const PetscInt j    = fieldA*Nb + g; /* Element matrix column */
	  const PetscInt fOff = i*totDim + j;
	  elemMat[fOff] = 0;
	  for (qj=0;qj<Nq;qj++) {
	    const PetscReal *B = TabBD[fieldA][0], *D = TabBD[fieldA][1], *BJq = &B[qj*Nb], *DIq = &D[qj*Nb*dim], *DJq = &D[qj*Nb*dim];
	    for (d = 0; d < dim; ++d) {
	      elemMat[fOff] += DIq[f*dim+d]*(*g2)[qj][0][fieldA][d]*BJq[g];
	      for (d2 = 0; d2 < dim; ++d2) {
		elemMat[fOff] += DIq[f*dim + d]*(*g3)[qj][0][fieldA][d][d2]*DJq[g*dim + d2];
	      }
	    }
	  }
	}
      }
    }
    if (myelem==-6) {
      if (threadIdx.x==0 && threadIdx.y==0) {
	__syncthreads();
	printf("GPU Element matrix\n"); 
	for (d = 0; d < totDim; ++d){
	  for (f = 0; f < totDim; ++f) printf(" %17.10e", elemMat[d*totDim + f]);
	  printf("\n");
	}
      } else {
	__syncthreads();
      }
    }
  }
}

struct _ISColoring_ctx {
  ISColoring coloring;
};
typedef struct _ISColoring_ctx * ISColoring_ctx;

static PetscErrorCode destroy_coloring (void *obj)
{
  PetscErrorCode    ierr;
  ISColoring_ctx    coloring_ctx = (ISColoring_ctx)obj;
  ISColoring        isc = coloring_ctx->coloring;
  PetscFunctionBegin;
  ierr = ISColoringDestroy(&isc);CHKERRQ(ierr);
  ierr = PetscFree(coloring_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

__global__ void assemble_kernel(const PetscInt nidx_arr[], PetscInt *idx_arr[], PetscScalar *el_mats[], const ISColoringValue colors[], Mat_SeqAIJ mats[]);
static PetscErrorCode assemble_omp_private(PetscInt cStart, PetscInt cEnd, PetscInt totDim, DM plex, PetscSection section, PetscSection globalSection, Mat JacP, PetscScalar elemMats[], PetscContainer container);
static PetscErrorCode assemble_cuda_private(PetscInt cStart, PetscInt cEnd, PetscInt totDim, DM plex, PetscSection section, PetscSection globalSection, Mat JacP, PetscScalar elemMats[], PetscContainer container, const PetscLogEvent events[]);

PetscErrorCode FPLandauCUDAJacobian( DM plex, const PetscInt Nq, const PetscReal nu_alpha[],const PetscReal nu_beta[],
				     const PetscReal invMass[], const PetscReal Eq_m[], const PetscReal * const IPDataGlobal,
				     const PetscReal wiGlobal[], const PetscReal invJj[], const PetscInt num_sub_blocks, const PetscLogEvent events[], PetscBool quarter3DDomain, 
				     Mat JacP)
{
  PetscErrorCode    ierr;
  PetscInt          ii,ej,*Nbf,Nb,nip_dim2,cStart,cEnd,Nf,dim,numGCells,totDim,nip,szf=sizeof(PetscReal);
  PetscReal         *d_TabBD,*d_invJj,*d_wiGlobal,*d_nu_alpha,*d_nu_beta,*d_invMass,*d_Eq_m;
  PetscScalar       *elemMats,*d_elemMats,  *iTab;
  PetscLogDouble    flops;
  PetscTabulation   *Tf;
  PetscDS           prob;
  PetscSection      section, globalSection;
  PetscReal        *d_IPDataGlobal;
  PetscContainer    container = NULL;
  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(events[3],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = DMGetDimension(plex, &dim);CHKERRQ(ierr);
  if (dim!=FP_DIM) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "FP_DIM != dim");
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  numGCells = cEnd - cStart;
  nip  = numGCells*Nq; /* length of inner global iteration */
  ierr = DMGetDS(plex, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nbf);CHKERRQ(ierr); Nb = Nbf[0];
  if (Nq != Nb) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Nq != Nb. %D  %D",Nq,Nb);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &Tf);CHKERRQ(ierr);
  ierr = DMGetLocalSection(plex, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(plex, &globalSection);CHKERRQ(ierr);
  // create data
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_IPDataGlobal, nip*(dim + Nf*(dim+1))*szf )); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_nu_alpha, Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_nu_beta,  Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_invMass,  Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_Eq_m,     Nf*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(d_IPDataGlobal, IPDataGlobal, nip*(dim + Nf*(dim+1))*szf, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_nu_alpha, nu_alpha, Nf*szf,                             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_nu_beta,  nu_beta,  Nf*szf,                             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_invMass,  invMass,  Nf*szf,                             cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_Eq_m,     Eq_m,     Nf*szf,                             cudaMemcpyHostToDevice));

  CUDA_SAFE_CALL(cudaMalloc((void **)&d_TabBD,    Nf*Nq*Nb*(1+dim)*szf)); // kernel input
  for (ii=0,iTab=d_TabBD;ii<Nf;ii++,iTab += Nq*Nb*(1+dim)) {
    CUDA_SAFE_CALL(cudaMemcpy( iTab,        Tf[ii]->T[0], Nq*Nb*szf,     cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(&iTab[Nq*Nb], Tf[ii]->T[1], Nq*Nb*dim*szf, cudaMemcpyHostToDevice));
  }
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_wiGlobal,           Nq*numGCells*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_wiGlobal, wiGlobal, Nq*numGCells*szf,   cudaMemcpyHostToDevice));
  // collect geometry
  flops = (PetscLogDouble)numGCells*(PetscLogDouble)Nq*(PetscLogDouble)(5.*dim*dim*Nf*Nf + 165.);
  nip_dim2 = Nq*numGCells*dim*dim;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_invJj, nip_dim2*szf)); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(d_invJj, invJj, nip_dim2*szf,       cudaMemcpyHostToDevice));
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[3],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(events[4],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(flops*nip);CHKERRQ(ierr);
#endif
  {
    PetscReal  *d_g2g3;
    dim3 dimBlock(Nq,num_sub_blocks);
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_elemMats, totDim*totDim*numGCells*sizeof(PetscScalar))); // kernel output
    ii = FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM*(1+FP_DIM)*FP_MAX_SUB_THREAD_BLOCKS;
#if defined(FP_USE_SHARED_GPU_MEM)
    /* PetscPrintf(PETSC_COMM_SELF,"Call land_kernel with %D words shared memory\n",ii); */
    land_kernel<<<numGCells,dimBlock,ii*szf>>>( nip,dim,totDim,Nf,Nb,d_invJj,d_nu_alpha,d_nu_beta,d_invMass,d_Eq_m,
						d_TabBD, d_IPDataGlobal, d_wiGlobal, quarter3DDomain, d_elemMats);
    CHECK_LAUNCH_ERROR();
#else
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_g2g3, ii*szf*numGCells)); // kernel input
    PetscReal  *g2 = &d_g2g3[0];
    PetscReal  *g3 = &d_g2g3[FP_MAX_SUB_THREAD_BLOCKS*FP_MAX_NQ*FP_MAX_SPECIES*FP_DIM*numGCells];
    land_kernel<<<numGCells,dimBlock>>>( nip,dim,totDim,Nf,Nb,d_invJj,d_nu_alpha,d_nu_beta,d_invMass,d_Eq_m,
					 d_TabBD, d_IPDataGlobal, d_wiGlobal, g2, g3, quarter3DDomain, d_elemMats);
    CHECK_LAUNCH_ERROR();
    CUDA_SAFE_CALL (cudaDeviceSynchronize());
    CUDA_SAFE_CALL(cudaFree(d_g2g3));
  }
#endif
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[4],0,0,0,0);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(events[5],0,0,0,0);CHKERRQ(ierr);
#endif
  // delete device data
  CUDA_SAFE_CALL(cudaFree(d_IPDataGlobal));
  CUDA_SAFE_CALL(cudaFree(d_invJj));
  CUDA_SAFE_CALL(cudaFree(d_wiGlobal));
  CUDA_SAFE_CALL(cudaFree(d_nu_alpha));
  CUDA_SAFE_CALL(cudaFree(d_nu_beta));
  CUDA_SAFE_CALL(cudaFree(d_invMass));
  CUDA_SAFE_CALL(cudaFree(d_Eq_m));
  CUDA_SAFE_CALL(cudaFree(d_TabBD));
  ierr = PetscMalloc1(totDim*totDim*numGCells,&elemMats);CHKERRQ(ierr);
  CUDA_SAFE_CALL(cudaMemcpy(elemMats, d_elemMats, totDim*totDim*numGCells*sizeof(PetscScalar), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_elemMats));
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[5],0,0,0,0);CHKERRQ(ierr);
#endif
  /* coloring */
  ierr = PetscObjectQuery((PetscObject)JacP,"coloring",(PetscObject*)&container);CHKERRQ(ierr);
  if (!container) {
    PetscInt        cell,i,nc,Nv;
    ISColoring      iscoloring = NULL;
    ISColoring_ctx  coloring_ctx = NULL;
    Mat             G,Q;
    PetscScalar     ones[64];
    MatColoring     mc;
    IS             *is;
    PetscInt        csize,colour,j,k;
    const PetscInt *indices;
    PetscInt       numComp[1];
    PetscInt       numDof[4];
    PetscFE        fe;
    DM             colordm;
    PetscSection   csection;
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventBegin(events[8],0,0,0,0);CHKERRQ(ierr);
#endif
    /* create cell centered DM */
    ierr = DMClone(plex, &colordm);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) plex), dim, 1, PETSC_FALSE, "color_", PETSC_DECIDE, &fe);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) fe, "color");CHKERRQ(ierr);
    ierr = DMSetField(colordm, 0, NULL, (PetscObject)fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
    for (i = 0; i < (dim+1); ++i) numDof[i] = 0;
    numDof[dim] = 1;
    numComp[0] = 1;
    ierr = DMPlexCreateSection(colordm, NULL, numComp, numDof, 0, 0, NULL, NULL, NULL, &csection);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldName(csection, 0, "color");CHKERRQ(ierr);
    ierr = DMSetLocalSection(colordm, csection);CHKERRQ(ierr);
    ierr = DMViewFromOptions(colordm,NULL,"-color_dm_view");CHKERRQ(ierr);
    /* get vertex to element map Q and colroing graph G */
    ierr = MatGetSize(JacP,NULL,&Nv);CHKERRQ(ierr);
    ierr = MatCreateAIJ(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,numGCells,Nv,totDim,NULL,0,NULL,&Q);CHKERRQ(ierr);
    for(i=0;i<64;i++) ones[i] = 1.0;
    for (cell = cStart, ej = 0 ; cell < cEnd; ++cell, ++ej) {
      PetscInt numindices,*indices;
      ierr = DMPlexGetClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, NULL);CHKERRQ(ierr);
      ierr = MatSetValues(Q,1,&ej,numindices,indices,ones,ADD_VALUES);CHKERRQ(ierr);
      ierr = DMPlexRestoreClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, NULL);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatMatTransposeMult(Q,Q,MAT_INITIAL_MATRIX,4.0,&G);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) Q, "Q");CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) G, "coloring graph");CHKERRQ(ierr);
    ierr = MatViewFromOptions(G,NULL,"-coloring_mat_view");CHKERRQ(ierr);
    ierr = MatViewFromOptions(Q,NULL,"-coloring_mat_view");CHKERRQ(ierr);
    ierr = MatDestroy(&Q);CHKERRQ(ierr);
    /* coloring */
    ierr = MatColoringCreate(G,&mc);CHKERRQ(ierr);
    ierr = MatColoringSetDistance(mc,1);CHKERRQ(ierr);
    ierr = MatColoringSetType(mc,MATCOLORINGJP);CHKERRQ(ierr);
    ierr = MatColoringSetFromOptions(mc);CHKERRQ(ierr);
    ierr = MatColoringApply(mc,&iscoloring);CHKERRQ(ierr);
    ierr = MatColoringDestroy(&mc);CHKERRQ(ierr);
    /* view */
    ierr = ISColoringViewFromOptions(iscoloring,NULL,"-coloring_is_view");CHKERRQ(ierr);
    ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&nc,&is);CHKERRQ(ierr);
    if (0) {
      PetscViewer    viewer;
      Vec            color_vec, eidx_vec;
      ierr = DMGetGlobalVector(colordm, &color_vec);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(colordm, &eidx_vec);CHKERRQ(ierr);
      for (colour=0; colour<nc; colour++) {
	ierr = ISGetLocalSize(is[colour],&csize);CHKERRQ(ierr);
	ierr = ISGetIndices(is[colour],&indices);CHKERRQ(ierr);
	for (j=0; j<csize; j++) {
	  PetscScalar v = (PetscScalar)colour;
	  k = indices[j];
	  ierr = VecSetValues(color_vec,1,&k,&v,INSERT_VALUES);
	  v = (PetscScalar)k;
	  ierr = VecSetValues(eidx_vec,1,&k,&v,INSERT_VALUES);
	}
	ierr = ISRestoreIndices(is[colour],&indices);CHKERRQ(ierr);
      }
      /* view */
      //ierr = VecViewFromOptions(color_vec, NULL, "-color_vec_view");CHKERRQ(ierr);
      //ierr = VecViewFromOptions(eidx_vec, NULL, "-eidx_vec_view");CHKERRQ(ierr);
      ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERVTK);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "color.vtk");CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) color_vec, "color");CHKERRQ(ierr);
      ierr = VecView(color_vec, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERVTK);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "eidx.vtk");CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) eidx_vec, "element-idx");CHKERRQ(ierr);
      ierr = VecView(eidx_vec, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(colordm, &color_vec);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(colordm, &eidx_vec);CHKERRQ(ierr);
    }
    ierr = PetscSectionDestroy(&csection);CHKERRQ(ierr);
    ierr = DMDestroy(&colordm);CHKERRQ(ierr);
    ierr = ISColoringRestoreIS(iscoloring,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
    ierr = MatDestroy(&G);CHKERRQ(ierr);
    /* stash coloring */
    ierr = PetscContainerCreate(PETSC_COMM_SELF, &container);CHKERRQ(ierr);
    ierr = PetscNew(&coloring_ctx);CHKERRQ(ierr);
    coloring_ctx->coloring = iscoloring;
    ierr = PetscContainerSetPointer(container,(void*)coloring_ctx);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container, destroy_coloring);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)JacP,"coloring",(PetscObject)container);CHKERRQ(ierr);
#if defined(PETSC_HAVE_OPENMP)
    if (1) {
      int thread_id,num_threads;
      //char name[MPI_MAX_PROCESSOR_NAME];
      // int resultlength;
      //MPI_Get_processor_name(name, &resultlength);
#pragma omp parallel default(shared) private(thread_id)
      {
	thread_id = omp_get_thread_num();
	num_threads = omp_get_num_threads();
	PetscPrintf(PETSC_COMM_WORLD, "Made coloring with %D colors. OMP_threadID %d of %d\n", nc, thread_id, num_threads);
      }
    }
#endif
#if defined(PETSC_USE_LOG)
    ierr = PetscLogEventEnd(events[8],0,0,0,0);CHKERRQ(ierr);
#endif
  }
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(events[6],0,0,0,0);CHKERRQ(ierr);
#endif
  if (1) {
    PetscScalar *elMat;
    for (ej = cStart, elMat = elemMats ; ej < cEnd; ++ej, elMat += totDim*totDim) {
      ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, ej, elMat, ADD_VALUES);CHKERRQ(ierr);
    }
  } else if (0) { /* OMP assembly */
    ierr = assemble_omp_private(cStart, cEnd, totDim, plex, section, globalSection, JacP, elemMats, container);CHKERRQ(ierr);
  } else {  /* gpu assembly */
    ierr = assemble_cuda_private(cStart, cEnd, totDim, plex, section, globalSection, JacP, elemMats, container, events);CHKERRQ(ierr);
  }
  ierr = PetscFree(elemMats);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[6],0,0,0,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

__global__
void assemble_kernel(const PetscInt nidx_arr[], PetscInt *idx_arr[], PetscScalar *el_mats[], const ISColoringValue colors[], Mat_SeqAIJ mats[])
{
  const PetscInt     myelem = (gridDim.x==1) ? threadIdx.x : blockIdx.x;
  Mat_SeqAIJ         a = mats[colors[myelem]]; /* copy to GPU */
  const PetscScalar *v = el_mats[myelem];
  const PetscInt    *in = idx_arr[myelem], *im = idx_arr[myelem], n = nidx_arr[myelem], m = nidx_arr[myelem];
  /* mat set values */
  PetscInt          *rp,k,low,high,t,row,nrow,i,col,l;
  PetscInt          *ai = a.i,*ailen = a.ilen;
  PetscInt          *aj = a.j,lastcol = -1;
  MatScalar         *ap=NULL,value=0.0,*aa = a.a;
  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
    rp   = aj + ai[row];
    ap = aa + ai[row];
    nrow = ailen[row];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      /* if (in[l] < 0) { */
      /* 	printf("\t\tin[l] < 0 ?????\n"); */
      /* 	continue; */
      /* } */
      while (l<n && (value = v[l + k*n]) == 0.0) l++;
      if (l==n) break;
      col = in[l];
      if (col <= lastcol) low = 0;
      else high = nrow;
      lastcol = col;
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else low = t;
      }
      for (i=low; i<high; i++) {
        // if (rp[i] > col) break;
        if (rp[i] == col) {
	  ap[i] += value;
	  low = i + 1;
          goto noinsert;
        }
      }
      printf("\t\t\t ERROR in assemble_kernel\n");
    noinsert:;
    }
  }
}
static PetscErrorCode assemble_omp_private(PetscInt cStart, PetscInt cEnd, PetscInt totDim, DM plex, PetscSection section, PetscSection globalSection, Mat JacP, PetscScalar elemMats[], PetscContainer container)
{
  PetscErrorCode  ierr;
  IS             *is;
  PetscInt        nc,colour,j;
  const PetscInt *clr_idxs;
  ISColoring_ctx  coloring_ctx = NULL;
  ISColoring      iscoloring;
  ierr = PetscContainerGetPointer(container,(void**)&coloring_ctx);CHKERRQ(ierr);
  iscoloring = coloring_ctx->coloring;
  ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,&nc,&is);CHKERRQ(ierr);
  for (colour=0; colour<nc; colour++) {
    PetscInt    *idx_arr[64];
    PetscScalar *new_el_mats[64];
    PetscInt     idx_size[64],csize;
    ierr = ISGetLocalSize(is[colour],&csize);CHKERRQ(ierr);
    if (csize>64) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "too many elements in color. %D > %D",csize,64);
    ierr = ISGetIndices(is[colour],&clr_idxs);CHKERRQ(ierr);
    /* get indices and mats */
    for (j=0; j<csize; j++) {
      PetscInt cell = cStart + clr_idxs[j];
      PetscInt numindices,*indices;
      PetscScalar *elMat = &elemMats[clr_idxs[j]*totDim*totDim];
      PetscScalar *valuesOrig = elMat;
      ierr = DMPlexGetClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
      idx_size[j] = numindices;
      ierr = PetscMalloc2(numindices,&idx_arr[j],numindices*numindices,&new_el_mats[j]);CHKERRQ(ierr);
      ierr = PetscMemcpy(idx_arr[j],indices,numindices*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemcpy(new_el_mats[j],elMat,numindices*numindices*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = DMPlexRestoreClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
      if (elMat != valuesOrig) {ierr = DMRestoreWorkArray(plex, numindices*numindices, MPIU_SCALAR, &elMat);}
    }
    /* assemble matrix */
#pragma omp parallel for shared(JacP,idx_size,idx_arr,new_el_mats,colour,clr_idxs) private(j) schedule(static)
    for (j=0; j<csize; j++) {
      PetscInt numindices = idx_size[j], *indices = idx_arr[j];
      PetscScalar *elMat = new_el_mats[j];
      MatSetValues(JacP,numindices,indices,numindices,indices,elMat,ADD_VALUES);
    }
    /* free */
    ierr = ISRestoreIndices(is[colour],&clr_idxs);CHKERRQ(ierr);
    for (j=0; j<csize; j++) {
      ierr = PetscFree2(idx_arr[j],new_el_mats[j]);CHKERRQ(ierr);
    }
  }
  ierr = ISColoringRestoreIS(iscoloring,PETSC_USE_POINTER,&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode assemble_cuda_private(PetscInt cStart, PetscInt cEnd, PetscInt totDim, DM plex, PetscSection section, PetscSection globalSection, Mat JacP, PetscScalar elemMats[], PetscContainer container, const PetscLogEvent events[])
{
  PetscErrorCode    ierr;
#define FP_MAX_COLORS 16
#define FP_MAX_ELEMS 512
  Mat_SeqAIJ             h_mats[FP_MAX_COLORS], *jaca = (Mat_SeqAIJ *)JacP->data, *d_mats;
  const PetscInt         nelems = cEnd - cStart, nnz = jaca->i[JacP->rmap->n], N = JacP->rmap->n;  /* serial */
  const ISColoringValue *colors;
  ISColoringValue       *d_colors,colour;
  PetscInt              *h_idx_arr[FP_MAX_ELEMS], h_nidx_arr[FP_MAX_ELEMS], *d_nidx_arr, **d_idx_arr,nc,ej,j,cell;
  PetscScalar           *h_new_el_mats[FP_MAX_ELEMS], *val_buf, **d_new_el_mats;
  ISColoring_ctx         coloring_ctx = NULL;
  ISColoring             iscoloring;
  ierr = PetscContainerGetPointer(container,(void**)&coloring_ctx);CHKERRQ(ierr);
  iscoloring = coloring_ctx->coloring;
  /* get colors */
  ierr = ISColoringGetColors(iscoloring, &j, &nc, &colors);CHKERRQ(ierr);
  if (nelems>FP_MAX_ELEMS) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "too many elements. %D > %D",nelems,FP_MAX_ELEMS);
  if (nc>FP_MAX_COLORS) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "too many colors. %D > %D",nc,FP_MAX_COLORS);
  /* colors for kernel */
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_colors,         nelems*sizeof(ISColoringValue))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_colors, colors, nelems*sizeof(ISColoringValue), cudaMemcpyHostToDevice));
  /* get indices and element matrices */
  for (cell = cStart, ej = 0 ; cell < cEnd; ++cell, ++ej) {
    PetscInt numindices,*indices;
    PetscScalar *elMat = &elemMats[ej*totDim*totDim];
    PetscScalar *valuesOrig = elMat;
    ierr = DMPlexGetClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
    h_nidx_arr[ej] = numindices;
    CUDA_SAFE_CALL(cudaMalloc((void **)&h_idx_arr[ej],            numindices*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          h_idx_arr[ej],   indices, numindices*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&h_new_el_mats[ej],        numindices*numindices*sizeof(PetscScalar))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          h_new_el_mats[ej], elMat, numindices*numindices*sizeof(PetscScalar), cudaMemcpyHostToDevice));
    ierr = DMPlexRestoreClosureIndices(plex, section, globalSection, cell, PETSC_TRUE, &numindices, &indices, NULL, (PetscScalar **) &elMat);CHKERRQ(ierr);
    if (elMat != valuesOrig) {ierr = DMRestoreWorkArray(plex, numindices*numindices, MPIU_SCALAR, &elMat);CHKERRQ(ierr);}
  }
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_nidx_arr,                  nelems*sizeof(PetscInt))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_nidx_arr,    h_nidx_arr,   nelems*sizeof(PetscInt), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_idx_arr,                   nelems*sizeof(PetscInt*))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_idx_arr,     h_idx_arr,    nelems*sizeof(PetscInt*), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_new_el_mats,               nelems*sizeof(PetscScalar*))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy(          d_new_el_mats, h_new_el_mats,nelems*sizeof(PetscScalar*), cudaMemcpyHostToDevice));
  /* make matrix buffers */
  for (colour=0; colour<nc; colour++) {
    Mat_SeqAIJ *a = &h_mats[colour];
    /* create on GPU and copy to GPU */
    CUDA_SAFE_CALL(cudaMalloc((void **)&a->i,               (N+1)*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          a->i,    jaca->i,   (N+1)*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&a->ilen,            (N)*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          a->ilen, jaca->ilen,(N)*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&a->j,               (nnz)*sizeof(PetscInt))); // kernel input
    CUDA_SAFE_CALL(cudaMemcpy(          a->j,    jaca->j,   (nnz)*sizeof(PetscInt), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&a->a,               (nnz)*sizeof(PetscScalar))); // kernel output
    CUDA_SAFE_CALL(cudaMemset(          a->a, 0,            (nnz)*sizeof(PetscScalar)));
  }
  CUDA_SAFE_CALL(cudaMalloc(&d_mats,         nc*sizeof(Mat_SeqAIJ))); // kernel input
  CUDA_SAFE_CALL(cudaMemcpy( d_mats, h_mats, nc*sizeof(Mat_SeqAIJ), cudaMemcpyHostToDevice));
  /* do it */
  assemble_kernel<<<nelems,1>>>(d_nidx_arr, d_idx_arr, d_new_el_mats, d_colors, d_mats);
  CHECK_LAUNCH_ERROR();
  /* cleanup */
  CUDA_SAFE_CALL(cudaFree(d_colors));
  CUDA_SAFE_CALL(cudaFree(d_nidx_arr));
  for (ej = cStart ; ej < nelems; ++ej) {
    CUDA_SAFE_CALL(cudaFree(h_idx_arr[ej]));
    CUDA_SAFE_CALL(cudaFree(h_new_el_mats[ej]));
  }
  CUDA_SAFE_CALL(cudaFree(d_idx_arr));
  CUDA_SAFE_CALL(cudaFree(d_new_el_mats));
  /* copy & add Mat data back to CPU to JacP */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventBegin(events[2],0,0,0,0);CHKERRQ(ierr);
#endif
  ierr = PetscMalloc1(nnz,&val_buf);CHKERRQ(ierr);
  ierr = PetscMemzero(jaca->a,nnz*sizeof(PetscScalar));CHKERRQ(ierr);
  for (colour=0; colour<nc; colour++) {
    Mat_SeqAIJ *a = &h_mats[colour];
    CUDA_SAFE_CALL(cudaMemcpy(val_buf, a->a, (nnz)*sizeof(PetscScalar), cudaMemcpyDeviceToHost));
    PetscKernelAXPY(jaca->a,1.0,val_buf,nnz);
  }
  ierr = PetscFree(val_buf);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogEventEnd(events[2],0,0,0,0);CHKERRQ(ierr);
#endif
  for (colour=0; colour<nc; colour++) {
    Mat_SeqAIJ *a = &h_mats[colour];
    /* destroy mat */
    CUDA_SAFE_CALL(cudaFree(a->i));
    CUDA_SAFE_CALL(cudaFree(a->ilen));
    CUDA_SAFE_CALL(cudaFree(a->j));
    CUDA_SAFE_CALL(cudaFree(a->a));
  }
  CUDA_SAFE_CALL(cudaFree(d_mats));
  PetscFunctionReturn(0);
}
