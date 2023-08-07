#include <petsc/private/vecimpl.h>
#include <petsc/private/deviceimpl.h>
#include <petscmanagedmemory.hpp>

#include <vector>

using namespace Petsc;

class MarkVecGuard {
public:
  explicit MarkVecGuard(PetscDeviceContext dctx, std::initializer_list<std::pair<PetscMemoryAccessMode, Vec>> list) noexcept : dctx_{dctx}
  {
    PetscFunctionBegin;
    for (auto &&pair : list) {
      const auto obj = PetscObjectCast(pair.second);

      PetscCallCXXAbort(PETSC_COMM_SELF, {
        id_list_.emplace_back(obj->id);
        mode_list_.emplace_back(pair.first);
        name_list_.emplace_back(obj->name);
      });
    }
    PetscCallAbort(PetscObjectComm(dctx_), PetscDeviceContextMarkIntentFromIDBeginGroup(dctx_, id_list_.size(), id_list_.data(), mode_list_.data(), name_list_.data()));
    PetscFunctionReturnVoid();
  }

  explicit MarkVecGuard(PetscDeviceContext dctx, std::initializer_list<std::tuple<PetscMemoryAccessMode, PetscInt, Vec *>> list) noexcept : dctx_{dctx}
  {
    PetscFunctionBegin;
    for (auto &&tup : list) {
      const auto mode = std::get<0>(tup);
      const auto nvec = std::get<1>(tup);
      const auto vecs = std::get<2>(tup);
      for (PetscInt i = 0; i < nvec; ++i) {
        const auto obj = PetscObjectCast(vecs[i]);

        PetscCallCXXAbort(PETSC_COMM_SELF, {
          id_list_.emplace_back(obj->id);
          mode_list_.emplace_back(mode);
          name_list_.emplace_back(obj->name);
        });
      }
    }
    PetscCallAbort(PetscObjectComm(dctx_), PetscDeviceContextMarkIntentFromIDBeginGroup(dctx_, id_list_.size(), id_list_.data(), mode_list_.data(), name_list_.data()));
    PetscFunctionReturnVoid();
  }

  ~MarkVecGuard() noexcept
  {
    PetscFunctionBegin;
    PetscCallAbort(PetscObjectComm(dctx_), PetscDeviceContextMarkIntentFromIDEndGroup(dctx_, id_list_.size(), id_list_.data(), mode_list_.data(), name_list_.data()));
    id_list_.clear();
    mode_list_.clear();
    name_list_.clear();
    PetscFunctionReturnVoid();
  }

private:
  PetscDeviceContext                        dctx_;
  static std::vector<PetscObjectId>         id_list_;
  static std::vector<PetscMemoryAccessMode> mode_list_;
  static std::vector<const char *>          name_list_;
};

std::vector<PetscObjectId>         MarkVecGuard::id_list_;
std::vector<PetscMemoryAccessMode> MarkVecGuard::mode_list_;
std::vector<const char *>          MarkVecGuard::name_list_;

PetscErrorCode VecCopyAsync(Vec x, Vec y, PetscDeviceContext dctx)
{
  PetscBool flgs[4];
  PetscReal norms[4] = {0.0, 0.0, 0.0, 0.0};

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  if (x == y) PetscFunctionReturn(PETSC_SUCCESS);
  VecCheckSameLocalSize(x, 1, y, 2);
  VecCheckAssembled(x);
  PetscCall(VecSetErrorIfLocked(y, 2));

  for (PetscInt i = 0; i < 4; i++) PetscCall(PetscObjectComposedDataGetReal(PetscObjectCast(x), NormIds[i], norms[i], flgs[i]));

  {
    auto _ = MarkVecGuard{
      dctx, {{PETSC_MEMORY_ACCESS_READ, x}, {PETSC_MEMORY_ACCESS_WRITE, y}}
    };

    PetscCall(PetscLogEventBegin(VEC_Copy, x, y, nullptr, nullptr));
    PetscUseTypeMethod(x, copy_async, y, dctx);
    PetscCall(PetscLogEventEnd(VEC_Copy, x, y, nullptr, nullptr));
  }

  PetscCall(PetscObjectStateIncrease(PetscObjectCast(y)));
  for (PetscInt i = 0; i < 4; i++) {
    if (flgs[i]) PetscCall(PetscObjectComposedDataSetReal(PetscObjectCast(y), NormIds[i], norms[i]));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecAYPXAsync(Vec y, const ManagedScalar &beta, Vec x, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidType(x, 3);
  PetscValidType(y, 1);
  PetscCheckSameTypeAndComm(x, 3, y, 1);
  VecCheckSameSize(x, 1, y, 3);
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  // PetscValidLogicalCollectiveScalar(y, beta.front(dctx), 2);
  PetscCall(VecSetErrorIfLocked(y, 1));
  if (x == y) {
    // ASYNC TODO
    PetscCall(VecScale(y, beta.cfront(dctx) + 1.0));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecLockReadPush(x));
  if (beta.KnownAndEqual(0.0)) {
    PetscCall(VecCopyAsync(x, y, dctx));
  } else {
    auto _ = MarkVecGuard{
      dctx, {{PETSC_MEMORY_ACCESS_READ, x}, {PETSC_MEMORY_ACCESS_READ_WRITE, y}}
    };

    PetscCall(PetscLogEventBegin(VEC_AYPX, x, y, nullptr, nullptr));
    PetscUseTypeMethod(y, aypx_async, beta, x, dctx);
    PetscCall(PetscLogEventEnd(VEC_AYPX, x, y, nullptr, nullptr));
    PetscCall(PetscObjectStateIncrease(PetscObjectCast(y)));
  }
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecDotAsync(Vec x, Vec y, ManagedScalar *val, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  //PetscValidScalarPointer(val, 3);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  PetscCheckSameTypeAndComm(x, 1, y, 2);
  VecCheckSameSize(x, 1, y, 2);
  VecCheckAssembled(x);
  VecCheckAssembled(y);

  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  {
    auto _ = MarkVecGuard{
      dctx, {{PETSC_MEMORY_ACCESS_READ, x}, {PETSC_MEMORY_ACCESS_READ, y}}
    };

    PetscCall(PetscLogEventBegin(VEC_Dot, x, y, nullptr, nullptr));
    PetscUseTypeMethod(x, dot_async, y, *val, dctx);
    PetscCall(PetscLogEventEnd(VEC_Dot, x, y, nullptr, nullptr));
  }
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecTDotAsync(Vec x, Vec y, ManagedScalar *val, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  //PetscValidScalarPointer(val, 3);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  PetscCheckSameTypeAndComm(x, 1, y, 2);
  VecCheckSameSize(x, 1, y, 2);
  VecCheckAssembled(x);
  VecCheckAssembled(y);

  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  {
    auto _ = MarkVecGuard{
      dctx, {{PETSC_MEMORY_ACCESS_READ, x}, {PETSC_MEMORY_ACCESS_READ, y}}
    };

    PetscCall(PetscLogEventBegin(VEC_TDot, x, y, nullptr, nullptr));
    PetscUseTypeMethod(x, tdot_async, y, *val, dctx);
    PetscCall(PetscLogEventEnd(VEC_TDot, x, y, nullptr, nullptr));
  }
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecNormAsync(Vec x, NormType type, ManagedReal *val, PetscDeviceContext dctx) noexcept
{
  PetscReal avail;
  PetscBool flg = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  PetscValidLogicalCollectiveEnum(x, type, 2);
  //PetscValidRealPointer(val, 3);

  /* Cached data? */
  PetscCall(VecNormAvailable(x, type, &flg, &avail));
  if (flg) {
    val->front(dctx) = avail;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(VecLockReadPush(x));
  {
    auto _ = MarkVecGuard{dctx, {{PETSC_MEMORY_ACCESS_READ, x}}};

    PetscCall(PetscLogEventBegin(VEC_Norm, x, nullptr, nullptr, nullptr));
    PetscUseTypeMethod(x, norm_async, type, val, dctx);
    PetscCall(PetscLogEventEnd(VEC_Norm, x, nullptr, nullptr, nullptr));
  }
  PetscCall(VecLockReadPop(x));

  // if (type != NORM_1_AND_2) {
  //   const PetscReal &v = val->cfront(dctx);

  //   // ASYNC TODO
  //   PetscCall(PetscObjectComposedDataSetReal(PetscObjectCast(x), NormIds[type], v));
  // }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecAXPYAsync(Vec y, const ManagedScalar &alpha, Vec x, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidType(x, 3);
  PetscValidType(y, 1);
  PetscCheckSameTypeAndComm(x, 3, y, 1);
  VecCheckSameSize(x, 3, y, 1);
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  //PetscValidLogicalCollectiveScalar(y, alpha, 2);
  if (alpha.KnownAndEqual(0.0)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecSetErrorIfLocked(y, 1));
  if (x == y) {
    PetscCall(VecScale(y, alpha.cfront(dctx) + 1.0));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecLockReadPush(x));
  {
    auto _ = MarkVecGuard{
      dctx, {{PETSC_MEMORY_ACCESS_READ, x}, {PETSC_MEMORY_ACCESS_READ_WRITE, y}}
    };

    PetscCall(PetscLogEventBegin(VEC_AXPY, x, y, nullptr, nullptr));
    PetscUseTypeMethod(y, axpy_async, alpha, x, dctx);
    PetscCall(PetscLogEventEnd(VEC_AXPY, x, y, nullptr, nullptr));
  }
  PetscCall(VecLockReadPop(x));
  PetscCall(PetscObjectStateIncrease(PetscObjectCast(y)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecPointwiseMultAsync(Vec w, Vec x, Vec y, PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(w, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 3);
  PetscValidType(w, 1);
  PetscValidType(x, 2);
  PetscValidType(y, 3);
  PetscCheckSameTypeAndComm(x, 2, y, 3);
  PetscCheckSameTypeAndComm(y, 3, w, 1);
  VecCheckSameSize(w, 1, x, 2);
  VecCheckSameSize(w, 1, y, 3);
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  PetscCall(VecSetErrorIfLocked(w, 1));

  {
    auto _ = MarkVecGuard{
      dctx, {{PETSC_MEMORY_ACCESS_READ, x}, {PETSC_MEMORY_ACCESS_READ, y}, {PETSC_MEMORY_ACCESS_WRITE, w}}
    };

    PetscCall(PetscLogEventBegin(VEC_PointwiseMult, x, y, w, nullptr));
    PetscUseTypeMethod(w, pointwisemult_async, x, y, dctx);
    PetscCall(PetscLogEventEnd(VEC_PointwiseMult, x, y, w, nullptr));
  }
  PetscCall(PetscObjectStateIncrease(PetscObjectCast(w)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecScaleAsync(Vec x, const ManagedScalar &alpha, PetscDeviceContext dctx) noexcept
{
  // PetscReal norms[4];
  // PetscBool flgs[4];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  PetscCall(VecSetErrorIfLocked(x, 1));
  if (alpha.KnownAndEqual(1.0)) PetscFunctionReturn(PETSC_SUCCESS);

  /* get current stashed norms */
  //for (PetscInt i = 0; i < 4; i++) PetscCall(PetscObjectComposedDataGetReal((PetscObject)x, NormIds[i], norms[i], flgs[i]));

  {
    auto _ = MarkVecGuard{dctx, {std::make_pair(PETSC_MEMORY_ACCESS_READ_WRITE, x)}};

    PetscCall(PetscLogEventBegin(VEC_Scale, x, nullptr, nullptr, nullptr));
    PetscUseTypeMethod(x, scale_async, alpha, dctx);
    PetscCall(PetscLogEventEnd(VEC_Scale, x, nullptr, nullptr, nullptr));
  }

  PetscCall(PetscObjectStateIncrease(PetscObjectCast(x)));
  /* put the scaled stashed norms back into the Vec */
  // for (PetscInt i = 0; i < 4; i++) {
  //   if (flgs[i]) PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[i], PetscAbsScalar(alpha) * norms[i]));
  // }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecSetAsync(Vec x, const ManagedScalar &alpha, PetscDeviceContext dctx) noexcept
{
  const auto pobj = PetscObjectCast(x);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  //PetscValidLogicalCollectiveScalar(x, alpha, 2);
  PetscCall(VecSetErrorIfLocked(x, 1));

  {
    auto _ = MarkVecGuard{dctx, {std::make_pair(PETSC_MEMORY_ACCESS_WRITE, x)}};

    PetscCall(PetscLogEventBegin(VEC_Set, x, nullptr, nullptr, nullptr));
    PetscUseTypeMethod(x, set_async, alpha, dctx);
    PetscCall(PetscLogEventEnd(VEC_Set, x, nullptr, nullptr, nullptr));
  }

  PetscCall(PetscObjectStateIncrease(pobj));

  /*  norms can be simply set (if |alpha|*N not too large) */
  {
    const auto N = x->map->N;

    if (N == 0) {
      PetscCall(PetscObjectComposedDataSetReal(pobj, NormIds[NORM_1], 0.0l));
      PetscCall(PetscObjectComposedDataSetReal(pobj, NormIds[NORM_INFINITY], 0.0));
      PetscCall(PetscObjectComposedDataSetReal(pobj, NormIds[NORM_2], 0.0));
      PetscCall(PetscObjectComposedDataSetReal(pobj, NormIds[NORM_FROBENIUS], 0.0));
    } else {
#if 0
      PetscReal val = PetscAbsScalar(alpha);

      if (val > PETSC_MAX_REAL / N) {
        PetscCall(PetscObjectComposedDataSetReal(pobj, NormIds[NORM_INFINITY], val));
      } else {
        PetscCall(PetscObjectComposedDataSetReal(pobj, NormIds[NORM_1], N * val));
        PetscCall(PetscObjectComposedDataSetReal(pobj, NormIds[NORM_INFINITY], val));
        val *= PetscSqrtReal(static_cast<PetscReal>(N));
        PetscCall(PetscObjectComposedDataSetReal(pobj, NormIds[NORM_2], val));
        PetscCall(PetscObjectComposedDataSetReal(pobj, NormIds[NORM_FROBENIUS], val));
      }
#endif
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecWAXPYAsync(Vec w, const ManagedScalar &alpha, Vec x, Vec y, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(w, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 4);
  PetscValidType(w, 1);
  PetscValidType(x, 3);
  PetscValidType(y, 4);
  PetscCheckSameTypeAndComm(x, 3, y, 4);
  PetscCheckSameTypeAndComm(y, 4, w, 1);
  VecCheckSameSize(x, 3, y, 4);
  VecCheckSameSize(x, 3, w, 1);
  PetscCheck(w != y, PETSC_COMM_SELF, PETSC_ERR_SUP, "Result vector w cannot be same as input vector y, suggest VecAXPY()");
  PetscCheck(w != x, PETSC_COMM_SELF, PETSC_ERR_SUP, "Result vector w cannot be same as input vector x, suggest VecAYPX()");
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  PetscCall(VecSetErrorIfLocked(w, 1));

  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (alpha.KnownAndEqual(0.0)) {
    PetscCall(VecCopyAsync(y, w, dctx));
  } else {
    auto _ = MarkVecGuard{
      dctx, {{PETSC_MEMORY_ACCESS_WRITE, w}, {PETSC_MEMORY_ACCESS_READ, x}, {PETSC_MEMORY_ACCESS_READ, y}}
    };

    PetscCall(PetscLogEventBegin(VEC_WAXPY, x, y, w, nullptr));
    PetscUseTypeMethod(w, waxpy_async, alpha, x, y, dctx);
    PetscCall(PetscLogEventEnd(VEC_WAXPY, x, y, w, nullptr));
    PetscCall(PetscObjectStateIncrease(PetscObjectCast(w)));
  }
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecNormalizeAsync(Vec x, ManagedReal *val, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(VecSetErrorIfLocked(x, 1));
  PetscAssertPointer(val, 2);
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(PetscLogEventBegin(VEC_Normalize, x, nullptr, nullptr, nullptr));
  PetscCall(VecNormAsync(x, NORM_2, val, dctx));
  {
    ManagedScalar scal{Eval(1.0 / (*val), dctx)};

    PetscCall(VecScaleAsync(x, scal, dctx));
  }
  PetscCall(PetscLogEventEnd(VEC_Normalize, x, nullptr, nullptr, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecMAXPYAsync(Vec y, PetscInt nv, const Petsc::ManagedScalar *alpha, Vec x[], PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  VecCheckAssembled(y);
  PetscValidLogicalCollectiveInt(y, nv, 2);
  PetscCall(VecSetErrorIfLocked(y, 1));
  PetscCheck(nv >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of vectors (given %" PetscInt_FMT ") cannot be negative", nv);
  if (nv) {
    PetscInt zeros = 0;

    PetscAssertPointer(alpha, 3);
    PetscAssertPointer(x, 4);
    for (PetscInt i = 0; i < nv; ++i) {
      //PetscValidLogicalCollectiveScalar(y, alpha[i], 3);
      PetscValidHeaderSpecific(x[i], VEC_CLASSID, 4);
      PetscValidType(x[i], 4);
      PetscCheckSameTypeAndComm(y, 1, x[i], 4);
      VecCheckSameSize(y, 1, x[i], 4);
      PetscCheck(y != x[i], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Array of vectors 'x' cannot contain y, found x[%" PetscInt_FMT "] == y", i);
      VecCheckAssembled(x[i]);
      PetscCall(VecLockReadPush(x[i]));
      zeros += alpha[i].KnownAndEqual(0.0);
    }

    if (zeros < nv) {
      PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));

      auto _ = MarkVecGuard{
        dctx, {{PETSC_MEMORY_ACCESS_READ_WRITE, 1, &y}, {PETSC_MEMORY_ACCESS_READ, nv, x}}
      };

      PetscCall(PetscLogEventBegin(VEC_MAXPY, y, *x, nullptr, nullptr));
      PetscUseTypeMethod(y, maxpy_async, nv, alpha, x, dctx);
      PetscCall(PetscLogEventEnd(VEC_MAXPY, y, *x, nullptr, nullptr));
      PetscCall(PetscObjectStateIncrease((PetscObject)y));
    }

    for (PetscInt i = 0; i < nv; ++i) PetscCall(VecLockReadPop(x[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
