#ifndef KALMAN_WRAPPER_H
#define KALMAN_WRAPPER_H

// This forces all inline functions to be static (file-local)
// which prevents the multiple definition errors
#define EXTERN_INLINE_MATRIX static inline
#define EXTERN_INLINE_KALMAN static inline

// Now include the actual headers
#include "compiler.h"
#include "matrix.h"
#include "kalman.h"
#include "cholesky.h"

#endif // KALMAN_WRAPPER_H