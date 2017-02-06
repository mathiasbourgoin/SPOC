/******************************************************************************
 * Mathias Bourgoin, Université Pierre et Marie Curie (2011)
 *
 * Mathias.Bourgoin@gmail.com
 *
 * This software is a computer program whose purpose is to allow
 * GPU programming with the OCaml language.
 *
 * This software is governed by the CeCILL-B license under French law and
 * abiding by the rules of distribution of free software.  You can  use,
 * modify and/ or redistribute the software under the terms of the CeCILL-B
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and,  more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-B license and that you accept its terms.
*******************************************************************************/
#ifndef _SPOC_CUBLAS_H_
#define _SPOC_CUBLAS_H_

#include <stdio.h>
#include <stdlib.h>

#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/callback.h>
#include <caml/fail.h>
#include <caml/bigarray.h>
#include <caml/signals.h>

#include <cublas.h>

#include "Spoc.h"

#define DEBUG 1

#ifdef _WIN32
#define FUNCTION __FUNCTION__
#else
#define FUNCTION __func__
#endif

#ifdef DEBUG
#define PRINT_FUNC \
	printf("IN: %s @ %d\n", FUNCTION,__LINE__ ); fflush(stdout);
#else
#define PRINT_FUNC \
	printf("IN: %s@ %d\n", FUNCTION,__LINE__ ); fflush(stdout);
#endif


#define RAISE_CUBLAS_ERROR \
	switch (cublas_error){ \
		case  CUBLAS_STATUS_NOT_INITIALIZED: \
			raise_constant(*caml_named_value("cublas_status_not_initialised")) ; \
			break; \
		case  CUBLAS_STATUS_ALLOC_FAILED: \
			raise_constant(*caml_named_value("cublas_status_alloc_failed")) ; \
			break; \
		case  CUBLAS_STATUS_INVALID_VALUE: \
			raise_constant(*caml_named_value("cublas_status_invalid_value")) ; \
			break; \
		case  CUBLAS_STATUS_ARCH_MISMATCH: \
			raise_constant(*caml_named_value("cublas_status_arch_mismatch")) ; \
			break; \
		case  CUBLAS_STATUS_MAPPING_ERROR: \
			raise_constant(*caml_named_value("cublas_status_mapping_error")) ; \
			break; \
		case  CUBLAS_STATUS_EXECUTION_FAILED: \
			raise_constant(*caml_named_value("cublas_status_execution_failed")) ; \
			break; \
		case  CUBLAS_STATUS_INTERNAL_ERROR: \
			raise_constant(*caml_named_value("cublas_status_internal_error")) ; \
			break; \
		default: \
			raise_constant(*caml_named_value("cublas_error_unknown")) ; \
			break;\
	}\

#define CUBLAS_CHECK_CALL(func) \
		cublas_error = func; \
		if (CUBLAS_STATUS_SUCCESS != cublas_error ) { \
			printf("IN: %s@ %d\n", FUNCTION,__LINE__ ); fflush(stdout); \
			RAISE_CUBLAS_ERROR\
		}

#define CUBLAS_GET_CONTEXT \
	{CUcontext ctx; \
	CUstream queue[2]; \
	spoc_cu_context *spoc_ctx; \
	enum cudaError_enum cuda_error = 0; \
	cublasStatus cublas_error = CUBLAS_STATUS_SUCCESS; \
	spoc_ctx = (spoc_cu_context*)Field(gi, 8); \
	ctx = spoc_ctx->ctx; \
	queue[0] = spoc_ctx->queue[0];\
	queue[1] = spoc_ctx->queue[1];\
	CUDA_CHECK_CALL(cuCtxSetCurrent(ctx)); \
	caml_enter_blocking_section();


#define CUBLAS_RESTORE_CONTEXT \
  caml_leave_blocking_section();       \
  spoc_ctx->queue[0] = queue[0];       \
  spoc_ctx->queue[1] = queue[1];		\
  Store_field(gi,8, (value)spoc_ctx); }


#endif
