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
#include "Spoc_cublas.h"

//cublas library

CAMLprim value spoc_cublasInit(){
	cublasStatus cublas_error = CUBLAS_STATUS_SUCCESS;
	CUBLAS_CHECK_CALL(cublasInit());
	return Val_unit;
}

CAMLprim value spoc_cublasShutdown(){
	cublasStatus cublas_error= CUBLAS_STATUS_SUCCESS;
	CUBLAS_CHECK_CALL(cublasShutdown());
	return Val_unit;
}

CAMLprim value spoc_cublasGetError(){
	cublasStatus cublas_error = CUBLAS_STATUS_SUCCESS;
	CUBLAS_CHECK_CALL(cublasGetError());
	return Val_unit;
}

/*********************************************************/

#define GET_HOST_VEC(vec, h_vec) \
	bigArray = Field (Field(vec, 1), 0); \
	h_vec = (void*) Data_bigarray_val(bigArray);

#define GET_VEC(vec, d_vec) \
	dev_vec_array = Field(vec, 2); \
	gi = Field(dev, 0);	       \
	id = Int_val(Field(gi, 7));    \
	dev_vec =Field(dev_vec_array, id); \	
	d_vec = CUdeviceptr_val(Field(dev_vec, 1));

/*********************************************************/

#define CUBLAS_FUN(fun, arg, arg2) \
		CAMLprim value fun##_n arg {\
			return fun arg2 ;\
		}\
		CAMLprim value fun##_b arg {\
			return fun arg2 ;\
		}

#define CUBLAS_FUN6(fun, arg, arg2) \
		CAMLprim value fun##_n arg {\
			return fun arg2 ;\
		}\
		CAMLprim value fun##_b (value * tab_val, int num_val){\
			return fun (tab_val[0], tab_val[1], tab_val[2], tab_val[3], tab_val[4], tab_val[5]) ;\
		}

#define CUBLAS_FUN7(fun, arg, arg2) \
		CAMLprim value fun##_n arg {\
			return fun arg2 ;\
		}\
		CAMLprim value fun##_b (value * tab_val, int num_val){\
			return fun (tab_val[0], tab_val[1], tab_val[2], tab_val[3], tab_val[4], tab_val[5], tab_val[6]) ;\
		}


#define CUBLAS_FUN8(fun, arg, arg2) \
		CAMLprim value fun##_n arg {\
			return fun arg2 ;\
		}\
		CAMLprim value fun##_b (value * tab_val, int num_val){\
			return fun (tab_val[0], tab_val[1], tab_val[2], tab_val[3], tab_val[4], tab_val[5], tab_val[6], tab_val[7]) ;\
		}

#define CUBLAS_FUN9(fun, arg, arg2) \
		CAMLprim value fun##_n arg {\
			return fun arg2 ;\
		}\
		CAMLprim value fun##_b (value * tab_val, int num_val){\
			return fun (tab_val[0], tab_val[1], tab_val[2], tab_val[3], tab_val[4], tab_val[5], tab_val[6], tab_val[7], tab_val[8]) ;\
		}

#define CUBLAS_FUN10(fun, arg, arg2) \
		CAMLprim value fun##_n arg {\
			return fun arg2 ;\
		}\
		CAMLprim value fun##_b (value * tab_val, int num_val){\
			return fun (tab_val[0], tab_val[1], tab_val[2], tab_val[3], tab_val[4], tab_val[5], tab_val[6], tab_val[7], tab_val[8], tab_val[9]) ;\
		}

#define CUBLAS_FUN11(fun, arg, arg2) \
		CAMLprim value fun##_n arg {\
			return fun arg2 ;\
		}\
		CAMLprim value fun##_b (value * tab_val, int num_val){\
			return fun (tab_val[0], tab_val[1], tab_val[2], tab_val[3], tab_val[4], tab_val[5], tab_val[6], tab_val[7], tab_val[8], tab_val[9], tab_val[10]) ;\
		}

#define CUBLAS_FUN12(fun, arg, arg2) \
		CAMLprim value fun##_n arg {\
			return fun arg2 ;\
		}\
		CAMLprim value fun##_b (value * tab_val, int num_val){\
			return fun (tab_val[0], tab_val[1], tab_val[2], tab_val[3], tab_val[4], tab_val[5], tab_val[6], tab_val[7], tab_val[8], tab_val[9], tab_val[10], tab_val[11]) ;\
		}

#define CUBLAS_FUN13(fun, arg, arg2) \
		CAMLprim value fun##_n arg {\
			return fun arg2 ;\
		}\
		CAMLprim value fun##_b (value * tab_val, int num_val){\
			return fun (tab_val[0], tab_val[1], tab_val[2], tab_val[3], tab_val[4], tab_val[5], tab_val[6], tab_val[7], tab_val[8], tab_val[9], tab_val[10], tab_val[11], tab_val[12]) ;\
		}

#define CUBLAS_FUN14(fun, arg, arg2) \
		CAMLprim value fun##_n arg {\
			return fun arg2 ;\
		}\
		CAMLprim value fun##_b (value * tab_val, int num_val){\
			return fun (tab_val[0], tab_val[1], tab_val[2], tab_val[3], tab_val[4], tab_val[5], tab_val[6], tab_val[7], tab_val[8], tab_val[9], tab_val[10], tab_val[11], tab_val[12], tab_val[13]) ;\
		}

#define CUBLAS_FUN15(fun, arg, arg2) \
		CAMLprim value fun##_n arg {\
			return fun arg2 ;\
		}\
		CAMLprim value fun##_b (value * tab_val, int num_val){\
			return fun (tab_val[0], tab_val[1], tab_val[2], tab_val[3], tab_val[4], tab_val[5], tab_val[6], tab_val[7], tab_val[8], tab_val[9], tab_val[10], tab_val[11], tab_val[12], tab_val[13], tab_val[14]) ;\
		}

#define CUBLAS_FUN16(fun, arg, arg2) \
		CAMLprim value fun##_n arg {\
			return fun arg2 ;\
		}\
		CAMLprim value fun##_b (value * tab_val, int num_val){\
			return fun (tab_val[0], tab_val[1], tab_val[2], tab_val[3], tab_val[4], tab_val[5], tab_val[6], tab_val[7], tab_val[8], tab_val[9], tab_val[10], tab_val[11], tab_val[12], tab_val[13], tab_val[14], tab_val[15]) ;\
		}
/******************************* BLAS1 ***************************************/

/* Single-Precision BLAS1 functions */

CAMLprim value spoc_cublasIsamax(value n, value x, value incx, value dev){
	CAMLparam4(n,x,incx, dev);
	CAMLlocal3(dev_vec_array, dev_vec, gi);
	int res;
	int id;
	CUdeviceptr d_A;
	GET_VEC(x, d_A);
	CUBLAS_GET_CONTEXT;
	res = cublasIsamax(Int_val(n), (float*)d_A, Int_val(incx));
	CUBLAS_CHECK_CALL(cublasGetError());
	CUDA_RESTORE_CONTEXT;

	CAMLreturn(Val_int(res));
}

CAMLprim value spoc_cublasIsamin(value n, value x, value incx, value dev){
	CAMLparam4(n,x,incx, dev);
	CAMLlocal3(dev_vec_array, dev_vec, gi);
	int res;
	int id;
	CUdeviceptr d_A;
	GET_VEC(x, d_A);
	CUBLAS_GET_CONTEXT;
	res = cublasIsamin(Int_val(n), (float*)d_A, Int_val(incx));
	CUBLAS_CHECK_CALL(cublasGetError());
	CUDA_RESTORE_CONTEXT;
	CAMLreturn(Val_int(res));
}

CAMLprim value spoc_cublasSasum (value n, value x, value incx, value dev){
	CAMLparam4(n,x,incx, dev);
	CAMLlocal4(dev_vec_array, dev_vec, res, gi);
	CUdeviceptr d_A;
	float result;
	int id;
	GET_VEC(x, d_A);
	CUBLAS_GET_CONTEXT;
	result = cublasSasum(Int_val(n), (float*)d_A, Int_val(incx));
	CUBLAS_CHECK_CALL(cublasGetError());
	res = caml_copy_double((double)result);
	CUDA_RESTORE_CONTEXT;
	CAMLreturn((res));
}

CAMLprim value spoc_cublasSaxpy (value n, value alpha, value x, value incx, value y, value incy, value dev){
	CAMLparam5(n,alpha, x,incx, y);
	CAMLxparam2(incy, dev);
	CAMLlocal3(dev_vec_array, dev_vec, gi);
	int id;
	CUdeviceptr d_A;
	CUdeviceptr d_B;
	GET_VEC(x, d_A);
	GET_VEC(y, d_B);
	CUBLAS_GET_CONTEXT;
	cublasSaxpy(Int_val(n), (float)(Double_val(alpha)), (float*)d_A, Int_val(incx), (float*)d_B, Int_val(incy));
	CUBLAS_CHECK_CALL(cublasGetError());
	CUDA_RESTORE_CONTEXT;
	CAMLreturn(Val_unit);
}

CAMLprim value spoc_cublasScopy (value n, value x, value incx, value y, value incy, value dev){
	CAMLparam5(n,x,incx, y, incy);
	CAMLxparam1(dev);
	CAMLlocal3(dev_vec_array, dev_vec, gi);
	int id;
	CUdeviceptr d_A;
	CUdeviceptr d_B;
	GET_VEC(x, d_A);
	GET_VEC(y, d_B);
	CUBLAS_GET_CONTEXT;
	cublasScopy(Int_val(n), (float*)d_A, Int_val(incx), (float*)d_B, Int_val(incy));
	CUBLAS_CHECK_CALL(cublasGetError());
	CUDA_RESTORE_CONTEXT;
	CAMLreturn(Val_unit);
}

CAMLprim value spoc_cublasSdot (value n, value x, value incx, value y, value incy, value dev){
	CAMLparam5(n,x,incx, y, incy);
	CAMLxparam1(dev);
	CAMLlocal4(dev_vec_array, dev_vec, res, gi);
	float result;
	int id;
	CUdeviceptr d_A;
	CUdeviceptr d_B;
	GET_VEC(x, d_A);
	GET_VEC(y, d_B);
	CUBLAS_GET_CONTEXT;
	result = cublasSdot(Int_val(n), (float*)d_A, Int_val(incx), (float*)d_B, Int_val(incy));
	CUBLAS_CHECK_CALL(cublasGetError());
	res = caml_copy_double((double)result);
	CUDA_RESTORE_CONTEXT;
	CAMLreturn(res);
}

CAMLprim value spoc_cublasSnrm2 (value n, value x, value incx, value dev){
	CAMLparam4(n,x,incx, dev);
	CAMLlocal4(dev_vec_array, dev_vec, res, gi);
	CUdeviceptr d_A;
	int id;
	float result;
	GET_VEC(x, d_A);
	CUBLAS_GET_CONTEXT;
	result = cublasSnrm2(Int_val(n), (float*)d_A, Int_val(incx));
	CUBLAS_CHECK_CALL(cublasGetError());
	res = caml_copy_double((double)result);
	CUDA_RESTORE_CONTEXT;
	CAMLreturn((res));
}

CAMLprim value spoc_cublasSrot (value n, value x, value incx, value y, value incy, value sc, value ss, value dev){
	CAMLparam5(n,x,incx, y, incy);
	CAMLxparam3(sc, ss, dev);
	CAMLlocal4(dev_vec_array, dev_vec, res, gi);
	int id;
	CUdeviceptr d_A;
	CUdeviceptr d_B;
	float result;
	GET_VEC(x, d_A);
	GET_VEC(y, d_B);
	CUBLAS_GET_CONTEXT;

	cublasSrot(Int_val(n), (float*)d_A, Int_val(incx), (float*)d_B, Int_val(incy), (float)(Double_val(sc)), (float)(Double_val(ss)));
	CUBLAS_CHECK_CALL(cublasGetError());
	CUDA_RESTORE_CONTEXT;
	CAMLreturn(Val_unit);
}

CAMLprim value spoc_cublasSrotg (value host_sa, value host_sb, value host_sc, value host_ss){
	CAMLparam4(host_sa, host_sb, host_sc, host_ss);
	CAMLlocal2(bigArray, gi);
	int id;
	enum cudaError_enum cuda_error = 0;
	cublasStatus cublas_error = CUBLAS_STATUS_SUCCESS;
	float* h_A;
	float* h_B;
	float* h_C;
	float* h_D;
	float result;
	GET_HOST_VEC(host_sa, h_A);
	GET_HOST_VEC(host_sb, h_B);
	GET_HOST_VEC(host_sc, h_C);
	GET_HOST_VEC(host_ss, h_D);
	cublasSrotg(h_A, h_B, h_C, h_D);
	CUBLAS_CHECK_CALL(cublasGetError());
	CAMLreturn(Val_unit);
}

CAMLprim value spoc_cublasSrotm (value n, value x, value incx, value y, value incy, value sparam, value dev){
	CAMLparam5(n,x,incx, y, incy);
	CAMLxparam2(sparam, dev);
	CAMLlocal4(dev_vec_array, dev_vec, res, gi);
	CUdeviceptr d_A;
	CUdeviceptr d_B;
	CUdeviceptr d_C;
	float result;
	int id;
	GET_VEC(x, d_A);
	GET_VEC(y, d_B);
	GET_VEC(sparam, d_C);
	CUBLAS_GET_CONTEXT;

	cublasSrotm(Int_val(n), (float*)d_A, Int_val(incx), (float*)d_B, Int_val(incy), (float*)sparam);
	CUBLAS_CHECK_CALL(cublasGetError());
	CUBLAS_RESTORE_CONTEXT;
	CAMLreturn(Val_unit);
}

CAMLprim value spoc_cublasSrotmg (value host_psd1, value host_psd2, value host_psx1, value host_psy1, value host_sparam){
	CAMLparam5(host_psd1, host_psd2, host_psx1, host_psy1, host_sparam);
	CAMLlocal2(bigArray, gi);
	int id;
	enum cudaError_enum cuda_error = 0;
	cublasStatus cublas_error = CUBLAS_STATUS_SUCCESS;
	float* h_A;
	float* h_B;
	float* h_C;
	float* h_D;
	float* h_E;
	GET_HOST_VEC(host_psd1, h_A);
	GET_HOST_VEC(host_psd2, h_B);
	GET_HOST_VEC(host_psx1, h_C);
	GET_HOST_VEC(host_psy1, h_D);
	GET_HOST_VEC(host_sparam, h_E);
	CUBLAS_GET_CONTEXT;

	cublasSrotmg(h_A, h_B, h_C, h_D, h_E);
	CUBLAS_CHECK_CALL(cublasGetError());
	CUBLAS_RESTORE_CONTEXT;
	CAMLreturn(Val_unit);
}

CAMLprim value spoc_cublasSscal (value n, value alpha, value x, value incx, value dev){
	CAMLparam5(n, alpha, x,incx, dev);
	CAMLlocal3(dev_vec_array, dev_vec, gi);
	CUdeviceptr d_A;
	int id;
	GET_VEC(x, d_A);
	CUBLAS_GET_CONTEXT;

	cublasSscal(Int_val(n), (float)(Double_val(alpha)), (float*)d_A, Int_val(incx));
	CUBLAS_CHECK_CALL(cublasGetError());
	CUDA_RESTORE_CONTEXT;
	CAMLreturn(Val_unit);
}

CAMLprim value spoc_cublasSswap (value n, value x, value incx, value y, value incy, value dev){
	CAMLparam5(n, x,incx, y, incy);
	CAMLxparam1(dev);
	CAMLlocal3(dev_vec_array, dev_vec, gi);
	CUdeviceptr d_A;
	CUdeviceptr d_B;
	int id;
	GET_VEC(x, d_A);
	GET_VEC(y, d_B);
	CUBLAS_GET_CONTEXT;

	cublasSswap(Int_val(n), (float*)d_A, Int_val(incx), (float*)d_B, Int_val(incy));
	CUBLAS_CHECK_CALL(cublasGetError());
	CUDA_RESTORE_CONTEXT;
	CAMLreturn(Val_unit);
}

CUBLAS_FUN(spoc_cublasIsamax, (value n, value x, value incx, value dev), (n, x, incx, dev));
CUBLAS_FUN(spoc_cublasIsamin, (value n, value x, value incx, value dev), (n, x, incx, dev));
CUBLAS_FUN(spoc_cublasSasum, (value n, value x, value incx, value dev), (n, x, incx, dev));
CUBLAS_FUN7(spoc_cublasSaxpy, (value n, value alpha, value x, value incx, value y, value incy, value dev), (n, alpha, x, incx, y, incy, dev));
CUBLAS_FUN6(spoc_cublasScopy, (value n, value x, value incx, value y, value incy, value dev), (n, x, incx, y, incy, dev));
CUBLAS_FUN6(spoc_cublasSdot, (value n, value x, value incx, value y, value incy, value dev), (n, x, incx, y, incy, dev));
CUBLAS_FUN(spoc_cublasSnrm2, (value n, value x, value incx, value dev), (n, x, incx, dev));
CUBLAS_FUN8(spoc_cublasSrot, (value n, value x, value incx, value y, value incy, value sc, value ss, value dev), (n, x, incx, y, incy, sc, ss, dev));
CUBLAS_FUN(spoc_cublasSrotg, (value host_sa, value host_sb, value host_sc, value host_ss), (host_sa, host_sb, host_sc, host_ss));
CUBLAS_FUN7(spoc_cublasSrotm, (value n, value x, value incx, value y, value incy, value sparam, value dev), (n, x, incx, y, incy, sparam, dev));
CUBLAS_FUN(spoc_cublasSrotmg, (value host_psd1, value host_psd2, value host_psx1, value host_psy1, value host_sparam), (host_psd1, host_psd2, host_psx1, host_psy1, host_sparam));
CUBLAS_FUN(spoc_cublasSscal, (value n, value alpha, value x, value incx, value dev), (n, alpha, x, incx, dev));
CUBLAS_FUN6(spoc_cublasSswap, (value n, value x, value incx, value y, value incy, value dev), (n, x, incx, y, incy, dev));

/* Single-Precision Complex BLAS1 functions */


CAMLprim value spoc_cublasCaxpy (value n, value alpha, value x, value incx, value y, value incy, value dev){
	CAMLparam5(n,alpha, x,incx, y);
	CAMLxparam2(incy, dev);
	CAMLlocal3(dev_vec_array, dev_vec, gi);
	CUdeviceptr d_A;
	CUdeviceptr d_B;
	int id;
	GET_VEC(x, d_A);
	GET_VEC(y, d_B);
	CUBLAS_GET_CONTEXT;
	cublasCaxpy(Int_val(n), Complex_val(alpha), (cuComplex*)d_A, Int_val(incx), (cuComplex*)d_B, Int_val(incy));
	CUBLAS_CHECK_CALL(cublasGetError());
	CUDA_RESTORE_CONTEXT;
	CAMLreturn(Val_unit);
}


CAMLprim value spoc_cublasScasum (value n, value x, value incx, value dev){
	CAMLparam4(n,x,incx, dev);
	CAMLlocal4(dev_vec_array, dev_vec, res, gi);
	CUdeviceptr d_A;
	float result;
	int id;
	GET_VEC(x, d_A);
	CUBLAS_GET_CONTEXT;

	result = cublasScasum(Int_val(n), (cuComplex*)d_A, Int_val(incx));
	CUBLAS_CHECK_CALL(cublasGetError());
	res = caml_copy_double((double)result);
	CUDA_RESTORE_CONTEXT;
	CAMLreturn((res));
}

CUBLAS_FUN7(spoc_cublasCaxpy, (value n, value alpha, value x, value incx, value y, value incy, value dev), (n, alpha, x, incx, y, incy, dev));
CUBLAS_FUN(spoc_cublasScasum, (value n, value x, value incx, value dev), (n, x, incx, dev));


/******************************* BLAS3 ***************************************/

/* Single-Precision BLAS3 functions */



CAMLprim value spoc_cublasSgemm(value transa, value transb,
		value m, value n, value k,
		value alpha, value a, value lda,
		value b, value ldb, value beta, value c, value ldc, value dev){
	CAMLparam5(transa, transb, m, n, k);
	CAMLxparam5(alpha, a, lda, b, ldb);
	CAMLxparam4(beta, c, ldc, dev);
	CAMLlocal3(dev_vec_array, dev_vec, gi);
	CUdeviceptr d_A;
	CUdeviceptr d_B;
	CUdeviceptr d_C;
	int id;

	GET_VEC(a, d_A);
	GET_VEC(b, d_B);
	GET_VEC(c, d_C);
	CUBLAS_GET_CONTEXT;

	cublasSgemm (Int_val(transa), Int_val(transb), Int_val(m), Int_val(n),
	Int_val(k), (float)Double_val(alpha), (float*) d_A, Int_val(lda),
	(float*) d_B, Int_val(ldb), (float) Double_val(beta),
	(float *)d_C, Int_val(ldc));

	CUDA_RESTORE_CONTEXT;

	CAMLreturn(Val_unit);
}

CUBLAS_FUN14(spoc_cublasSgemm, (value transa, value transb, value m, value n, value k, value alpha, value a, value lda, value b, value ldb, value beta, value c, value ldc, value dev),(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, dev) );


/* Double-Precision BLAS3 functions */



CAMLprim value spoc_cublasDgemm(value transa, value transb,
		value m, value n, value k,
		value alpha, value a, value lda,
		value b, value ldb, value beta, value c, value ldc, value dev){
	CAMLparam5(transa, transb, m, n, k);
	CAMLxparam5(alpha, a, lda, b, ldb);
	CAMLxparam4(beta, c, ldc, dev);
	CAMLlocal3(dev_vec_array, dev_vec, gi);
	CUdeviceptr d_A;
	CUdeviceptr d_B;
	CUdeviceptr d_C;
	int id;
	gi = Field(dev, 0);
	id = Int_val(Field(gi, 7));

	GET_VEC(a, d_A);
	GET_VEC(b, d_B);
	GET_VEC(c, d_C);

	//CUBLAS_GET_CONTEXT;
	CUBLAS_GET_CONTEXT;


	cublasDgemm (Int_val(transa), Int_val(transb), Int_val(m), Int_val(n),
	Int_val(k), (double)Double_val(alpha), (double*) d_A, Int_val(lda),
	(double*) d_B, Int_val(ldb), (double) Double_val(beta),
	(double *)d_C, Int_val(ldc));

	CUDA_RESTORE_CONTEXT;

	CAMLreturn(Val_unit);
}

CUBLAS_FUN14(spoc_cublasDgemm, (value transa, value transb, value m, value n, value k, value alpha, value a, value lda, value b, value ldb, value beta, value c, value ldc, value dev),(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, dev) );


CAMLprim value spoc_cublasSetMatrix (value rows, value cols, value a, value lda, value b, value ldb, value dev){
	CAMLparam5(rows, cols, a, lda, b);
	CAMLxparam2(ldb, dev);
	CAMLlocal4(dev_vec_array, dev_vec, gi, bigArray);
	CUdeviceptr d_B;
	void* h_A;
	int type_size = sizeof(double);
	int tag;
	int id;
	gi = Field(dev, 0);
	id = Int_val(Field(gi, 7));
	GET_VEC(b, d_B);
	GET_HOST_VEC (a, h_A);

	CUBLAS_GET_CONTEXT;
	int custom = 0;
	GET_TYPE_SIZE;

	//printf("rows : %d, col: %d, type_size : %d, lda :%d, ldb : %d\n", Int_val(rows), Int_val(cols), type_size, Int_val (lda), Int_val(ldb));
	//fflush(stdout);
	CUBLAS_CHECK_CALL(cublasSetMatrix(Int_val(rows), Int_val(cols), type_size, h_A, Int_val(lda), (void*) d_B, Int_val(ldb)));
	
	CUBLAS_RESTORE_CONTEXT;
	CAMLreturn(Val_unit);
}
CUBLAS_FUN7(spoc_cublasSetMatrix, (value rows, value cols, value a, value lda, value b, value ldb, value dev), (rows, cols, a, lda, b, ldb, dev));
