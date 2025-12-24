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
#ifdef __cplusplus
extern "C" {
#endif


#include <assert.h>
#include <string.h>
#include "Spoc.h"

/* Use macros from Spoc.h for custom array access */
#define Custom_ptr_val(v) Custom_array_val(v)
#define Set_custom_ptr(v, x) Set_custom_array(v, x)

/* Finalizer for custom arrays that OWN their buffer */
static void custom_array_finalize(value v) {
	void* f = Custom_ptr_val(v);
	if (f) {
	  if (noCuda)
	    free(f);
	  else
	    cuMemFreeHost(f);
	  Set_custom_ptr(v, NULL);
	}
}

/* Finalizer for sub-custom arrays that do NOT own their buffer */
static void custom_array_nonowning_finalize(value v) {
	/* Nothing to free - parent owns the buffer */
	(void)v;
}

/* Custom operations for custom arrays that own their buffer */
static struct custom_operations custom_array_ops = {
  .identifier = "spoc.custom_array",
  .finalize = custom_array_finalize,
  .compare = custom_compare_default,
  .hash = custom_hash_default,
  .serialize = custom_serialize_default,
  .deserialize = custom_deserialize_default,
  .compare_ext = custom_compare_ext_default,
  .fixed_length = custom_fixed_length_default
};

/* Custom operations for sub-custom arrays that do NOT own their buffer */
static struct custom_operations custom_array_nonowning_ops = {
  .identifier = "spoc.custom_array_nonowning",
  .finalize = custom_array_nonowning_finalize,
  .compare = custom_compare_default,
  .hash = custom_hash_default,
  .serialize = custom_serialize_default,
  .deserialize = custom_deserialize_default,
  .compare_ext = custom_compare_ext_default,
  .fixed_length = custom_fixed_length_default
};

CAMLprim value spoc_create_custom (value custom, value size)
{
	CAMLparam2(custom, size);
	CAMLlocal2(customSize, ret);
	void* res;
	/* OCaml 5 compatible: allocate custom block with space for void pointer */
	ret = caml_alloc_custom(&custom_array_ops, sizeof(void*), 0, 1);
	customSize = Field(custom, 0);
	if (noCuda){
	  if (0 != posix_memalign(&res, OPENCL_PAGE_ALIGN,
			 ((Int_val(customSize)*Int_val(size) - 1)/OPENCL_CACHE_ALIGN + 1) * OPENCL_CACHE_ALIGN)) {
	    caml_failwith("spoc_create_custom: posix_memalign failed");
	  }
	}
	else
	{
	    cuMemAllocHost(&res, Int_val(customSize)*Int_val(size));
	}
	Set_custom_ptr(ret, res);
	CAMLreturn(ret);
}


CAMLprim value spoc_sub_custom_array(value customArray, value custom, value start)
{
	CAMLparam3(customArray, custom, start);
	CAMLlocal2(customSize, ret);
	char* res;
	int elemSize = Int_val(Field(custom, 0))/sizeof(char);
	/* OCaml 5 compatible: use non-owning custom block since parent owns the memory */
	ret = caml_alloc_custom(&custom_array_nonowning_ops, sizeof(void*), 0, 1);
	res = (((char*)Custom_ptr_val(customArray)) + (elemSize * Int_val(start)));
	Set_custom_ptr(ret, res);
	CAMLreturn(ret);
}

#ifdef __cplusplus
extern }
#endif
