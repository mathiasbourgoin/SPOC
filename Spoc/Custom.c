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

#include "Spoc.h"
#include <assert.h>
#include <string.h>


void free_custom (value v) {
	void* f = (void*)(Field(v, 1));
	if (f) free(f);
}



CAMLprim value spoc_create_custom (value custom, value size)
{
	CAMLparam2(custom, size);
	CAMLlocal2(customSize, ret);
	char* res;
	ret = alloc_final(2, free_custom, sizeof(char*), sizeof(char*)*1024);
	customSize = Field(custom, 1);
	res = (char*)malloc(Int_val(size)*Int_val(customSize));
	Store_field(ret, 1, (value)(res));
	CAMLreturn(ret);
}


CAMLprim value spoc_sub_custom_array(value customArray, value custom, value start)
{
	CAMLparam3(customArray, custom, start);
	CAMLlocal2(customSize, ret);
	char* res;
	customSize = Int_val(Field(custom, 1))/sizeof(char);
	ret = alloc_final(2, free_custom, sizeof(char*), sizeof(char*)*1024);
	res = (((char*)(Field(customArray,1)))+(customSize*Int_val(start)));
	//Store_field(ret, 1, (value)(res));
	CAMLreturn(ret);
}

#ifdef __cplusplus
extern }
#endif
