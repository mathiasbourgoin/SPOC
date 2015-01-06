#ifdef __cplusplus
extern "C" {
#endif

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


struct point{
	float x;
	float y;
};


CAMLprim value custom_getsizeofpoint()
{
	CAMLparam0();
	CAMLreturn(Val_int(sizeof(struct point)));
}


CAMLprim value custom_extget (value customArray, value idx)
{
	CAMLparam2(customArray, idx);
	CAMLlocal1(mlPoint);
	struct point *pt;
	pt = ((struct point*)(Field(customArray, 1)))+(Int_val(idx));
	mlPoint = caml_alloc(2, 0);
	Store_double_field(mlPoint, 0,(float)(pt->x));
	Store_double_field(mlPoint, 1, (float)(pt->y));
	CAMLreturn(mlPoint);
}

CAMLprim value custom_extset (value customArray, value idx, value v)
{
	CAMLparam3(customArray, idx, v);
	struct point *pt;
	pt = ((struct point*)(Field(customArray, 1)))+Int_val(idx);
	pt->x= (float)Double_field(v, 0);
	pt->y= (float)Double_field(v, 1);
	CAMLreturn(Val_unit);
}

#ifdef __cplusplus
extern }
#endif
