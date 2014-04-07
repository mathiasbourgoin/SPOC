// This program was compiled from OCaml by js_of_ocaml 1.99dev
(function(d){"use strict";var
dg="set_cuda_sources",dy=123,b3=";",fL=108,go="section1",df="reload_sources",b8="Map.bal",fZ=",",cb='"',ag=16777215,de="get_cuda_sources",ci=" / ",fK="double spoc_var",dp="args_to_list",ca=" * ",ak="(",gn=0.5,fy="float spoc_var",dn=65599,ch="if (",aV="return",fY=" ;\n",dx="exec",bt=115,br=";}\n",fJ=".ptx",z=512,dw=120,dd="..",fX=-512,M="]",dv=117,b7="; ",du="compile",gm=" (",$="0",dm="list_to_args",b6=248,fW=126,gl="fd ",dc="get_binaries",fI=" == ",au="(float)",dl="Kirc_Cuda.ml",cg=" + ",fV=") ",dt="x",fH=-97,fx="g",bp=1073741823,gk="parse concat",aC=105,dk="get_opencl_sources",gj=511,bq=110,gi=-88,ai=" = ",dj="set_opencl_sources",fU=200,N="[",b$="'",fw="Unix",b2="int_of_string",gh="(double) ",fT=982028505,bo="){\n",bs="e",gg="#define __FLOAT64_EXTENSION__ \n",aB="-",aU=-48,b_="(double) spoc_var",fv="++){\n",fG="__shared__ float spoc_var",gf="opencl_sources",fF=".cl",ds="reset_binaries",b1="\n",ge=101,dB=748841679,cf="index out of bounds",fu="spoc_init_opencl_device_vec",db=125,b9=" - ",gd=";}",r=255,gc="binaries",ce="}",gb=" < ",ft="__shared__ long spoc_var",aT=250,ga=" >= ",fs="input",fS=246,di=102,fR="Unix.Unix_error",g="",fr=" || ",aS=100,dr="Kirc_OpenCL.ml",f$="#ifndef __FLOAT64_EXTENSION__ \n",fQ="__shared__ int spoc_var",dA=103,b0=", ",fP="./",b5=1e3,fq="for (int ",f_="file_file",f9="spoc_var",al=".",fE="else{\n",b4="+",dz="run",cd=65535,dq="#endif\n",aR=";\n",aa="f",fD="Mandelbrot_js_js.ml",f8=785140586,f7="__shared__ double spoc_var",fC=-32,dh=111,fO=" > ",C=" ",f6="int spoc_var",aj=")",fN="cuda_sources",f5=256,fB="nan",da=116,f2="../",f3="kernel_name",f4=65520,f1="%.12g",fp=" && ",fA="/",fM="while (",c$="compile_and_run",cc=114,f0="* spoc_var",bZ=" <= ",n="number",fz=" % ",vn=d.spoc_opencl_part_device_to_cpu_b!==undefined?d.spoc_opencl_part_device_to_cpu_b:function(){o("spoc_opencl_part_device_to_cpu_b not implemented")},vm=d.spoc_opencl_part_cpu_to_device_b!==undefined?d.spoc_opencl_part_cpu_to_device_b:function(){o("spoc_opencl_part_cpu_to_device_b not implemented")},vk=d.spoc_opencl_load_param_int64!==undefined?d.spoc_opencl_load_param_int64:function(){o("spoc_opencl_load_param_int64 not implemented")},vi=d.spoc_opencl_load_param_float64!==undefined?d.spoc_opencl_load_param_float64:function(){o("spoc_opencl_load_param_float64 not implemented")},vh=d.spoc_opencl_load_param_float!==undefined?d.spoc_opencl_load_param_float:function(){o("spoc_opencl_load_param_float not implemented")},vc=d.spoc_opencl_custom_part_device_to_cpu_b!==undefined?d.spoc_opencl_custom_part_device_to_cpu_b:function(){o("spoc_opencl_custom_part_device_to_cpu_b not implemented")},vb=d.spoc_opencl_custom_part_cpu_to_device_b!==undefined?d.spoc_opencl_custom_part_cpu_to_device_b:function(){o("spoc_opencl_custom_part_cpu_to_device_b not implemented")},va=d.spoc_opencl_custom_device_to_cpu!==undefined?d.spoc_opencl_custom_device_to_cpu:function(){o("spoc_opencl_custom_device_to_cpu not implemented")},u$=d.spoc_opencl_custom_cpu_to_device!==undefined?d.spoc_opencl_custom_cpu_to_device:function(){o("spoc_opencl_custom_cpu_to_device not implemented")},u_=d.spoc_opencl_custom_alloc_vect!==undefined?d.spoc_opencl_custom_alloc_vect:function(){o("spoc_opencl_custom_alloc_vect not implemented")},uZ=d.spoc_cuda_part_device_to_cpu_b!==undefined?d.spoc_cuda_part_device_to_cpu_b:function(){o("spoc_cuda_part_device_to_cpu_b not implemented")},uY=d.spoc_cuda_part_cpu_to_device_b!==undefined?d.spoc_cuda_part_cpu_to_device_b:function(){o("spoc_cuda_part_cpu_to_device_b not implemented")},uX=d.spoc_cuda_load_param_vec_b!==undefined?d.spoc_cuda_load_param_vec_b:function(){o("spoc_cuda_load_param_vec_b not implemented")},uW=d.spoc_cuda_load_param_int_b!==undefined?d.spoc_cuda_load_param_int_b:function(){o("spoc_cuda_load_param_int_b not implemented")},uV=d.spoc_cuda_load_param_int64_b!==undefined?d.spoc_cuda_load_param_int64_b:function(){o("spoc_cuda_load_param_int64_b not implemented")},uU=d.spoc_cuda_load_param_float_b!==undefined?d.spoc_cuda_load_param_float_b:function(){o("spoc_cuda_load_param_float_b not implemented")},uT=d.spoc_cuda_load_param_float64_b!==undefined?d.spoc_cuda_load_param_float64_b:function(){o("spoc_cuda_load_param_float64_b not implemented")},uS=d.spoc_cuda_launch_grid_b!==undefined?d.spoc_cuda_launch_grid_b:function(){o("spoc_cuda_launch_grid_b not implemented")},uR=d.spoc_cuda_flush_all!==undefined?d.spoc_cuda_flush_all:function(){o("spoc_cuda_flush_all not implemented")},uQ=d.spoc_cuda_flush!==undefined?d.spoc_cuda_flush:function(){o("spoc_cuda_flush not implemented")},uP=d.spoc_cuda_device_to_cpu!==undefined?d.spoc_cuda_device_to_cpu:function(){o("spoc_cuda_device_to_cpu not implemented")},uN=d.spoc_cuda_custom_part_device_to_cpu_b!==undefined?d.spoc_cuda_custom_part_device_to_cpu_b:function(){o("spoc_cuda_custom_part_device_to_cpu_b not implemented")},uM=d.spoc_cuda_custom_part_cpu_to_device_b!==undefined?d.spoc_cuda_custom_part_cpu_to_device_b:function(){o("spoc_cuda_custom_part_cpu_to_device_b not implemented")},uL=d.spoc_cuda_custom_load_param_vec_b!==undefined?d.spoc_cuda_custom_load_param_vec_b:function(){o("spoc_cuda_custom_load_param_vec_b not implemented")},uK=d.spoc_cuda_custom_device_to_cpu!==undefined?d.spoc_cuda_custom_device_to_cpu:function(){o("spoc_cuda_custom_device_to_cpu not implemented")},uJ=d.spoc_cuda_custom_cpu_to_device!==undefined?d.spoc_cuda_custom_cpu_to_device:function(){o("spoc_cuda_custom_cpu_to_device not implemented")},gD=d.spoc_cuda_custom_alloc_vect!==undefined?d.spoc_cuda_custom_alloc_vect:function(){o("spoc_cuda_custom_alloc_vect not implemented")},uI=d.spoc_cuda_create_extra!==undefined?d.spoc_cuda_create_extra:function(){o("spoc_cuda_create_extra not implemented")},uH=d.spoc_cuda_cpu_to_device!==undefined?d.spoc_cuda_cpu_to_device:function(){o("spoc_cuda_cpu_to_device not implemented")},gC=d.spoc_cuda_alloc_vect!==undefined?d.spoc_cuda_alloc_vect:function(){o("spoc_cuda_alloc_vect not implemented")},uE=d.spoc_create_custom!==undefined?d.spoc_create_custom:function(){o("spoc_create_custom not implemented")},vq=1;function
gy(a,b){throw[0,a,b]}function
dL(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=d.console;b&&b.error&&b.error(a)}var
q=[0];function
bw(a,b){if(!a)return g;if(a&1)return bw(a-1,b)+b;var
c=bw(a>>1,b);return c+c}function
F(a){if(a!=null){this.bytes=this.fullBytes=a;this.last=this.len=a.length}}function
gB(){gy(q[4],new
F(cf))}F.prototype={string:null,bytes:null,fullBytes:null,array:null,len:null,last:0,toJsString:function(){var
a=this.getFullBytes();try{return this.string=decodeURIComponent(escape(a))}catch(f){dL('MlString.toJsString: wrong encoding for \"%s\" ',a);return a}},toBytes:function(){if(this.string!=null)try{var
a=unescape(encodeURIComponent(this.string))}catch(f){dL('MlString.toBytes: wrong encoding for \"%s\" ',this.string);var
a=this.string}else{var
a=g,c=this.array,d=c.length;for(var
b=0;b<d;b++)a+=String.fromCharCode(c[b])}this.bytes=this.fullBytes=a;this.last=this.len=a.length;return a},getBytes:function(){var
a=this.bytes;if(a==null)a=this.toBytes();return a},getFullBytes:function(){var
a=this.fullBytes;if(a!==null)return a;a=this.bytes;if(a==null)a=this.toBytes();if(this.last<this.len){this.bytes=a+=bw(this.len-this.last,"\0");this.last=this.len}this.fullBytes=a;return a},toArray:function(){var
c=this.bytes;if(c==null)c=this.toBytes();var
b=[],d=this.last;for(var
a=0;a<d;a++)b[a]=c.charCodeAt(a);for(d=this.len;a<d;a++)b[a]=0;this.string=this.bytes=this.fullBytes=null;this.last=this.len;this.array=b;return b},getArray:function(){var
a=this.array;if(!a)a=this.toArray();return a},getLen:function(){var
a=this.len;if(a!==null)return a;this.toBytes();return this.len},toString:function(){var
a=this.string;return a?a:this.toJsString()},valueOf:function(){var
a=this.string;return a?a:this.toJsString()},blitToArray:function(a,b,c,d){var
g=this.array;if(g)if(c<=a)for(var
e=0;e<d;e++)b[c+e]=g[a+e];else
for(var
e=d-1;e>=0;e--)b[c+e]=g[a+e];else{var
f=this.bytes;if(f==null)f=this.toBytes();var
h=this.last-a;if(d<=h)for(var
e=0;e<d;e++)b[c+e]=f.charCodeAt(a+e);else{for(var
e=0;e<h;e++)b[c+e]=f.charCodeAt(a+e);for(;e<d;e++)b[c+e]=0}}},get:function(a){var
c=this.array;if(c)return c[a];var
b=this.bytes;if(b==null)b=this.toBytes();return a<this.last?b.charCodeAt(a):0},safeGet:function(a){if(this.len==null)this.toBytes();if(a<0||a>=this.len)gB();return this.get(a)},set:function(a,b){var
c=this.array;if(!c){if(this.last==a){this.bytes+=String.fromCharCode(b&r);this.last++;return 0}c=this.toArray()}else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;c[a]=b&r;return 0},safeSet:function(a,b){if(this.len==null)this.toBytes();if(a<0||a>=this.len)gB();this.set(a,b)},fill:function(a,b,c){if(a>=this.last&&this.last&&c==0)return;var
d=this.array;if(!d)d=this.toArray();else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;var
f=a+b;for(var
e=a;e<f;e++)d[e]=c},compare:function(a){if(this.string!=null&&a.string!=null){if(this.string<a.string)return-1;if(this.string>a.string)return 1;return 0}var
b=this.getFullBytes(),c=a.getFullBytes();if(b<c)return-1;if(b>c)return 1;return 0},equal:function(a){if(this.string!=null&&a.string!=null)return this.string==a.string;return this.getFullBytes()==a.getFullBytes()},lessThan:function(a){if(this.string!=null&&a.string!=null)return this.string<a.string;return this.getFullBytes()<a.getFullBytes()},lessEqual:function(a){if(this.string!=null&&a.string!=null)return this.string<=a.string;return this.getFullBytes()<=a.getFullBytes()}};function
aD(a){this.string=a}aD.prototype=new
F();function
tu(a,b,c,d,e){if(d<=b)for(var
f=1;f<=e;f++)c[d+f]=a[b+f];else
for(var
f=e;f>=1;f--)c[d+f]=a[b+f]}function
tv(a){var
c=[0];while(a!==0){var
d=a[1];for(var
b=1;b<d.length;b++)c.push(d[b]);a=a[2]}return c}function
dK(a,b){gy(a,new
aD(b))}function
av(a){dK(q[4],a)}function
aW(){av(cf)}function
tw(a,b){if(b<0||b>=a.length-1)aW();return a[b+1]}function
tx(a,b,c){if(b<0||b>=a.length-1)aW();a[b+1]=c;return 0}var
dD;function
ty(a,b,c){if(c.length!=2)av("Bigarray.create: bad number of dimensions");if(b!=0)av("Bigarray.create: unsupported layout");if(c[1]<0)av("Bigarray.create: negative dimension");if(!dD){var
e=d;dD=[e.Float32Array,e.Float64Array,e.Int8Array,e.Uint8Array,e.Int16Array,e.Uint16Array,e.Int32Array,null,e.Int32Array,e.Int32Array,null,null,e.Uint8Array]}var
f=dD[a];if(!f)av("Bigarray.create: unsupported kind");return new
f(c[1])}function
tz(a,b){if(b<0||b>=a.length)aW();return a[b]}function
tA(a,b,c){if(b<0||b>=a.length)aW();a[b]=c;return 0}function
dE(a,b,c,d,e){if(e===0)return;if(d===c.last&&c.bytes!=null){var
f=a.bytes;if(f==null)f=a.toBytes();if(b>0||a.last>e)f=f.slice(b,b+e);c.bytes+=f;c.last+=f.length;return}var
g=c.array;if(!g)g=c.toArray();else
c.bytes=c.string=null;a.blitToArray(b,g,d,e)}function
am(c,b){if(c.fun)return am(c.fun,b);var
a=c.length,d=a-b.length;if(d==0)return c.apply(null,b);else
if(d<0)return am(c.apply(null,b.slice(0,a)),b.slice(a));else
return function(a){return am(c,b.concat([a]))}}function
tB(a){if(isFinite(a)){if(Math.abs(a)>=2.22507385850720138e-308)return 0;if(a!=0)return 1;return 2}return isNaN(a)?4:3}function
tN(a,b){var
c=a[3]<<16,d=b[3]<<16;if(c>d)return 1;if(c<d)return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
gv(a,b){if(a<b)return-1;if(a==b)return 0;return 1}function
dF(a,b,c){var
e=[];for(;;){if(!(c&&a===b))if(a
instanceof
F)if(b
instanceof
F){if(a!==b){var
d=a.compare(b);if(d!=0)return d}}else
return 1;else
if(a
instanceof
Array&&a[0]===(a[0]|0)){var
g=a[0];if(g===aT){a=a[1];continue}else
if(b
instanceof
Array&&b[0]===(b[0]|0)){var
h=b[0];if(h===aT){b=b[1];continue}else
if(g!=h)return g<h?-1:1;else
switch(g){case
b6:{var
d=gv(a[2],b[2]);if(d!=0)return d;break}case
251:av("equal: abstract value");case
r:{var
d=tN(a,b);if(d!=0)return d;break}default:if(a.length!=b.length)return a.length<b.length?-1:1;if(a.length>1)e.push(a,b,1)}}else
return 1}else
if(b
instanceof
F||b
instanceof
Array&&b[0]===(b[0]|0))return-1;else{if(a<b)return-1;if(a>b)return 1;if(c&&a!=b){if(a==a)return 1;if(b==b)return-1}}if(e.length==0)return 0;var
f=e.pop();b=e.pop();a=e.pop();if(f+1<a.length)e.push(a,b,f+1);a=a[f];b=b[f]}}function
gq(a,b){return dF(a,b,true)}function
gp(a){this.bytes=g;this.len=a}gp.prototype=new
F();function
gr(a){if(a<0)av("String.create");return new
gp(a)}function
dJ(a){throw[0,a]}function
gz(){dJ(q[6])}function
tC(a,b){if(b==0)gz();return a/b|0}function
tD(a,b){return+(dF(a,b,false)==0)}function
tE(a,b,c,d){a.fill(b,c,d)}function
dI(a){a=a.toString();var
e=a.length;if(e>31)av("format_int: format too long");var
b={justify:b4,signstyle:aB,filler:C,alternate:false,base:0,signedconv:false,width:0,uppercase:false,sign:1,prec:-1,conv:aa};for(var
d=0;d<e;d++){var
c=a.charAt(d);switch(c){case
aB:b.justify=aB;break;case
b4:case
C:b.signstyle=c;break;case
$:b.filler=$;break;case"#":b.alternate=true;break;case"1":case"2":case"3":case"4":case"5":case"6":case"7":case"8":case"9":b.width=0;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.width=b.width*10+c;d++}d--;break;case
al:b.prec=0;d++;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.prec=b.prec*10+c;d++}d--;case"d":case"i":b.signedconv=true;case"u":b.base=10;break;case
dt:b.base=16;break;case"X":b.base=16;b.uppercase=true;break;case"o":b.base=8;break;case
bs:case
aa:case
fx:b.signedconv=true;b.conv=c;break;case"E":case"F":case"G":b.signedconv=true;b.uppercase=true;b.conv=c.toLowerCase();break}}return b}function
dG(a,b){if(a.uppercase)b=b.toUpperCase();var
e=b.length;if(a.signedconv&&(a.sign<0||a.signstyle!=aB))e++;if(a.alternate){if(a.base==8)e+=1;if(a.base==16)e+=2}var
c=g;if(a.justify==b4&&a.filler==C)for(var
d=e;d<a.width;d++)c+=C;if(a.signedconv)if(a.sign<0)c+=aB;else
if(a.signstyle!=aB)c+=a.signstyle;if(a.alternate&&a.base==8)c+=$;if(a.alternate&&a.base==16)c+="0x";if(a.justify==b4&&a.filler==$)for(var
d=e;d<a.width;d++)c+=$;c+=b;if(a.justify==aB)for(var
d=e;d<a.width;d++)c+=C;return new
aD(c)}function
tF(a,b){var
c,f=dI(a),e=f.prec<0?6:f.prec;if(b<0){f.sign=-1;b=-b}if(isNaN(b)){c=fB;f.filler=C}else
if(!isFinite(b)){c="inf";f.filler=C}else
switch(f.conv){case
bs:var
c=b.toExponential(e),d=c.length;if(c.charAt(d-3)==bs)c=c.slice(0,d-1)+$+c.slice(d-1);break;case
aa:c=b.toFixed(e);break;case
fx:e=e?e:1;c=b.toExponential(e-1);var
i=c.indexOf(bs),h=+c.slice(i+1);if(h<-4||b.toFixed(0).length>e){var
d=i-1;while(c.charAt(d)==$)d--;if(c.charAt(d)==al)d--;c=c.slice(0,d+1)+c.slice(i);d=c.length;if(c.charAt(d-3)==bs)c=c.slice(0,d-1)+$+c.slice(d-1);break}else{var
g=e;if(h<0){g-=h+1;c=b.toFixed(g)}else
while(c=b.toFixed(g),c.length>e+1)g--;if(g){var
d=c.length-1;while(c.charAt(d)==$)d--;if(c.charAt(d)==al)d--;c=c.slice(0,d+1)}}break}return dG(f,c)}function
tG(a,b){if(a.toString()=="%d")return new
aD(g+b);var
c=dI(a);if(b<0)if(c.signedconv){c.sign=-1;b=-b}else
b>>>=0;var
d=b.toString(c.base);if(c.prec>=0){c.filler=C;var
e=c.prec-d.length;if(e>0)d=bw(e,$)+d}return dG(c,d)}function
tH(){return 0}function
tI(){return 0}var
ck=[];function
tJ(a,b,c){var
e=a[1],i=ck[c];if(i===null)for(var
h=ck.length;h<c;h++)ck[h]=0;else
if(e[i]===b)return e[i-1];var
d=3,g=e[1]*2+1,f;while(d<g){f=d+g>>1|1;if(b<e[f+1])g=f-2;else
d=f}ck[c]=d+1;return b==e[d+1]?e[d]:0}function
tK(a,b){return+(gq(a,b,false)>=0)}function
gs(a){if(!isFinite(a)){if(isNaN(a))return[r,1,0,f4];return a>0?[r,0,0,32752]:[r,0,0,f4]}var
f=a>=0?0:32768;if(f)a=-a;var
b=Math.floor(Math.LOG2E*Math.log(a))+1023;if(b<=0){b=0;a/=Math.pow(2,-1026)}else{a/=Math.pow(2,b-1027);if(a<16){a*=2;b-=1}if(b==0)a/=2}var
d=Math.pow(2,24),c=a|0;a=(a-c)*d;var
e=a|0;a=(a-e)*d;var
g=a|0;c=c&15|f|b<<4;return[r,g,e,c]}function
bv(a,b){return((a>>16)*b<<16)+(a&cd)*b|0}var
tL=function(){var
p=f5;function
c(a,b){return a<<b|a>>>32-b}function
g(a,b){b=bv(b,3432918353);b=c(b,15);b=bv(b,461845907);a^=b;a=c(a,13);return(a*5|0)+3864292196|0}function
t(a){a^=a>>>16;a=bv(a,2246822507);a^=a>>>13;a=bv(a,3266489909);a^=a>>>16;return a}function
u(a,b){var
d=b[1]|b[2]<<24,c=b[2]>>>8|b[3]<<16;a=g(a,d);a=g(a,c);return a}function
v(a,b){var
d=b[1]|b[2]<<24,c=b[2]>>>8|b[3]<<16;a=g(a,c^d);return a}function
x(a,b){var
e=b.length,c,d;for(c=0;c+4<=e;c+=4){d=b.charCodeAt(c)|b.charCodeAt(c+1)<<8|b.charCodeAt(c+2)<<16|b.charCodeAt(c+3)<<24;a=g(a,d)}d=0;switch(e&3){case
3:d=b.charCodeAt(c+2)<<16;case
2:d|=b.charCodeAt(c+1)<<8;case
1:d|=b.charCodeAt(c);a=g(a,d);default:}a^=e;return a}function
w(a,b){var
e=b.length,c,d;for(c=0;c+4<=e;c+=4){d=b[c]|b[c+1]<<8|b[c+2]<<16|b[c+3]<<24;a=g(a,d)}d=0;switch(e&3){case
3:d=b[c+2]<<16;case
2:d|=b[c+1]<<8;case
1:d|=b[c];a=g(a,d);default:}a^=e;return a}return function(a,b,c,d){var
k,l,m,i,h,f,e,j,o;i=b;if(i<0||i>p)i=p;h=a;f=c;k=[d];l=0;m=1;while(l<m&&h>0){e=k[l++];if(e
instanceof
Array&&e[0]===(e[0]|0))switch(e[0]){case
b6:f=g(f,e[2]);h--;break;case
aT:k[--l]=e[1];break;case
r:f=v(f,e);h--;break;default:var
s=e.length-1<<10|e[0];f=g(f,s);for(j=1,o=e.length;j<o;j++){if(m>=i)break;k[m++]=e[j]}break}else
if(e
instanceof
F){var
n=e.array;if(n)f=w(f,n);else{var
q=e.getFullBytes();f=x(f,q)}h--;break}else
if(e===(e|0)){f=g(f,e+e+1);h--}else
if(e===+e){f=u(f,gs(e));h--;break}}f=t(f);return f&bp}}();function
tV(a){return[a[3]>>8,a[3]&r,a[2]>>16,a[2]>>8&r,a[2]&r,a[1]>>16,a[1]>>8&r,a[1]&r]}function
tM(e,b,c){var
d=0;function
f(a){b--;if(e<0||b<0)return;if(a
instanceof
Array&&a[0]===(a[0]|0))switch(a[0]){case
b6:e--;d=d*dn+a[2]|0;break;case
aT:b++;f(a);break;case
r:e--;d=d*dn+a[1]+(a[2]<<24)|0;break;default:e--;d=d*19+a[0]|0;for(var
c=a.length-1;c>0;c--)f(a[c])}else
if(a
instanceof
F){e--;var
g=a.array,h=a.getLen();if(g)for(var
c=0;c<h;c++)d=d*19+g[c]|0;else{var
i=a.getFullBytes();for(var
c=0;c<h;c++)d=d*19+i.charCodeAt(c)|0}}else
if(a===(a|0)){e--;d=d*dn+a|0}else
if(a===+a){e--;var
j=tV(gs(a));for(var
c=7;c>=0;c--)d=d*19+j[c]|0}}f(c);return d&bp}function
tQ(a){return(a[3]|a[2]|a[1])==0}function
tT(a){return[r,a&ag,a>>24&ag,a>>31&cd]}function
tU(a,b){var
c=a[1]-b[1],d=a[2]-b[2]+(c>>24),e=a[3]-b[3]+(d>>24);return[r,c&ag,d&ag,e&cd]}function
gu(a,b){if(a[3]>b[3])return 1;if(a[3]<b[3])return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
gt(a){a[3]=a[3]<<1|a[2]>>23;a[2]=(a[2]<<1|a[1]>>23)&ag;a[1]=a[1]<<1&ag}function
tR(a){a[1]=(a[1]>>>1|a[2]<<23)&ag;a[2]=(a[2]>>>1|a[3]<<23)&ag;a[3]=a[3]>>>1}function
tX(a,b){var
e=0,d=a.slice(),c=b.slice(),f=[r,0,0,0];while(gu(d,c)>0){e++;gt(c)}while(e>=0){e--;gt(f);if(gu(d,c)>=0){f[1]++;d=tU(d,c)}tR(c)}return[0,f,d]}function
tW(a){return a[1]|a[2]<<24}function
tP(a){return a[3]<<16<0}function
tS(a){var
b=-a[1],c=-a[2]+(b>>24),d=-a[3]+(c>>24);return[r,b&ag,c&ag,d&cd]}function
tO(a,b){var
c=dI(a);if(c.signedconv&&tP(b)){c.sign=-1;b=tS(b)}var
d=g,i=tT(c.base),h="0123456789abcdef";do{var
f=tX(b,i);b=f[1];d=h.charAt(tW(f[2]))+d}while(!tQ(b));if(c.prec>=0){c.filler=C;var
e=c.prec-d.length;if(e>0)d=bw(e,$)+d}return dG(c,d)}function
uh(a){var
b=0,c=10,d=a.get(0)==45?(b++,-1):1;if(a.get(b)==48)switch(a.get(b+1)){case
dw:case
88:c=16;b+=2;break;case
dh:case
79:c=8;b+=2;break;case
98:case
66:c=2;b+=2;break}return[b,d,c]}function
gx(a){if(a>=48&&a<=57)return a-48;if(a>=65&&a<=90)return a-55;if(a>=97&&a<=122)return a-87;return-1}function
o(a){dK(q[3],a)}function
tY(a){var
g=uh(a),e=g[0],h=g[1],f=g[2],i=-1>>>0,d=a.get(e),c=gx(d);if(c<0||c>=f)o(b2);var
b=c;for(;;){e++;d=a.get(e);if(d==95)continue;c=gx(d);if(c<0||c>=f)break;b=f*b+c;if(b>i)o(b2)}if(e!=a.getLen())o(b2);b=h*b;if((b|0)!=b)o(b2);return b}function
tZ(a){return+(a>31&&a<127)}var
cj={amp:/&/g,lt:/</g,quot:/\"/g,all:/[&<\"]/};function
t0(a){if(!cj.all.test(a))return a;return a.replace(cj.amp,"&amp;").replace(cj.lt,"&lt;").replace(cj.quot,"&quot;")}function
t1(a){var
c=Array.prototype.slice;return function(){var
b=arguments.length>0?c.call(arguments):[undefined];return am(a,b)}}function
t2(a,b){var
d=[0];for(var
c=1;c<=a;c++)d[c]=b;return d}function
dC(a){var
b=a.length;this.array=a;this.len=this.last=b}dC.prototype=new
F();var
t3=function(){function
m(a,b){return a+b|0}function
l(a,b,c,d,e,f){b=m(m(b,a),m(d,f));return m(b<<e|b>>>32-e,c)}function
h(a,b,c,d,e,f,g){return l(b&c|~b&d,a,b,e,f,g)}function
i(a,b,c,d,e,f,g){return l(b&d|c&~d,a,b,e,f,g)}function
j(a,b,c,d,e,f,g){return l(b^c^d,a,b,e,f,g)}function
k(a,b,c,d,e,f,g){return l(c^(b|~d),a,b,e,f,g)}function
n(a,b){var
g=b;a[g>>2]|=128<<8*(g&3);for(g=(g&~3)+8;(g&63)<60;g+=4)a[(g>>2)-1]=0;a[(g>>2)-1]=b<<3;a[g>>2]=b>>29&536870911;var
l=[1732584193,4023233417,2562383102,271733878];for(g=0;g<a.length;g+=16){var
c=l[0],d=l[1],e=l[2],f=l[3];c=h(c,d,e,f,a[g+0],7,3614090360);f=h(f,c,d,e,a[g+1],12,3905402710);e=h(e,f,c,d,a[g+2],17,606105819);d=h(d,e,f,c,a[g+3],22,3250441966);c=h(c,d,e,f,a[g+4],7,4118548399);f=h(f,c,d,e,a[g+5],12,1200080426);e=h(e,f,c,d,a[g+6],17,2821735955);d=h(d,e,f,c,a[g+7],22,4249261313);c=h(c,d,e,f,a[g+8],7,1770035416);f=h(f,c,d,e,a[g+9],12,2336552879);e=h(e,f,c,d,a[g+10],17,4294925233);d=h(d,e,f,c,a[g+11],22,2304563134);c=h(c,d,e,f,a[g+12],7,1804603682);f=h(f,c,d,e,a[g+13],12,4254626195);e=h(e,f,c,d,a[g+14],17,2792965006);d=h(d,e,f,c,a[g+15],22,1236535329);c=i(c,d,e,f,a[g+1],5,4129170786);f=i(f,c,d,e,a[g+6],9,3225465664);e=i(e,f,c,d,a[g+11],14,643717713);d=i(d,e,f,c,a[g+0],20,3921069994);c=i(c,d,e,f,a[g+5],5,3593408605);f=i(f,c,d,e,a[g+10],9,38016083);e=i(e,f,c,d,a[g+15],14,3634488961);d=i(d,e,f,c,a[g+4],20,3889429448);c=i(c,d,e,f,a[g+9],5,568446438);f=i(f,c,d,e,a[g+14],9,3275163606);e=i(e,f,c,d,a[g+3],14,4107603335);d=i(d,e,f,c,a[g+8],20,1163531501);c=i(c,d,e,f,a[g+13],5,2850285829);f=i(f,c,d,e,a[g+2],9,4243563512);e=i(e,f,c,d,a[g+7],14,1735328473);d=i(d,e,f,c,a[g+12],20,2368359562);c=j(c,d,e,f,a[g+5],4,4294588738);f=j(f,c,d,e,a[g+8],11,2272392833);e=j(e,f,c,d,a[g+11],16,1839030562);d=j(d,e,f,c,a[g+14],23,4259657740);c=j(c,d,e,f,a[g+1],4,2763975236);f=j(f,c,d,e,a[g+4],11,1272893353);e=j(e,f,c,d,a[g+7],16,4139469664);d=j(d,e,f,c,a[g+10],23,3200236656);c=j(c,d,e,f,a[g+13],4,681279174);f=j(f,c,d,e,a[g+0],11,3936430074);e=j(e,f,c,d,a[g+3],16,3572445317);d=j(d,e,f,c,a[g+6],23,76029189);c=j(c,d,e,f,a[g+9],4,3654602809);f=j(f,c,d,e,a[g+12],11,3873151461);e=j(e,f,c,d,a[g+15],16,530742520);d=j(d,e,f,c,a[g+2],23,3299628645);c=k(c,d,e,f,a[g+0],6,4096336452);f=k(f,c,d,e,a[g+7],10,1126891415);e=k(e,f,c,d,a[g+14],15,2878612391);d=k(d,e,f,c,a[g+5],21,4237533241);c=k(c,d,e,f,a[g+12],6,1700485571);f=k(f,c,d,e,a[g+3],10,2399980690);e=k(e,f,c,d,a[g+10],15,4293915773);d=k(d,e,f,c,a[g+1],21,2240044497);c=k(c,d,e,f,a[g+8],6,1873313359);f=k(f,c,d,e,a[g+15],10,4264355552);e=k(e,f,c,d,a[g+6],15,2734768916);d=k(d,e,f,c,a[g+13],21,1309151649);c=k(c,d,e,f,a[g+4],6,4149444226);f=k(f,c,d,e,a[g+11],10,3174756917);e=k(e,f,c,d,a[g+2],15,718787259);d=k(d,e,f,c,a[g+9],21,3951481745);l[0]=m(c,l[0]);l[1]=m(d,l[1]);l[2]=m(e,l[2]);l[3]=m(f,l[3])}var
o=[];for(var
g=0;g<4;g++)for(var
n=0;n<4;n++)o[g*4+n]=l[g]>>8*n&r;return o}return function(a,b,c){var
h=[];if(a.array){var
f=a.array;for(var
d=0;d<c;d+=4){var
e=d+b;h[d>>2]=f[e]|f[e+1]<<8|f[e+2]<<16|f[e+3]<<24}for(;d<c;d++)h[d>>2]|=f[d+b]<<8*(d&3)}else{var
g=a.getFullBytes();for(var
d=0;d<c;d+=4){var
e=d+b;h[d>>2]=g.charCodeAt(e)|g.charCodeAt(e+1)<<8|g.charCodeAt(e+2)<<16|g.charCodeAt(e+3)<<24}for(;d<c;d++)h[d>>2]|=g.charCodeAt(d+b)<<8*(d&3)}return new
dC(n(h,c))}}();function
t4(a){return a.data.array.length}function
aw(a){dK(q[2],a)}function
dH(a){if(!a.opened)aw("Cannot flush a closed channel");if(a.buffer==g)return 0;if(a.output){switch(a.output.length){case
2:a.output(a,a.buffer);break;default:a.output(a.buffer)}}a.buffer=g}var
bu=new
Array();function
t5(a){dH(a);a.opened=false;delete
bu[a.fd];return 0}function
t6(a,b,c,d){var
e=a.data.array.length-a.data.offset;if(e<d)d=e;dE(new
dC(a.data.array),a.data.offset,b,c,d);a.data.offset+=d;return d}function
ui(){dJ(q[5])}function
t7(a){if(a.data.offset>=a.data.array.length)ui();if(a.data.offset<0||a.data.offset>a.data.array.length)aW();var
b=a.data.array[a.data.offset];a.data.offset++;return b}function
t8(a){var
b=a.data.offset,c=a.data.array.length;if(b>=c)return 0;while(true){if(b>=c)return-(b-a.data.offset);if(b<0||b>a.data.array.length)aW();if(a.data.array[b]==10)return b-a.data.offset+1;b++}}function
uk(a,b){if(!q.files)q.files={};if(b
instanceof
F)var
c=b.getArray();else
if(b
instanceof
Array)var
c=b;else
var
c=new
F(b).getArray();q.files[a
instanceof
F?a.toString():a]=c}function
ur(a){return q.files&&q.files[a.toString()]?1:q.auto_register_file===undefined?0:q.auto_register_file(a)}function
bx(a,b,c){if(q.fds===undefined)q.fds=new
Array();c=c?c:{};var
d={};d.array=b;d.offset=c.append?d.array.length:0;d.flags=c;q.fds[a]=d;q.fd_last_idx=a;return a}function
uv(a,b,c){var
d={};while(b){switch(b[1]){case
0:d.rdonly=1;break;case
1:d.wronly=1;break;case
2:d.append=1;break;case
3:d.create=1;break;case
4:d.truncate=1;break;case
5:d.excl=1;break;case
6:d.binary=1;break;case
7:d.text=1;break;case
8:d.nonblock=1;break}b=b[2]}var
e=a.toString();if(d.rdonly&&d.wronly)aw(e+" : flags Open_rdonly and Open_wronly are not compatible");if(d.text&&d.binary)aw(e+" : flags Open_text and Open_binary are not compatible");if(ur(a)){if(d.create&&d.excl)aw(e+" : file already exists");var
f=q.fd_last_idx?q.fd_last_idx:0;if(d.truncate)q.files[e]=g;return bx(f+1,q.files[e],d)}else
if(d.create){var
f=q.fd_last_idx?q.fd_last_idx:0;uk(e,[]);return bx(f+1,q.files[e],d)}else
aw(e+": no such file or directory")}bx(0,[]);bx(1,[]);bx(2,[]);function
t9(a){var
b=q.fds[a];if(b.flags.wronly)aw(gl+a+" is writeonly");return{data:b,fd:a,opened:true}}function
uC(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=d.console;b&&b.log&&b.log(a)}function
un(a,b){var
e=new
F(b),d=e.getLen();for(var
c=0;c<d;c++)a.data.array[a.data.offset+c]=e.get(c);a.data.offset+=d;return 0}function
t_(a){var
b;switch(a){case
1:b=uC;break;case
2:b=dL;break;default:b=un}var
d=q.fds[a];if(d.flags.rdonly)aw(gl+a+" is readonly");var
c={data:d,fd:a,opened:true,buffer:g,output:b};bu[c.fd]=c;return c}function
t$(){var
a=0;for(var
b
in
bu)if(bu[b].opened)a=[0,bu[b],a];return a}function
gw(a,b,c,d){if(!a.opened)aw("Cannot output to a closed channel");var
f;if(c==0&&b.getLen()==d)f=b;else{f=gr(d);dE(b,c,f,0,d)}var
e=f.toString(),g=e.lastIndexOf("\n");if(g<0)a.buffer+=e;else{a.buffer+=e.substr(0,g+1);dH(a);a.buffer+=e.substr(g+1)}}function
S(a){return new
F(a)}function
ua(a,b){var
c=S(String.fromCharCode(b));gw(a,c,0,1)}function
ub(a,b){if(b==0)gz();return a%b}function
ud(a,b){return+(dF(a,b,false)!=0)}function
ue(a,b){var
d=[a];for(var
c=1;c<=b;c++)d[c]=0;return d}function
uf(a,b){a[0]=b;return 0}function
ug(a){return a
instanceof
Array?a[0]:b5}function
ul(a,b){q[a+1]=b}var
uc={};function
um(a,b){uc[a]=b;return 0}function
uo(a,b){return a.compare(b)}function
gA(a,b){var
c=a.fullBytes,d=b.fullBytes;if(c!=null&&d!=null)return c==d?1:0;return a.getFullBytes()==b.getFullBytes()?1:0}function
up(a,b){return 1-gA(a,b)}function
uq(){return 32}function
us(){var
a=new
aD("a.out");return[0,a,[0,a]]}function
ut(){return[0,new
aD(fw),32,0]}function
uj(){dJ(q[7])}function
uu(){uj()}function
uw(){var
a=new
Date()^4294967295*Math.random();return{valueOf:function(){return a},0:0,1:a,length:2}}function
ux(){console.log("caml_sys_system_command");return 0}function
uy(a){var
b=1;while(a&&a.joo_tramp){a=a.joo_tramp.apply(null,a.joo_args);b++}return a}function
uz(a,b){return{joo_tramp:a,joo_args:b}}function
uA(a,b){if(typeof
b==="function"){a.fun=b;return 0}if(b.fun){a.fun=b.fun;return 0}var
c=b.length;while(c--)a[c]=b[c];return 0}function
uB(){return 0}var
dM=0;function
uD(){if(window.webcl==undefined){alert("Unfortunately your system does not support WebCL. "+"Make sure that you have both the OpenCL driver "+"and the WebCL browser extension installed.");dM=1}else{console.log("INIT OPENCL");dM=0}return 0}function
uF(){console.log(" spoc_cuInit");return 0}function
uG(){console.log(" spoc_cuda_compile");return 0}function
uO(){console.log(" spoc_cuda_debug_compile");return 0}function
u0(a,b,c){console.log(" spoc_debug_opencl_compile");console.log(a.bytes);var
e=c[9],f=e[0],d=f.createProgram(a.bytes),g=d.getInfo(WebCL.PROGRAM_DEVICES);d.build(g);var
h=d.createKernel(b.bytes);e[0]=f;c[9]=e;return h}function
u1(a){console.log("spoc_getCudaDevice");return 0}function
u2(){console.log(" spoc_getCudaDevicesCount");return 0}function
u3(a,b){console.log(" spoc_getOpenCLDevice");var
u=[["DEVICE_ADDRESS_BITS",WebCL.DEVICE_ADDRESS_BITS],["DEVICE_AVAILABLE",WebCL.DEVICE_AVAILABLE],["DEVICE_COMPILER_AVAILABLE",WebCL.DEVICE_COMPILER_AVAILABLE],["DEVICE_ENDIAN_LITTLE",WebCL.DEVICE_ENDIAN_LITTLE],["DEVICE_ERROR_CORRECTION_SUPPORT",WebCL.DEVICE_ERROR_CORRECTION_SUPPORT],["DEVICE_EXECUTION_CAPABILITIES",WebCL.DEVICE_EXECUTION_CAPABILITIES],["DEVICE_EXTENSIONS",WebCL.DEVICE_EXTENSIONS],["DEVICE_GLOBAL_MEM_CACHE_SIZE",WebCL.DEVICE_GLOBAL_MEM_CACHE_SIZE],["DEVICE_GLOBAL_MEM_CACHE_TYPE",WebCL.DEVICE_GLOBAL_MEM_CACHE_TYPE],["DEVICE_GLOBAL_MEM_CACHELINE_SIZE",WebCL.DEVICE_GLOBAL_MEM_CACHELINE_SIZE],["DEVICE_GLOBAL_MEM_SIZE",WebCL.DEVICE_GLOBAL_MEM_SIZE],["DEVICE_HALF_FP_CONFIG",WebCL.DEVICE_HALF_FP_CONFIG],["DEVICE_IMAGE_SUPPORT",WebCL.DEVICE_IMAGE_SUPPORT],["DEVICE_IMAGE2D_MAX_HEIGHT",WebCL.DEVICE_IMAGE2D_MAX_HEIGHT],["DEVICE_IMAGE2D_MAX_WIDTH",WebCL.DEVICE_IMAGE2D_MAX_WIDTH],["DEVICE_IMAGE3D_MAX_DEPTH",WebCL.DEVICE_IMAGE3D_MAX_DEPTH],["DEVICE_IMAGE3D_MAX_HEIGHT",WebCL.DEVICE_IMAGE3D_MAX_HEIGHT],["DEVICE_IMAGE3D_MAX_WIDTH",WebCL.DEVICE_IMAGE3D_MAX_WIDTH],["DEVICE_LOCAL_MEM_SIZE",WebCL.DEVICE_LOCAL_MEM_SIZE],["DEVICE_LOCAL_MEM_TYPE",WebCL.DEVICE_LOCAL_MEM_TYPE],["DEVICE_MAX_CLOCK_FREQUENCY",WebCL.DEVICE_MAX_CLOCK_FREQUENCY],["DEVICE_MAX_COMPUTE_UNITS",WebCL.DEVICE_MAX_COMPUTE_UNITS],["DEVICE_MAX_CONSTANT_ARGS",WebCL.DEVICE_MAX_CONSTANT_ARGS],["DEVICE_MAX_CONSTANT_BUFFER_SIZE",WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE],["DEVICE_MAX_MEM_ALLOC_SIZE",WebCL.DEVICE_MAX_MEM_ALLOC_SIZE],["DEVICE_MAX_PARAMETER_SIZE",WebCL.DEVICE_MAX_PARAMETER_SIZE],["DEVICE_MAX_READ_IMAGE_ARGS",WebCL.DEVICE_MAX_READ_IMAGE_ARGS],["DEVICE_MAX_SAMPLERS",WebCL.DEVICE_MAX_SAMPLERS],["DEVICE_MAX_WORK_GROUP_SIZE",WebCL.DEVICE_MAX_WORK_GROUP_SIZE],["DEVICE_MAX_WORK_ITEM_DIMENSIONS",WebCL.DEVICE_MAX_WORK_ITEM_DIMENSIONS],["DEVICE_MAX_WORK_ITEM_SIZES",WebCL.DEVICE_MAX_WORK_ITEM_SIZES],["DEVICE_MAX_WRITE_IMAGE_ARGS",WebCL.DEVICE_MAX_WRITE_IMAGE_ARGS],["DEVICE_MEM_BASE_ADDR_ALIGN",WebCL.DEVICE_MEM_BASE_ADDR_ALIGN],["DEVICE_NAME",WebCL.DEVICE_NAME],["DEVICE_PLATFORM",WebCL.DEVICE_PLATFORM],["DEVICE_PREFERRED_VECTOR_WIDTH_CHAR",WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_CHAR],["DEVICE_PREFERRED_VECTOR_WIDTH_SHORT",WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_SHORT],["DEVICE_PREFERRED_VECTOR_WIDTH_INT",WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_INT],["DEVICE_PREFERRED_VECTOR_WIDTH_LONG",WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_LONG],["DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT",WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT],["DEVICE_PROFILE",WebCL.DEVICE_PROFILE],["DEVICE_PROFILING_TIMER_RESOLUTION",WebCL.DEVICE_PROFILING_TIMER_RESOLUTION],["DEVICE_QUEUE_PROPERTIES",WebCL.DEVICE_QUEUE_PROPERTIES],["DEVICE_SINGLE_FP_CONFIG",WebCL.DEVICE_SINGLE_FP_CONFIG],["DEVICE_TYPE",WebCL.DEVICE_TYPE],["DEVICE_VENDOR",WebCL.DEVICE_VENDOR],["DEVICE_VENDOR_ID",WebCL.DEVICE_VENDOR_ID],["DEVICE_VERSION",WebCL.DEVICE_VERSION],["DRIVER_VERSION",WebCL.DRIVER_VERSION]],s=0,e=[0],p=[1],d=new
Array(48);d[0]=0;var
h=[0],k=webcl.getPlatforms();for(var
t
in
k){var
f=k[t],j=f.getDevices();s+=j.length}var
g=0;k=webcl.getPlatforms();for(var
o
in
k){console.log("here "+o);var
f=k[o],j=f.getDevices(),m=j.length;console.log("there "+g+C+m+C+a);if(g+m>=a)for(var
q
in
j){var
c=j[q];if(g==a){console.log("current ----------"+g);e[1]=S(c.getInfo(WebCL.DEVICE_NAME));console.log(e[1]);e[2]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_SIZE);e[3]=c.getInfo(WebCL.DEVICE_LOCAL_MEM_SIZE);e[4]=c.getInfo(WebCL.DEVICE_MAX_CLOCK_FREQUENCY);e[5]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE);e[6]=c.getInfo(WebCL.DEVICE_MAX_COMPUTE_UNITS);e[7]=c.getInfo(WebCL.DEVICE_ERROR_CORRECTION_SUPPORT);e[8]=b;var
i=new
Array(3);i[0]=webcl.createContext(c);i[1]=i[0].createCommandQueue();i[2]=i[0].createCommandQueue();e[9]=i;h[1]=S(f.getInfo(WebCL.PLATFORM_PROFILE));h[2]=S(f.getInfo(WebCL.PLATFORM_VERSION));h[3]=S(f.getInfo(WebCL.PLATFORM_NAME));h[4]=S(f.getInfo(WebCL.PLATFORM_VENDOR));h[5]=S(f.getInfo(WebCL.PLATFORM_EXTENSIONS));h[6]=m;var
l=c.getInfo(WebCL.DEVICE_TYPE),v=0;if(l&WebCL.DEVICE_TYPE_CPU)d[2]=0;if(l&WebCL.DEVICE_TYPE_GPU)d[2]=1;if(l&WebCL.DEVICE_TYPE_ACCELERATOR)d[2]=2;if(l&WebCL.DEVICE_TYPE_DEFAULT)d[2]=3;d[3]=S(c.getInfo(WebCL.DEVICE_PROFILE));d[4]=S(c.getInfo(WebCL.DEVICE_VERSION));d[5]=S(c.getInfo(WebCL.DEVICE_VENDOR));var
r=c.getInfo(WebCL.DEVICE_EXTENSIONS);d[6]=S(r);d[7]=c.getInfo(WebCL.DEVICE_VENDOR_ID);d[8]=c.getInfo(WebCL.DEVICE_MAX_WORK_ITEM_DIMENSIONS);d[9]=c.getInfo(WebCL.DEVICE_ADDRESS_BITS);d[10]=c.getInfo(WebCL.DEVICE_MAX_MEM_ALLOC_SIZE);d[11]=c.getInfo(WebCL.DEVICE_IMAGE_SUPPORT);d[12]=c.getInfo(WebCL.DEVICE_MAX_READ_IMAGE_ARGS);d[13]=c.getInfo(WebCL.DEVICE_MAX_WRITE_IMAGE_ARGS);d[14]=c.getInfo(WebCL.DEVICE_MAX_SAMPLERS);d[15]=c.getInfo(WebCL.DEVICE_MEM_BASE_ADDR_ALIGN);d[17]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHELINE_SIZE);d[18]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHE_SIZE);d[19]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_ARGS);d[20]=c.getInfo(WebCL.DEVICE_ENDIAN_LITTLE);d[21]=c.getInfo(WebCL.DEVICE_AVAILABLE);d[22]=c.getInfo(WebCL.DEVICE_COMPILER_AVAILABLE);d[23]=c.getInfo(WebCL.DEVICE_SINGLE_FP_CONFIG);d[24]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHE_TYPE);d[25]=c.getInfo(WebCL.DEVICE_QUEUE_PROPERTIES);d[26]=c.getInfo(WebCL.DEVICE_LOCAL_MEM_TYPE);d[28]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE);d[29]=c.getInfo(WebCL.DEVICE_EXECUTION_CAPABILITIES);d[31]=c.getInfo(WebCL.DEVICE_MAX_WORK_GROUP_SIZE);d[32]=c.getInfo(WebCL.DEVICE_IMAGE2D_MAX_HEIGHT);d[33]=c.getInfo(WebCL.DEVICE_IMAGE2D_MAX_WIDTH);d[34]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_DEPTH);d[35]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_HEIGHT);d[36]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_WIDTH);d[37]=c.getInfo(WebCL.DEVICE_MAX_PARAMETER_SIZE);d[38]=[0];var
n=c.getInfo(WebCL.DEVICE_MAX_WORK_ITEM_SIZES);d[38][1]=n[0];d[38][2]=n[1];d[38][3]=n[2];d[39]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);d[40]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);d[41]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_INT);d[42]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_LONG);d[43]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);d[45]=c.getInfo(WebCL.DEVICE_PROFILING_TIMER_RESOLUTION);d[46]=S(c.getInfo(WebCL.DRIVER_VERSION));g++;break}else
g++}else
g+=m}var
c=[0];d[1]=h;p[1]=d;c[1]=e;c[2]=p;return c}function
u4(){console.log(" spoc_getOpenCLDevicesCount");var
a=0,b=webcl.getPlatforms();for(var
d
in
b){var
e=b[d],c=e.getDevices();a+=c.length}return a}function
u5(){console.log(fu);return 0}function
u6(){console.log(fu);var
a=new
Array(3);a[0]=0;return a}function
dN(a){if(a[1]instanceof
Float32Array||a[1].constructor.name=="Float32Array")return 4;if(a[1]instanceof
Int32Array||a[1].constructor.name=="Int32Array")return 4;{console.log("unimplemented vector type");console.log(a[1].constructor.name);return 4}}function
u7(a,b,c){console.log("spoc_opencl_alloc_vect");var
f=a[2],i=a[4],h=i[b+1],j=a[5],k=dN(f),d=c[9],e=d[0],d=c[9],e=d[0],g=e.createBuffer(WebCL.MEM_READ_WRITE,j*k);h[2]=g;d[0]=e;c[9]=d;return 0}function
u8(){console.log(" spoc_opencl_compile");return 0}function
u9(a,b,c,d){console.log("spoc_opencl_cpu_to_device");var
f=a[2],k=a[4],j=k[b+1],l=a[5],m=dN(f),e=c[9],h=e[0],g=e[d+1],i=j[2];g.enqueueWriteBuffer(i,false,0,l*m,f[1]);e[d+1]=g;e[0]=h;c[9]=e;return 0}function
vd(a,b,c,d,e){console.log("spoc_opencl_device_to_cpu");var
g=a[2],l=a[4],k=l[b+1],n=a[5],o=dN(g),f=c[9],i=f[0],h=f[e+1],j=k[2],m=g[1];h.enqueueReadBuffer(j,false,0,n*o,m);f[e+1]=h;f[0]=i;c[9]=f;return 0}function
ve(a,b){console.log("spoc_opencl_flush");var
c=a[9][b+1];c.flush();a[9][b+1]=c;return 0}function
vf(){console.log(" spoc_opencl_is_available");return!dM}function
vg(a,b,c,d,e){console.log("spoc_opencl_launch_grid");var
m=b[1],n=b[2],o=b[3],h=c[1],i=c[2],j=c[3],g=new
Array(3);g[0]=m*h;g[1]=n*i;g[2]=o*j;var
f=new
Array(3);f[0]=h;f[1]=i;f[2]=j;var
l=d[9],k=l[e+1];if(h==1&&i==1&&j==1)k.enqueueNDRangeKernel(a,f.length,null,g);else
k.enqueueNDRangeKernel(a,f.length,null,g,f);return 0}function
vj(a,b,c,d){console.log("spoc_opencl_load_param_int");b.setArg(a[1],new
Uint32Array([c]));a[1]=a[1]+1;return 0}function
vl(a,b,c,d,e){console.log("spoc_opencl_load_param_vec");var
f=d[2];b.setArg(a[1],f);a[1]=a[1]+1;return 0}function
vo(){return new
Date().getTime()/b5}function
vp(){return 0}var
s=tw,m=tx,bi=dE,aP=gq,B=gr,at=tC,c2=tF,bU=tG,bj=tI,af=tJ,c6=tZ,fk=t0,w=t2,e$=t5,c4=dH,c8=t6,e9=t9,c3=t_,aQ=ub,x=bv,b=S,c7=ud,fc=ue,aO=ul,c5=um,fb=uo,bX=gA,y=up,bV=uu,e_=uv,fa=uw,fj=ux,_=uy,E=uz,fi=uB,fl=uD,fn=uF,fo=u2,fm=u4,ff=u5,fe=u6,fg=u7,fd=ve,bY=vp;function
j(a,b){return a.length==1?a(b):am(a,[b])}function
i(a,b,c){return a.length==2?a(b,c):am(a,[b,c])}function
p(a,b,c,d){return a.length==3?a(b,c,d):am(a,[b,c,d])}function
fh(a,b,c,d,e,f,g){return a.length==6?a(b,c,d,e,f,g):am(a,[b,c,d,e,f,g])}var
aX=[0,b("Failure")],by=[0,b("Invalid_argument")],bz=[0,b("End_of_file")],t=[0,b("Not_found")],H=[0,b("Assert_failure")],cI=b(al),cL=b(al),cN=b(al),eQ=b(g),eP=[0,b(f_),b(f3),b(fN),b(gf),b(gc)],e8=[0,1],e3=[0,b(gf),b(f3),b(f_),b(fN),b(gc)],e4=[0,b(du),b(c$),b(dc),b(de),b(dk),b(df),b(ds),b(dz),b(dg),b(dj)],e5=[0,b(dm),b(dx),b(dp)],c0=[0,b(dx),b(dc),b(de),b(dp),b(dm),b(c$),b(dz),b(dj),b(du),b(df),b(ds),b(dk),b(dg)];aO(6,t);aO(5,[0,b("Division_by_zero")]);aO(4,bz);aO(3,by);aO(2,aX);aO(1,[0,b("Sys_error")]);var
gK=b("really_input"),gJ=[0,0,[0,7,0]],gI=[0,1,[0,3,[0,4,[0,7,0]]]],gH=b(f1),gG=b(al),gE=b("true"),gF=b("false"),gL=b("Pervasives.do_at_exit"),gN=b("Array.blit"),gR=b("List.iter2"),gP=b("tl"),gO=b("hd"),gV=b("\\b"),gW=b("\\t"),gX=b("\\n"),gY=b("\\r"),gU=b("\\\\"),gT=b("\\'"),gS=b("Char.chr"),g1=b("String.contains_from"),g0=b("String.blit"),gZ=b("String.sub"),g_=b("Map.remove_min_elt"),g$=[0,0,0,0],ha=[0,b("map.ml"),270,10],hb=[0,0,0],g6=b(b8),g7=b(b8),g8=b(b8),g9=b(b8),hc=b("CamlinternalLazy.Undefined"),hf=b("Buffer.add: cannot grow buffer"),hv=b(g),hw=b(g),hz=b(f1),hA=b(cb),hB=b(cb),hx=b(b$),hy=b(b$),hu=b(fB),hs=b("neg_infinity"),ht=b("infinity"),hr=b(al),hq=b("printf: bad positional specification (0)."),hp=b("%_"),ho=[0,b("printf.ml"),143,8],hm=b(b$),hn=b("Printf: premature end of format string '"),hi=b(b$),hj=b(" in format string '"),hk=b(", at char number "),hl=b("Printf: bad conversion %"),hg=b("Sformat.index_of_int: negative argument "),hD=b(dt),hE=[0,987910699,495797812,364182224,414272206,318284740,990407751,383018966,270373319,840823159,24560019,536292337,512266505,189156120,730249596,143776328,51606627,140166561,366354223,1003410265,700563762,981890670,913149062,526082594,1021425055,784300257,667753350,630144451,949649812,48546892,415514493,258888527,511570777,89983870,283659902,308386020,242688715,482270760,865188196,1027664170,207196989,193777847,619708188,671350186,149669678,257044018,87658204,558145612,183450813,28133145,901332182,710253903,510646120,652377910,409934019,801085050],tq=b("OCAMLRUNPARAM"),to=b("CAMLRUNPARAM"),hG=b(g),h3=[0,b("camlinternalOO.ml"),287,50],h2=b(g),hI=b("CamlinternalOO.last_id"),iw=b(g),it=b(fP),is=b(".\\"),ir=b(f2),iq=b("..\\"),ih=b(fP),ig=b(f2),ib=b(g),ia=b(g),ic=b(dd),id=b(fA),tm=b("TMPDIR"),ij=b("/tmp"),ik=b("'\\''"),io=b(dd),ip=b("\\"),tk=b("TEMP"),iu=b(al),iz=b(dd),iA=b(fA),iD=b("Cygwin"),iE=b(fw),iF=b("Win32"),iG=[0,b("filename.ml"),189,9],iN=b("E2BIG"),iP=b("EACCES"),iQ=b("EAGAIN"),iR=b("EBADF"),iS=b("EBUSY"),iT=b("ECHILD"),iU=b("EDEADLK"),iV=b("EDOM"),iW=b("EEXIST"),iX=b("EFAULT"),iY=b("EFBIG"),iZ=b("EINTR"),i0=b("EINVAL"),i1=b("EIO"),i2=b("EISDIR"),i3=b("EMFILE"),i4=b("EMLINK"),i5=b("ENAMETOOLONG"),i6=b("ENFILE"),i7=b("ENODEV"),i8=b("ENOENT"),i9=b("ENOEXEC"),i_=b("ENOLCK"),i$=b("ENOMEM"),ja=b("ENOSPC"),jb=b("ENOSYS"),jc=b("ENOTDIR"),jd=b("ENOTEMPTY"),je=b("ENOTTY"),jf=b("ENXIO"),jg=b("EPERM"),jh=b("EPIPE"),ji=b("ERANGE"),jj=b("EROFS"),jk=b("ESPIPE"),jl=b("ESRCH"),jm=b("EXDEV"),jn=b("EWOULDBLOCK"),jo=b("EINPROGRESS"),jp=b("EALREADY"),jq=b("ENOTSOCK"),jr=b("EDESTADDRREQ"),js=b("EMSGSIZE"),jt=b("EPROTOTYPE"),ju=b("ENOPROTOOPT"),jv=b("EPROTONOSUPPORT"),jw=b("ESOCKTNOSUPPORT"),jx=b("EOPNOTSUPP"),jy=b("EPFNOSUPPORT"),jz=b("EAFNOSUPPORT"),jA=b("EADDRINUSE"),jB=b("EADDRNOTAVAIL"),jC=b("ENETDOWN"),jD=b("ENETUNREACH"),jE=b("ENETRESET"),jF=b("ECONNABORTED"),jG=b("ECONNRESET"),jH=b("ENOBUFS"),jI=b("EISCONN"),jJ=b("ENOTCONN"),jK=b("ESHUTDOWN"),jL=b("ETOOMANYREFS"),jM=b("ETIMEDOUT"),jN=b("ECONNREFUSED"),jO=b("EHOSTDOWN"),jP=b("EHOSTUNREACH"),jQ=b("ELOOP"),jR=b("EOVERFLOW"),jS=b("EUNKNOWNERR %d"),iO=b("Unix.Unix_error(Unix.%s, %S, %S)"),iJ=b(fR),iK=b(g),iL=b(g),iM=b(fR),jT=b("0.0.0.0"),jU=b("127.0.0.1"),tj=b("::"),ti=b("::1"),j5=[0,b("Vector.ml"),fW,25],j6=b("Cuda.No_Cuda_Device"),j7=b("Cuda.ERROR_DEINITIALIZED"),j8=b("Cuda.ERROR_NOT_INITIALIZED"),j9=b("Cuda.ERROR_INVALID_CONTEXT"),j_=b("Cuda.ERROR_INVALID_VALUE"),j$=b("Cuda.ERROR_OUT_OF_MEMORY"),ka=b("Cuda.ERROR_INVALID_DEVICE"),kb=b("Cuda.ERROR_NOT_FOUND"),kc=b("Cuda.ERROR_FILE_NOT_FOUND"),kd=b("Cuda.ERROR_UNKNOWN"),ke=b("Cuda.ERROR_LAUNCH_FAILED"),kf=b("Cuda.ERROR_LAUNCH_OUT_OF_RESOURCES"),kg=b("Cuda.ERROR_LAUNCH_TIMEOUT"),kh=b("Cuda.ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"),ki=b("no_cuda_device"),kj=b("cuda_error_deinitialized"),kk=b("cuda_error_not_initialized"),kl=b("cuda_error_invalid_context"),km=b("cuda_error_invalid_value"),kn=b("cuda_error_out_of_memory"),ko=b("cuda_error_invalid_device"),kp=b("cuda_error_not_found"),kq=b("cuda_error_file_not_found"),kr=b("cuda_error_launch_failed"),ks=b("cuda_error_launch_out_of_resources"),kt=b("cuda_error_launch_timeout"),ku=b("cuda_error_launch_incompatible_texturing"),kv=b("cuda_error_unknown"),kw=b("OpenCL.No_OpenCL_Device"),kx=b("OpenCL.OPENCL_ERROR_UNKNOWN"),ky=b("OpenCL.INVALID_CONTEXT"),kz=b("OpenCL.INVALID_DEVICE"),kA=b("OpenCL.INVALID_VALUE"),kB=b("OpenCL.INVALID_QUEUE_PROPERTIES"),kC=b("OpenCL.OUT_OF_RESOURCES"),kD=b("OpenCL.MEM_OBJECT_ALLOCATION_FAILURE"),kE=b("OpenCL.OUT_OF_HOST_MEMORY"),kF=b("OpenCL.FILE_NOT_FOUND"),kG=b("OpenCL.INVALID_PROGRAM"),kH=b("OpenCL.INVALID_BINARY"),kI=b("OpenCL.INVALID_BUILD_OPTIONS"),kJ=b("OpenCL.INVALID_OPERATION"),kK=b("OpenCL.COMPILER_NOT_AVAILABLE"),kL=b("OpenCL.BUILD_PROGRAM_FAILURE"),kM=b("OpenCL.INVALID_KERNEL"),kN=b("OpenCL.INVALID_ARG_INDEX"),kO=b("OpenCL.INVALID_ARG_VALUE"),kP=b("OpenCL.INVALID_MEM_OBJECT"),kQ=b("OpenCL.INVALID_SAMPLER"),kR=b("OpenCL.INVALID_ARG_SIZE"),kS=b("OpenCL.INVALID_COMMAND_QUEUE"),kT=b("no_opencl_device"),kU=b("opencl_error_unknown"),kV=b("opencl_invalid_context"),kW=b("opencl_invalid_device"),kX=b("opencl_invalid_value"),kY=b("opencl_invalid_queue_properties"),kZ=b("opencl_out_of_resources"),k0=b("opencl_mem_object_allocation_failure"),k1=b("opencl_out_of_host_memory"),k2=b("opencl_file_not_found"),k3=b("opencl_invalid_program"),k4=b("opencl_invalid_binary"),k5=b("opencl_invalid_build_options"),k6=b("opencl_invalid_operation"),k7=b("opencl_compiler_not_available"),k8=b("opencl_build_program_failure"),k9=b("opencl_invalid_kernel"),k_=b("opencl_invalid_arg_index"),k$=b("opencl_invalid_arg_value"),la=b("opencl_invalid_mem_object"),lb=b("opencl_invalid_sampler"),lc=b("opencl_invalid_arg_size"),ld=b("opencl_invalid_command_queue"),le=b(cf),lf=b(cf),lw=b(fJ),lv=b(fF),lu=b(fJ),lt=b(fF),ls=[0,1],lr=b(g),ln=b(b1),li=b("Cl LOAD ARG Type Not Implemented\n"),lh=b("CU LOAD ARG Type Not Implemented\n"),lg=[0,b(dj),b(dg),b(dz),b(ds),b(df),b(dm),b(dk),b(de),b(dc),b(dx),b(c$),b(du),b(dp)],lj=b("Kernel.ERROR_BLOCK_SIZE"),ll=b("Kernel.ERROR_GRID_SIZE"),lo=b("Kernel.No_source_for_device"),lz=b("Empty"),lA=b("Unit"),lB=b("Kern"),lC=b("Params"),lD=b("Plus"),lE=b("Plusf"),lF=b("Min"),lG=b("Minf"),lH=b("Mul"),lI=b("Mulf"),lJ=b("Div"),lK=b("Divf"),lL=b("Mod"),lM=b("Id "),lN=b("IdName "),lO=b("IntVar "),lP=b("FloatVar "),lQ=b("UnitVar "),lR=b("CastDoubleVar "),lS=b("DoubleVar "),lT=b("IntArr"),lU=b("Int32Arr"),lV=b("Int64Arr"),lW=b("Float32Arr"),lX=b("Float64Arr"),lY=b("VecVar "),lZ=b("Concat"),l0=b("Seq"),l1=b("Return"),l2=b("Set"),l3=b("Decl"),l4=b("SetV"),l5=b("SetLocalVar"),l6=b("Intrinsics"),l7=b(C),l8=b("IntId "),l9=b("Int "),l$=b("IntVecAcc"),ma=b("Local"),mb=b("Acc"),mc=b("Ife"),md=b("If"),me=b("Or"),mf=b("And"),mg=b("EqBool"),mh=b("LtBool"),mi=b("GtBool"),mj=b("LtEBool"),mk=b("GtEBool"),ml=b("DoLoop"),mm=b("While"),mn=b("App"),mo=b("GInt"),mp=b("GFloat"),l_=b("Float "),ly=b("  "),lx=b("%s\n"),n3=b(fZ),n4=[0,b(dl),166,14],ms=b(g),mt=b(b1),mu=b("\n}\n#ifdef __cplusplus\n}\n#endif"),mv=b(" ) {\n"),mw=b(g),mx=b(b0),mz=b(g),my=b('#ifdef __cplusplus\nextern "C" {\n#endif\n\n__global__ void spoc_dummy ( '),mA=b(aj),mB=b(cg),mC=b(ak),mD=b(aj),mE=b(cg),mF=b(ak),mG=b(aj),mH=b(b9),mI=b(ak),mJ=b(aj),mK=b(b9),mL=b(ak),mM=b(aj),mN=b(ca),mO=b(ak),mP=b(aj),mQ=b(ca),mR=b(ak),mS=b(aj),mT=b(ci),mU=b(ak),mV=b(aj),mW=b(ci),mX=b(ak),mY=b(aj),mZ=b(fz),m0=b(ak),m1=b(f6),m2=b(fy),m3=[0,b(dl),65,17],m4=b(b_),m5=b(fK),m6=b(M),m7=b(N),m8=b(fQ),m9=b(M),m_=b(N),m$=b(ft),na=b(M),nb=b(N),nc=b(fG),nd=b(M),ne=b(N),nf=b(f7),ng=b(f0),ni=b("int"),nj=b("float"),nk=b("double"),nh=[0,b(dl),60,12],nm=b(b0),nl=b(gk),nn=b(fY),no=b(g),np=b(g),ns=b(b3),nt=b(ai),nu=b(aR),nw=b(b3),nv=b(ai),nx=b(aa),ny=b(M),nz=b(N),nA=b("}\n"),nB=b(aR),nC=b(aR),nD=b("{"),nE=b(br),nF=b(fE),nG=b(br),nH=b(bo),nI=b(ch),nJ=b(br),nK=b(bo),nL=b(ch),nM=b(fr),nN=b(fp),nO=b(fI),nP=b(gb),nQ=b(fO),nR=b(bZ),nS=b(ga),nT=b(ce),nU=b(fv),nV=b(b7),nW=b(bZ),nX=b(b7),nY=b(ai),nZ=b(fq),n0=b(ce),n1=b(bo),n2=b(fM),n7=b(aV),n8=b(aV),n9=b(C),n_=b(C),n5=b(fV),n6=b(gm),n$=b(aa),nq=b(b3),nr=b(ai),oa=b(M),ob=b(N),od=b(b_),oe=b(aa),of=b(gh),og=b(M),oh=b(N),oi=b(aa),oc=b("cuda error parse_float"),mq=[0,b(g),b(g)],pE=b(fZ),pF=[0,b(dr),162,14],ol=b(g),om=b(b1),on=b(ce),oo=b(" ) \n{\n"),op=b(g),oq=b(b0),os=b(g),or=b("__kernel void spoc_dummy ( "),ot=b(cg),ou=b(cg),ov=b(b9),ow=b(b9),ox=b(ca),oy=b(ca),oz=b(ci),oA=b(ci),oB=b(fz),oC=b(f6),oD=b(fy),oE=[0,b(dr),65,17],oF=b(b_),oG=b(fK),oH=b(M),oI=b(N),oJ=b(fQ),oK=b(M),oL=b(N),oM=b(ft),oN=b(M),oO=b(N),oP=b(fG),oQ=b(M),oR=b(N),oS=b(f7),oT=b(f0),oV=b("__global int"),oW=b("__global float"),oX=b("__global double"),oU=[0,b(dr),60,12],oZ=b(b0),oY=b(gk),o0=b(fY),o1=b(g),o2=b(g),o4=b(b3),o5=b(ai),o6=b(aR),o7=b(ai),o8=b(aa),o9=b(M),o_=b(N),o$=b(g),pa=b(b1),pb=b(aR),pc=b(g),pd=b(br),pe=b(fE),pf=b(br),pg=b(bo),ph=b(ch),pi=b(ce),pj=b(aR),pk=b("{\n"),pl=b(")\n"),pm=b(ch),pn=b(fr),po=b(fp),pp=b(fI),pq=b(gb),pr=b(fO),ps=b(bZ),pt=b(ga),pu=b(gd),pv=b(fv),pw=b(b7),px=b(bZ),py=b(b7),pz=b(ai),pA=b(fq),pB=b(gd),pC=b(bo),pD=b(fM),pI=b(aV),pJ=b(aV),pK=b(C),pL=b(C),pG=b(fV),pH=b(gm),pM=b(aa),o3=b(ai),pN=b(M),pO=b(N),pQ=b(b_),pR=b(aa),pS=b(gh),pT=b(M),pU=b(N),pV=b(aa),pP=b("opencl error parse_float"),oj=[0,b(g),b(g)],qU=[0,0],qV=[0,0],qW=[0,1],qX=[0,1],qO=b("kirc_kernel.cu"),qP=b("nvcc -m64 -arch=sm_10 -O3 -ptx kirc_kernel.cu -o kirc_kernel.ptx"),qQ=b("kirc_kernel.ptx"),qR=b("rm kirc_kernel.cu kirc_kernel.ptx"),qL=[0,b(g),b(g)],qN=b(g),qM=[0,b("Kirc.ml"),407,81],qS=b(ai),qT=b(f9),qG=[33,0],qD=b(f9),pW=b("int spoc_xor (int a, int b ) { return (a^b);}\n"),pX=b("int spoc_powint (int a, int b ) { return ((int) pow (((float) a), ((float) b)));}\n"),pY=b("int logical_and (int a, int b ) { return (a & b);}\n"),pZ=b("float spoc_fdiv ( float a, float b ) { return (a / b);}\n"),p0=b("float spoc_fmul ( float a, float b ) { return (a * b);}\n"),p1=b("float spoc_fminus ( float a, float b ) { return (a - b);}\n"),p2=b("float spoc_fadd ( float a, float b ) { return (a + b);}\n"),p3=b("float spoc_fdiv ( float a, float b );\n"),p4=b("float spoc_fmul ( float a, float b );\n"),p5=b("float spoc_fminus ( float a, float b );\n"),p6=b("float spoc_fadd ( float a, float b );\n"),p8=b(dq),p9=b("double spoc_ddiv ( double a, double b ) { return (a / b);}\n"),p_=b("double spoc_dmul ( double a, double b ) { return (a * b);}\n"),p$=b("double spoc_dminus ( double a, double b ) { return (a - b);}\n"),qa=b("double spoc_dadd ( double a, double b ) { return (a + b);}\n"),qb=b("double spoc_ddiv ( double a, double b );\n"),qc=b("double spoc_dmul ( double a, double b );\n"),qd=b("double spoc_dminus ( double a, double b );\n"),qe=b("double spoc_dadd ( double a, double b );\n"),qf=b(dq),qg=b("#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"),qh=b("#elif defined(cl_amd_fp64)  // AMD extension available?\n"),qi=b("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"),qj=b("#if defined(cl_khr_fp64)  // Khronos extension available?\n"),qk=b(gg),ql=b(f$),qn=b(dq),qo=b("__device__ double spoc_ddiv ( double a, double b ) { return (a / b);}\n"),qp=b("__device__ double spoc_dmul ( double a, double b ) { return (a * b);}\n"),qq=b("__device__ double spoc_dminus ( double a, double b ) { return (a - b);}\n"),qr=b("__device__ double spoc_dadd ( double a, double b ) { return (a + b);}\n"),qs=b(gg),qt=b(f$),qv=b("__device__ int spoc_xor (int a, int b ) { return (a^b);}\n"),qw=b("__device__ int spoc_powint (int a, int b ) { return ((int) pow (((double) a), ((double) b)));}\n"),qx=b("__device__ int logical_and (int a, int b ) { return (a & b);}\n"),qy=b("__device__ float spoc_fdiv ( float a, float b ) { return (a / b);}\n"),qz=b("__device__ float spoc_fmul ( float a, float b ) { return (a * b);}\n"),qA=b("__device__ float spoc_fminus ( float a, float b ) { return (a - b);}\n"),qB=b("__device__ float spoc_fadd ( float a, float b ) { return (a + b);}\n"),qJ=[0,b(g),b(g)],re=b("canvas"),rb=b("span"),ra=b("a"),q$=b("br"),q_=b(fs),q9=b("select"),q8=b("option"),rc=b("Dom_html.Canvas_not_available"),tg=[0,b(fD),170,17],td=b("Will use device : %s!"),te=[0,1],tf=b(g),tc=[0,196,fU,fU],tb=b("Time %s : %Fs\n%!"),rp=b("spoc_dummy"),rq=b("kirc_kernel"),rn=b("spoc_kernel_extension error"),rf=[0,b(fD),12,15],rs=[7,[0,0,0]],r5=b(au),r6=b(au),r9=b(au),r_=b(au),se=b(au),sf=b(au),si=b(au),sj=b(au),sB=b(aV),sC=b(aV),sI=b("(get_local_size (0))"),sJ=b("blockDim.x"),sL=b("(get_group_id (0))"),sM=b("blockIdx.x"),sO=b("(get_local_id (0))"),sP=b("threadIdx.x"),sS=b("(get_local_size (1))"),sT=b("blockDim.y"),sV=b("(get_group_id (1))"),sW=b("blockIdx.y"),sY=b("(get_local_id (1))"),sZ=b("threadIdx.y");function
T(a){throw[0,aX,a]}function
G(a){throw[0,by,a]}function
h(a,b){var
c=a.getLen(),e=b.getLen(),d=B(c+e|0);bi(a,0,d,0,c);bi(b,0,d,c,e);return d}function
k(a){return b(g+a)}function
O(a){var
c=c2(gH,a),b=0,f=c.getLen();for(;;){if(f<=b)var
e=h(c,gG);else{var
d=c.safeGet(b),g=48<=d?58<=d?0:1:45===d?1:0;if(g){var
b=b+1|0;continue}var
e=c}return e}}function
cl(a,b){if(a){var
c=a[1];return[0,c,cl(a[2],b)]}return b}e9(0);var
dO=c3(1);c3(2);function
dP(a,b){return gw(a,b,0,b.getLen())}function
dQ(a){return e9(e_(a,gJ,0))}function
dR(a){var
b=t$(0);for(;;){if(b){var
c=b[2],d=b[1];try{c4(d)}catch(f){}var
b=c;continue}return 0}}c5(gL,dR);function
dS(a){return e$(a)}function
gM(a,b){return ua(a,b)}function
dT(a){return c4(a)}function
dU(a,b){var
d=b.length-1-1|0,e=0;if(!(d<0)){var
c=e;for(;;){j(a,b[c+1]);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return 0}function
aY(a,b){var
d=b.length-1;if(0===d)return[0];var
e=w(d,j(a,b[0+1])),f=d-1|0,g=1;if(!(f<1)){var
c=g;for(;;){e[c+1]=j(a,b[c+1]);var
h=c+1|0;if(f!==c){var
c=h;continue}break}}return e}function
cm(a,b){var
d=b.length-1-1|0,e=0;if(!(d<0)){var
c=e;for(;;){i(a,c,b[c+1]);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return 0}function
aZ(a){var
b=a.length-1-1|0,c=0;for(;;){if(0<=b){var
d=[0,a[b+1],c],b=b-1|0,c=d;continue}return c}}function
dV(a,b,c){var
e=[0,b],f=c.length-1-1|0,g=0;if(!(f<0)){var
d=g;for(;;){e[1]=i(a,e[1],c[d+1]);var
h=d+1|0;if(f!==d){var
d=h;continue}break}}return e[1]}function
dW(a){var
b=a,c=0;for(;;){if(b){var
d=[0,b[1],c],b=b[2],c=d;continue}return c}}function
cn(a,b){if(b){var
c=b[2],d=j(a,b[1]);return[0,d,cn(a,c)]}return 0}function
cp(a,b,c){if(b){var
d=b[1];return i(a,d,cp(a,b[2],c))}return c}function
dY(a,b,c){var
e=b,d=c;for(;;){if(e){if(d){var
f=d[2],g=e[2];i(a,e[1],d[1]);var
e=g,d=f;continue}}else
if(!d)return 0;return G(gR)}}function
cq(a,b){var
c=b;for(;;){if(c){var
e=c[2],d=0===aP(c[1],a)?1:0;if(d)return d;var
c=e;continue}return 0}}function
cr(a){if(0<=a)if(!(r<a))return a;return G(gS)}function
dZ(a){var
b=65<=a?90<a?0:1:0;if(!b){var
c=192<=a?214<a?0:1:0;if(!c){var
d=216<=a?222<a?1:0:1;if(d)return a}}return a+32|0}function
an(a,b){var
c=B(a);tE(c,0,a,b);return c}function
u(a,b,c){if(0<=b)if(0<=c)if(!((a.getLen()-c|0)<b)){var
d=B(c);bi(a,b,d,0,c);return d}return G(gZ)}function
bB(a,b,c,d,e){if(0<=e)if(0<=b)if(!((a.getLen()-e|0)<b))if(0<=d)if(!((c.getLen()-e|0)<d))return bi(a,b,c,d,e);return G(g0)}function
d0(a){var
c=a.getLen();if(0===c)var
f=a;else{var
d=B(c),e=c-1|0,g=0;if(!(e<0)){var
b=g;for(;;){d.safeSet(b,dZ(a.safeGet(b)));var
h=b+1|0;if(e!==b){var
b=h;continue}break}}var
f=d}return f}var
ct=ut(0)[1],aE=uq(0),cu=(1<<(aE-10|0))-1|0,a0=x(aE/8|0,cu)-1|0,g3=us(0)[2],g4=b6,g5=aT;function
cv(k){function
h(a){return a?a[5]:0}function
e(a,b,c,d){var
e=h(a),f=h(d),g=f<=e?e+1|0:f+1|0;return[0,a,b,c,d,g]}function
q(a,b){return[0,0,a,b,0,1]}function
f(a,b,c,d){var
i=a?a[5]:0,j=d?d[5]:0;if((j+2|0)<i){if(a){var
f=a[4],m=a[3],n=a[2],k=a[1],q=h(f);if(q<=h(k))return e(k,n,m,e(f,b,c,d));if(f){var
r=f[3],s=f[2],t=f[1],u=e(f[4],b,c,d);return e(e(k,n,m,t),s,r,u)}return G(g6)}return G(g7)}if((i+2|0)<j){if(d){var
l=d[4],o=d[3],p=d[2],g=d[1],v=h(g);if(v<=h(l))return e(e(a,b,c,g),p,o,l);if(g){var
w=g[3],x=g[2],y=g[1],z=e(g[4],p,o,l);return e(e(a,b,c,y),x,w,z)}return G(g8)}return G(g9)}var
A=j<=i?i+1|0:j+1|0;return[0,a,b,c,d,A]}var
a=0;function
I(a){return a?0:1}function
r(a,b,c){if(c){var
d=c[4],h=c[3],e=c[2],g=c[1],l=c[5],j=i(k[1],a,e);return 0===j?[0,g,a,b,d,l]:0<=j?f(g,e,h,r(a,b,d)):f(r(a,b,g),e,h,d)}return[0,0,a,b,0,1]}function
J(a,b){var
c=b;for(;;){if(c){var
e=c[4],f=c[3],g=c[1],d=i(k[1],a,c[2]);if(0===d)return f;var
h=0<=d?e:g,c=h;continue}throw[0,t]}}function
K(a,b){var
c=b;for(;;){if(c){var
f=c[4],g=c[1],d=i(k[1],a,c[2]),e=0===d?1:0;if(e)return e;var
h=0<=d?f:g,c=h;continue}return 0}}function
n(a){var
b=a;for(;;){if(b){var
c=b[1];if(c){var
b=c;continue}return[0,b[2],b[3]]}throw[0,t]}}function
L(a){var
b=a;for(;;){if(b){var
c=b[4],d=b[3],e=b[2];if(c){var
b=c;continue}return[0,e,d]}throw[0,t]}}function
s(a){if(a){var
b=a[1];if(b){var
c=a[4],d=a[3],e=a[2];return f(s(b),e,d,c)}return a[4]}return G(g_)}function
u(a,b){if(b){var
c=b[4],j=b[3],e=b[2],d=b[1],l=i(k[1],a,e);if(0===l){if(d)if(c){var
h=n(c),m=h[2],o=h[1],g=f(d,o,m,s(c))}else
var
g=d;else
var
g=c;return g}return 0<=l?f(d,e,j,u(a,c)):f(u(a,d),e,j,c)}return 0}function
y(a,b){var
c=b;for(;;){if(c){var
d=c[4],e=c[3],f=c[2];y(a,c[1]);i(a,f,e);var
c=d;continue}return 0}}function
c(a,b){if(b){var
d=b[5],e=b[4],f=b[3],g=b[2],h=c(a,b[1]),i=j(a,f);return[0,h,g,i,c(a,e),d]}return 0}function
v(a,b){if(b){var
c=b[2],d=b[5],e=b[4],f=b[3],g=v(a,b[1]),h=i(a,c,f);return[0,g,c,h,v(a,e),d]}return 0}function
z(a,b,c){var
d=b,e=c;for(;;){if(d){var
f=d[4],g=d[3],h=d[2],i=p(a,h,g,z(a,d[1],e)),d=f,e=i;continue}return e}}function
A(a,b){var
c=b;for(;;){if(c){var
g=c[4],h=c[1],d=i(a,c[2],c[3]);if(d){var
e=A(a,h);if(e){var
c=g;continue}var
f=e}else
var
f=d;return f}return 1}}function
B(a,b){var
c=b;for(;;){if(c){var
g=c[4],h=c[1],d=i(a,c[2],c[3]);if(d)var
e=d;else{var
f=B(a,h);if(!f){var
c=g;continue}var
e=f}return e}return 0}}function
C(a,b,c){if(c){var
d=c[4],e=c[3],g=c[2];return f(C(a,b,c[1]),g,e,d)}return q(a,b)}function
D(a,b,c){if(c){var
d=c[3],e=c[2],g=c[1];return f(g,e,d,D(a,b,c[4]))}return q(a,b)}function
g(a,b,c,d){if(a){if(d){var
h=d[5],i=a[5],j=d[4],k=d[3],l=d[2],m=d[1],n=a[4],o=a[3],p=a[2],q=a[1];return(h+2|0)<i?f(q,p,o,g(n,b,c,d)):(i+2|0)<h?f(g(a,b,c,m),l,k,j):e(a,b,c,d)}return D(b,c,a)}return C(b,c,d)}function
o(a,b){if(a){if(b){var
c=n(b),d=c[2],e=c[1];return g(a,e,d,s(b))}return a}return b}function
E(a,b,c,d){return c?g(a,b,c[1],d):o(a,d)}function
l(a,b){if(b){var
c=b[4],d=b[3],e=b[2],f=b[1],m=i(k[1],a,e);if(0===m)return[0,f,[0,d],c];if(0<=m){var
h=l(a,c),n=h[3],o=h[2];return[0,g(f,e,d,h[1]),o,n]}var
j=l(a,f),p=j[2],q=j[1];return[0,q,p,g(j[3],e,d,c)]}return g$}function
m(a,b,c){if(b){var
d=b[2],i=b[5],j=b[4],k=b[3],n=b[1];if(h(c)<=i){var
e=l(d,c),o=e[2],q=e[1],r=m(a,j,e[3]),s=p(a,d,[0,k],o);return E(m(a,n,q),d,s,r)}}else
if(!c)return 0;if(c){var
f=c[2],t=c[4],u=c[3],v=c[1],g=l(f,b),w=g[2],x=g[1],y=m(a,g[3],t),z=p(a,f,w,[0,u]);return E(m(a,x,v),f,z,y)}throw[0,H,ha]}function
w(a,b){if(b){var
c=b[3],d=b[2],h=b[4],e=w(a,b[1]),j=i(a,d,c),f=w(a,h);return j?g(e,d,c,f):o(e,f)}return 0}function
x(a,b){if(b){var
c=b[3],d=b[2],m=b[4],e=x(a,b[1]),f=e[2],h=e[1],n=i(a,d,c),j=x(a,m),k=j[2],l=j[1];if(n){var
p=o(f,k);return[0,g(h,d,c,l),p]}var
q=g(f,d,c,k);return[0,o(h,l),q]}return hb}function
d(a,b){var
c=a,d=b;for(;;){if(c){var
e=[0,c[2],c[3],c[4],d],c=c[1],d=e;continue}return d}}function
M(a,b,c){var
s=d(c,0),f=d(b,0),e=s;for(;;){if(f)if(e){var
l=e[4],m=e[3],n=e[2],o=f[4],p=f[3],q=f[2],h=i(k[1],f[1],e[1]);if(0===h){var
j=i(a,q,n);if(0===j){var
r=d(m,l),f=d(p,o),e=r;continue}var
g=j}else
var
g=h}else
var
g=1;else
var
g=e?-1:0;return g}}function
N(a,b,c){var
t=d(c,0),f=d(b,0),e=t;for(;;){if(f)if(e){var
m=e[4],n=e[3],o=e[2],p=f[4],q=f[3],r=f[2],h=0===i(k[1],f[1],e[1])?1:0;if(h){var
j=i(a,r,o);if(j){var
s=d(n,m),f=d(q,p),e=s;continue}var
l=j}else
var
l=h;var
g=l}else
var
g=0;else
var
g=e?0:1;return g}}function
b(a){if(a){var
c=a[1],d=b(a[4]);return(b(c)+1|0)+d|0}return 0}function
F(a,b){var
d=a,c=b;for(;;){if(c){var
e=c[3],f=c[2],g=c[1],d=[0,[0,f,e],F(d,c[4])],c=g;continue}return d}}return[0,a,I,K,r,q,u,m,M,N,y,z,A,B,w,x,b,function(a){return F(0,a)},n,L,n,l,J,c,v]}var
hd=[0,hc];function
he(a){throw[0,hd]}function
a1(a){var
b=1<=a?a:1,c=a0<b?a0:b,d=B(c);return[0,d,0,c,d]}function
a2(a){return u(a[1],0,a[2])}function
d3(a,b){var
c=[0,a[3]];for(;;){if(c[1]<(a[2]+b|0)){c[1]=2*c[1]|0;continue}if(a0<c[1])if((a[2]+b|0)<=a0)c[1]=a0;else
T(hf);var
d=B(c[1]);bB(a[1],0,d,0,a[2]);a[1]=d;a[3]=c[1];return 0}}function
I(a,b){var
c=a[2];if(a[3]<=c)d3(a,1);a[1].safeSet(c,b);a[2]=c+1|0;return 0}function
bD(a,b){var
c=b.getLen(),d=a[2]+c|0;if(a[3]<d)d3(a,c);bB(b,0,a[1],a[2],c);a[2]=d;return 0}function
cw(a){return 0<=a?a:T(h(hg,k(a)))}function
d4(a,b){return cw(a+b|0)}var
hh=1;function
d5(a){return d4(hh,a)}function
d6(a){return u(a,0,a.getLen())}function
d7(a,b,c){var
d=h(hj,h(a,hi)),e=h(hk,h(k(b),d));return G(h(hl,h(an(1,c),e)))}function
a3(a,b,c){return d7(d6(a),b,c)}function
bE(a){return G(h(hn,h(d6(a),hm)))}function
ax(e,b,c,d){function
h(a){if((e.safeGet(a)+aU|0)<0||9<(e.safeGet(a)+aU|0))return a;var
b=a+1|0;for(;;){var
c=e.safeGet(b);if(48<=c){if(!(58<=c)){var
b=b+1|0;continue}var
d=0}else
if(36===c){var
f=b+1|0,d=1}else
var
d=0;if(!d)var
f=a;return f}}var
i=h(b+1|0),f=a1((c-i|0)+10|0);I(f,37);var
a=i,g=dW(d);for(;;){if(a<=c){var
j=e.safeGet(a);if(42===j){if(g){var
l=g[2];bD(f,k(g[1]));var
a=h(a+1|0),g=l;continue}throw[0,H,ho]}I(f,j);var
a=a+1|0;continue}return a2(f)}}function
d8(a,b,c,d,e){var
f=ax(b,c,d,e);if(78!==a)if(bq!==a)return f;f.safeSet(f.getLen()-1|0,dv);return f}function
d9(a){return function(c,b){var
m=c.getLen();function
n(a,b){var
o=40===a?41:db;function
k(a){var
d=a;for(;;){if(m<=d)return bE(c);if(37===c.safeGet(d)){var
e=d+1|0;if(m<=e)var
f=bE(c);else{var
g=c.safeGet(e),h=g-40|0;if(h<0||1<h){var
l=h-83|0;if(l<0||2<l)var
j=1;else
switch(l){case
1:var
j=1;break;case
2:var
i=1,j=0;break;default:var
i=0,j=0}if(j){var
f=k(e+1|0),i=2}}else
var
i=0===h?0:1;switch(i){case
1:var
f=g===o?e+1|0:a3(c,b,g);break;case
2:break;default:var
f=k(n(g,e+1|0)+1|0)}}return f}var
d=d+1|0;continue}}return k(b)}return n(a,b)}}function
d_(j,b,c){var
m=j.getLen()-1|0;function
s(a){var
l=a;a:for(;;){if(l<m){if(37===j.safeGet(l)){var
e=0,h=l+1|0;for(;;){if(m<h)var
w=bE(j);else{var
n=j.safeGet(h);if(58<=n){if(95===n){var
e=1,h=h+1|0;continue}}else
if(32<=n)switch(n+fC|0){case
1:case
2:case
4:case
5:case
6:case
7:case
8:case
9:case
12:case
15:break;case
0:case
3:case
11:case
13:var
h=h+1|0;continue;case
10:var
h=p(b,e,h,aC);continue;default:var
h=h+1|0;continue}var
d=h;b:for(;;){if(m<d)var
f=bE(j);else{var
k=j.safeGet(d);if(fW<=k)var
g=0;else
switch(k){case
78:case
88:case
aS:case
aC:case
dh:case
dv:case
dw:var
f=p(b,e,d,aC),g=1;break;case
69:case
70:case
71:case
ge:case
di:case
dA:var
f=p(b,e,d,di),g=1;break;case
33:case
37:case
44:case
64:var
f=d+1|0,g=1;break;case
83:case
91:case
bt:var
f=p(b,e,d,bt),g=1;break;case
97:case
cc:case
da:var
f=p(b,e,d,k),g=1;break;case
76:case
fL:case
bq:var
t=d+1|0;if(m<t){var
f=p(b,e,d,aC),g=1}else{var
q=j.safeGet(t)+gi|0;if(q<0||32<q)var
r=1;else
switch(q){case
0:case
12:case
17:case
23:case
29:case
32:var
f=i(c,p(b,e,d,k),aC),g=1,r=0;break;default:var
r=1}if(r){var
f=p(b,e,d,aC),g=1}}break;case
67:case
99:var
f=p(b,e,d,99),g=1;break;case
66:case
98:var
f=p(b,e,d,66),g=1;break;case
41:case
db:var
f=p(b,e,d,k),g=1;break;case
40:var
f=s(p(b,e,d,k)),g=1;break;case
dy:var
u=p(b,e,d,k),v=i(d9(k),j,u),o=u;for(;;){if(o<(v-2|0)){var
o=i(c,o,j.safeGet(o));continue}var
d=v-1|0;continue b}default:var
g=0}if(!g)var
f=a3(j,d,k)}var
w=f;break}}var
l=w;continue a}}var
l=l+1|0;continue}return l}}s(0);return 0}function
d$(a){var
d=[0,0,0,0];function
b(a,b,c){var
f=41!==c?1:0,g=f?db!==c?1:0:f;if(g){var
e=97===c?2:1;if(cc===c)d[3]=d[3]+1|0;if(a)d[2]=d[2]+e|0;else
d[1]=d[1]+e|0}return b+1|0}d_(a,b,function(a,b){return a+1|0});return d[1]}function
ea(a,b,c){var
h=a.safeGet(c);if((h+aU|0)<0||9<(h+aU|0))return i(b,0,c);var
e=h+aU|0,d=c+1|0;for(;;){var
f=a.safeGet(d);if(48<=f){if(!(58<=f)){var
e=(10*e|0)+(f+aU|0)|0,d=d+1|0;continue}var
g=0}else
if(36===f)if(0===e){var
j=T(hq),g=1}else{var
j=i(b,[0,cw(e-1|0)],d+1|0),g=1}else
var
g=0;if(!g)var
j=i(b,0,c);return j}}function
P(a,b){return a?b:d5(b)}function
eb(a,b){return a?a[1]:b}function
ec(aI,b,c,d,e,f,g){var
D=j(b,g);function
af(a){return i(d,D,a)}function
aJ(a,b,m,aK){var
k=m.getLen();function
E(l,b){var
p=b;for(;;){if(k<=p)return j(a,D);var
d=m.safeGet(p);if(37===d){var
o=function(a,b){return s(aK,eb(a,b))},at=function(g,f,c,d){var
a=d;for(;;){var
aa=m.safeGet(a)+fC|0;if(!(aa<0||25<aa))switch(aa){case
1:case
2:case
4:case
5:case
6:case
7:case
8:case
9:case
12:case
15:break;case
10:return ea(m,function(a,b){var
d=[0,o(a,f),c];return at(g,P(a,f),d,b)},a+1|0);default:var
a=a+1|0;continue}var
q=m.safeGet(a);if(124<=q)var
k=0;else
switch(q){case
78:case
88:case
aS:case
aC:case
dh:case
dv:case
dw:var
a8=o(g,f),a9=bU(d8(q,m,p,a,c),a8),l=r(P(g,f),a9,a+1|0),k=1;break;case
69:case
71:case
ge:case
di:case
dA:var
aY=o(g,f),aZ=c2(ax(m,p,a,c),aY),l=r(P(g,f),aZ,a+1|0),k=1;break;case
76:case
fL:case
bq:var
ad=m.safeGet(a+1|0)+gi|0;if(ad<0||32<ad)var
ag=1;else
switch(ad){case
0:case
12:case
17:case
23:case
29:case
32:var
U=a+1|0,ae=q-108|0;if(ae<0||2<ae)var
ah=0;else{switch(ae){case
1:var
ah=0,ai=0;break;case
2:var
a7=o(g,f),aA=bU(ax(m,p,U,c),a7),ai=1;break;default:var
a6=o(g,f),aA=bU(ax(m,p,U,c),a6),ai=1}if(ai){var
az=aA,ah=1}}if(!ah){var
a5=o(g,f),az=tO(ax(m,p,U,c),a5)}var
l=r(P(g,f),az,U+1|0),k=1,ag=0;break;default:var
ag=1}if(ag){var
a0=o(g,f),a4=bU(d8(bq,m,p,a,c),a0),l=r(P(g,f),a4,a+1|0),k=1}break;case
37:case
64:var
l=r(f,an(1,q),a+1|0),k=1;break;case
83:case
bt:var
y=o(g,f);if(bt===q)var
z=y;else{var
b=[0,0],am=y.getLen()-1|0,aL=0;if(!(am<0)){var
M=aL;for(;;){var
x=y.safeGet(M),bd=14<=x?34===x?1:92===x?1:0:11<=x?13<=x?1:0:8<=x?1:0,aO=bd?2:c6(x)?1:4;b[1]=b[1]+aO|0;var
aP=M+1|0;if(am!==M){var
M=aP;continue}break}}if(b[1]===y.getLen())var
aD=y;else{var
n=B(b[1]);b[1]=0;var
ao=y.getLen()-1|0,aM=0;if(!(ao<0)){var
L=aM;for(;;){var
w=y.safeGet(L),A=w-34|0;if(A<0||58<A)if(-20<=A)var
V=1;else{switch(A+34|0){case
8:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],98);var
K=1;break;case
9:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],da);var
K=1;break;case
10:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],bq);var
K=1;break;case
13:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],cc);var
K=1;break;default:var
V=1,K=0}if(K)var
V=0}else
var
V=(A-1|0)<0||56<(A-1|0)?(n.safeSet(b[1],92),b[1]++,n.safeSet(b[1],w),0):1;if(V)if(c6(w))n.safeSet(b[1],w);else{n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],48+(w/aS|0)|0);b[1]++;n.safeSet(b[1],48+((w/10|0)%10|0)|0);b[1]++;n.safeSet(b[1],48+(w%10|0)|0)}b[1]++;var
aN=L+1|0;if(ao!==L){var
L=aN;continue}break}}var
aD=n}var
z=h(hB,h(aD,hA))}if(a===(p+1|0))var
aB=z;else{var
J=ax(m,p,a,c);try{var
W=0,t=1;for(;;){if(J.getLen()<=t)var
ap=[0,0,W];else{var
X=J.safeGet(t);if(49<=X)if(58<=X)var
aj=0;else{var
ap=[0,tY(u(J,t,(J.getLen()-t|0)-1|0)),W],aj=1}else{if(45===X){var
W=1,t=t+1|0;continue}var
aj=0}if(!aj){var
t=t+1|0;continue}}var
Z=ap;break}}catch(f){if(f[1]!==aX)throw f;var
Z=d7(J,0,bt)}var
N=Z[1],C=z.getLen(),aQ=Z[2],O=0,aR=32;if(N===C)if(0===O){var
_=z,ak=1}else
var
ak=0;else
var
ak=0;if(!ak)if(N<=C)var
_=u(z,O,C);else{var
Y=an(N,aR);if(aQ)bB(z,O,Y,0,C);else
bB(z,O,Y,N-C|0,C);var
_=Y}var
aB=_}var
l=r(P(g,f),aB,a+1|0),k=1;break;case
67:case
99:var
s=o(g,f);if(99===q)var
aw=an(1,s);else{if(39===s)var
v=gT;else
if(92===s)var
v=gU;else{if(14<=s)var
F=0;else
switch(s){case
8:var
v=gV,F=1;break;case
9:var
v=gW,F=1;break;case
10:var
v=gX,F=1;break;case
13:var
v=gY,F=1;break;default:var
F=0}if(!F)if(c6(s)){var
al=B(1);al.safeSet(0,s);var
v=al}else{var
G=B(4);G.safeSet(0,92);G.safeSet(1,48+(s/aS|0)|0);G.safeSet(2,48+((s/10|0)%10|0)|0);G.safeSet(3,48+(s%10|0)|0);var
v=G}}var
aw=h(hy,h(v,hx))}var
l=r(P(g,f),aw,a+1|0),k=1;break;case
66:case
98:var
aV=a+1|0,aW=o(g,f)?gE:gF,l=r(P(g,f),aW,aV),k=1;break;case
40:case
dy:var
T=o(g,f),au=i(d9(q),m,a+1|0);if(dy===q){var
Q=a1(T.getLen()),aq=function(a,b){I(Q,b);return a+1|0};d_(T,function(a,b,c){if(a)bD(Q,hp);else
I(Q,37);return aq(b,c)},aq);var
aT=a2(Q),l=r(P(g,f),aT,au),k=1}else{var
av=P(g,f),bc=d4(d$(T),av),l=aJ(function(a){return E(bc,au)},av,T,aK),k=1}break;case
33:j(e,D);var
l=E(f,a+1|0),k=1;break;case
41:var
l=r(f,hv,a+1|0),k=1;break;case
44:var
l=r(f,hw,a+1|0),k=1;break;case
70:var
ab=o(g,f);if(0===c)var
ay=hz;else{var
$=ax(m,p,a,c);if(70===q)$.safeSet($.getLen()-1|0,dA);var
ay=$}var
as=tB(ab);if(3===as)var
ac=ab<0?hs:ht;else
if(4<=as)var
ac=hu;else{var
S=c2(ay,ab),R=0,aU=S.getLen();for(;;){if(aU<=R)var
ar=h(S,hr);else{var
H=S.safeGet(R)-46|0,be=H<0||23<H?55===H?1:0:(H-1|0)<0||21<(H-1|0)?1:0;if(!be){var
R=R+1|0;continue}var
ar=S}var
ac=ar;break}}var
l=r(P(g,f),ac,a+1|0),k=1;break;case
91:var
l=a3(m,a,q),k=1;break;case
97:var
aE=o(g,f),aF=d5(eb(g,f)),aG=o(0,aF),a_=a+1|0,a$=P(g,aF);if(aI)af(i(aE,0,aG));else
i(aE,D,aG);var
l=E(a$,a_),k=1;break;case
cc:var
l=a3(m,a,q),k=1;break;case
da:var
aH=o(g,f),ba=a+1|0,bb=P(g,f);if(aI)af(j(aH,0));else
j(aH,D);var
l=E(bb,ba),k=1;break;default:var
k=0}if(!k)var
l=a3(m,a,q);return l}},f=p+1|0,g=0;return ea(m,function(a,b){return at(a,l,g,b)},f)}i(c,D,d);var
p=p+1|0;continue}}function
r(a,b,c){af(b);return E(a,c)}return E(b,0)}var
o=cw(0);function
k(a,b){return aJ(f,o,a,b)}var
l=d$(g);if(l<0||6<l){var
n=function(f,b){if(l<=f){var
h=w(l,0),i=function(a,b){return m(h,(l-a|0)-1|0,b)},c=0,a=b;for(;;){if(a){var
d=a[2],e=a[1];if(d){i(c,e);var
c=c+1|0,a=d;continue}i(c,e)}return k(g,h)}}return function(a){return n(f+1|0,[0,a,b])}},a=n(0,0)}else
switch(l){case
1:var
a=function(a){var
b=w(1,0);m(b,0,a);return k(g,b)};break;case
2:var
a=function(a,b){var
c=w(2,0);m(c,0,a);m(c,1,b);return k(g,c)};break;case
3:var
a=function(a,b,c){var
d=w(3,0);m(d,0,a);m(d,1,b);m(d,2,c);return k(g,d)};break;case
4:var
a=function(a,b,c,d){var
e=w(4,0);m(e,0,a);m(e,1,b);m(e,2,c);m(e,3,d);return k(g,e)};break;case
5:var
a=function(a,b,c,d,e){var
f=w(5,0);m(f,0,a);m(f,1,b);m(f,2,c);m(f,3,d);m(f,4,e);return k(g,f)};break;case
6:var
a=function(a,b,c,d,e,f){var
h=w(6,0);m(h,0,a);m(h,1,b);m(h,2,c);m(h,3,d);m(h,4,e);m(h,5,f);return k(g,h)};break;default:var
a=k(g,[0])}return a}function
ed(a){function
b(a){return 0}return ec(0,function(a){return dO},gM,dP,dT,b,a)}function
hC(a){return a1(2*a.getLen()|0)}function
ee(c){function
b(a){var
b=a2(a);a[2]=0;return j(c,b)}function
d(a){return 0}var
e=1;return function(a){return ec(e,hC,I,bD,d,b,a)}}function
ef(a){return j(ee(function(a){return a}),a)}var
eg=[0,0];function
eh(a){eg[1]=[0,a,eg[1]];return 0}function
ei(a,b){var
j=0===b.length-1?[0,0]:b,f=j.length-1,p=0,q=54;if(!(54<0)){var
d=p;for(;;){m(a[1],d,d);var
w=d+1|0;if(q!==d){var
d=w;continue}break}}var
g=[0,hD],l=0,r=55,t=tK(55,f)?r:f,n=54+t|0;if(!(n<l)){var
c=l;for(;;){var
o=c%55|0,u=g[1],i=h(u,k(s(j,aQ(c,f))));g[1]=t3(i,0,i.getLen());var
e=g[1];m(a[1],o,(s(a[1],o)^(((e.safeGet(0)+(e.safeGet(1)<<8)|0)+(e.safeGet(2)<<16)|0)+(e.safeGet(3)<<24)|0))&bp);var
v=c+1|0;if(n!==c){var
c=v;continue}break}}a[2]=0;return 0}32===aE;var
hF=[0,hE.slice(),0];try{var
tr=bV(tq),cx=tr}catch(f){if(f[1]!==t)throw f;try{var
tp=bV(to),ej=tp}catch(f){if(f[1]!==t)throw f;var
ej=hG}var
cx=ej}var
d1=cx.getLen(),hH=82,d2=0;if(0<=0)if(d1<d2)var
bW=0;else
try{var
bC=d2;for(;;){if(d1<=bC)throw[0,t];if(cx.safeGet(bC)!==hH){var
bC=bC+1|0;continue}var
g2=1,cs=g2,bW=1;break}}catch(f){if(f[1]!==t)throw f;var
cs=0,bW=1}else
var
bW=0;if(!bW)var
cs=G(g1);var
ao=[fS,function(a){var
b=[0,w(55,0),0];ei(b,fa(0));return b}];function
ek(a,b){var
l=a?a[1]:cs,d=16;for(;;){if(!(b<=d))if(!(cu<(d*2|0))){var
d=d*2|0;continue}if(l){var
h=ug(ao);if(aT===h)var
c=ao[1];else
if(fS===h){var
k=ao[0+1];ao[0+1]=he;try{var
e=j(k,0);ao[0+1]=e;uf(ao,g5)}catch(f){ao[0+1]=function(a){throw f};throw f}var
c=e}else
var
c=ao;c[2]=(c[2]+1|0)%55|0;var
f=s(c[1],c[2]),g=(s(c[1],(c[2]+24|0)%55|0)+(f^f>>>25&31)|0)&bp;m(c[1],c[2],g);var
i=g}else
var
i=0;return[0,0,w(d,0),i,d]}}function
cy(a,b){return 3<=a.length-1?tL(10,aS,a[3],b)&(a[2].length-1-1|0):aQ(tM(10,aS,b),a[2].length-1)}function
bF(a,b){var
i=cy(a,b),d=s(a[2],i);if(d){var
e=d[3],j=d[2];if(0===aP(b,d[1]))return j;if(e){var
f=e[3],k=e[2];if(0===aP(b,e[1]))return k;if(f){var
l=f[3],m=f[2];if(0===aP(b,f[1]))return m;var
c=l;for(;;){if(c){var
g=c[3],h=c[2];if(0===aP(b,c[1]))return h;var
c=g;continue}throw[0,t]}}throw[0,t]}throw[0,t]}throw[0,t]}function
l(a,b){return c5(a,b[0+1])}var
cz=[0,0];c5(hI,cz);var
hJ=2;function
hK(a){var
b=[0,0],d=a.getLen()-1|0,e=0;if(!(d<0)){var
c=e;for(;;){b[1]=(223*b[1]|0)+a.safeGet(c)|0;var
g=c+1|0;if(d!==c){var
c=g;continue}break}}b[1]=b[1]&((1<<31)-1|0);var
f=bp<b[1]?b[1]-(1<<31)|0:b[1];return f}var
ah=cv([0,function(a,b){return fb(a,b)}]),ay=cv([0,function(a,b){return fb(a,b)}]),ap=cv([0,function(a,b){return gv(a,b)}]),el=fc(0,0),hL=[0,0];function
em(a){return 2<a?em((a+1|0)/2|0)*2|0:a}function
en(a){hL[1]++;var
c=a.length-1,d=w((c*2|0)+2|0,el);m(d,0,c);m(d,1,(x(em(c),aE)/8|0)-1|0);var
e=c-1|0,f=0;if(!(e<0)){var
b=f;for(;;){m(d,(b*2|0)+3|0,s(a,b));var
g=b+1|0;if(e!==b){var
b=g;continue}break}}return[0,hJ,d,ay[1],ap[1],0,0,ah[1],0]}function
cA(a,b){var
c=a[2].length-1,g=c<b?1:0;if(g){var
d=w(b,el),h=a[2],e=0,f=0,j=0<=c?0<=f?(h.length-1-c|0)<f?0:0<=e?(d.length-1-c|0)<e?0:(tu(h,f,d,e,c),1):0:0:0;if(!j)G(gN);a[2]=d;var
i=0}else
var
i=g;return i}var
eo=[0,0],hM=[0,0];function
cB(a){var
b=a[2].length-1;cA(a,b+1|0);return b}function
a4(a,b){try{var
d=i(ay[22],b,a[3])}catch(f){if(f[1]===t){var
c=cB(a);a[3]=p(ay[4],b,c,a[3]);a[4]=p(ap[4],c,1,a[4]);return c}throw f}return d}function
cD(a){return a===0?0:aZ(a)}function
eu(a,b){try{var
d=i(ah[22],b,a[7])}catch(f){if(f[1]===t){var
c=a[1];a[1]=c+1|0;if(y(b,h2))a[7]=p(ah[4],b,c,a[7]);return c}throw f}return d}function
cF(a){return tD(a,0)?[0]:a}function
ew(a,b){if(a)return a;var
c=fc(g4,b[1]);c[0+1]=b[2];var
d=cz[1];c[1+1]=d;cz[1]=d+1|0;return c}function
bG(a){var
b=cB(a);if(0===(b%2|0))var
d=0;else
if((2+at(s(a[2],1)*16|0,aE)|0)<b)var
d=0;else{var
c=cB(a),d=1}if(!d)var
c=b;m(a[2],c,0);return c}function
ex(a,ao){var
g=[0,0],aq=ao.length-1;for(;;){if(g[1]<aq){var
k=s(ao,g[1]),e=function(a){g[1]++;return s(ao,g[1])},l=e(0);if(typeof
l===n)switch(l){case
1:var
p=e(0),f=function(p){return function(a){return a[p+1]}}(p);break;case
2:var
q=e(0),b=e(0),f=function(q,b){return function(a){return a[q+1][b+1]}}(q,b);break;case
3:var
r=e(0),f=function(r){return function(a){return j(a[1][r+1],a)}}(r);break;case
4:var
t=e(0),f=function(t){return function(a,b){a[t+1]=b;return 0}}(t);break;case
5:var
u=e(0),v=e(0),f=function(u,v){return function(a){return j(u,v)}}(u,v);break;case
6:var
w=e(0),x=e(0),f=function(w,x){return function(a){return j(w,a[x+1])}}(w,x);break;case
7:var
y=e(0),z=e(0),c=e(0),f=function(y,z,c){return function(a){return j(y,a[z+1][c+1])}}(y,z,c);break;case
8:var
A=e(0),B=e(0),f=function(A,B){return function(a){return j(A,j(a[1][B+1],a))}}(A,B);break;case
9:var
C=e(0),D=e(0),E=e(0),f=function(C,D,E){return function(a){return i(C,D,E)}}(C,D,E);break;case
10:var
F=e(0),G=e(0),H=e(0),f=function(F,G,H){return function(a){return i(F,G,a[H+1])}}(F,G,H);break;case
11:var
I=e(0),J=e(0),K=e(0),d=e(0),f=function(I,J,K,d){return function(a){return i(I,J,a[K+1][d+1])}}(I,J,K,d);break;case
12:var
L=e(0),M=e(0),N=e(0),f=function(L,M,N){return function(a){return i(L,M,j(a[1][N+1],a))}}(L,M,N);break;case
13:var
O=e(0),P=e(0),Q=e(0),f=function(O,P,Q){return function(a){return i(O,a[P+1],Q)}}(O,P,Q);break;case
14:var
R=e(0),S=e(0),T=e(0),U=e(0),f=function(R,S,T,U){return function(a){return i(R,a[S+1][T+1],U)}}(R,S,T,U);break;case
15:var
V=e(0),W=e(0),X=e(0),f=function(V,W,X){return function(a){return i(V,j(a[1][W+1],a),X)}}(V,W,X);break;case
16:var
Y=e(0),Z=e(0),f=function(Y,Z){return function(a){return i(a[1][Y+1],a,Z)}}(Y,Z);break;case
17:var
_=e(0),$=e(0),f=function(_,$){return function(a){return i(a[1][_+1],a,a[$+1])}}(_,$);break;case
18:var
aa=e(0),ab=e(0),ac=e(0),f=function(aa,ab,ac){return function(a){return i(a[1][aa+1],a,a[ab+1][ac+1])}}(aa,ab,ac);break;case
19:var
ad=e(0),ae=e(0),f=function(ad,ae){return function(a){var
b=j(a[1][ae+1],a);return i(a[1][ad+1],a,b)}}(ad,ae);break;case
20:var
ag=e(0),h=e(0);bG(a);var
f=function(ag,h){return function(a){return j(af(h,ag,0),h)}}(ag,h);break;case
21:var
ah=e(0),ai=e(0);bG(a);var
f=function(ah,ai){return function(a){var
b=a[ai+1];return j(af(b,ah,0),b)}}(ah,ai);break;case
22:var
aj=e(0),ak=e(0),al=e(0);bG(a);var
f=function(aj,ak,al){return function(a){var
b=a[ak+1][al+1];return j(af(b,aj,0),b)}}(aj,ak,al);break;case
23:var
am=e(0),an=e(0);bG(a);var
f=function(am,an){return function(a){var
b=j(a[1][an+1],a);return j(af(b,am,0),b)}}(am,an);break;default:var
o=e(0),f=function(o){return function(a){return o}}(o)}else
var
f=l;hM[1]++;if(i(ap[22],k,a[4])){cA(a,k+1|0);m(a[2],k,f)}else
a[6]=[0,[0,k,f],a[6]];g[1]++;continue}return 0}}function
cG(a,b,c){if(bX(c,ia))return b;var
d=c.getLen()-1|0;for(;;){if(0<=d){if(i(a,c,d)){var
d=d-1|0;continue}var
f=d+1|0,e=d;for(;;){if(0<=e){if(!i(a,c,e)){var
e=e-1|0;continue}var
g=u(c,e+1|0,(f-e|0)-1|0)}else
var
g=u(c,0,f);var
h=g;break}}else
var
h=u(c,0,1);return h}}function
cH(a,b,c){if(bX(c,ib))return b;var
d=c.getLen()-1|0;for(;;){if(0<=d){if(i(a,c,d)){var
d=d-1|0;continue}var
e=d;for(;;){if(0<=e){if(!i(a,c,e)){var
e=e-1|0;continue}var
f=e;for(;;){if(0<=f){if(i(a,c,f)){var
f=f-1|0;continue}var
h=u(c,0,f+1|0)}else
var
h=u(c,0,1);var
g=h;break}}else
var
g=b;var
j=g;break}}else
var
j=u(c,0,1);return j}}function
cJ(a,b){return 47===a.safeGet(b)?1:0}function
ey(a){var
b=a.getLen()<1?1:0,c=b||(47!==a.safeGet(0)?1:0);return c}function
ie(a){var
c=ey(a);if(c){var
e=a.getLen()<2?1:0,d=e||y(u(a,0,2),ih);if(d){var
f=a.getLen()<3?1:0,b=f||y(u(a,0,3),ig)}else
var
b=d}else
var
b=c;return b}function
ii(a,b){var
c=b.getLen()<=a.getLen()?1:0,d=c?bX(u(a,a.getLen()-b.getLen()|0,b.getLen()),b):c;return d}try{var
tn=bV(tm),cK=tn}catch(f){if(f[1]!==t)throw f;var
cK=ij}function
ez(a){var
d=a.getLen(),b=a1(d+20|0);I(b,39);var
e=d-1|0,f=0;if(!(e<0)){var
c=f;for(;;){if(39===a.safeGet(c))bD(b,ik);else
I(b,a.safeGet(c));var
g=c+1|0;if(e!==c){var
c=g;continue}break}}I(b,39);return a2(b)}function
il(a){return cG(cJ,cI,a)}function
im(a){return cH(cJ,cI,a)}function
aG(a,b){var
c=a.safeGet(b),d=47===c?1:0;if(d)var
e=d;else{var
f=92===c?1:0,e=f||(58===c?1:0)}return e}function
cM(a){var
e=a.getLen()<1?1:0,c=e||(47!==a.safeGet(0)?1:0);if(c){var
f=a.getLen()<1?1:0,d=f||(92!==a.safeGet(0)?1:0);if(d){var
g=a.getLen()<2?1:0,b=g||(58!==a.safeGet(1)?1:0)}else
var
b=d}else
var
b=c;return b}function
eA(a){var
c=cM(a);if(c){var
g=a.getLen()<2?1:0,d=g||y(u(a,0,2),it);if(d){var
h=a.getLen()<2?1:0,e=h||y(u(a,0,2),is);if(e){var
i=a.getLen()<3?1:0,f=i||y(u(a,0,3),ir);if(f){var
j=a.getLen()<3?1:0,b=j||y(u(a,0,3),iq)}else
var
b=f}else
var
b=e}else
var
b=d}else
var
b=c;return b}function
eB(a,b){var
c=b.getLen()<=a.getLen()?1:0;if(c){var
e=u(a,a.getLen()-b.getLen()|0,b.getLen()),f=d0(b),d=bX(d0(e),f)}else
var
d=c;return d}try{var
tl=bV(tk),eC=tl}catch(f){if(f[1]!==t)throw f;var
eC=iu}function
iv(h){var
i=h.getLen(),e=a1(i+20|0);I(e,34);function
g(a,b){var
c=b;for(;;){if(c===i)return I(e,34);var
f=h.safeGet(c);if(34===f)return a<50?d(1+a,0,c):E(d,[0,0,c]);if(92===f)return a<50?d(1+a,0,c):E(d,[0,0,c]);I(e,f);var
c=c+1|0;continue}}function
d(a,b,c){var
f=b,d=c;for(;;){if(d===i){I(e,34);return a<50?j(1+a,f):E(j,[0,f])}var
l=h.safeGet(d);if(34===l){k((2*f|0)+1|0);I(e,34);return a<50?g(1+a,d+1|0):E(g,[0,d+1|0])}if(92===l){var
f=f+1|0,d=d+1|0;continue}k(f);return a<50?g(1+a,d):E(g,[0,d])}}function
j(a,b){var
d=1;if(!(b<1)){var
c=d;for(;;){I(e,92);var
f=c+1|0;if(b!==c){var
c=f;continue}break}}return 0}function
a(b){return _(g(0,b))}function
b(b,c){return _(d(0,b,c))}function
k(b){return _(j(0,b))}a(0);return a2(e)}function
eD(a){var
c=2<=a.getLen()?1:0;if(c){var
b=a.safeGet(0),g=91<=b?(b+fH|0)<0||25<(b+fH|0)?0:1:65<=b?1:0,d=g?1:0,e=d?58===a.safeGet(1)?1:0:d}else
var
e=c;if(e){var
f=u(a,2,a.getLen()-2|0);return[0,u(a,0,2),f]}return[0,iw,a]}function
ix(a){var
b=eD(a),c=b[1];return h(c,cH(aG,cL,b[2]))}function
iy(a){return cG(aG,cL,eD(a)[2])}function
iB(a){return cG(aG,cN,a)}function
iC(a){return cH(aG,cN,a)}if(y(ct,iD))if(y(ct,iE)){if(y(ct,iF))throw[0,H,iG];var
bH=[0,cL,io,ip,aG,cM,eA,eB,eC,iv,iy,ix]}else
var
bH=[0,cI,ic,id,cJ,ey,ie,ii,cK,ez,il,im];else
var
bH=[0,cN,iz,iA,aG,cM,eA,eB,cK,ez,iB,iC];var
eE=[0,iJ],iH=bH[11],iI=bH[3];l(iM,[0,eE,0,iL,iK]);eh(function(a){if(a[1]===eE){var
c=a[2],d=a[4],e=a[3];if(typeof
c===n)switch(c){case
1:var
b=iP;break;case
2:var
b=iQ;break;case
3:var
b=iR;break;case
4:var
b=iS;break;case
5:var
b=iT;break;case
6:var
b=iU;break;case
7:var
b=iV;break;case
8:var
b=iW;break;case
9:var
b=iX;break;case
10:var
b=iY;break;case
11:var
b=iZ;break;case
12:var
b=i0;break;case
13:var
b=i1;break;case
14:var
b=i2;break;case
15:var
b=i3;break;case
16:var
b=i4;break;case
17:var
b=i5;break;case
18:var
b=i6;break;case
19:var
b=i7;break;case
20:var
b=i8;break;case
21:var
b=i9;break;case
22:var
b=i_;break;case
23:var
b=i$;break;case
24:var
b=ja;break;case
25:var
b=jb;break;case
26:var
b=jc;break;case
27:var
b=jd;break;case
28:var
b=je;break;case
29:var
b=jf;break;case
30:var
b=jg;break;case
31:var
b=jh;break;case
32:var
b=ji;break;case
33:var
b=jj;break;case
34:var
b=jk;break;case
35:var
b=jl;break;case
36:var
b=jm;break;case
37:var
b=jn;break;case
38:var
b=jo;break;case
39:var
b=jp;break;case
40:var
b=jq;break;case
41:var
b=jr;break;case
42:var
b=js;break;case
43:var
b=jt;break;case
44:var
b=ju;break;case
45:var
b=jv;break;case
46:var
b=jw;break;case
47:var
b=jx;break;case
48:var
b=jy;break;case
49:var
b=jz;break;case
50:var
b=jA;break;case
51:var
b=jB;break;case
52:var
b=jC;break;case
53:var
b=jD;break;case
54:var
b=jE;break;case
55:var
b=jF;break;case
56:var
b=jG;break;case
57:var
b=jH;break;case
58:var
b=jI;break;case
59:var
b=jJ;break;case
60:var
b=jK;break;case
61:var
b=jL;break;case
62:var
b=jM;break;case
63:var
b=jN;break;case
64:var
b=jO;break;case
65:var
b=jP;break;case
66:var
b=jQ;break;case
67:var
b=jR;break;default:var
b=iN}else{var
f=c[1],b=j(ef(jS),f)}return[0,p(ef(iO),b,e,d)]}return 0});bY(jT);bY(jU);try{bY(tj)}catch(f){if(f[1]!==aX)throw f}try{bY(ti)}catch(f){if(f[1]!==aX)throw f}ek(0,7);function
eF(a){return vo(a)}an(32,r);var
jV=6,jW=0,j1=B(f5),j2=0,j3=r;if(!(r<0)){var
bh=j2;for(;;){j1.safeSet(bh,dZ(cr(bh)));var
th=bh+1|0;if(j3!==bh){var
bh=th;continue}break}}var
cO=an(32,0);cO.safeSet(10>>>3,cr(cO.safeGet(10>>>3)|1<<(10&7)));var
jX=B(32),jY=0,jZ=31;if(!(31<0)){var
a8=jY;for(;;){jX.safeSet(a8,cr(cO.safeGet(a8)^r));var
j0=a8+1|0;if(jZ!==a8){var
a8=j0;continue}break}}var
aH=[0,0],aI=[0,0],eG=[0,0];function
J(a){return aH[1]}function
eH(a){return aI[1]}function
Q(a,b,c){return 0===a[2][0]?b?uQ(a[1],a,b[1]):uR(a[1],a):b?fd(a[1],b[1]):fd(a[1],0)}var
cP=[0,0],j4=[3,jV];function
aJ(e,b,c){cP[1]++;switch(e[0]){case
7:case
8:throw[0,H,j5];case
6:var
g=e[1],m=cP[1],n=fe(0),o=w(eH(0)+1|0,n),p=ff(0),q=w(J(0)+1|0,p),f=[0,-1,[1,[0,uE(g,c),g]],q,o,c,0,e,0,0,m,0];break;default:var
h=e[1],i=cP[1],j=fe(0),k=w(eH(0)+1|0,j),l=ff(0),f=[0,-1,[0,ty(h,jW,[0,c])],w(J(0)+1|0,l),k,c,0,e,0,0,i,0]}if(b){var
d=b[1],a=function(a){{if(0===d[2][0])return 6===e[0]?gD(f,d[1][8],d[1]):gC(f,d[1][8],d[1]);{var
b=d[1],c=J(0);return fg(f,d[1][8]-c|0,b)}}};try{a(0)}catch(f){bj(0);a(0)}f[6]=[0,d]}return f}function
Y(a){return a[5]}function
a9(a){return a[6]}function
bI(a){return a[8]}function
bJ(a){return a[7]}function
ab(a){return a[2]}function
bK(a,b,c){a[1]=b;a[6]=c;return 0}function
cQ(a,b,c){return dB<=b?s(a[3],c):s(a[4],c)}function
cR(a,b){var
e=b[3].length-1-2|0,g=0;if(!(e<0)){var
d=g;for(;;){m(b[3],d,s(a[3],d));var
j=d+1|0;if(e!==d){var
d=j;continue}break}}var
f=b[4].length-1-2|0,h=0;if(!(f<0)){var
c=h;for(;;){m(b[4],c,s(a[4],c));var
i=c+1|0;if(f!==c){var
c=i;continue}break}}return 0}function
bL(a,b){b[8]=a[8];return 0}var
az=[0,j$];l(ki,[0,[0,j6]]);l(kj,[0,[0,j7]]);l(kk,[0,[0,j8]]);l(kl,[0,[0,j9]]);l(km,[0,[0,j_]]);l(kn,[0,az]);l(ko,[0,[0,ka]]);l(kp,[0,[0,kb]]);l(kq,[0,[0,kc]]);l(kr,[0,[0,ke]]);l(ks,[0,[0,kf]]);l(kt,[0,[0,kg]]);l(ku,[0,[0,kh]]);l(kv,[0,[0,kd]]);var
cS=[0,kD];l(kT,[0,[0,kw]]);l(kU,[0,[0,kx]]);l(kV,[0,[0,ky]]);l(kW,[0,[0,kz]]);l(kX,[0,[0,kA]]);l(kY,[0,[0,kB]]);l(kZ,[0,[0,kC]]);l(k0,[0,cS]);l(k1,[0,[0,kE]]);l(k2,[0,[0,kF]]);l(k3,[0,[0,kG]]);l(k4,[0,[0,kH]]);l(k5,[0,[0,kI]]);l(k6,[0,[0,kJ]]);l(k7,[0,[0,kK]]);l(k8,[0,[0,kL]]);l(k9,[0,[0,kM]]);l(k_,[0,[0,kN]]);l(k$,[0,[0,kO]]);l(la,[0,[0,kP]]);l(lb,[0,[0,kQ]]);l(lc,[0,[0,kR]]);l(ld,[0,[0,kS]]);var
bM=1,eI=0;function
a_(a,b,c){var
d=a[2];if(0===d[0])var
f=tA(d[1],b,c);else{var
e=d[1],f=p(e[2][4],e[1],b,c)}return f}function
a$(a,b){var
c=a[2];if(0===c[0])var
e=tz(c[1],b);else{var
d=c[1],e=i(d[2][3],d[1],b)}return e}function
eJ(a,b){Q(a,0,0);eO(b,0,0);return Q(a,0,0)}function
aK(a,b,c){var
f=a,d=b;for(;;){if(eI)return a_(f,d,c);var
m=d<0?1:0,o=m||(Y(f)<=d?1:0);if(o)throw[0,by,le];if(bM){var
i=a9(f);if(typeof
i!==n)eJ(i[1],f)}var
j=bI(f);if(j){var
e=j[1];if(1===e[1]){var
k=e[4],g=e[3],l=e[2];return 0===k?a_(e[5],l+d|0,c):a_(e[5],(l+x(at(d,g),k+g|0)|0)+aQ(d,g)|0,c)}var
h=e[3],f=e[5],d=(e[2]+x(at(d,h),e[4]+h|0)|0)+aQ(d,h)|0;continue}return a_(f,d,c)}}function
aL(a,b){var
e=a,c=b;for(;;){if(eI)return a$(e,c);var
l=c<0?1:0,m=l||(Y(e)<=c?1:0);if(m)throw[0,by,lf];if(bM){var
h=a9(e);if(typeof
h!==n)eJ(h[1],e)}var
i=bI(e);if(i){var
d=i[1];if(1===d[1]){var
j=d[4],f=d[3],k=d[2];return 0===j?a$(d[5],k+c|0):a$(d[5],(k+x(at(c,f),j+f|0)|0)+aQ(c,f)|0)}var
g=d[3],e=d[5],c=(d[2]+x(at(c,g),d[4]+g|0)|0)+aQ(c,g)|0;continue}return a$(e,c)}}function
eK(a){if(a[8]){var
b=aJ(a[7],0,a[5]);b[1]=a[1];b[6]=a[6];cR(a,b);var
c=b}else
var
c=a;return c}function
eL(d,b,c){{if(0===c[2][0]){var
a=function(a){return 0===ab(d)[0]?uH(d,c[1][8],c[1],c[3],b):uJ(d,c[1][8],c[1],b)};try{var
f=a(0)}catch(f){if(f[1]===az){try{Q(c,0,0);var
g=a(0)}catch(f){bj(0);return a(0)}return g}throw f}return f}var
e=function(a){{if(0===ab(d)[0]){var
e=c[1],f=J(0);return u9(d,c[1][8]-f|0,e,b)}var
g=c[1],h=J(0);return u$(d,c[1][8]-h|0,g,b)}};try{var
i=e(0)}catch(f){try{Q(c,0,0);var
h=e(0)}catch(f){bj(0);return e(0)}return h}return i}}function
eM(d,b,c){{if(0===c[2][0]){var
a=function(a){return 0===ab(d)[0]?uP(d,c[1][8],c[1],c,b):uK(d,c[1][8],c[1],b)};try{var
f=a(0)}catch(f){if(f[1]===az){try{Q(c,0,0);var
g=a(0)}catch(f){bj(0);return a(0)}return g}throw f}return f}var
e=function(a){{if(0===ab(d)[0]){var
e=c[2],f=c[1],g=J(0);return vd(d,c[1][8]-g|0,f,e,b)}var
h=c[2],i=c[1],j=J(0);return va(d,c[1][8]-j|0,i,h,b)}};try{var
i=e(0)}catch(f){try{Q(c,0,0);var
h=e(0)}catch(f){bj(0);return e(0)}return h}return i}}function
ba(a,b,c,d,e,f,g,h){{if(0===d[2][0])return 0===ab(a)[0]?uY(a,b,d[1][8],d[1],d[3],c,e,f,g,h):uM(a,b,d[1][8],d[1],d[3],c,e,f,g,h);{if(0===ab(a)[0]){var
i=d[3],j=d[1],k=J(0);return vm(a,b,d[1][8]-k|0,j,i,c,e,f,g,h)}var
l=d[3],m=d[1],n=J(0);return vb(a,b,d[1][8]-n|0,m,l,c,e,f,g,h)}}}function
bb(a,b,c,d,e,f,g,h){{if(0===d[2][0])return 0===ab(a)[0]?uZ(a,b,d[1][8],d[1],d[3],c,e,f,g,h):uN(a,b,d[1][8],d[1],d[3],c,e,f,g,h);{if(0===ab(a)[0]){var
i=d[3],j=d[1],k=J(0);return vn(a,b,d[1][8]-k|0,j,i,c,e,f,g,h)}var
l=d[3],m=d[1],n=J(0);return vc(a,b,d[1][8]-n|0,m,l,c,e,f,g,h)}}}function
eN(a,b,c){var
p=b;for(;;){var
d=p?p[1]:0,q=a9(a);if(typeof
q===n){bK(a,c[1][8],[1,c]);try{cT(a,c)}catch(f){if(f[1]!==az)f[1]===cS;try{Q(c,[0,d],0);cT(a,c)}catch(f){if(f[1]!==az)if(f[1]!==cS)throw f;Q(c,0,0);tH(0);cT(a,c)}}var
y=bI(a);if(y){var
j=y[1];if(1===j[1]){var
k=j[5],r=j[4],f=j[3],l=j[2];if(0===f)ba(k,a,d,c,0,0,l,Y(a));else
if(z<f){var
h=0,m=Y(a);for(;;){if(f<m){ba(k,a,d,c,x(h,f+r|0),x(h,f),l,f);var
h=h+1|0,m=m-f|0;continue}if(0<m)ba(k,a,d,c,x(h,f+r|0),x(h,f),l,m);break}}else{var
e=0,i=0,g=Y(a);for(;;){if(z<g){var
u=aJ(bJ(a),0,z);bL(a,u);var
A=e+gj|0;if(!(A<e)){var
s=e;for(;;){aK(u,s,aL(a,e));var
H=s+1|0;if(A!==s){var
s=H;continue}break}}ba(k,u,d,c,x(i,z+r|0),i*z|0,l,z);var
e=e+z|0,i=i+1|0,g=g+fX|0;continue}if(0<g){var
v=aJ(bJ(a),0,g),B=(e+g|0)-1|0;if(!(B<e)){var
t=e;for(;;){aK(v,t,aL(a,e));var
I=t+1|0;if(B!==t){var
t=I;continue}break}}bL(a,v);ba(k,v,d,c,x(i,z+r|0),i*z|0,l,g)}break}}}else{var
w=eK(a),C=Y(a)-1|0,J=0;if(!(C<0)){var
o=J;for(;;){a_(w,o,aL(a,o));var
K=o+1|0;if(C!==o){var
o=K;continue}break}}eL(w,d,c);cR(w,a)}}else
eL(a,d,c);return bK(a,c[1][8],[0,c])}else{if(0===q[0]){var
D=q[1],E=c7(D,c);if(E){eO(a,[0,d],0);Q(D,0,0);var
p=[0,d];continue}return E}var
F=q[1],G=c7(F,c);if(G){Q(F,0,0);var
p=[0,d];continue}return G}}}function
cT(a,b){{if(0===b[2][0])return 0===ab(a)[0]?gC(a,b[1][8],b[1]):gD(a,b[1][8],b[1]);{if(0===ab(a)[0]){var
c=b[1],d=J(0);return fg(a,b[1][8]-d|0,c)}var
e=b[1],f=J(0);return u_(a,b[1][8]-f|0,e)}}}function
eO(a,b,c){var
v=b;for(;;){var
f=v?v[1]:0,q=a9(a);if(typeof
q===n)return 0;else{if(0===q[0]){var
d=q[1];bK(a,d[1][8],[1,d]);var
A=bI(a);if(A){var
k=A[1];if(1===k[1]){var
l=k[5],r=k[4],e=k[3],m=k[2];if(0===e)bb(l,a,f,d,0,0,m,Y(a));else
if(z<e){var
i=0,o=Y(a);for(;;){if(e<o){bb(l,a,f,d,x(i,e+r|0),x(i,e),m,e);var
i=i+1|0,o=o-e|0;continue}if(0<o)bb(l,a,f,d,x(i,e+r|0),x(i,e),m,o);break}}else{var
j=0,h=Y(a),g=0;for(;;){if(z<h){var
w=aJ(bJ(a),0,z);bL(a,w);var
B=g+gj|0;if(!(B<g)){var
s=g;for(;;){aK(w,s,aL(a,g));var
E=s+1|0;if(B!==s){var
s=E;continue}break}}bb(l,w,f,d,x(j,z+r|0),j*z|0,m,z);var
j=j+1|0,h=h+fX|0;continue}if(0<h){var
y=aJ(bJ(a),0,h),C=(g+h|0)-1|0;if(!(C<g)){var
t=g;for(;;){aK(y,t,aL(a,g));var
F=t+1|0;if(C!==t){var
t=F;continue}break}}bL(a,y);bb(l,y,f,d,x(j,z+r|0),j*z|0,m,h)}break}}}else{var
u=eK(a);cR(u,a);eM(u,f,d);var
D=Y(u)-1|0,G=0;if(!(D<0)){var
p=G;for(;;){aK(a,p,a$(u,p));var
H=p+1|0;if(D!==p){var
p=H;continue}break}}}}else
eM(a,f,d);return bK(a,d[1][8],0)}Q(q[1],0,0);var
v=[0,f];continue}}}var
lk=[0,lj],lm=[0,ll];function
bN(a,b){var
p=s(g3,0),q=h(iI,h(a,b)),f=dQ(h(iH(p),q));try{var
n=eQ,g=eQ;a:for(;;){if(1){var
k=function(a,b,c){var
e=b,d=c;for(;;){if(d){var
g=d[1],f=g.getLen(),h=d[2];bi(g,0,a,e-f|0,f);var
e=e-f|0,d=h;continue}return a}},d=0,e=0;for(;;){var
c=t8(f);if(0===c){if(!d)throw[0,bz];var
j=k(B(e),e,d)}else{if(!(0<c)){var
m=B(-c|0);c8(f,m,0,-c|0);var
d=[0,m,d],e=e-c|0;continue}var
i=B(c-1|0);c8(f,i,0,c-1|0);t7(f);if(d){var
l=(e+c|0)-1|0,j=k(B(l),l,[0,i,d])}else
var
j=i}var
g=h(g,h(j,ln)),n=g;continue a}}var
o=g;break}}catch(f){if(f[1]!==bz)throw f;var
o=n}dS(f);return o}var
eR=[0,lo],cU=[],lp=0,lq=0;uA(cU,[0,0,function(f){var
k=eu(f,lr),e=cF(lg),d=e.length-1,n=eP.length-1,a=w(d+n|0,0),o=d-1|0,u=0;if(!(o<0)){var
c=u;for(;;){m(a,c,a4(f,s(e,c)));var
y=c+1|0;if(o!==c){var
c=y;continue}break}}var
q=n-1|0,v=0;if(!(q<0)){var
b=v;for(;;){m(a,b+d|0,eu(f,s(eP,b)));var
x=b+1|0;if(q!==b){var
b=x;continue}break}}var
r=a[10],l=a[12],h=a[15],i=a[16],j=a[17],g=a[18],z=a[1],A=a[2],B=a[3],C=a[4],D=a[5],E=a[7],F=a[8],G=a[9],H=a[11],I=a[14];function
J(a,b,c,d,e,f){var
h=d?d[1]:d;p(a[1][l+1],a,[0,h],f);var
i=bF(a[g+1],f);return fh(a[1][r+1],a,b,[0,c[1],c[2]],e,f,i)}function
K(a,b,c,d,e){try{var
f=bF(a[g+1],e),h=f}catch(f){if(f[1]!==t)throw f;try{p(a[1][l+1],a,ls,e)}catch(f){throw f}var
h=bF(a[g+1],e)}return fh(a[1][r+1],a,b,[0,c[1],c[2]],d,e,h)}function
L(a,b,c){var
y=b?b[1]:b;try{bF(a[g+1],c);var
f=0}catch(f){if(f[1]===t){if(0===c[2][0]){var
z=a[i+1];if(!z)throw[0,eR,c];var
A=z[1],H=y?uO(A,a[h+1],c[1]):uG(A,a[h+1],c[1]),B=H}else{var
D=a[j+1];if(!D)throw[0,eR,c];var
E=D[1],I=y?u0(E,a[h+1],c[1]):u8(E,a[h+1],c[1]),B=I}var
d=a[g+1],v=cy(d,c);m(d[2],v,[0,c,B,s(d[2],v)]);d[1]=d[1]+1|0;var
x=d[2].length-1<<1<d[1]?1:0;if(x){var
l=d[2],n=l.length-1,o=n*2|0,p=o<cu?1:0;if(p){var
k=w(o,0);d[2]=k;var
q=function(a){if(a){var
b=a[1],e=a[2];q(a[3]);var
c=cy(d,b);return m(k,c,[0,b,e,s(k,c)])}return 0},r=n-1|0,F=0;if(!(r<0)){var
e=F;for(;;){q(s(l,e));var
G=e+1|0;if(r!==e){var
e=G;continue}break}}var
u=0}else
var
u=p;var
C=u}else
var
C=x;return C}throw f}return f}function
M(a,b){try{var
f=[0,bN(a[k+1],lu),0],c=f}catch(f){var
c=0}a[i+1]=c;try{var
e=[0,bN(a[k+1],lt),0],d=e}catch(f){var
d=0}a[j+1]=d;return 0}function
N(a,b){a[j+1]=[0,b,0];return 0}function
O(a,b){return a[j+1]}function
P(a,b){a[i+1]=[0,b,0];return 0}function
Q(a,b){return a[i+1]}function
R(a,b){var
d=a[g+1];d[1]=0;var
e=d[2].length-1-1|0,f=0;if(!(e<0)){var
c=f;for(;;){m(d[2],c,0);var
h=c+1|0;if(e!==c){var
c=h;continue}break}}return 0}ex(f,[0,G,function(a,b){return a[g+1]},C,R,F,Q,A,P,E,O,z,N,D,M,l,L,B,K,H,J]);return function(a,b,c,d){var
e=ew(b,f);e[k+1]=c;e[I+1]=c;e[h+1]=d;try{var
o=[0,bN(c,lw),0],l=o}catch(f){var
l=0}e[i+1]=l;try{var
n=[0,bN(c,lv),0],m=n}catch(f){var
m=0}e[j+1]=m;e[g+1]=ek(0,8);return e}},lq,lp]);fi(0);fi(0);function
cV(a){function
e(a,b){var
d=a-1|0,e=0;if(!(d<0)){var
c=e;for(;;){ed(ly);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return j(ed(lx),b)}function
f(a,b){var
c=a,d=b;for(;;)if(typeof
d===n)return 0===d?e(c,lz):e(c,lA);else
switch(d[0]){case
0:e(c,lB);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
1:e(c,lC);var
c=c+1|0,d=d[1];continue;case
2:e(c,lD);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
3:e(c,lE);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
4:e(c,lF);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
5:e(c,lG);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
6:e(c,lH);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
7:e(c,lI);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
8:e(c,lJ);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
9:e(c,lK);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
10:e(c,lL);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
11:return e(c,h(lM,d[1]));case
12:return e(c,h(lN,d[1]));case
13:return e(c,h(lO,k(d[1])));case
14:return e(c,h(lP,k(d[1])));case
15:return e(c,h(lQ,k(d[1])));case
16:return e(c,h(lR,k(d[1])));case
17:return e(c,h(lS,k(d[1])));case
18:return e(c,lT);case
19:return e(c,lU);case
20:return e(c,lV);case
21:return e(c,lW);case
22:return e(c,lX);case
23:return e(c,h(lY,k(d[2])));case
24:e(c,lZ);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
25:e(c,l0);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
26:e(c,l1);var
c=c+1|0,d=d[1];continue;case
27:e(c,l2);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
28:e(c,l3);var
c=c+1|0,d=d[1];continue;case
29:e(c,l4);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
30:e(c,l5);f(c+1|0,d[1]);f(c+1|0,d[2]);var
c=c+1|0,d=d[3];continue;case
31:return e(c,l6);case
32:var
g=h(l7,k(d[2]));return e(c,h(l8,h(d[1],g)));case
33:return e(c,h(l9,k(d[1])));case
36:e(c,l$);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
37:e(c,ma);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
38:e(c,mb);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
39:e(c,mc);f(c+1|0,d[1]);f(c+1|0,d[2]);var
c=c+1|0,d=d[3];continue;case
40:e(c,md);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
41:e(c,me);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
42:e(c,mf);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
43:e(c,mg);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
44:e(c,mh);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
45:e(c,mi);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
46:e(c,mj);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
47:e(c,mk);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
48:e(c,ml);f(c+1|0,d[1]);f(c+1|0,d[2]);f(c+1|0,d[3]);var
c=c+1|0,d=d[4];continue;case
49:e(c,mm);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
50:e(c,mn);f(c+1|0,d[1]);var
i=d[2],j=c+1|0;return dU(function(a){return f(j,a)},i);case
51:return e(c,mo);case
52:return e(c,mp);default:return e(c,h(l_,O(d[1])))}}return f(0,a)}function
K(a){return an(a,32)}var
bc=[0,mq];function
bl(a,b,c){var
d=c;for(;;)if(typeof
d===n)return ms;else
switch(d[0]){case
18:case
19:var
S=h(m7,h(e(b,d[2]),m6));return h(m8,h(k(d[1]),S));case
27:case
38:var
ac=d[1],ad=h(nr,h(e(b,d[2]),nq));return h(e(b,ac),ad);case
0:var
g=d[2],B=e(b,d[1]);if(typeof
g===n)var
r=0;else
if(25===g[0]){var
t=e(b,g),r=1}else
var
r=0;if(!r){var
C=h(mt,K(b)),t=h(e(b,g),C)}return h(h(B,t),mu);case
1:var
D=h(e(b,d[1]),mv),F=y(bc[1][1],mw)?h(bc[1][1],mx):mz;return h(my,h(F,D));case
2:var
G=h(mB,h(U(b,d[2]),mA));return h(mC,h(U(b,d[1]),G));case
3:var
I=h(mE,h(aq(b,d[2]),mD));return h(mF,h(aq(b,d[1]),I));case
4:var
J=h(mH,h(U(b,d[2]),mG));return h(mI,h(U(b,d[1]),J));case
5:var
L=h(mK,h(aq(b,d[2]),mJ));return h(mL,h(aq(b,d[1]),L));case
6:var
M=h(mN,h(U(b,d[2]),mM));return h(mO,h(U(b,d[1]),M));case
7:var
N=h(mQ,h(aq(b,d[2]),mP));return h(mR,h(aq(b,d[1]),N));case
8:var
P=h(mT,h(U(b,d[2]),mS));return h(mU,h(U(b,d[1]),P));case
9:var
Q=h(mW,h(aq(b,d[2]),mV));return h(mX,h(aq(b,d[1]),Q));case
10:var
R=h(mZ,h(U(b,d[2]),mY));return h(m0,h(U(b,d[1]),R));case
13:return h(m1,k(d[1]));case
14:return h(m2,k(d[1]));case
15:throw[0,H,m3];case
16:return h(m4,k(d[1]));case
17:return h(m5,k(d[1]));case
20:var
V=h(m_,h(e(b,d[2]),m9));return h(m$,h(k(d[1]),V));case
21:var
W=h(nb,h(e(b,d[2]),na));return h(nc,h(k(d[1]),W));case
22:var
X=h(ne,h(e(b,d[2]),nd));return h(nf,h(k(d[1]),X));case
23:var
Y=h(ng,k(d[2])),u=d[1];if(typeof
u===n)var
f=0;else
switch(u[0]){case
33:var
o=ni,f=1;break;case
34:var
o=nj,f=1;break;case
35:var
o=nk,f=1;break;default:var
f=0}if(f)return h(o,Y);throw[0,H,nh];case
24:var
i=d[2],v=d[1];if(typeof
i===n){if(0===i){var
d=v;continue}}else
if(24===i[0]){var
Z=h(nm,e(b,i));return h(e(b,v),Z)}return T(nl);case
25:var
_=e(b,d[2]),$=h(nn,h(K(b),_));return h(e(b,d[1]),$);case
26:var
aa=e(b,d[1]),ab=y(bc[1][2],no)?bc[1][2]:np;return h(ab,aa);case
28:var
d=d[1];continue;case
29:var
ae=h(nt,h(e(b,d[2]),ns));return h(e(b,d[1]),ae);case
30:var
l=d[2],af=e(b,d[3]),ag=h(nu,h(K(b),af));if(typeof
l===n)var
s=0;else
if(31===l[0]){var
w=h(mr(l[1]),nw),s=1}else
var
s=0;if(!s)var
w=e(b,l);var
ah=h(nv,h(w,ag));return h(e(b,d[1]),ah);case
31:return a<50?bk(1+a,d[1]):E(bk,[0,d[1]]);case
33:return k(d[1]);case
34:return h(O(d[1]),nx);case
35:return O(d[1]);case
36:var
ai=h(nz,h(e(b,d[2]),ny));return h(e(b,d[1]),ai);case
37:var
aj=h(nB,h(K(b),nA)),ak=h(e(b,d[2]),aj),al=h(nC,h(K(b),ak)),am=h(nD,h(e(b,d[1]),al));return h(K(b),am);case
39:var
an=h(nE,K(b)),ao=h(e(b+2|0,d[3]),an),ap=h(nF,h(K(b+2|0),ao)),ar=h(nG,h(K(b),ap)),as=h(e(b+2|0,d[2]),ar),at=h(nH,h(K(b+2|0),as));return h(nI,h(e(b,d[1]),at));case
40:var
au=h(nJ,K(b)),av=h(e(b+2|0,d[2]),au),aw=h(nK,h(K(b+2|0),av));return h(nL,h(e(b,d[1]),aw));case
41:var
ax=h(nM,e(b,d[2]));return h(e(b,d[1]),ax);case
42:var
ay=h(nN,e(b,d[2]));return h(e(b,d[1]),ay);case
43:var
az=h(nO,e(b,d[2]));return h(e(b,d[1]),az);case
44:var
aA=h(nP,e(b,d[2]));return h(e(b,d[1]),aA);case
45:var
aB=h(nQ,e(b,d[2]));return h(e(b,d[1]),aB);case
46:var
aC=h(nR,e(b,d[2]));return h(e(b,d[1]),aC);case
47:var
aD=h(nS,e(b,d[2]));return h(e(b,d[1]),aD);case
48:var
p=e(b,d[1]),aE=e(b,d[2]),aF=e(b,d[3]),aG=h(e(b+2|0,d[4]),nT);return h(nZ,h(p,h(nY,h(aE,h(nX,h(p,h(nW,h(aF,h(nV,h(p,h(nU,h(K(b+2|0),aG))))))))))));case
49:var
aH=e(b,d[1]),aI=h(e(b+2|0,d[2]),n0);return h(n2,h(aH,h(n1,h(K(b+2|0),aI))));case
50:var
x=d[2],m=d[1],z=e(b,m),q=function(a){if(a){var
c=a[2],d=a[1];if(c){var
f=h(n3,q(c));return h(e(b,d),f)}return e(b,d)}throw[0,H,n4]};if(typeof
m!==n)if(31===m[0]){var
A=m[1];if(!y(A[1],n7))if(!y(A[2],n8))return h(z,h(n_,h(q(aZ(x)),n9)))}return h(z,h(n6,h(q(aZ(x)),n5)));case
51:return k(j(d[1],0));case
52:return h(O(j(d[1],0)),n$);default:return d[1]}}function
ts(a,b,c){if(typeof
c!==n)switch(c[0]){case
2:case
4:case
6:case
8:case
10:case
50:return a<50?bl(1+a,b,c):E(bl,[0,b,c]);case
32:return c[1];case
33:return k(c[1]);case
36:var
d=h(ob,h(U(b,c[2]),oa));return h(e(b,c[1]),d);case
51:return k(j(c[1],0));default:}return a<50?c9(1+a,b,c):E(c9,[0,b,c])}function
c9(a,b,c){if(typeof
c!==n)switch(c[0]){case
3:case
5:case
7:case
9:case
29:case
50:return a<50?bl(1+a,b,c):E(bl,[0,b,c]);case
16:return h(od,k(c[1]));case
31:return a<50?bk(1+a,c[1]):E(bk,[0,c[1]]);case
32:return c[1];case
34:return h(O(c[1]),oe);case
35:return h(of,O(c[1]));case
36:var
d=h(oh,h(U(b,c[2]),og));return h(e(b,c[1]),d);case
52:return h(O(j(c[1],0)),oi);default:}cV(c);return T(oc)}function
bk(a,b){return b[1]}function
e(b,c){return _(bl(0,b,c))}function
U(b,c){return _(ts(0,b,c))}function
aq(b,c){return _(c9(0,b,c))}function
mr(b){return _(bk(0,b))}function
A(a){return an(a,32)}var
bd=[0,oj];function
bn(a,b,c){var
d=c;for(;;)if(typeof
d===n)return ol;else
switch(d[0]){case
18:case
19:var
U=h(oI,h(f(b,d[2]),oH));return h(oJ,h(k(d[1]),U));case
27:case
38:var
ac=d[1],ad=h(o3,R(b,d[2]));return h(f(b,ac),ad);case
0:var
g=d[2],C=f(b,d[1]);if(typeof
g===n)var
r=0;else
if(25===g[0]){var
t=f(b,g),r=1}else
var
r=0;if(!r){var
D=h(om,A(b)),t=h(f(b,g),D)}return h(h(C,t),on);case
1:var
F=h(f(b,d[1]),oo),G=y(bd[1][1],op)?h(bd[1][1],oq):os;return h(or,h(G,F));case
2:var
I=h(ot,R(b,d[2]));return h(R(b,d[1]),I);case
3:var
J=h(ou,ar(b,d[2]));return h(ar(b,d[1]),J);case
4:var
K=h(ov,R(b,d[2]));return h(R(b,d[1]),K);case
5:var
L=h(ow,ar(b,d[2]));return h(ar(b,d[1]),L);case
6:var
M=h(ox,R(b,d[2]));return h(R(b,d[1]),M);case
7:var
N=h(oy,ar(b,d[2]));return h(ar(b,d[1]),N);case
8:var
P=h(oz,R(b,d[2]));return h(R(b,d[1]),P);case
9:var
Q=h(oA,ar(b,d[2]));return h(ar(b,d[1]),Q);case
10:var
S=h(oB,R(b,d[2]));return h(R(b,d[1]),S);case
13:return h(oC,k(d[1]));case
14:return h(oD,k(d[1]));case
15:throw[0,H,oE];case
16:return h(oF,k(d[1]));case
17:return h(oG,k(d[1]));case
20:var
V=h(oL,h(f(b,d[2]),oK));return h(oM,h(k(d[1]),V));case
21:var
W=h(oO,h(f(b,d[2]),oN));return h(oP,h(k(d[1]),W));case
22:var
X=h(oR,h(f(b,d[2]),oQ));return h(oS,h(k(d[1]),X));case
23:var
Y=h(oT,k(d[2])),u=d[1];if(typeof
u===n)var
e=0;else
switch(u[0]){case
33:var
o=oV,e=1;break;case
34:var
o=oW,e=1;break;case
35:var
o=oX,e=1;break;default:var
e=0}if(e)return h(o,Y);throw[0,H,oU];case
24:var
i=d[2],v=d[1];if(typeof
i===n){if(0===i){var
d=v;continue}}else
if(24===i[0]){var
Z=h(oZ,f(b,i));return h(f(b,v),Z)}return T(oY);case
25:var
_=f(b,d[2]),$=h(o0,h(A(b),_));return h(f(b,d[1]),$);case
26:var
aa=f(b,d[1]),ab=y(bd[1][2],o1)?bd[1][2]:o2;return h(ab,aa);case
28:var
d=d[1];continue;case
29:var
ae=h(o5,h(f(b,d[2]),o4));return h(f(b,d[1]),ae);case
30:var
l=d[2],af=f(b,d[3]),ag=h(o6,h(A(b),af));if(typeof
l===n)var
s=0;else
if(31===l[0]){var
w=ok(l[1]),s=1}else
var
s=0;if(!s)var
w=f(b,l);var
ah=h(o7,h(w,ag));return h(f(b,d[1]),ah);case
31:return a<50?bm(1+a,d[1]):E(bm,[0,d[1]]);case
33:return k(d[1]);case
34:return h(O(d[1]),o8);case
35:return O(d[1]);case
36:var
ai=h(o_,h(f(b,d[2]),o9));return h(f(b,d[1]),ai);case
37:var
aj=h(pa,h(A(b),o$)),ak=h(f(b,d[2]),aj),al=h(pb,h(A(b),ak)),am=h(pc,h(f(b,d[1]),al));return h(A(b),am);case
39:var
an=h(pd,A(b)),ao=h(f(b+2|0,d[3]),an),ap=h(pe,h(A(b+2|0),ao)),aq=h(pf,h(A(b),ap)),as=h(f(b+2|0,d[2]),aq),at=h(pg,h(A(b+2|0),as));return h(ph,h(f(b,d[1]),at));case
40:var
au=h(pi,A(b)),av=h(pj,h(A(b),au)),aw=h(f(b+2|0,d[2]),av),ax=h(pk,h(A(b+2|0),aw)),ay=h(pl,h(A(b),ax));return h(pm,h(f(b,d[1]),ay));case
41:var
az=h(pn,f(b,d[2]));return h(f(b,d[1]),az);case
42:var
aA=h(po,f(b,d[2]));return h(f(b,d[1]),aA);case
43:var
aB=h(pp,f(b,d[2]));return h(f(b,d[1]),aB);case
44:var
aC=h(pq,f(b,d[2]));return h(f(b,d[1]),aC);case
45:var
aD=h(pr,f(b,d[2]));return h(f(b,d[1]),aD);case
46:var
aE=h(ps,f(b,d[2]));return h(f(b,d[1]),aE);case
47:var
aF=h(pt,f(b,d[2]));return h(f(b,d[1]),aF);case
48:var
p=f(b,d[1]),aG=f(b,d[2]),aH=f(b,d[3]),aI=h(f(b+2|0,d[4]),pu);return h(pA,h(p,h(pz,h(aG,h(py,h(p,h(px,h(aH,h(pw,h(p,h(pv,h(A(b+2|0),aI))))))))))));case
49:var
aJ=f(b,d[1]),aK=h(f(b+2|0,d[2]),pB);return h(pD,h(aJ,h(pC,h(A(b+2|0),aK))));case
50:var
x=d[2],m=d[1],z=f(b,m),q=function(a){if(a){var
c=a[2],d=a[1];if(c){var
e=h(pE,q(c));return h(f(b,d),e)}return f(b,d)}throw[0,H,pF]};if(typeof
m!==n)if(31===m[0]){var
B=m[1];if(!y(B[1],pI))if(!y(B[2],pJ))return h(z,h(pL,h(q(aZ(x)),pK)))}return h(z,h(pH,h(q(aZ(x)),pG)));case
51:return k(j(d[1],0));case
52:return h(O(j(d[1],0)),pM);default:return d[1]}}function
tt(a,b,c){if(typeof
c!==n)switch(c[0]){case
2:case
4:case
6:case
8:case
10:case
50:return a<50?bn(1+a,b,c):E(bn,[0,b,c]);case
32:return c[1];case
33:return k(c[1]);case
36:var
d=h(pO,h(R(b,c[2]),pN));return h(f(b,c[1]),d);case
51:return k(j(c[1],0));default:}return a<50?c_(1+a,b,c):E(c_,[0,b,c])}function
c_(a,b,c){if(typeof
c!==n)switch(c[0]){case
3:case
5:case
7:case
9:case
50:return a<50?bn(1+a,b,c):E(bn,[0,b,c]);case
16:return h(pQ,k(c[1]));case
31:return a<50?bm(1+a,c[1]):E(bm,[0,c[1]]);case
32:return c[1];case
34:return h(O(c[1]),pR);case
35:return h(pS,O(c[1]));case
36:var
d=h(pU,h(R(b,c[2]),pT));return h(f(b,c[1]),d);case
52:return h(O(j(c[1],0)),pV);default:}cV(c);return T(pP)}function
bm(a,b){return b[2]}function
f(b,c){return _(bn(0,b,c))}function
R(b,c){return _(tt(0,b,c))}function
ar(b,c){return _(c_(0,b,c))}function
ok(b){return _(bm(0,b))}var
p7=h(p6,h(p5,h(p4,h(p3,h(p2,h(p1,h(p0,h(pZ,h(pY,h(pX,pW)))))))))),qm=h(ql,h(qk,h(qj,h(qi,h(qh,h(qg,h(qf,h(qe,h(qd,h(qc,h(qb,h(qa,h(p$,h(p_,h(p9,p8))))))))))))))),qu=h(qt,h(qs,h(qr,h(qq,h(qp,h(qo,qn)))))),qC=h(qB,h(qA,h(qz,h(qy,h(qx,h(qw,qv))))));function
a(a){return[32,h(qD,k(a)),a]}function
v(a,b){return[25,a,b]}function
be(a,b){return[50,a,b]}function
eS(a){return[33,a]}function
aM(a){return[51,a]}function
ac(a){return[34,a]}function
bO(a,b){return[2,a,b]}function
bP(a,b){return[3,a,b]}function
cW(a,b){return[5,a,b]}function
cX(a,b){return[6,a,b]}function
ad(a,b){return[7,a,b]}function
eT(a,b){return[9,a,b]}function
bf(a){return[13,a]}function
aA(a){return[14,a]}function
Z(a,b){return[31,[0,a,b]]}function
V(a,b){return[37,a,b]}function
W(a,b){return[27,a,b]}function
X(a){return[28,a]}function
aN(a,b){return[38,a,b]}function
eU(a,b){return[47,a,b]}function
eV(a){var
e=[0,0];function
b(a){var
c=a;for(;;){if(typeof
c!==n)switch(c[0]){case
0:var
s=b(c[2]);return[0,b(c[1]),s];case
1:return[1,b(c[1])];case
2:var
t=b(c[2]);return[2,b(c[1]),t];case
3:var
h=c[2],i=c[1];if(typeof
i!==n)if(34===i[0])if(typeof
h!==n)if(34===h[0]){e[1]=1;return[34,i[1]+h[1]]}var
u=b(h);return[3,b(i),u];case
4:var
v=b(c[2]);return[4,b(c[1]),v];case
5:var
j=c[2],k=c[1];if(typeof
k!==n)if(34===k[0])if(typeof
j!==n)if(34===j[0]){e[1]=1;return[34,k[1]+j[1]]}var
w=b(j);return[5,b(k),w];case
6:var
x=b(c[2]);return[6,b(c[1]),x];case
7:var
l=c[2],m=c[1];if(typeof
m!==n)if(34===m[0])if(typeof
l!==n)if(34===l[0]){e[1]=1;return[34,m[1]+l[1]]}var
y=b(l);return[7,b(m),y];case
8:var
z=b(c[2]);return[8,b(c[1]),z];case
9:var
o=c[2],p=c[1];if(typeof
p!==n)if(34===p[0])if(typeof
o!==n)if(34===o[0]){e[1]=1;return[34,p[1]+o[1]]}var
A=b(o);return[9,b(p),A];case
10:var
B=b(c[2]);return[10,b(c[1]),B];case
15:return[25,c,c];case
18:return[18,c[1],c[2]];case
19:return[19,c[1],c[2]];case
20:return[20,c[1],c[2]];case
21:return[21,c[1],c[2]];case
22:return[22,c[1],c[2]];case
23:var
C=c[2];return[23,b(c[1]),C];case
24:var
D=b(c[2]);return[24,b(c[1]),D];case
25:var
E=b(c[2]);return[25,b(c[1]),E];case
26:var
d=c[1];if(typeof
d!==n)switch(d[0]){case
25:e[1]=1;var
F=b([26,d[2]]);return[25,b(d[1]),F];case
39:e[1]=1;var
G=b([26,d[3]]),H=b([26,d[2]]);return[39,b(d[1]),H,G];case
40:e[1]=1;var
I=b([26,d[2]]);return[40,b(d[1]),I];case
48:e[1]=1;var
J=b([26,d[4]]),K=b(d[3]),L=b(d[2]);return[48,b(d[1]),L,K,J];case
49:e[1]=1;var
M=b([26,d[2]]);return[49,b(d[1]),M];default:}return[26,b(d)];case
27:var
N=b(c[2]);return[27,b(c[1]),N];case
28:var
c=c[1];continue;case
29:var
f=c[2],q=c[1];if(typeof
f!==n)switch(f[0]){case
25:var
P=b(f[2]),Q=[29,b(q),P];return[25,f[1],Q];case
39:e[1]=1;var
R=f[3],S=[29,b(q),R],T=b(f[2]),U=[29,b(q),T];return[39,b(f[1]),U,S];default:}var
O=b(f);return[29,b(q),O];case
30:var
V=b(c[3]),W=b(c[2]);return[30,b(c[1]),W,V];case
36:var
g=c[2],r=c[1];if(typeof
g!==n)if(25===g[0]){var
Y=b(g[2]),Z=[36,b(r),Y];return[25,g[1],Z]}var
X=b(g);return[36,b(r),X];case
37:var
_=b(c[2]);return[37,b(c[1]),_];case
38:var
$=b(c[2]);return[38,b(c[1]),$];case
39:var
aa=b(c[3]),ab=b(c[2]);return[39,b(c[1]),ab,aa];case
40:var
ac=b(c[2]);return[40,b(c[1]),ac];case
41:var
ad=b(c[2]);return[41,b(c[1]),ad];case
42:var
ae=b(c[2]);return[42,b(c[1]),ae];case
43:var
af=b(c[2]);return[43,b(c[1]),af];case
44:var
ag=b(c[2]);return[44,b(c[1]),ag];case
45:var
ah=b(c[2]);return[45,b(c[1]),ah];case
46:var
ai=b(c[2]);return[46,b(c[1]),ai];case
47:var
aj=b(c[2]);return[47,b(c[1]),aj];case
48:var
ak=b(c[4]),al=b(c[3]),am=b(c[2]);return[48,b(c[1]),am,al,ak];case
49:var
an=b(c[2]);return[49,b(c[1]),an];case
50:var
ao=aY(b,c[2]);return[50,b(c[1]),ao];default:}return c}}var
c=b(a);for(;;){if(e[1]){e[1]=0;var
c=b(c);continue}return c}}var
qK=[0,qJ];function
bg(a,b,c){var
g=c[2],d=c[1],t=a?a[1]:a,u=b?b[1]:2,m=g[3],p=g[2];qK[1]=qL;var
j=m[1];if(typeof
j===n)var
l=1===j?0:1;else
switch(j[0]){case
23:case
29:case
36:var
l=0;break;case
13:case
14:case
17:var
o=h(qT,h(k(j[1]),qS)),l=2;break;default:var
l=1}switch(l){case
1:cV(m[1]);dT(dO);throw[0,H,qM];case
2:break;default:var
o=qN}var
q=[0,e(0,m[1]),o];if(t){bc[1]=q;bd[1]=q}function
r(a){var
q=g[4],r=dV(function(a,b){return 0===b?a:h(qu,a)},qC,q),s=h(r,e(0,eV(p))),j=c3(e_(qO,gI,438));dP(j,s);c4(j);e$(j);fj(qP);var
m=dQ(qQ),c=t4(m),n=B(c),o=0;if(0<=0)if(0<=c)if((n.getLen()-c|0)<o)var
f=0;else{var
k=o,b=c;for(;;){if(0<b){var
l=c8(m,n,k,b);if(0===l)throw[0,bz];var
k=k+l|0,b=b-l|0;continue}var
f=1;break}}else
var
f=0;else
var
f=0;if(!f)G(gK);dS(m);i(af(d,723535973,3),d,n);fj(qR);return 0}function
s(a){var
b=g[4],c=dV(function(a,b){return 0===b?a:h(qm,a)},p7,b);return i(af(d,56985577,4),d,h(c,f(0,eV(p))))}switch(u){case
1:s(0);break;case
2:r(0);s(0);break;default:r(0)}i(af(d,345714255,5),d,0);return[0,d,g]}var
cY=d,eW=null,qY=1,qZ=1,q0=1,q1=1,q2=1,q3=1,q4=undefined;function
eX(a,b){return a==eW?j(b,0):a}var
eY=Array,q5=true,q6=false;eh(function(a){return a
instanceof
eY?0:[0,new
aD(a.toString())]});function
D(a,b){a.appendChild(b);return 0}function
eZ(d){return t1(function(a){if(a){var
e=j(d,a);if(!(e|0))a.preventDefault();return e}var
c=event,b=j(d,c);if(!(b|0))c.returnValue=b;return b})}var
L=cY.document,q7="2d";function
bQ(a,b){return a?j(b,a[1]):0}function
bR(a,b){return a.createElement(b.toString())}function
bS(a,b){return bR(a,b)}var
e0=[0,f8];function
e1(a,b,c,d){for(;;){if(0===a)if(0===b)return bR(c,d);var
h=e0[1];if(f8===h){try{var
j=L.createElement('<input name="x">'),k=j.tagName.toLowerCase()===fs?1:0,m=k?j.name===dt?1:0:k,i=m}catch(f){var
i=0}var
l=i?fT:-1003883683;e0[1]=l;continue}if(fT<=h){var
e=new
eY();e.push("<",d.toString());bQ(a,function(a){e.push(' type="',fk(a),cb);return 0});bQ(b,function(a){e.push(' name="',fk(a),cb);return 0});e.push(">");return c.createElement(e.join(g))}var
f=bR(c,d);bQ(a,function(a){return f.type=a});bQ(b,function(a){return f.name=a});return f}}function
e2(a){return bS(a,q$)}var
rd=[0,rc];cY.HTMLElement===q4;function
e6(a){return e2(L)}function
e7(a){function
c(a){throw[0,H,rf]}var
b=eX(L.getElementById(go),c);return j(ee(function(a){D(b,e6(0));D(b,L.createTextNode(a.toString()));return D(b,e6(0))}),a)}var
ae=b5,as=b5,cZ=5e3;function
rg(a){var
k=[0,[4,a]];return function(a,b,c,d){var
h=a[2],i=a[1],l=c[2];if(0===l[0]){var
g=l[1],e=[0,0],f=uI(k.length-1),m=g[7][1]<i[1]?1:0;if(m)var
n=m;else{var
s=g[7][2]<i[2]?1:0,n=s||(g[7][3]<i[3]?1:0)}if(n)throw[0,lk];var
o=g[8][1]<h[1]?1:0;if(o)var
p=o;else{var
r=g[8][2]<h[2]?1:0,p=r||(g[8][3]<h[3]?1:0)}if(p)throw[0,lm];cm(function(a,b){function
h(a){if(bM)try{eN(a,0,c);Q(c,0,0)}catch(f){if(f[1]===az)throw[0,az];throw f}return 11===b[0]?uL(e,f,cQ(b[1],dB,c[1][8]),a):uX(e,f,cQ(a,dB,c[1][8]),a,c)}switch(b[0]){case
6:case
7:case
8:case
9:case
10:switch(b[0]){case
6:var
d=uW(e,f,b[1]);break;case
7:var
d=uV(e,f,b[1]);break;case
8:var
d=uU(e,f,b[1]);break;case
9:var
d=uT(e,f,b[1]);break;default:var
d=T(lh)}var
g=d;break;case
11:var
g=h(b[1]);break;default:var
g=h(b[1])}return g},k);var
q=uS(e,d,h,i,f,c[1],b)}else{var
j=[0,0];cm(function(a,b){switch(b[0]){case
6:case
7:case
8:case
9:case
10:switch(b[0]){case
6:var
e=vj(j,d,b[1],c[1]);break;case
7:var
e=vk(j,d,b[1],c[1]);break;case
8:var
e=vh(j,d,b[1],c[1]);break;case
9:var
e=vi(j,d,b[1],c[1]);break;default:var
e=T(li)}var
g=e;break;default:var
f=b[1];if(bM){if(c7(a9(f),[0,c]))eN(f,0,c);Q(c,0,0)}var
h=c[1],i=J(0),g=vl(j,d,a,cQ(f,-701974253,c[1][8]-i|0),h)}return g},k);var
q=vg(d,h,i,c[1],b)}return q}}if(c0===0)var
c=en([0]);else{var
a7=en(aY(hK,c0));cm(function(a,b){var
c=(a*2|0)+2|0;a7[3]=p(ay[4],b,c,a7[3]);a7[4]=p(ap[4],c,1,a7[4]);return 0},c0);var
c=a7}var
cC=aY(function(a){return a4(c,a)},e5),ev=cU[2],rh=cC[1],ri=cC[2],rj=cC[3],h4=cU[4],ep=cD(e3),eq=cD(e5),er=cD(e4),rk=1,cE=cn(function(a){return a4(c,a)},eq),hN=cn(function(a){return a4(c,a)},er);c[5]=[0,[0,c[3],c[4],c[6],c[7],cE,ep],c[5]];var
hO=ah[1],hP=c[7];function
hQ(a,b,c){return cq(a,ep)?p(ah[4],a,b,c):c}c[7]=p(ah[11],hQ,hP,hO);var
a5=[0,ay[1]],a6=[0,ap[1]];dY(function(a,b){a5[1]=p(ay[4],a,b,a5[1]);var
e=a6[1];try{var
f=i(ap[22],b,c[4]),d=f}catch(f){if(f[1]!==t)throw f;var
d=1}a6[1]=p(ap[4],b,d,e);return 0},er,hN);dY(function(a,b){a5[1]=p(ay[4],a,b,a5[1]);a6[1]=p(ap[4],b,0,a6[1]);return 0},eq,cE);c[3]=a5[1];c[4]=a6[1];var
hR=0,hS=c[6];c[6]=cp(function(a,b){return cq(a[1],cE)?b:[0,a,b]},hS,hR);var
h5=rk?i(ev,c,h4):j(ev,c),es=c[5],aF=es?es[1]:T(gO),et=c[5],hT=aF[6],hU=aF[5],hV=aF[4],hW=aF[3],hX=aF[2],hY=aF[1],hZ=et?et[2]:T(gP);c[5]=hZ;var
co=hV,bA=hT;for(;;){if(bA){var
dX=bA[1],gQ=bA[2],h0=i(ah[22],dX,c[7]),co=p(ah[4],dX,h0,co),bA=gQ;continue}c[7]=co;c[3]=hY;c[4]=hX;var
h1=c[6];c[6]=cp(function(a,b){return cq(a[1],hU)?b:[0,a,b]},h1,hW);var
h6=0,h7=cF(e4),h8=[0,aY(function(a){var
e=a4(c,a);try{var
b=c[6];for(;;){if(!b)throw[0,t];var
d=b[1],f=b[2],h=d[2];if(0!==aP(d[1],e)){var
b=f;continue}var
g=h;break}}catch(f){if(f[1]!==t)throw f;var
g=s(c[2],e)}return g},h7),h6],h9=cF(e3),rl=tv([0,[0,h5],[0,aY(function(a){try{var
b=i(ah[22],a,c[7])}catch(f){if(f[1]===t)throw[0,H,h3];throw f}return b},h9),h8]])[1],rm=function(a,b){if(1===b.length-1){var
c=b[0+1];if(4===c[0])return c[1]}return T(rn)};ex(c,[0,ri,0,rg,rj,function(a,b){return[0,[4,b]]},rh,rm]);var
ro=function(a,b){var
e=ew(b,c);p(rl,e,rq,rp);if(!b){var
f=c[8];if(0!==f){var
d=f;for(;;){if(d){var
g=d[2];j(d[1],e);var
d=g;continue}break}}}return e};eo[1]=(eo[1]+c[1]|0)-1|0;c[8]=dW(c[8]);cA(c,3+at(s(c[2],1)*16|0,aE)|0);var
h_=0,h$=function(a){var
b=a;return ro(h_,b)},ru=a(6),rr=[0,0],rt=[0,1,rs],rv=a(3),rw=aM(function(a){return ae}),rx=bO(cX(a(2),rw),rv),qH=[29,[36,a(0),rx],ru],ry=a(8),rz=ad(a(8),ry),rA=a(7),rB=bP(ad(a(7),rA),rz),rC=aN(a(13),rB),rD=a(10),rE=v(aN(a(8),rD),rC),rF=a(9),rG=v(aN(a(7),rF),rE),rH=a(12),rI=a(8),rJ=a(7),rK=bP(ad(ad(ac(2),rJ),rI),rH),rL=v(aN(a(10),rK),rG),rM=a(11),rN=a(8),rO=ad(a(8),rN),rP=a(7),rQ=bP(cW(ad(a(7),rP),rO),rM),rR=v(aN(a(9),rQ),rL),rS=eS(1),rT=bO(a(6),rS),rU=v(aN(a(6),rT),rR),rV=ac(4),qI=[46,a(13),rV],rW=aM(function(a){return cZ}),rX=v([49,[42,[44,a(6),rW],qI],rU],qH),rY=a(8),rZ=ad(a(8),rY),r0=a(7),r1=bP(ad(a(7),r0),rZ),r2=v(W(a(13),r1),rX),r3=ac(2),r4=[0,aM(function(a){return as})],r7=be(Z(r6,r5),r4),r8=[0,a(5)],r$=eT(be(Z(r_,r9),r8),r7),sa=cW(ad(ac(4),r$),r3),sb=v(W(a(12),sa),r2),sc=ac(2),sd=[0,aM(function(a){return ae})],sg=be(Z(sf,se),sd),sh=[0,a(4)],sk=eT(be(Z(sj,si),sh),sg),sl=cW(ad(ac(4),sk),sc),sm=v(W(a(11),sl),sb),sn=ac(0),so=v(W(a(10),sn),sm),sp=ac(0),sq=v(W(a(9),sp),so),sr=ac(0),ss=v(W(a(8),sr),sq),st=ac(0),su=v(W(a(7),st),ss),sv=eS(0),sw=v(W(a(6),sv),su),sx=a(2),sy=v(W(a(5),sx),sw),sz=a(3),sA=v(W(a(4),sz),sy),sD=be(Z(sC,sB),[0,1]),sE=aM(function(a){return ae}),sF=eU(a(3),sE),sG=aM(function(a){return as}),sH=v([40,[41,eU(a(2),sG),sF],sD],sA),sK=Z(sJ,sI),sN=cX(Z(sM,sL),sK),sQ=bO(Z(sP,sO),sN),sR=v(W(a(3),sQ),sH),sU=Z(sT,sS),sX=cX(Z(sW,sV),sU),s0=bO(Z(sZ,sY),sX),qF=[26,v(W(a(2),s0),sR)],s1=V(X(aA(13)),qF),s2=V(X(aA(12)),s1),s3=V(X(aA(11)),s2),s4=V(X(aA(10)),s3),s5=V(X(aA(9)),s4),s6=V(X(aA(8)),s5),s7=V(X(aA(7)),s6),s8=V(X(bf(6)),s7),s9=V(X(bf(5)),s8),s_=V(X(bf(4)),s9),s$=V(X(bf(3)),s_),qE=[0,[1,[24,[23,qG,0],0]],V(X(bf(2)),s$)],ta=[0,function(a){var
e=qZ+(q1*q3|0)|0,f=qY+(q0*q2|0)|0,j=as<=e?1:0,m=j||(ae<=f?1:0),i=0*0+0*0,d=0,c=0,b=0,k=4*(f/ae)-2,l=4*(e/as)-2;for(;;){if(b<cZ)if(i<=4){var
g=c*c-d*d+k,h=2*c*d+l,i=g*g+h*h,d=h,c=g,b=b+1|0;continue}return aK(a,(e*ae|0)+f|0,b)}},qE,rt,rr],c1=[0,h$(0),ta],bT=function(a){return e2(L)};cY.onload=eZ(function(a){function
R(a){throw[0,H,tg]}var
c=eX(L.getElementById(go),R);D(c,bT(0));var
h=e1(0,0,L,q9),G=bR(L,rb);D(G,L.createTextNode("Choose a computing device : "));D(c,G);h.style.margin="10px";D(c,h);var
F=bS(L,ra);D(c,F);D(c,bT(0));var
e=bS(L,re);if(1-(e.getContext==eW?1:0)){e.width=ae;e.height=as;var
I=e.getContext(q7);D(c,bT(0));D(c,e);var
O=e8?e8[1]:2;switch(O){case
1:fl(0);aI[1]=fm(0);break;case
2:fn(0);aH[1]=fo(0);fl(0);aI[1]=fm(0);break;default:fn(0);aH[1]=fo(0)}eG[1]=aH[1]+aI[1]|0;var
y=aH[1]-1|0,x=0,P=0;if(y<0)var
z=x;else{var
g=P,C=x;for(;;){var
E=cl(C,[0,u1(g),0]),Q=g+1|0;if(y!==g){var
g=Q,C=E;continue}var
z=E;break}}var
q=0,d=0,b=z;for(;;){if(q<aI[1]){if(vf(d)){var
B=d+1|0,A=cl(b,[0,u3(d,d+aH[1]|0),0])}else{var
B=d,A=b}var
q=q+1|0,d=B,b=A;continue}var
p=0,o=b;for(;;){if(o){var
p=p+1|0,o=o[2];continue}eG[1]=p;aI[1]=d;if(b){var
m=0,k=b,K=b[2],M=b[1];for(;;){if(k){var
m=m+1|0,k=k[2];continue}var
v=w(m,M),n=1,f=K;for(;;){if(f){var
N=f[2];v[n+1]=f[1];var
n=n+1|0,f=N;continue}var
t=v;break}break}}else
var
t=[0];var
J=I.getImageData(0,0,ae,as),l=J.data;D(c,bT(0));dU(function(a){var
b=bS(L,q8);D(b,L.createTextNode(a[1][1].toString()));return D(h,b)},t);var
S=function(a){var
f=s(t,h.selectedIndex+0|0),z=f[1][1];j(e7(td),z);var
n=aJ(j4,0,ae*as|0);ei(hF,fa(0));var
p=f[2];if(0===p[0])var
d=16;else{var
E=0===p[1][2]?1:16,d=E}bg(0,te,c1);var
v=eF(0),e=c1[2],b=c1[1],A=0,B=[0,[0,d,d,1],[0,at((ae+d|0)-1|0,d),at((as+d|0)-1|0,d),1]],q=0,o=0?q[1]:q;if(0===f[2][0]){if(o)bg(0,qU,[0,b,e]);else
if(!i(af(b,-723625231,7),b,0))bg(0,qV,[0,b,e])}else
if(o)bg(0,qW,[0,b,e]);else
if(!i(af(b,649483637,8),b,0))bg(0,qX,[0,b,e]);(function(a,b,c,d,e,f){return a.length==5?a(b,c,d,e,f):am(a,[b,c,d,e,f])}(af(b,5695307,6),b,n,B,A,f));var
u=Y(n)-1|0,C=0;if(!(u<0)){var
c=C;for(;;){var
g=aL(n,c);if(g===cZ)var
k=tc;else{var
m=function(a){return r*(gn+gn*Math.sin(a*0.1))|0},x=m(g),y=m(g+16|0),k=[0,m(g+32|0),y,x]}l[c*4|0]=k[1];l[(c*4|0)+1|0]=k[2];l[(c*4|0)+2|0]=k[3];l[(c*4|0)+3|0]=r;var
D=c+1|0;if(u!==c){var
c=D;continue}break}}var
w=eF(0)-v;i(e7(tb),tf,w);I.putImageData(J,0,0);return q5},u=e1([0,"button"],0,L,q_);u.value="Go";u.onclick=eZ(S);D(F,u);return q6}}}throw[0,rd]});dR(0);return}}(this));
