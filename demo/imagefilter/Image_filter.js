// This program was compiled from OCaml by js_of_ocaml 1.99dev
(function(d){"use strict";var
c9="set_cuda_sources",dp=123,bP=";",fA=108,gb="section1",c8="reload_sources",bT="Map.bal",fN=",",bZ='"',_=16777215,c7="get_cuda_sources",b7=" / ",fz="double spoc_var",df="args_to_list",bY=" * ",ae="(",fn="float spoc_var",de=65599,b6="if (",bX="return",fM=" ;\n",dn="exec",bh=115,bf=";}\n",fy=".ptx",p=512,dm=120,c6="..",fL=-512,L="]",dl=117,bS="; ",dk="compile",ga=" (",W="0",dd="list_to_args",bR=248,fK=126,f$="fd ",c5="get_binaries",fx=" == ",dc="Kirc_Cuda.ml",b5=" + ",fJ=") ",dj="x",fw=-97,fm="g",bd=1073741823,f_="parse concat",av=105,db="get_opencl_sources",f9=511,be=110,f8=-88,ac=" = ",da="set_opencl_sources",M="[",bW="'",fl="Unix",bO="int_of_string",f7="(double) ",fI=982028505,bc="){\n",bg="e",f6="#define __FLOAT64_EXTENSION__ \n",au="-",aK=-48,bV="(double) spoc_var",fk="++){\n",fv="__shared__ float spoc_var",f4="Image_filter_js.ml",f5="opencl_sources",fu=".cl",di="reset_binaries",bN="\n",f3=101,ds=748841679,b4="index out of bounds",fj="spoc_init_opencl_device_vec",c4=125,bU=" - ",f2=";}",r=255,f1="binaries",b3="}",f0=" < ",fi="__shared__ long spoc_var",aJ=250,fZ=" >= ",fh="input",fH=246,c$=102,fG="Unix.Unix_error",g="",fg=" || ",aI=100,dh="Kirc_OpenCL.ml",fY="#ifndef __FLOAT64_EXTENSION__ \n",fF="__shared__ int spoc_var",dr=103,bM=", ",fE="./",ft=1e3,ff="for (int ",fX="file_file",fW="spoc_var",af=".",fs="else{\n",bQ="+",dq="run",b2=65535,dg="#endif\n",aH=";\n",X="f",fV=785140586,fU="__shared__ double spoc_var",fr=-32,c_=111,fD=" > ",B=" ",fT="int spoc_var",ad=")",fC="cuda_sources",b1=256,fq="nan",c3=116,fQ="../",fR="kernel_name",fS=65520,fP="%.12g",fe=" && ",fp="/",fB="while (",c2="compile_and_run",b0=114,fO="* spoc_var",bL=" <= ",m="number",fo=" % ",tY=d.spoc_opencl_part_device_to_cpu_b!==undefined?d.spoc_opencl_part_device_to_cpu_b:function(){n("spoc_opencl_part_device_to_cpu_b not implemented")},tX=d.spoc_opencl_part_cpu_to_device_b!==undefined?d.spoc_opencl_part_cpu_to_device_b:function(){n("spoc_opencl_part_cpu_to_device_b not implemented")},tV=d.spoc_opencl_load_param_int64!==undefined?d.spoc_opencl_load_param_int64:function(){n("spoc_opencl_load_param_int64 not implemented")},tT=d.spoc_opencl_load_param_float64!==undefined?d.spoc_opencl_load_param_float64:function(){n("spoc_opencl_load_param_float64 not implemented")},tS=d.spoc_opencl_load_param_float!==undefined?d.spoc_opencl_load_param_float:function(){n("spoc_opencl_load_param_float not implemented")},tN=d.spoc_opencl_custom_part_device_to_cpu_b!==undefined?d.spoc_opencl_custom_part_device_to_cpu_b:function(){n("spoc_opencl_custom_part_device_to_cpu_b not implemented")},tM=d.spoc_opencl_custom_part_cpu_to_device_b!==undefined?d.spoc_opencl_custom_part_cpu_to_device_b:function(){n("spoc_opencl_custom_part_cpu_to_device_b not implemented")},tL=d.spoc_opencl_custom_device_to_cpu!==undefined?d.spoc_opencl_custom_device_to_cpu:function(){n("spoc_opencl_custom_device_to_cpu not implemented")},tK=d.spoc_opencl_custom_cpu_to_device!==undefined?d.spoc_opencl_custom_cpu_to_device:function(){n("spoc_opencl_custom_cpu_to_device not implemented")},tJ=d.spoc_opencl_custom_alloc_vect!==undefined?d.spoc_opencl_custom_alloc_vect:function(){n("spoc_opencl_custom_alloc_vect not implemented")},ty=d.spoc_cuda_part_device_to_cpu_b!==undefined?d.spoc_cuda_part_device_to_cpu_b:function(){n("spoc_cuda_part_device_to_cpu_b not implemented")},tx=d.spoc_cuda_part_cpu_to_device_b!==undefined?d.spoc_cuda_part_cpu_to_device_b:function(){n("spoc_cuda_part_cpu_to_device_b not implemented")},tw=d.spoc_cuda_load_param_vec_b!==undefined?d.spoc_cuda_load_param_vec_b:function(){n("spoc_cuda_load_param_vec_b not implemented")},tv=d.spoc_cuda_load_param_int_b!==undefined?d.spoc_cuda_load_param_int_b:function(){n("spoc_cuda_load_param_int_b not implemented")},tu=d.spoc_cuda_load_param_int64_b!==undefined?d.spoc_cuda_load_param_int64_b:function(){n("spoc_cuda_load_param_int64_b not implemented")},tt=d.spoc_cuda_load_param_float_b!==undefined?d.spoc_cuda_load_param_float_b:function(){n("spoc_cuda_load_param_float_b not implemented")},ts=d.spoc_cuda_load_param_float64_b!==undefined?d.spoc_cuda_load_param_float64_b:function(){n("spoc_cuda_load_param_float64_b not implemented")},tr=d.spoc_cuda_launch_grid_b!==undefined?d.spoc_cuda_launch_grid_b:function(){n("spoc_cuda_launch_grid_b not implemented")},tq=d.spoc_cuda_flush_all!==undefined?d.spoc_cuda_flush_all:function(){n("spoc_cuda_flush_all not implemented")},tp=d.spoc_cuda_flush!==undefined?d.spoc_cuda_flush:function(){n("spoc_cuda_flush not implemented")},to=d.spoc_cuda_device_to_cpu!==undefined?d.spoc_cuda_device_to_cpu:function(){n("spoc_cuda_device_to_cpu not implemented")},tm=d.spoc_cuda_custom_part_device_to_cpu_b!==undefined?d.spoc_cuda_custom_part_device_to_cpu_b:function(){n("spoc_cuda_custom_part_device_to_cpu_b not implemented")},tl=d.spoc_cuda_custom_part_cpu_to_device_b!==undefined?d.spoc_cuda_custom_part_cpu_to_device_b:function(){n("spoc_cuda_custom_part_cpu_to_device_b not implemented")},tk=d.spoc_cuda_custom_load_param_vec_b!==undefined?d.spoc_cuda_custom_load_param_vec_b:function(){n("spoc_cuda_custom_load_param_vec_b not implemented")},tj=d.spoc_cuda_custom_device_to_cpu!==undefined?d.spoc_cuda_custom_device_to_cpu:function(){n("spoc_cuda_custom_device_to_cpu not implemented")},ti=d.spoc_cuda_custom_cpu_to_device!==undefined?d.spoc_cuda_custom_cpu_to_device:function(){n("spoc_cuda_custom_cpu_to_device not implemented")},gq=d.spoc_cuda_custom_alloc_vect!==undefined?d.spoc_cuda_custom_alloc_vect:function(){n("spoc_cuda_custom_alloc_vect not implemented")},th=d.spoc_cuda_create_extra!==undefined?d.spoc_cuda_create_extra:function(){n("spoc_cuda_create_extra not implemented")},tg=d.spoc_cuda_cpu_to_device!==undefined?d.spoc_cuda_cpu_to_device:function(){n("spoc_cuda_cpu_to_device not implemented")},gp=d.spoc_cuda_alloc_vect!==undefined?d.spoc_cuda_alloc_vect:function(){n("spoc_cuda_alloc_vect not implemented")},td=d.spoc_create_custom!==undefined?d.spoc_create_custom:function(){n("spoc_create_custom not implemented")},t1=1;function
gl(a,b){throw[0,a,b]}function
dC(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=d.console;b&&b.error&&b.error(a)}var
q=[0];function
bk(a,b){if(!a)return g;if(a&1)return bk(a-1,b)+b;var
c=bk(a>>1,b);return c+c}function
D(a){if(a!=null){this.bytes=this.fullBytes=a;this.last=this.len=a.length}}function
go(){gl(q[4],new
D(b4))}D.prototype={string:null,bytes:null,fullBytes:null,array:null,len:null,last:0,toJsString:function(){var
a=this.getFullBytes();try{return this.string=decodeURIComponent(escape(a))}catch(f){dC('MlString.toJsString: wrong encoding for \"%s\" ',a);return a}},toBytes:function(){if(this.string!=null)try{var
a=unescape(encodeURIComponent(this.string))}catch(f){dC('MlString.toBytes: wrong encoding for \"%s\" ',this.string);var
a=this.string}else{var
a=g,c=this.array,d=c.length;for(var
b=0;b<d;b++)a+=String.fromCharCode(c[b])}this.bytes=this.fullBytes=a;this.last=this.len=a.length;return a},getBytes:function(){var
a=this.bytes;if(a==null)a=this.toBytes();return a},getFullBytes:function(){var
a=this.fullBytes;if(a!==null)return a;a=this.bytes;if(a==null)a=this.toBytes();if(this.last<this.len){this.bytes=a+=bk(this.len-this.last,"\0");this.last=this.len}this.fullBytes=a;return a},toArray:function(){var
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
b=this.bytes;if(b==null)b=this.toBytes();return a<this.last?b.charCodeAt(a):0},safeGet:function(a){if(this.len==null)this.toBytes();if(a<0||a>=this.len)go();return this.get(a)},set:function(a,b){var
c=this.array;if(!c){if(this.last==a){this.bytes+=String.fromCharCode(b&r);this.last++;return 0}c=this.toArray()}else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;c[a]=b&r;return 0},safeSet:function(a,b){if(this.len==null)this.toBytes();if(a<0||a>=this.len)go();this.set(a,b)},fill:function(a,b,c){if(a>=this.last&&this.last&&c==0)return;var
d=this.array;if(!d)d=this.toArray();else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;var
f=a+b;for(var
e=a;e<f;e++)d[e]=c},compare:function(a){if(this.string!=null&&a.string!=null){if(this.string<a.string)return-1;if(this.string>a.string)return 1;return 0}var
b=this.getFullBytes(),c=a.getFullBytes();if(b<c)return-1;if(b>c)return 1;return 0},equal:function(a){if(this.string!=null&&a.string!=null)return this.string==a.string;return this.getFullBytes()==a.getFullBytes()},lessThan:function(a){if(this.string!=null&&a.string!=null)return this.string<a.string;return this.getFullBytes()<a.getFullBytes()},lessEqual:function(a){if(this.string!=null&&a.string!=null)return this.string<=a.string;return this.getFullBytes()<=a.getFullBytes()}};function
aw(a){this.string=a}aw.prototype=new
D();function
r5(a,b,c,d,e){if(d<=b)for(var
f=1;f<=e;f++)c[d+f]=a[b+f];else
for(var
f=e;f>=1;f--)c[d+f]=a[b+f]}function
r6(a){var
c=[0];while(a!==0){var
d=a[1];for(var
b=1;b<d.length;b++)c.push(d[b]);a=a[2]}return c}function
dB(a,b){gl(a,new
aw(b))}function
an(a){dB(q[4],a)}function
aL(){an(b4)}function
r7(a,b){if(b<0||b>=a.length-1)aL();return a[b+1]}function
r8(a,b,c){if(b<0||b>=a.length-1)aL();a[b+1]=c;return 0}var
du;function
r9(a,b,c){if(c.length!=2)an("Bigarray.create: bad number of dimensions");if(b!=0)an("Bigarray.create: unsupported layout");if(c[1]<0)an("Bigarray.create: negative dimension");if(!du){var
e=d;du=[e.Float32Array,e.Float64Array,e.Int8Array,e.Uint8Array,e.Int16Array,e.Uint16Array,e.Int32Array,null,e.Int32Array,e.Int32Array,null,null,e.Uint8Array]}var
f=du[a];if(!f)an("Bigarray.create: unsupported kind");return new
f(c[1])}function
r_(a,b){if(b<0||b>=a.length)aL();return a[b]}function
r$(a,b,c){if(b<0||b>=a.length)aL();a[b]=c;return 0}function
dv(a,b,c,d,e){if(e===0)return;if(d===c.last&&c.bytes!=null){var
f=a.bytes;if(f==null)f=a.toBytes();if(b>0||a.last>e)f=f.slice(b,b+e);c.bytes+=f;c.last+=f.length;return}var
g=c.array;if(!g)g=c.toArray();else
c.bytes=c.string=null;a.blitToArray(b,g,d,e)}function
ag(c,b){if(c.fun)return ag(c.fun,b);var
a=c.length,d=a-b.length;if(d==0)return c.apply(null,b);else
if(d<0)return ag(c.apply(null,b.slice(0,a)),b.slice(a));else
return function(a){return ag(c,b.concat([a]))}}function
sa(a){if(isFinite(a)){if(Math.abs(a)>=2.22507385850720138e-308)return 0;if(a!=0)return 1;return 2}return isNaN(a)?4:3}function
sm(a,b){var
c=a[3]<<16,d=b[3]<<16;if(c>d)return 1;if(c<d)return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
gi(a,b){if(a<b)return-1;if(a==b)return 0;return 1}function
dw(a,b,c){var
e=[];for(;;){if(!(c&&a===b))if(a
instanceof
D)if(b
instanceof
D){if(a!==b){var
d=a.compare(b);if(d!=0)return d}}else
return 1;else
if(a
instanceof
Array&&a[0]===(a[0]|0)){var
g=a[0];if(g===aJ){a=a[1];continue}else
if(b
instanceof
Array&&b[0]===(b[0]|0)){var
h=b[0];if(h===aJ){b=b[1];continue}else
if(g!=h)return g<h?-1:1;else
switch(g){case
bR:{var
d=gi(a[2],b[2]);if(d!=0)return d;break}case
251:an("equal: abstract value");case
r:{var
d=sm(a,b);if(d!=0)return d;break}default:if(a.length!=b.length)return a.length<b.length?-1:1;if(a.length>1)e.push(a,b,1)}}else
return 1}else
if(b
instanceof
D||b
instanceof
Array&&b[0]===(b[0]|0))return-1;else{if(a<b)return-1;if(a>b)return 1;if(c&&a!=b){if(a==a)return 1;if(b==b)return-1}}if(e.length==0)return 0;var
f=e.pop();b=e.pop();a=e.pop();if(f+1<a.length)e.push(a,b,f+1);a=a[f];b=b[f]}}function
gd(a,b){return dw(a,b,true)}function
gc(a){this.bytes=g;this.len=a}gc.prototype=new
D();function
ge(a){if(a<0)an("String.create");return new
gc(a)}function
dA(a){throw[0,a]}function
gm(){dA(q[6])}function
sb(a,b){if(b==0)gm();return a/b|0}function
sc(a,b){return+(dw(a,b,false)==0)}function
sd(a,b,c,d){a.fill(b,c,d)}function
dz(a){a=a.toString();var
e=a.length;if(e>31)an("format_int: format too long");var
b={justify:bQ,signstyle:au,filler:B,alternate:false,base:0,signedconv:false,width:0,uppercase:false,sign:1,prec:-1,conv:X};for(var
d=0;d<e;d++){var
c=a.charAt(d);switch(c){case
au:b.justify=au;break;case
bQ:case
B:b.signstyle=c;break;case
W:b.filler=W;break;case"#":b.alternate=true;break;case"1":case"2":case"3":case"4":case"5":case"6":case"7":case"8":case"9":b.width=0;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.width=b.width*10+c;d++}d--;break;case
af:b.prec=0;d++;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.prec=b.prec*10+c;d++}d--;case"d":case"i":b.signedconv=true;case"u":b.base=10;break;case
dj:b.base=16;break;case"X":b.base=16;b.uppercase=true;break;case"o":b.base=8;break;case
bg:case
X:case
fm:b.signedconv=true;b.conv=c;break;case"E":case"F":case"G":b.signedconv=true;b.uppercase=true;b.conv=c.toLowerCase();break}}return b}function
dx(a,b){if(a.uppercase)b=b.toUpperCase();var
e=b.length;if(a.signedconv&&(a.sign<0||a.signstyle!=au))e++;if(a.alternate){if(a.base==8)e+=1;if(a.base==16)e+=2}var
c=g;if(a.justify==bQ&&a.filler==B)for(var
d=e;d<a.width;d++)c+=B;if(a.signedconv)if(a.sign<0)c+=au;else
if(a.signstyle!=au)c+=a.signstyle;if(a.alternate&&a.base==8)c+=W;if(a.alternate&&a.base==16)c+="0x";if(a.justify==bQ&&a.filler==W)for(var
d=e;d<a.width;d++)c+=W;c+=b;if(a.justify==au)for(var
d=e;d<a.width;d++)c+=B;return new
aw(c)}function
se(a,b){var
c,f=dz(a),e=f.prec<0?6:f.prec;if(b<0){f.sign=-1;b=-b}if(isNaN(b)){c=fq;f.filler=B}else
if(!isFinite(b)){c="inf";f.filler=B}else
switch(f.conv){case
bg:var
c=b.toExponential(e),d=c.length;if(c.charAt(d-3)==bg)c=c.slice(0,d-1)+W+c.slice(d-1);break;case
X:c=b.toFixed(e);break;case
fm:e=e?e:1;c=b.toExponential(e-1);var
i=c.indexOf(bg),h=+c.slice(i+1);if(h<-4||b.toFixed(0).length>e){var
d=i-1;while(c.charAt(d)==W)d--;if(c.charAt(d)==af)d--;c=c.slice(0,d+1)+c.slice(i);d=c.length;if(c.charAt(d-3)==bg)c=c.slice(0,d-1)+W+c.slice(d-1);break}else{var
g=e;if(h<0){g-=h+1;c=b.toFixed(g)}else
while(c=b.toFixed(g),c.length>e+1)g--;if(g){var
d=c.length-1;while(c.charAt(d)==W)d--;if(c.charAt(d)==af)d--;c=c.slice(0,d+1)}}break}return dx(f,c)}function
sf(a,b){if(a.toString()=="%d")return new
aw(g+b);var
c=dz(a);if(b<0)if(c.signedconv){c.sign=-1;b=-b}else
b>>>=0;var
d=b.toString(c.base);if(c.prec>=0){c.filler=B;var
e=c.prec-d.length;if(e>0)d=bk(e,W)+d}return dx(c,d)}function
sg(){return 0}function
sh(){return 0}var
b9=[];function
si(a,b,c){var
e=a[1],i=b9[c];if(i===null)for(var
h=b9.length;h<c;h++)b9[h]=0;else
if(e[i]===b)return e[i-1];var
d=3,g=e[1]*2+1,f;while(d<g){f=d+g>>1|1;if(b<e[f+1])g=f-2;else
d=f}b9[c]=d+1;return b==e[d+1]?e[d]:0}function
sj(a,b){return+(gd(a,b,false)>=0)}function
gf(a){if(!isFinite(a)){if(isNaN(a))return[r,1,0,fS];return a>0?[r,0,0,32752]:[r,0,0,fS]}var
f=a>=0?0:32768;if(f)a=-a;var
b=Math.floor(Math.LOG2E*Math.log(a))+1023;if(b<=0){b=0;a/=Math.pow(2,-1026)}else{a/=Math.pow(2,b-1027);if(a<16){a*=2;b-=1}if(b==0)a/=2}var
d=Math.pow(2,24),c=a|0;a=(a-c)*d;var
e=a|0;a=(a-e)*d;var
g=a|0;c=c&15|f|b<<4;return[r,g,e,c]}function
bj(a,b){return((a>>16)*b<<16)+(a&b2)*b|0}var
sk=function(){var
p=b1;function
c(a,b){return a<<b|a>>>32-b}function
g(a,b){b=bj(b,3432918353);b=c(b,15);b=bj(b,461845907);a^=b;a=c(a,13);return(a*5|0)+3864292196|0}function
t(a){a^=a>>>16;a=bj(a,2246822507);a^=a>>>13;a=bj(a,3266489909);a^=a>>>16;return a}function
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
bR:f=g(f,e[2]);h--;break;case
aJ:k[--l]=e[1];break;case
r:f=v(f,e);h--;break;default:var
s=e.length-1<<10|e[0];f=g(f,s);for(j=1,o=e.length;j<o;j++){if(m>=i)break;k[m++]=e[j]}break}else
if(e
instanceof
D){var
n=e.array;if(n)f=w(f,n);else{var
q=e.getFullBytes();f=x(f,q)}h--;break}else
if(e===(e|0)){f=g(f,e+e+1);h--}else
if(e===+e){f=u(f,gf(e));h--;break}}f=t(f);return f&bd}}();function
su(a){return[a[3]>>8,a[3]&r,a[2]>>16,a[2]>>8&r,a[2]&r,a[1]>>16,a[1]>>8&r,a[1]&r]}function
sl(e,b,c){var
d=0;function
f(a){b--;if(e<0||b<0)return;if(a
instanceof
Array&&a[0]===(a[0]|0))switch(a[0]){case
bR:e--;d=d*de+a[2]|0;break;case
aJ:b++;f(a);break;case
r:e--;d=d*de+a[1]+(a[2]<<24)|0;break;default:e--;d=d*19+a[0]|0;for(var
c=a.length-1;c>0;c--)f(a[c])}else
if(a
instanceof
D){e--;var
g=a.array,h=a.getLen();if(g)for(var
c=0;c<h;c++)d=d*19+g[c]|0;else{var
i=a.getFullBytes();for(var
c=0;c<h;c++)d=d*19+i.charCodeAt(c)|0}}else
if(a===(a|0)){e--;d=d*de+a|0}else
if(a===+a){e--;var
j=su(gf(a));for(var
c=7;c>=0;c--)d=d*19+j[c]|0}}f(c);return d&bd}function
sp(a){return(a[3]|a[2]|a[1])==0}function
ss(a){return[r,a&_,a>>24&_,a>>31&b2]}function
st(a,b){var
c=a[1]-b[1],d=a[2]-b[2]+(c>>24),e=a[3]-b[3]+(d>>24);return[r,c&_,d&_,e&b2]}function
gh(a,b){if(a[3]>b[3])return 1;if(a[3]<b[3])return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
gg(a){a[3]=a[3]<<1|a[2]>>23;a[2]=(a[2]<<1|a[1]>>23)&_;a[1]=a[1]<<1&_}function
sq(a){a[1]=(a[1]>>>1|a[2]<<23)&_;a[2]=(a[2]>>>1|a[3]<<23)&_;a[3]=a[3]>>>1}function
sw(a,b){var
e=0,d=a.slice(),c=b.slice(),f=[r,0,0,0];while(gh(d,c)>0){e++;gg(c)}while(e>=0){e--;gg(f);if(gh(d,c)>=0){f[1]++;d=st(d,c)}sq(c)}return[0,f,d]}function
sv(a){return a[1]|a[2]<<24}function
so(a){return a[3]<<16<0}function
sr(a){var
b=-a[1],c=-a[2]+(b>>24),d=-a[3]+(c>>24);return[r,b&_,c&_,d&b2]}function
sn(a,b){var
c=dz(a);if(c.signedconv&&so(b)){c.sign=-1;b=sr(b)}var
d=g,i=ss(c.base),h="0123456789abcdef";do{var
f=sw(b,i);b=f[1];d=h.charAt(sv(f[2]))+d}while(!sp(b));if(c.prec>=0){c.filler=B;var
e=c.prec-d.length;if(e>0)d=bk(e,W)+d}return dx(c,d)}function
sS(a){var
b=0,c=10,d=a.get(0)==45?(b++,-1):1;if(a.get(b)==48)switch(a.get(b+1)){case
dm:case
88:c=16;b+=2;break;case
c_:case
79:c=8;b+=2;break;case
98:case
66:c=2;b+=2;break}return[b,d,c]}function
gk(a){if(a>=48&&a<=57)return a-48;if(a>=65&&a<=90)return a-55;if(a>=97&&a<=122)return a-87;return-1}function
n(a){dB(q[3],a)}function
sx(a){var
g=sS(a),e=g[0],h=g[1],f=g[2],i=-1>>>0,d=a.get(e),c=gk(d);if(c<0||c>=f)n(bO);var
b=c;for(;;){e++;d=a.get(e);if(d==95)continue;c=gk(d);if(c<0||c>=f)break;b=f*b+c;if(b>i)n(bO)}if(e!=a.getLen())n(bO);b=h*b;if((b|0)!=b)n(bO);return b}function
sy(a){return+(a>31&&a<127)}var
b8={amp:/&/g,lt:/</g,quot:/\"/g,all:/[&<\"]/};function
sz(a){if(!b8.all.test(a))return a;return a.replace(b8.amp,"&amp;").replace(b8.lt,"&lt;").replace(b8.quot,"&quot;")}function
sA(a){var
c=Array.prototype.slice;return function(){var
b=arguments.length>0?c.call(arguments):[undefined];return ag(a,b)}}function
sB(a,b){var
d=[0];for(var
c=1;c<=a;c++)d[c]=b;return d}function
dt(a){var
b=a.length;this.array=a;this.len=this.last=b}dt.prototype=new
D();var
sC=function(){function
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
dt(n(h,c))}}();function
sD(a){return a.data.array.length}function
ao(a){dB(q[2],a)}function
dy(a){if(!a.opened)ao("Cannot flush a closed channel");if(a.buffer==g)return 0;if(a.output){switch(a.output.length){case
2:a.output(a,a.buffer);break;default:a.output(a.buffer)}}a.buffer=g}var
bi=new
Array();function
sE(a){dy(a);a.opened=false;delete
bi[a.fd];return 0}function
sF(a,b,c,d){var
e=a.data.array.length-a.data.offset;if(e<d)d=e;dv(new
dt(a.data.array),a.data.offset,b,c,d);a.data.offset+=d;return d}function
sT(){dA(q[5])}function
sG(a){if(a.data.offset>=a.data.array.length)sT();if(a.data.offset<0||a.data.offset>a.data.array.length)aL();var
b=a.data.array[a.data.offset];a.data.offset++;return b}function
sH(a){var
b=a.data.offset,c=a.data.array.length;if(b>=c)return 0;while(true){if(b>=c)return-(b-a.data.offset);if(b<0||b>a.data.array.length)aL();if(a.data.array[b]==10)return b-a.data.offset+1;b++}}function
sV(a,b){if(!q.files)q.files={};if(b
instanceof
D)var
c=b.getArray();else
if(b
instanceof
Array)var
c=b;else
var
c=new
D(b).getArray();q.files[a
instanceof
D?a.toString():a]=c}function
s2(a){return q.files&&q.files[a.toString()]?1:q.auto_register_file===undefined?0:q.auto_register_file(a)}function
bl(a,b,c){if(q.fds===undefined)q.fds=new
Array();c=c?c:{};var
d={};d.array=b;d.offset=c.append?d.array.length:0;d.flags=c;q.fds[a]=d;q.fd_last_idx=a;return a}function
s6(a,b,c){var
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
e=a.toString();if(d.rdonly&&d.wronly)ao(e+" : flags Open_rdonly and Open_wronly are not compatible");if(d.text&&d.binary)ao(e+" : flags Open_text and Open_binary are not compatible");if(s2(a)){if(d.create&&d.excl)ao(e+" : file already exists");var
f=q.fd_last_idx?q.fd_last_idx:0;if(d.truncate)q.files[e]=g;return bl(f+1,q.files[e],d)}else
if(d.create){var
f=q.fd_last_idx?q.fd_last_idx:0;sV(e,[]);return bl(f+1,q.files[e],d)}else
ao(e+": no such file or directory")}bl(0,[]);bl(1,[]);bl(2,[]);function
sI(a){var
b=q.fds[a];if(b.flags.wronly)ao(f$+a+" is writeonly");return{data:b,fd:a,opened:true}}function
tb(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=d.console;b&&b.log&&b.log(a)}function
sY(a,b){var
e=new
D(b),d=e.getLen();for(var
c=0;c<d;c++)a.data.array[a.data.offset+c]=e.get(c);a.data.offset+=d;return 0}function
sJ(a){var
b;switch(a){case
1:b=tb;break;case
2:b=dC;break;default:b=sY}var
d=q.fds[a];if(d.flags.rdonly)ao(f$+a+" is readonly");var
c={data:d,fd:a,opened:true,buffer:g,output:b};bi[c.fd]=c;return c}function
sK(){var
a=0;for(var
b
in
bi)if(bi[b].opened)a=[0,bi[b],a];return a}function
gj(a,b,c,d){if(!a.opened)ao("Cannot output to a closed channel");var
f;if(c==0&&b.getLen()==d)f=b;else{f=ge(d);dv(b,c,f,0,d)}var
e=f.toString(),g=e.lastIndexOf("\n");if(g<0)a.buffer+=e;else{a.buffer+=e.substr(0,g+1);dy(a);a.buffer+=e.substr(g+1)}}function
R(a){return new
D(a)}function
sL(a,b){var
c=R(String.fromCharCode(b));gj(a,c,0,1)}function
sM(a,b){if(b==0)gm();return a%b}function
sO(a,b){return+(dw(a,b,false)!=0)}function
sP(a,b){var
d=[a];for(var
c=1;c<=b;c++)d[c]=0;return d}function
sQ(a,b){a[0]=b;return 0}function
sR(a){return a
instanceof
Array?a[0]:ft}function
sW(a,b){q[a+1]=b}var
sN={};function
sX(a,b){sN[a]=b;return 0}function
sZ(a,b){return a.compare(b)}function
gn(a,b){var
c=a.fullBytes,d=b.fullBytes;if(c!=null&&d!=null)return c==d?1:0;return a.getFullBytes()==b.getFullBytes()?1:0}function
s0(a,b){return 1-gn(a,b)}function
s1(){return 32}function
s3(){var
a=new
aw("a.out");return[0,a,[0,a]]}function
s4(){return[0,new
aw(fl),32,0]}function
sU(){dA(q[7])}function
s5(){sU()}function
s7(){var
a=new
Date()^4294967295*Math.random();return{valueOf:function(){return a},0:0,1:a,length:2}}function
s8(){console.log("caml_sys_system_command");return 0}function
s9(a){var
b=1;while(a&&a.joo_tramp){a=a.joo_tramp.apply(null,a.joo_args);b++}return a}function
s_(a,b){return{joo_tramp:a,joo_args:b}}function
s$(a,b){if(typeof
b==="function"){a.fun=b;return 0}if(b.fun){a.fun=b.fun;return 0}var
c=b.length;while(c--)a[c]=b[c];return 0}function
ta(){return 0}var
dD=0;function
tc(){if(window.webcl==undefined){alert("Unfortunately your system does not support WebCL. "+"Make sure that you have both the OpenCL driver "+"and the WebCL browser extension installed.");dD=1}else{console.log("INIT OPENCL");dD=0}return 0}function
te(){console.log(" spoc_cuInit");return 0}function
tf(){console.log(" spoc_cuda_compile");return 0}function
tn(){console.log(" spoc_cuda_debug_compile");return 0}function
tz(a,b,c){console.log(" spoc_debug_opencl_compile");console.log(a.bytes);var
e=c[9],f=e[0],d=f.createProgram(a.bytes),g=d.getInfo(WebCL.PROGRAM_DEVICES);d.build(g);var
h=d.createKernel(b.bytes);e[0]=f;c[9]=e;return h}function
tA(a){console.log("spoc_getCudaDevice");return 0}function
tB(){console.log(" spoc_getCudaDevicesCount");return 0}function
tC(a,b){console.log(" spoc_getOpenCLDevice");var
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
f=k[o],j=f.getDevices(),m=j.length;console.log("there "+g+B+m+B+a);if(g+m>=a)for(var
q
in
j){var
c=j[q];if(g==a){console.log("current ----------"+g);e[1]=R(c.getInfo(WebCL.DEVICE_NAME));console.log(e[1]);e[2]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_SIZE);e[3]=c.getInfo(WebCL.DEVICE_LOCAL_MEM_SIZE);e[4]=c.getInfo(WebCL.DEVICE_MAX_CLOCK_FREQUENCY);e[5]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE);e[6]=c.getInfo(WebCL.DEVICE_MAX_COMPUTE_UNITS);e[7]=c.getInfo(WebCL.DEVICE_ERROR_CORRECTION_SUPPORT);e[8]=b;var
i=new
Array(3);i[0]=webcl.createContext(c);i[1]=i[0].createCommandQueue();i[2]=i[0].createCommandQueue();e[9]=i;h[1]=R(f.getInfo(WebCL.PLATFORM_PROFILE));h[2]=R(f.getInfo(WebCL.PLATFORM_VERSION));h[3]=R(f.getInfo(WebCL.PLATFORM_NAME));h[4]=R(f.getInfo(WebCL.PLATFORM_VENDOR));h[5]=R(f.getInfo(WebCL.PLATFORM_EXTENSIONS));h[6]=m;var
l=c.getInfo(WebCL.DEVICE_TYPE),v=0;if(l&WebCL.DEVICE_TYPE_CPU)d[2]=0;if(l&WebCL.DEVICE_TYPE_GPU)d[2]=1;if(l&WebCL.DEVICE_TYPE_ACCELERATOR)d[2]=2;if(l&WebCL.DEVICE_TYPE_DEFAULT)d[2]=3;d[3]=R(c.getInfo(WebCL.DEVICE_PROFILE));d[4]=R(c.getInfo(WebCL.DEVICE_VERSION));d[5]=R(c.getInfo(WebCL.DEVICE_VENDOR));var
r=c.getInfo(WebCL.DEVICE_EXTENSIONS);d[6]=R(r);d[7]=c.getInfo(WebCL.DEVICE_VENDOR_ID);d[8]=c.getInfo(WebCL.DEVICE_MAX_WORK_ITEM_DIMENSIONS);d[9]=c.getInfo(WebCL.DEVICE_ADDRESS_BITS);d[10]=c.getInfo(WebCL.DEVICE_MAX_MEM_ALLOC_SIZE);d[11]=c.getInfo(WebCL.DEVICE_IMAGE_SUPPORT);d[12]=c.getInfo(WebCL.DEVICE_MAX_READ_IMAGE_ARGS);d[13]=c.getInfo(WebCL.DEVICE_MAX_WRITE_IMAGE_ARGS);d[14]=c.getInfo(WebCL.DEVICE_MAX_SAMPLERS);d[15]=c.getInfo(WebCL.DEVICE_MEM_BASE_ADDR_ALIGN);d[17]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHELINE_SIZE);d[18]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHE_SIZE);d[19]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_ARGS);d[20]=c.getInfo(WebCL.DEVICE_ENDIAN_LITTLE);d[21]=c.getInfo(WebCL.DEVICE_AVAILABLE);d[22]=c.getInfo(WebCL.DEVICE_COMPILER_AVAILABLE);d[23]=c.getInfo(WebCL.DEVICE_SINGLE_FP_CONFIG);d[24]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHE_TYPE);d[25]=c.getInfo(WebCL.DEVICE_QUEUE_PROPERTIES);d[26]=c.getInfo(WebCL.DEVICE_LOCAL_MEM_TYPE);d[28]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE);d[29]=c.getInfo(WebCL.DEVICE_EXECUTION_CAPABILITIES);d[31]=c.getInfo(WebCL.DEVICE_MAX_WORK_GROUP_SIZE);d[32]=c.getInfo(WebCL.DEVICE_IMAGE2D_MAX_HEIGHT);d[33]=c.getInfo(WebCL.DEVICE_IMAGE2D_MAX_WIDTH);d[34]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_DEPTH);d[35]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_HEIGHT);d[36]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_WIDTH);d[37]=c.getInfo(WebCL.DEVICE_MAX_PARAMETER_SIZE);d[38]=[0];var
n=c.getInfo(WebCL.DEVICE_MAX_WORK_ITEM_SIZES);d[38][1]=n[0];d[38][2]=n[1];d[38][3]=n[2];d[39]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);d[40]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);d[41]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_INT);d[42]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_LONG);d[43]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);d[45]=c.getInfo(WebCL.DEVICE_PROFILING_TIMER_RESOLUTION);d[46]=R(c.getInfo(WebCL.DRIVER_VERSION));g++;break}else
g++}else
g+=m}var
c=[0];d[1]=h;p[1]=d;c[1]=e;c[2]=p;return c}function
tD(){console.log(" spoc_getOpenCLDevicesCount");var
a=0,b=webcl.getPlatforms();for(var
d
in
b){var
e=b[d],c=e.getDevices();a+=c.length}return a}function
tE(){console.log(fj);return 0}function
tF(){console.log(fj);var
a=new
Array(3);a[0]=0;return a}function
dE(a){if(a[1]instanceof
Float32Array||a[1].constructor.name=="Float32Array")return 4;if(a[1]instanceof
Int32Array||a[1].constructor.name=="Int32Array")return 4;{console.log("unimplemented vector type");console.log(a[1].constructor.name);return 4}}function
tG(a,b,c){console.log("spoc_opencl_alloc_vect");var
f=a[2],i=a[4],h=i[b+1],j=a[5],k=dE(f),d=c[9],e=d[0],d=c[9],e=d[0],g=e.createBuffer(WebCL.MEM_READ_WRITE,j*k);h[2]=g;d[0]=e;c[9]=d;return 0}function
tH(){console.log(" spoc_opencl_compile");return 0}function
tI(a,b,c,d){console.log("spoc_opencl_cpu_to_device");var
f=a[2],k=a[4],j=k[b+1],l=a[5],m=dE(f),e=c[9],h=e[0],g=e[d+1],i=j[2];g.enqueueWriteBuffer(i,false,0,l*m,f[1]);e[d+1]=g;e[0]=h;c[9]=e;return 0}function
tO(a,b,c,d,e){console.log("spoc_opencl_device_to_cpu");var
g=a[2],l=a[4],k=l[b+1],n=a[5],o=dE(g),f=c[9],i=f[0],h=f[e+1],j=k[2],m=g[1];h.enqueueReadBuffer(j,false,0,n*o,m);f[e+1]=h;f[0]=i;c[9]=f;return 0}function
tP(a,b){console.log("spoc_opencl_flush");var
c=a[9][b+1];c.flush();a[9][b+1]=c;return 0}function
tQ(){console.log(" spoc_opencl_is_available");return!dD}function
tR(a,b,c,d,e){console.log("spoc_opencl_launch_grid");var
m=b[1],n=b[2],o=b[3],h=c[1],i=c[2],j=c[3],g=new
Array(3);g[0]=m*h;g[1]=n*i;g[2]=o*j;var
f=new
Array(3);f[0]=h;f[1]=i;f[2]=j;var
l=d[9],k=l[e+1];if(h==1&&i==1&&j==1)k.enqueueNDRangeKernel(a,f.length,null,g);else
k.enqueueNDRangeKernel(a,f.length,null,g,f);return 0}function
tU(a,b,c,d){console.log("spoc_opencl_load_param_int");b.setArg(a[1],new
Uint32Array([c]));a[1]=a[1]+1;return 0}function
tW(a,b,c,d,e){console.log("spoc_opencl_load_param_vec");var
f=d[2];b.setArg(a[1],f);a[1]=a[1]+1;return 0}function
tZ(){return new
Date().getTime()/ft}function
t0(){return 0}var
s=r7,l=r8,a8=dv,aF=gd,A=ge,at=sb,cT=se,bG=sf,a9=sh,Z=si,cX=sy,e$=sz,w=sB,e0=sE,cV=dy,cZ=sF,eY=sI,cU=sJ,aG=sM,x=bj,b=R,cY=sO,e3=sP,aE=sW,cW=sX,e2=sZ,bJ=gn,y=s0,bH=s5,eZ=s6,e1=s7,e_=s8,V=s9,C=s_,e9=ta,fa=tc,fc=te,fd=tB,fb=tD,e6=tE,e5=tF,e7=tG,e4=tP,bK=t0;function
j(a,b){return a.length==1?a(b):ag(a,[b])}function
i(a,b,c){return a.length==2?a(b,c):ag(a,[b,c])}function
o(a,b,c,d){return a.length==3?a(b,c,d):ag(a,[b,c,d])}function
e8(a,b,c,d,e,f,g){return a.length==6?a(b,c,d,e,f,g):ag(a,[b,c,d,e,f,g])}var
aM=[0,b("Failure")],bm=[0,b("Invalid_argument")],bn=[0,b("End_of_file")],t=[0,b("Not_found")],F=[0,b("Assert_failure")],cv=b(af),cy=b(af),cA=b(af),eI=b(g),eH=[0,b(fX),b(fR),b(fC),b(f5),b(f1)],eX=[0,1],eS=[0,b(f5),b(fR),b(fX),b(fC),b(f1)],eT=[0,b(dk),b(c2),b(c5),b(c7),b(db),b(c8),b(di),b(dq),b(c9),b(da)],eU=[0,b(dd),b(dn),b(df)],cR=[0,b(dn),b(c5),b(c7),b(df),b(dd),b(c2),b(dq),b(da),b(dk),b(c8),b(di),b(db),b(c9)];aE(6,t);aE(5,[0,b("Division_by_zero")]);aE(4,bn);aE(3,bm);aE(2,aM);aE(1,[0,b("Sys_error")]);var
gx=b("really_input"),gw=[0,0,[0,7,0]],gv=[0,1,[0,3,[0,4,[0,7,0]]]],gu=b(fP),gt=b(af),gr=b("true"),gs=b("false"),gy=b("Pervasives.do_at_exit"),gA=b("Array.blit"),gE=b("List.iter2"),gC=b("tl"),gB=b("hd"),gI=b("\\b"),gJ=b("\\t"),gK=b("\\n"),gL=b("\\r"),gH=b("\\\\"),gG=b("\\'"),gF=b("Char.chr"),gO=b("String.contains_from"),gN=b("String.blit"),gM=b("String.sub"),gX=b("Map.remove_min_elt"),gY=[0,0,0,0],gZ=[0,b("map.ml"),270,10],g0=[0,0,0],gT=b(bT),gU=b(bT),gV=b(bT),gW=b(bT),g1=b("CamlinternalLazy.Undefined"),g4=b("Buffer.add: cannot grow buffer"),hi=b(g),hj=b(g),hm=b(fP),hn=b(bZ),ho=b(bZ),hk=b(bW),hl=b(bW),hh=b(fq),hf=b("neg_infinity"),hg=b("infinity"),he=b(af),hd=b("printf: bad positional specification (0)."),hc=b("%_"),hb=[0,b("printf.ml"),143,8],g$=b(bW),ha=b("Printf: premature end of format string '"),g7=b(bW),g8=b(" in format string '"),g9=b(", at char number "),g_=b("Printf: bad conversion %"),g5=b("Sformat.index_of_int: negative argument "),hq=b(dj),hr=[0,987910699,495797812,364182224,414272206,318284740,990407751,383018966,270373319,840823159,24560019,536292337,512266505,189156120,730249596,143776328,51606627,140166561,366354223,1003410265,700563762,981890670,913149062,526082594,1021425055,784300257,667753350,630144451,949649812,48546892,415514493,258888527,511570777,89983870,283659902,308386020,242688715,482270760,865188196,1027664170,207196989,193777847,619708188,671350186,149669678,257044018,87658204,558145612,183450813,28133145,901332182,710253903,510646120,652377910,409934019,801085050],r1=b("OCAMLRUNPARAM"),rZ=b("CAMLRUNPARAM"),ht=b(g),hQ=[0,b("camlinternalOO.ml"),287,50],hP=b(g),hv=b("CamlinternalOO.last_id"),ii=b(g),ie=b(fE),id=b(".\\"),ic=b(fQ),ib=b("..\\"),h5=b(fE),h4=b(fQ),h0=b(g),hZ=b(g),h1=b(c6),h2=b(fp),rX=b("TMPDIR"),h7=b("/tmp"),h8=b("'\\''"),h$=b(c6),ia=b("\\"),rV=b("TEMP"),ig=b(af),il=b(c6),im=b(fp),iq=b("Cygwin"),ir=b(fl),is=b("Win32"),it=[0,b("filename.ml"),189,9],iA=b("E2BIG"),iC=b("EACCES"),iD=b("EAGAIN"),iE=b("EBADF"),iF=b("EBUSY"),iG=b("ECHILD"),iH=b("EDEADLK"),iI=b("EDOM"),iJ=b("EEXIST"),iK=b("EFAULT"),iL=b("EFBIG"),iM=b("EINTR"),iN=b("EINVAL"),iO=b("EIO"),iP=b("EISDIR"),iQ=b("EMFILE"),iR=b("EMLINK"),iS=b("ENAMETOOLONG"),iT=b("ENFILE"),iU=b("ENODEV"),iV=b("ENOENT"),iW=b("ENOEXEC"),iX=b("ENOLCK"),iY=b("ENOMEM"),iZ=b("ENOSPC"),i0=b("ENOSYS"),i1=b("ENOTDIR"),i2=b("ENOTEMPTY"),i3=b("ENOTTY"),i4=b("ENXIO"),i5=b("EPERM"),i6=b("EPIPE"),i7=b("ERANGE"),i8=b("EROFS"),i9=b("ESPIPE"),i_=b("ESRCH"),i$=b("EXDEV"),ja=b("EWOULDBLOCK"),jb=b("EINPROGRESS"),jc=b("EALREADY"),jd=b("ENOTSOCK"),je=b("EDESTADDRREQ"),jf=b("EMSGSIZE"),jg=b("EPROTOTYPE"),jh=b("ENOPROTOOPT"),ji=b("EPROTONOSUPPORT"),jj=b("ESOCKTNOSUPPORT"),jk=b("EOPNOTSUPP"),jl=b("EPFNOSUPPORT"),jm=b("EAFNOSUPPORT"),jn=b("EADDRINUSE"),jo=b("EADDRNOTAVAIL"),jp=b("ENETDOWN"),jq=b("ENETUNREACH"),jr=b("ENETRESET"),js=b("ECONNABORTED"),jt=b("ECONNRESET"),ju=b("ENOBUFS"),jv=b("EISCONN"),jw=b("ENOTCONN"),jx=b("ESHUTDOWN"),jy=b("ETOOMANYREFS"),jz=b("ETIMEDOUT"),jA=b("ECONNREFUSED"),jB=b("EHOSTDOWN"),jC=b("EHOSTUNREACH"),jD=b("ELOOP"),jE=b("EOVERFLOW"),jF=b("EUNKNOWNERR %d"),iB=b("Unix.Unix_error(Unix.%s, %S, %S)"),iw=b(fG),ix=b(g),iy=b(g),iz=b(fG),jG=b("0.0.0.0"),jH=b("127.0.0.1"),rU=b("::"),rT=b("::1"),jR=[0,b("Vector.ml"),fK,25],jS=b("Cuda.No_Cuda_Device"),jT=b("Cuda.ERROR_DEINITIALIZED"),jU=b("Cuda.ERROR_NOT_INITIALIZED"),jV=b("Cuda.ERROR_INVALID_CONTEXT"),jW=b("Cuda.ERROR_INVALID_VALUE"),jX=b("Cuda.ERROR_OUT_OF_MEMORY"),jY=b("Cuda.ERROR_INVALID_DEVICE"),jZ=b("Cuda.ERROR_NOT_FOUND"),j0=b("Cuda.ERROR_FILE_NOT_FOUND"),j1=b("Cuda.ERROR_UNKNOWN"),j2=b("Cuda.ERROR_LAUNCH_FAILED"),j3=b("Cuda.ERROR_LAUNCH_OUT_OF_RESOURCES"),j4=b("Cuda.ERROR_LAUNCH_TIMEOUT"),j5=b("Cuda.ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"),j6=b("no_cuda_device"),j7=b("cuda_error_deinitialized"),j8=b("cuda_error_not_initialized"),j9=b("cuda_error_invalid_context"),j_=b("cuda_error_invalid_value"),j$=b("cuda_error_out_of_memory"),ka=b("cuda_error_invalid_device"),kb=b("cuda_error_not_found"),kc=b("cuda_error_file_not_found"),kd=b("cuda_error_launch_failed"),ke=b("cuda_error_launch_out_of_resources"),kf=b("cuda_error_launch_timeout"),kg=b("cuda_error_launch_incompatible_texturing"),kh=b("cuda_error_unknown"),ki=b("OpenCL.No_OpenCL_Device"),kj=b("OpenCL.OPENCL_ERROR_UNKNOWN"),kk=b("OpenCL.INVALID_CONTEXT"),kl=b("OpenCL.INVALID_DEVICE"),km=b("OpenCL.INVALID_VALUE"),kn=b("OpenCL.INVALID_QUEUE_PROPERTIES"),ko=b("OpenCL.OUT_OF_RESOURCES"),kp=b("OpenCL.MEM_OBJECT_ALLOCATION_FAILURE"),kq=b("OpenCL.OUT_OF_HOST_MEMORY"),kr=b("OpenCL.FILE_NOT_FOUND"),ks=b("OpenCL.INVALID_PROGRAM"),kt=b("OpenCL.INVALID_BINARY"),ku=b("OpenCL.INVALID_BUILD_OPTIONS"),kv=b("OpenCL.INVALID_OPERATION"),kw=b("OpenCL.COMPILER_NOT_AVAILABLE"),kx=b("OpenCL.BUILD_PROGRAM_FAILURE"),ky=b("OpenCL.INVALID_KERNEL"),kz=b("OpenCL.INVALID_ARG_INDEX"),kA=b("OpenCL.INVALID_ARG_VALUE"),kB=b("OpenCL.INVALID_MEM_OBJECT"),kC=b("OpenCL.INVALID_SAMPLER"),kD=b("OpenCL.INVALID_ARG_SIZE"),kE=b("OpenCL.INVALID_COMMAND_QUEUE"),kF=b("no_opencl_device"),kG=b("opencl_error_unknown"),kH=b("opencl_invalid_context"),kI=b("opencl_invalid_device"),kJ=b("opencl_invalid_value"),kK=b("opencl_invalid_queue_properties"),kL=b("opencl_out_of_resources"),kM=b("opencl_mem_object_allocation_failure"),kN=b("opencl_out_of_host_memory"),kO=b("opencl_file_not_found"),kP=b("opencl_invalid_program"),kQ=b("opencl_invalid_binary"),kR=b("opencl_invalid_build_options"),kS=b("opencl_invalid_operation"),kT=b("opencl_compiler_not_available"),kU=b("opencl_build_program_failure"),kV=b("opencl_invalid_kernel"),kW=b("opencl_invalid_arg_index"),kX=b("opencl_invalid_arg_value"),kY=b("opencl_invalid_mem_object"),kZ=b("opencl_invalid_sampler"),k0=b("opencl_invalid_arg_size"),k1=b("opencl_invalid_command_queue"),k2=b(b4),k3=b(b4),li=b(fy),lh=b(fu),lg=b(fy),lf=b(fu),le=[0,1],ld=b(g),k$=b(bN),k6=b("Cl LOAD ARG Type Not Implemented\n"),k5=b("CU LOAD ARG Type Not Implemented\n"),k4=[0,b(da),b(c9),b(dq),b(di),b(c8),b(dd),b(db),b(c7),b(c5),b(dn),b(c2),b(dk),b(df)],k7=b("Kernel.ERROR_BLOCK_SIZE"),k9=b("Kernel.ERROR_GRID_SIZE"),la=b("Kernel.No_source_for_device"),ll=b("Empty"),lm=b("Unit"),ln=b("Kern"),lo=b("Params"),lp=b("Plus"),lq=b("Plusf"),lr=b("Min"),ls=b("Minf"),lt=b("Mul"),lu=b("Mulf"),lv=b("Div"),lw=b("Divf"),lx=b("Mod"),ly=b("Id "),lz=b("IdName "),lA=b("IntVar "),lB=b("FloatVar "),lC=b("UnitVar "),lD=b("CastDoubleVar "),lE=b("DoubleVar "),lF=b("IntArr"),lG=b("Int32Arr"),lH=b("Int64Arr"),lI=b("Float32Arr"),lJ=b("Float64Arr"),lK=b("VecVar "),lL=b("Concat"),lM=b("Seq"),lN=b("Return"),lO=b("Set"),lP=b("Decl"),lQ=b("SetV"),lR=b("SetLocalVar"),lS=b("Intrinsics"),lT=b(B),lU=b("IntId "),lV=b("Int "),lX=b("IntVecAcc"),lY=b("Local"),lZ=b("Acc"),l0=b("Ife"),l1=b("If"),l2=b("Or"),l3=b("And"),l4=b("EqBool"),l5=b("LtBool"),l6=b("GtBool"),l7=b("LtEBool"),l8=b("GtEBool"),l9=b("DoLoop"),l_=b("While"),l$=b("App"),ma=b("GInt"),mb=b("GFloat"),lW=b("Float "),lk=b("  "),lj=b("%s\n"),nP=b(fN),nQ=[0,b(dc),166,14],me=b(g),mf=b(bN),mg=b("\n}\n#ifdef __cplusplus\n}\n#endif"),mh=b(" ) {\n"),mi=b(g),mj=b(bM),ml=b(g),mk=b('#ifdef __cplusplus\nextern "C" {\n#endif\n\n__global__ void spoc_dummy ( '),mm=b(ad),mn=b(b5),mo=b(ae),mp=b(ad),mq=b(b5),mr=b(ae),ms=b(ad),mt=b(bU),mu=b(ae),mv=b(ad),mw=b(bU),mx=b(ae),my=b(ad),mz=b(bY),mA=b(ae),mB=b(ad),mC=b(bY),mD=b(ae),mE=b(ad),mF=b(b7),mG=b(ae),mH=b(ad),mI=b(b7),mJ=b(ae),mK=b(ad),mL=b(fo),mM=b(ae),mN=b(fT),mO=b(fn),mP=[0,b(dc),65,17],mQ=b(bV),mR=b(fz),mS=b(L),mT=b(M),mU=b(fF),mV=b(L),mW=b(M),mX=b(fi),mY=b(L),mZ=b(M),m0=b(fv),m1=b(L),m2=b(M),m3=b(fU),m4=b(fO),m6=b("int"),m7=b("float"),m8=b("double"),m5=[0,b(dc),60,12],m_=b(bM),m9=b(f_),m$=b(fM),na=b(g),nb=b(g),ne=b(bP),nf=b(ac),ng=b(aH),ni=b(bP),nh=b(ac),nj=b(X),nk=b(L),nl=b(M),nm=b("}\n"),nn=b(aH),no=b(aH),np=b("{"),nq=b(bf),nr=b(fs),ns=b(bf),nt=b(bc),nu=b(b6),nv=b(bf),nw=b(bc),nx=b(b6),ny=b(fg),nz=b(fe),nA=b(fx),nB=b(f0),nC=b(fD),nD=b(bL),nE=b(fZ),nF=b(b3),nG=b(fk),nH=b(bS),nI=b(bL),nJ=b(bS),nK=b(ac),nL=b(ff),nM=b(b3),nN=b(bc),nO=b(fB),nT=b(bX),nU=b(bX),nV=b(B),nW=b(B),nR=b(fJ),nS=b(ga),nX=b(X),nc=b(bP),nd=b(ac),nY=b(L),nZ=b(M),n1=b(bV),n2=b(X),n3=b(f7),n4=b(L),n5=b(M),n6=b(X),n0=b("cuda error parse_float"),mc=[0,b(g),b(g)],pq=b(fN),pr=[0,b(dh),162,14],n9=b(g),n_=b(bN),n$=b(b3),oa=b(" ) \n{\n"),ob=b(g),oc=b(bM),oe=b(g),od=b("__kernel void spoc_dummy ( "),of=b(b5),og=b(b5),oh=b(bU),oi=b(bU),oj=b(bY),ok=b(bY),ol=b(b7),om=b(b7),on=b(fo),oo=b(fT),op=b(fn),oq=[0,b(dh),65,17],or=b(bV),os=b(fz),ot=b(L),ou=b(M),ov=b(fF),ow=b(L),ox=b(M),oy=b(fi),oz=b(L),oA=b(M),oB=b(fv),oC=b(L),oD=b(M),oE=b(fU),oF=b(fO),oH=b("__global int"),oI=b("__global float"),oJ=b("__global double"),oG=[0,b(dh),60,12],oL=b(bM),oK=b(f_),oM=b(fM),oN=b(g),oO=b(g),oQ=b(bP),oR=b(ac),oS=b(aH),oT=b(ac),oU=b(X),oV=b(L),oW=b(M),oX=b(g),oY=b(bN),oZ=b(aH),o0=b(g),o1=b(bf),o2=b(fs),o3=b(bf),o4=b(bc),o5=b(b6),o6=b(b3),o7=b(aH),o8=b("{\n"),o9=b(")\n"),o_=b(b6),o$=b(fg),pa=b(fe),pb=b(fx),pc=b(f0),pd=b(fD),pe=b(bL),pf=b(fZ),pg=b(f2),ph=b(fk),pi=b(bS),pj=b(bL),pk=b(bS),pl=b(ac),pm=b(ff),pn=b(f2),po=b(bc),pp=b(fB),pu=b(bX),pv=b(bX),pw=b(B),px=b(B),ps=b(fJ),pt=b(ga),py=b(X),oP=b(ac),pz=b(L),pA=b(M),pC=b(bV),pD=b(X),pE=b(f7),pF=b(L),pG=b(M),pH=b(X),pB=b("opencl error parse_float"),n7=[0,b(g),b(g)],qG=[0,0],qH=[0,0],qI=[0,1],qJ=[0,1],qA=b("kirc_kernel.cu"),qB=b("nvcc -m64 -arch=sm_10 -O3 -ptx kirc_kernel.cu -o kirc_kernel.ptx"),qC=b("kirc_kernel.ptx"),qD=b("rm kirc_kernel.cu kirc_kernel.ptx"),qx=[0,b(g),b(g)],qz=b(g),qy=[0,b("Kirc.ml"),407,81],qE=b(ac),qF=b(fW),qu=[33,0],qp=b(fW),pI=b("int spoc_xor (int a, int b ) { return (a^b);}\n"),pJ=b("int spoc_powint (int a, int b ) { return ((int) pow (((float) a), ((float) b)));}\n"),pK=b("int logical_and (int a, int b ) { return (a & b);}\n"),pL=b("float spoc_fdiv ( float a, float b ) { return (a / b);}\n"),pM=b("float spoc_fmul ( float a, float b ) { return (a * b);}\n"),pN=b("float spoc_fminus ( float a, float b ) { return (a - b);}\n"),pO=b("float spoc_fadd ( float a, float b ) { return (a + b);}\n"),pP=b("float spoc_fdiv ( float a, float b );\n"),pQ=b("float spoc_fmul ( float a, float b );\n"),pR=b("float spoc_fminus ( float a, float b );\n"),pS=b("float spoc_fadd ( float a, float b );\n"),pU=b(dg),pV=b("double spoc_ddiv ( double a, double b ) { return (a / b);}\n"),pW=b("double spoc_dmul ( double a, double b ) { return (a * b);}\n"),pX=b("double spoc_dminus ( double a, double b ) { return (a - b);}\n"),pY=b("double spoc_dadd ( double a, double b ) { return (a + b);}\n"),pZ=b("double spoc_ddiv ( double a, double b );\n"),p0=b("double spoc_dmul ( double a, double b );\n"),p1=b("double spoc_dminus ( double a, double b );\n"),p2=b("double spoc_dadd ( double a, double b );\n"),p3=b(dg),p4=b("#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"),p5=b("#elif defined(cl_amd_fp64)  // AMD extension available?\n"),p6=b("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"),p7=b("#if defined(cl_khr_fp64)  // Khronos extension available?\n"),p8=b(f6),p9=b(fY),p$=b(dg),qa=b("__device__ double spoc_ddiv ( double a, double b ) { return (a / b);}\n"),qb=b("__device__ double spoc_dmul ( double a, double b ) { return (a * b);}\n"),qc=b("__device__ double spoc_dminus ( double a, double b ) { return (a - b);}\n"),qd=b("__device__ double spoc_dadd ( double a, double b ) { return (a + b);}\n"),qe=b(f6),qf=b(fY),qh=b("__device__ int spoc_xor (int a, int b ) { return (a^b);}\n"),qi=b("__device__ int spoc_powint (int a, int b ) { return ((int) pow (((double) a), ((double) b)));}\n"),qj=b("__device__ int logical_and (int a, int b ) { return (a & b);}\n"),qk=b("__device__ float spoc_fdiv ( float a, float b ) { return (a / b);}\n"),ql=b("__device__ float spoc_fmul ( float a, float b ) { return (a * b);}\n"),qm=b("__device__ float spoc_fminus ( float a, float b ) { return (a - b);}\n"),qn=b("__device__ float spoc_fadd ( float a, float b ) { return (a + b);}\n"),qv=[0,b(g),b(g)],qZ=b("canvas"),qW=b("span"),qV=b("img"),qU=b("br"),qT=b(fh),qS=b("select"),qR=b("option"),qX=b("Dom_html.Canvas_not_available"),rR=[0,b(f4),135,17],rO=b("Will use device : %s!"),rP=[0,1],rQ=b(g),rN=b("Time %s : %Fs\n%!"),q_=b("spoc_dummy"),q$=b("kirc_kernel"),q8=b("spoc_kernel_extension error"),q0=[0,b(f4),12,15],rB=b("(get_group_id (0))"),rC=b("blockIdx.x"),rE=b("(get_local_size (0))"),rF=b("blockDim.x"),rH=b("(get_local_id (0))"),rI=b("threadIdx.x");function
S(a){throw[0,aM,a]}function
E(a){throw[0,bm,a]}function
h(a,b){var
c=a.getLen(),e=b.getLen(),d=A(c+e|0);a8(a,0,d,0,c);a8(b,0,d,c,e);return d}function
k(a){return b(g+a)}function
N(a){var
c=cT(gu,a),b=0,f=c.getLen();for(;;){if(f<=b)var
e=h(c,gt);else{var
d=c.safeGet(b),g=48<=d?58<=d?0:1:45===d?1:0;if(g){var
b=b+1|0;continue}var
e=c}return e}}function
b_(a,b){if(a){var
c=a[1];return[0,c,b_(a[2],b)]}return b}eY(0);var
dF=cU(1);cU(2);function
dG(a,b){return gj(a,b,0,b.getLen())}function
dH(a){return eY(eZ(a,gw,0))}function
dI(a){var
b=sK(0);for(;;){if(b){var
c=b[2],d=b[1];try{cV(d)}catch(f){}var
b=c;continue}return 0}}cW(gy,dI);function
dJ(a){return e0(a)}function
gz(a,b){return sL(a,b)}function
dK(a){return cV(a)}function
dL(a,b){var
d=b.length-1-1|0,e=0;if(!(d<0)){var
c=e;for(;;){j(a,b[c+1]);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return 0}function
aN(a,b){var
d=b.length-1;if(0===d)return[0];var
e=w(d,j(a,b[0+1])),f=d-1|0,g=1;if(!(f<1)){var
c=g;for(;;){e[c+1]=j(a,b[c+1]);var
h=c+1|0;if(f!==c){var
c=h;continue}break}}return e}function
b$(a,b){var
d=b.length-1-1|0,e=0;if(!(d<0)){var
c=e;for(;;){i(a,c,b[c+1]);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return 0}function
aO(a){var
b=a.length-1-1|0,c=0;for(;;){if(0<=b){var
d=[0,a[b+1],c],b=b-1|0,c=d;continue}return c}}function
dM(a,b,c){var
e=[0,b],f=c.length-1-1|0,g=0;if(!(f<0)){var
d=g;for(;;){e[1]=i(a,e[1],c[d+1]);var
h=d+1|0;if(f!==d){var
d=h;continue}break}}return e[1]}function
dN(a){var
b=a,c=0;for(;;){if(b){var
d=[0,b[1],c],b=b[2],c=d;continue}return c}}function
ca(a,b){if(b){var
c=b[2],d=j(a,b[1]);return[0,d,ca(a,c)]}return 0}function
cc(a,b,c){if(b){var
d=b[1];return i(a,d,cc(a,b[2],c))}return c}function
dP(a,b,c){var
e=b,d=c;for(;;){if(e){if(d){var
f=d[2],g=e[2];i(a,e[1],d[1]);var
e=g,d=f;continue}}else
if(!d)return 0;return E(gE)}}function
cd(a,b){var
c=b;for(;;){if(c){var
e=c[2],d=0===aF(c[1],a)?1:0;if(d)return d;var
c=e;continue}return 0}}function
ce(a){if(0<=a)if(!(r<a))return a;return E(gF)}function
dQ(a){var
b=65<=a?90<a?0:1:0;if(!b){var
c=192<=a?214<a?0:1:0;if(!c){var
d=216<=a?222<a?1:0:1;if(d)return a}}return a+32|0}function
ah(a,b){var
c=A(a);sd(c,0,a,b);return c}function
u(a,b,c){if(0<=b)if(0<=c)if(!((a.getLen()-c|0)<b)){var
d=A(c);a8(a,b,d,0,c);return d}return E(gM)}function
bp(a,b,c,d,e){if(0<=e)if(0<=b)if(!((a.getLen()-e|0)<b))if(0<=d)if(!((c.getLen()-e|0)<d))return a8(a,b,c,d,e);return E(gN)}function
dR(a){var
c=a.getLen();if(0===c)var
f=a;else{var
d=A(c),e=c-1|0,g=0;if(!(e<0)){var
b=g;for(;;){d.safeSet(b,dQ(a.safeGet(b)));var
h=b+1|0;if(e!==b){var
b=h;continue}break}}var
f=d}return f}var
cg=s4(0)[1],ax=s1(0),ch=(1<<(ax-10|0))-1|0,aP=x(ax/8|0,ch)-1|0,gQ=s3(0)[2],gR=bR,gS=aJ;function
ci(k){function
h(a){return a?a[5]:0}function
e(a,b,c,d){var
e=h(a),f=h(d),g=f<=e?e+1|0:f+1|0;return[0,a,b,c,d,g]}function
q(a,b){return[0,0,a,b,0,1]}function
f(a,b,c,d){var
i=a?a[5]:0,j=d?d[5]:0;if((j+2|0)<i){if(a){var
f=a[4],m=a[3],n=a[2],k=a[1],q=h(f);if(q<=h(k))return e(k,n,m,e(f,b,c,d));if(f){var
r=f[3],s=f[2],t=f[1],u=e(f[4],b,c,d);return e(e(k,n,m,t),s,r,u)}return E(gT)}return E(gU)}if((i+2|0)<j){if(d){var
l=d[4],o=d[3],p=d[2],g=d[1],v=h(g);if(v<=h(l))return e(e(a,b,c,g),p,o,l);if(g){var
w=g[3],x=g[2],y=g[1],z=e(g[4],p,o,l);return e(e(a,b,c,y),x,w,z)}return E(gV)}return E(gW)}var
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
c=a[4],d=a[3],e=a[2];return f(s(b),e,d,c)}return a[4]}return E(gX)}function
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
f=d[4],g=d[3],h=d[2],i=o(a,h,g,z(a,d[1],e)),d=f,e=i;continue}return e}}function
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
p(a,b){if(a){if(b){var
c=n(b),d=c[2],e=c[1];return g(a,e,d,s(b))}return a}return b}function
G(a,b,c,d){return c?g(a,b,c[1],d):p(a,d)}function
l(a,b){if(b){var
c=b[4],d=b[3],e=b[2],f=b[1],m=i(k[1],a,e);if(0===m)return[0,f,[0,d],c];if(0<=m){var
h=l(a,c),n=h[3],o=h[2];return[0,g(f,e,d,h[1]),o,n]}var
j=l(a,f),p=j[2],q=j[1];return[0,q,p,g(j[3],e,d,c)]}return gY}function
m(a,b,c){if(b){var
d=b[2],i=b[5],j=b[4],k=b[3],n=b[1];if(h(c)<=i){var
e=l(d,c),p=e[2],q=e[1],r=m(a,j,e[3]),s=o(a,d,[0,k],p);return G(m(a,n,q),d,s,r)}}else
if(!c)return 0;if(c){var
f=c[2],t=c[4],u=c[3],v=c[1],g=l(f,b),w=g[2],x=g[1],y=m(a,g[3],t),z=o(a,f,w,[0,u]);return G(m(a,x,v),f,z,y)}throw[0,F,gZ]}function
w(a,b){if(b){var
c=b[3],d=b[2],h=b[4],e=w(a,b[1]),j=i(a,d,c),f=w(a,h);return j?g(e,d,c,f):p(e,f)}return 0}function
x(a,b){if(b){var
c=b[3],d=b[2],m=b[4],e=x(a,b[1]),f=e[2],h=e[1],n=i(a,d,c),j=x(a,m),k=j[2],l=j[1];if(n){var
o=p(f,k);return[0,g(h,d,c,l),o]}var
q=g(f,d,c,k);return[0,p(h,l),q]}return g0}function
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
H(a,b){var
d=a,c=b;for(;;){if(c){var
e=c[3],f=c[2],g=c[1],d=[0,[0,f,e],H(d,c[4])],c=g;continue}return d}}return[0,a,I,K,r,q,u,m,M,N,y,z,A,B,w,x,b,function(a){return H(0,a)},n,L,n,l,J,c,v]}var
g2=[0,g1];function
g3(a){throw[0,g2]}function
aQ(a){var
b=1<=a?a:1,c=aP<b?aP:b,d=A(c);return[0,d,0,c,d]}function
aR(a){return u(a[1],0,a[2])}function
dU(a,b){var
c=[0,a[3]];for(;;){if(c[1]<(a[2]+b|0)){c[1]=2*c[1]|0;continue}if(aP<c[1])if((a[2]+b|0)<=aP)c[1]=aP;else
S(g4);var
d=A(c[1]);bp(a[1],0,d,0,a[2]);a[1]=d;a[3]=c[1];return 0}}function
G(a,b){var
c=a[2];if(a[3]<=c)dU(a,1);a[1].safeSet(c,b);a[2]=c+1|0;return 0}function
br(a,b){var
c=b.getLen(),d=a[2]+c|0;if(a[3]<d)dU(a,c);bp(b,0,a[1],a[2],c);a[2]=d;return 0}function
cj(a){return 0<=a?a:S(h(g5,k(a)))}function
dV(a,b){return cj(a+b|0)}var
g6=1;function
dW(a){return dV(g6,a)}function
dX(a){return u(a,0,a.getLen())}function
dY(a,b,c){var
d=h(g8,h(a,g7)),e=h(g9,h(k(b),d));return E(h(g_,h(ah(1,c),e)))}function
aS(a,b,c){return dY(dX(a),b,c)}function
bs(a){return E(h(ha,h(dX(a),g$)))}function
ap(e,b,c,d){function
h(a){if((e.safeGet(a)+aK|0)<0||9<(e.safeGet(a)+aK|0))return a;var
b=a+1|0;for(;;){var
c=e.safeGet(b);if(48<=c){if(!(58<=c)){var
b=b+1|0;continue}var
d=0}else
if(36===c){var
f=b+1|0,d=1}else
var
d=0;if(!d)var
f=a;return f}}var
i=h(b+1|0),f=aQ((c-i|0)+10|0);G(f,37);var
a=i,g=dN(d);for(;;){if(a<=c){var
j=e.safeGet(a);if(42===j){if(g){var
l=g[2];br(f,k(g[1]));var
a=h(a+1|0),g=l;continue}throw[0,F,hb]}G(f,j);var
a=a+1|0;continue}return aR(f)}}function
dZ(a,b,c,d,e){var
f=ap(b,c,d,e);if(78!==a)if(be!==a)return f;f.safeSet(f.getLen()-1|0,dl);return f}function
d0(a){return function(c,b){var
m=c.getLen();function
n(a,b){var
o=40===a?41:c4;function
k(a){var
d=a;for(;;){if(m<=d)return bs(c);if(37===c.safeGet(d)){var
e=d+1|0;if(m<=e)var
f=bs(c);else{var
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
f=g===o?e+1|0:aS(c,b,g);break;case
2:break;default:var
f=k(n(g,e+1|0)+1|0)}}return f}var
d=d+1|0;continue}}return k(b)}return n(a,b)}}function
d1(j,b,c){var
m=j.getLen()-1|0;function
s(a){var
l=a;a:for(;;){if(l<m){if(37===j.safeGet(l)){var
e=0,h=l+1|0;for(;;){if(m<h)var
w=bs(j);else{var
n=j.safeGet(h);if(58<=n){if(95===n){var
e=1,h=h+1|0;continue}}else
if(32<=n)switch(n+fr|0){case
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
h=o(b,e,h,av);continue;default:var
h=h+1|0;continue}var
d=h;b:for(;;){if(m<d)var
f=bs(j);else{var
k=j.safeGet(d);if(fK<=k)var
g=0;else
switch(k){case
78:case
88:case
aI:case
av:case
c_:case
dl:case
dm:var
f=o(b,e,d,av),g=1;break;case
69:case
70:case
71:case
f3:case
c$:case
dr:var
f=o(b,e,d,c$),g=1;break;case
33:case
37:case
44:case
64:var
f=d+1|0,g=1;break;case
83:case
91:case
bh:var
f=o(b,e,d,bh),g=1;break;case
97:case
b0:case
c3:var
f=o(b,e,d,k),g=1;break;case
76:case
fA:case
be:var
t=d+1|0;if(m<t){var
f=o(b,e,d,av),g=1}else{var
q=j.safeGet(t)+f8|0;if(q<0||32<q)var
r=1;else
switch(q){case
0:case
12:case
17:case
23:case
29:case
32:var
f=i(c,o(b,e,d,k),av),g=1,r=0;break;default:var
r=1}if(r){var
f=o(b,e,d,av),g=1}}break;case
67:case
99:var
f=o(b,e,d,99),g=1;break;case
66:case
98:var
f=o(b,e,d,66),g=1;break;case
41:case
c4:var
f=o(b,e,d,k),g=1;break;case
40:var
f=s(o(b,e,d,k)),g=1;break;case
dp:var
u=o(b,e,d,k),v=i(d0(k),j,u),p=u;for(;;){if(p<(v-2|0)){var
p=i(c,p,j.safeGet(p));continue}var
d=v-1|0;continue b}default:var
g=0}if(!g)var
f=aS(j,d,k)}var
w=f;break}}var
l=w;continue a}}var
l=l+1|0;continue}return l}}s(0);return 0}function
d2(a){var
d=[0,0,0,0];function
b(a,b,c){var
f=41!==c?1:0,g=f?c4!==c?1:0:f;if(g){var
e=97===c?2:1;if(b0===c)d[3]=d[3]+1|0;if(a)d[2]=d[2]+e|0;else
d[1]=d[1]+e|0}return b+1|0}d1(a,b,function(a,b){return a+1|0});return d[1]}function
d3(a,b,c){var
h=a.safeGet(c);if((h+aK|0)<0||9<(h+aK|0))return i(b,0,c);var
e=h+aK|0,d=c+1|0;for(;;){var
f=a.safeGet(d);if(48<=f){if(!(58<=f)){var
e=(10*e|0)+(f+aK|0)|0,d=d+1|0;continue}var
g=0}else
if(36===f)if(0===e){var
j=S(hd),g=1}else{var
j=i(b,[0,cj(e-1|0)],d+1|0),g=1}else
var
g=0;if(!g)var
j=i(b,0,c);return j}}function
O(a,b){return a?b:dW(b)}function
d4(a,b){return a?a[1]:b}function
d5(aJ,b,c,d,e,f,g){var
D=j(b,g);function
af(a){return i(d,D,a)}function
aK(a,b,m,aL){var
k=m.getLen();function
E(l,b){var
p=b;for(;;){if(k<=p)return j(a,D);var
d=m.safeGet(p);if(37===d){var
o=function(a,b){return s(aL,d4(a,b))},au=function(g,f,c,d){var
a=d;for(;;){var
aa=m.safeGet(a)+fr|0;if(!(aa<0||25<aa))switch(aa){case
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
10:return d3(m,function(a,b){var
d=[0,o(a,f),c];return au(g,O(a,f),d,b)},a+1|0);default:var
a=a+1|0;continue}var
q=m.safeGet(a);if(124<=q)var
k=0;else
switch(q){case
78:case
88:case
aI:case
av:case
c_:case
dl:case
dm:var
a8=o(g,f),a9=bG(dZ(q,m,p,a,c),a8),l=r(O(g,f),a9,a+1|0),k=1;break;case
69:case
71:case
f3:case
c$:case
dr:var
a1=o(g,f),a2=cT(ap(m,p,a,c),a1),l=r(O(g,f),a2,a+1|0),k=1;break;case
76:case
fA:case
be:var
ad=m.safeGet(a+1|0)+f8|0;if(ad<0||32<ad)var
ag=1;else
switch(ad){case
0:case
12:case
17:case
23:case
29:case
32:var
U=a+1|0,ae=q-108|0;if(ae<0||2<ae)var
ai=0;else{switch(ae){case
1:var
ai=0,aj=0;break;case
2:var
a7=o(g,f),aB=bG(ap(m,p,U,c),a7),aj=1;break;default:var
a6=o(g,f),aB=bG(ap(m,p,U,c),a6),aj=1}if(aj){var
aA=aB,ai=1}}if(!ai){var
a5=o(g,f),aA=sn(ap(m,p,U,c),a5)}var
l=r(O(g,f),aA,U+1|0),k=1,ag=0;break;default:var
ag=1}if(ag){var
a3=o(g,f),a4=bG(dZ(be,m,p,a,c),a3),l=r(O(g,f),a4,a+1|0),k=1}break;case
37:case
64:var
l=r(f,ah(1,q),a+1|0),k=1;break;case
83:case
bh:var
y=o(g,f);if(bh===q)var
z=y;else{var
b=[0,0],an=y.getLen()-1|0,aN=0;if(!(an<0)){var
M=aN;for(;;){var
x=y.safeGet(M),bd=14<=x?34===x?1:92===x?1:0:11<=x?13<=x?1:0:8<=x?1:0,aT=bd?2:cX(x)?1:4;b[1]=b[1]+aT|0;var
aU=M+1|0;if(an!==M){var
M=aU;continue}break}}if(b[1]===y.getLen())var
aD=y;else{var
n=A(b[1]);b[1]=0;var
ao=y.getLen()-1|0,aO=0;if(!(ao<0)){var
L=aO;for(;;){var
w=y.safeGet(L),B=w-34|0;if(B<0||58<B)if(-20<=B)var
V=1;else{switch(B+34|0){case
8:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],98);var
K=1;break;case
9:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],c3);var
K=1;break;case
10:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],be);var
K=1;break;case
13:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],b0);var
K=1;break;default:var
V=1,K=0}if(K)var
V=0}else
var
V=(B-1|0)<0||56<(B-1|0)?(n.safeSet(b[1],92),b[1]++,n.safeSet(b[1],w),0):1;if(V)if(cX(w))n.safeSet(b[1],w);else{n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],48+(w/aI|0)|0);b[1]++;n.safeSet(b[1],48+((w/10|0)%10|0)|0);b[1]++;n.safeSet(b[1],48+(w%10|0)|0)}b[1]++;var
aP=L+1|0;if(ao!==L){var
L=aP;continue}break}}var
aD=n}var
z=h(ho,h(aD,hn))}if(a===(p+1|0))var
aC=z;else{var
J=ap(m,p,a,c);try{var
W=0,t=1;for(;;){if(J.getLen()<=t)var
aq=[0,0,W];else{var
X=J.safeGet(t);if(49<=X)if(58<=X)var
ak=0;else{var
aq=[0,sx(u(J,t,(J.getLen()-t|0)-1|0)),W],ak=1}else{if(45===X){var
W=1,t=t+1|0;continue}var
ak=0}if(!ak){var
t=t+1|0;continue}}var
Z=aq;break}}catch(f){if(f[1]!==aM)throw f;var
Z=dY(J,0,bh)}var
N=Z[1],C=z.getLen(),aV=Z[2],P=0,aW=32;if(N===C)if(0===P){var
_=z,al=1}else
var
al=0;else
var
al=0;if(!al)if(N<=C)var
_=u(z,P,C);else{var
Y=ah(N,aW);if(aV)bp(z,P,Y,0,C);else
bp(z,P,Y,N-C|0,C);var
_=Y}var
aC=_}var
l=r(O(g,f),aC,a+1|0),k=1;break;case
67:case
99:var
s=o(g,f);if(99===q)var
ay=ah(1,s);else{if(39===s)var
v=gG;else
if(92===s)var
v=gH;else{if(14<=s)var
F=0;else
switch(s){case
8:var
v=gI,F=1;break;case
9:var
v=gJ,F=1;break;case
10:var
v=gK,F=1;break;case
13:var
v=gL,F=1;break;default:var
F=0}if(!F)if(cX(s)){var
am=A(1);am.safeSet(0,s);var
v=am}else{var
H=A(4);H.safeSet(0,92);H.safeSet(1,48+(s/aI|0)|0);H.safeSet(2,48+((s/10|0)%10|0)|0);H.safeSet(3,48+(s%10|0)|0);var
v=H}}var
ay=h(hl,h(v,hk))}var
l=r(O(g,f),ay,a+1|0),k=1;break;case
66:case
98:var
aZ=a+1|0,a0=o(g,f)?gr:gs,l=r(O(g,f),a0,aZ),k=1;break;case
40:case
dp:var
T=o(g,f),aw=i(d0(q),m,a+1|0);if(dp===q){var
Q=aQ(T.getLen()),ar=function(a,b){G(Q,b);return a+1|0};d1(T,function(a,b,c){if(a)br(Q,hc);else
G(Q,37);return ar(b,c)},ar);var
aX=aR(Q),l=r(O(g,f),aX,aw),k=1}else{var
ax=O(g,f),bc=dV(d2(T),ax),l=aK(function(a){return E(bc,aw)},ax,T,aL),k=1}break;case
33:j(e,D);var
l=E(f,a+1|0),k=1;break;case
41:var
l=r(f,hi,a+1|0),k=1;break;case
44:var
l=r(f,hj,a+1|0),k=1;break;case
70:var
ab=o(g,f);if(0===c)var
az=hm;else{var
$=ap(m,p,a,c);if(70===q)$.safeSet($.getLen()-1|0,dr);var
az=$}var
at=sa(ab);if(3===at)var
ac=ab<0?hf:hg;else
if(4<=at)var
ac=hh;else{var
S=cT(az,ab),R=0,aY=S.getLen();for(;;){if(aY<=R)var
as=h(S,he);else{var
I=S.safeGet(R)-46|0,bf=I<0||23<I?55===I?1:0:(I-1|0)<0||21<(I-1|0)?1:0;if(!bf){var
R=R+1|0;continue}var
as=S}var
ac=as;break}}var
l=r(O(g,f),ac,a+1|0),k=1;break;case
91:var
l=aS(m,a,q),k=1;break;case
97:var
aE=o(g,f),aF=dW(d4(g,f)),aG=o(0,aF),a_=a+1|0,a$=O(g,aF);if(aJ)af(i(aE,0,aG));else
i(aE,D,aG);var
l=E(a$,a_),k=1;break;case
b0:var
l=aS(m,a,q),k=1;break;case
c3:var
aH=o(g,f),ba=a+1|0,bb=O(g,f);if(aJ)af(j(aH,0));else
j(aH,D);var
l=E(bb,ba),k=1;break;default:var
k=0}if(!k)var
l=aS(m,a,q);return l}},f=p+1|0,g=0;return d3(m,function(a,b){return au(a,l,g,b)},f)}i(c,D,d);var
p=p+1|0;continue}}function
r(a,b,c){af(b);return E(a,c)}return E(b,0)}var
o=cj(0);function
k(a,b){return aK(f,o,a,b)}var
m=d2(g);if(m<0||6<m){var
n=function(f,b){if(m<=f){var
h=w(m,0),i=function(a,b){return l(h,(m-a|0)-1|0,b)},c=0,a=b;for(;;){if(a){var
d=a[2],e=a[1];if(d){i(c,e);var
c=c+1|0,a=d;continue}i(c,e)}return k(g,h)}}return function(a){return n(f+1|0,[0,a,b])}},a=n(0,0)}else
switch(m){case
1:var
a=function(a){var
b=w(1,0);l(b,0,a);return k(g,b)};break;case
2:var
a=function(a,b){var
c=w(2,0);l(c,0,a);l(c,1,b);return k(g,c)};break;case
3:var
a=function(a,b,c){var
d=w(3,0);l(d,0,a);l(d,1,b);l(d,2,c);return k(g,d)};break;case
4:var
a=function(a,b,c,d){var
e=w(4,0);l(e,0,a);l(e,1,b);l(e,2,c);l(e,3,d);return k(g,e)};break;case
5:var
a=function(a,b,c,d,e){var
f=w(5,0);l(f,0,a);l(f,1,b);l(f,2,c);l(f,3,d);l(f,4,e);return k(g,f)};break;case
6:var
a=function(a,b,c,d,e,f){var
h=w(6,0);l(h,0,a);l(h,1,b);l(h,2,c);l(h,3,d);l(h,4,e);l(h,5,f);return k(g,h)};break;default:var
a=k(g,[0])}return a}function
d6(a){function
b(a){return 0}return d5(0,function(a){return dF},gz,dG,dK,b,a)}function
hp(a){return aQ(2*a.getLen()|0)}function
d7(c){function
b(a){var
b=aR(a);a[2]=0;return j(c,b)}function
d(a){return 0}var
e=1;return function(a){return d5(e,hp,G,br,d,b,a)}}function
d8(a){return j(d7(function(a){return a}),a)}var
d9=[0,0];function
d_(a){d9[1]=[0,a,d9[1]];return 0}function
d$(a,b){var
j=0===b.length-1?[0,0]:b,f=j.length-1,p=0,q=54;if(!(54<0)){var
d=p;for(;;){l(a[1],d,d);var
w=d+1|0;if(q!==d){var
d=w;continue}break}}var
g=[0,hq],m=0,r=55,t=sj(55,f)?r:f,n=54+t|0;if(!(n<m)){var
c=m;for(;;){var
o=c%55|0,u=g[1],i=h(u,k(s(j,aG(c,f))));g[1]=sC(i,0,i.getLen());var
e=g[1];l(a[1],o,(s(a[1],o)^(((e.safeGet(0)+(e.safeGet(1)<<8)|0)+(e.safeGet(2)<<16)|0)+(e.safeGet(3)<<24)|0))&bd);var
v=c+1|0;if(n!==c){var
c=v;continue}break}}a[2]=0;return 0}32===ax;var
hs=[0,hr.slice(),0];try{var
r2=bH(r1),ck=r2}catch(f){if(f[1]!==t)throw f;try{var
r0=bH(rZ),ea=r0}catch(f){if(f[1]!==t)throw f;var
ea=ht}var
ck=ea}var
dS=ck.getLen(),hu=82,dT=0;if(0<=0)if(dS<dT)var
bI=0;else
try{var
bq=dT;for(;;){if(dS<=bq)throw[0,t];if(ck.safeGet(bq)!==hu){var
bq=bq+1|0;continue}var
gP=1,cf=gP,bI=1;break}}catch(f){if(f[1]!==t)throw f;var
cf=0,bI=1}else
var
bI=0;if(!bI)var
cf=E(gO);var
ai=[fH,function(a){var
b=[0,w(55,0),0];d$(b,e1(0));return b}];function
eb(a,b){var
m=a?a[1]:cf,d=16;for(;;){if(!(b<=d))if(!(ch<(d*2|0))){var
d=d*2|0;continue}if(m){var
h=sR(ai);if(aJ===h)var
c=ai[1];else
if(fH===h){var
k=ai[0+1];ai[0+1]=g3;try{var
e=j(k,0);ai[0+1]=e;sQ(ai,gS)}catch(f){ai[0+1]=function(a){throw f};throw f}var
c=e}else
var
c=ai;c[2]=(c[2]+1|0)%55|0;var
f=s(c[1],c[2]),g=(s(c[1],(c[2]+24|0)%55|0)+(f^f>>>25&31)|0)&bd;l(c[1],c[2],g);var
i=g}else
var
i=0;return[0,0,w(d,0),i,d]}}function
cl(a,b){return 3<=a.length-1?sk(10,aI,a[3],b)&(a[2].length-1-1|0):aG(sl(10,aI,b),a[2].length-1)}function
bt(a,b){var
i=cl(a,b),d=s(a[2],i);if(d){var
e=d[3],j=d[2];if(0===aF(b,d[1]))return j;if(e){var
f=e[3],k=e[2];if(0===aF(b,e[1]))return k;if(f){var
l=f[3],m=f[2];if(0===aF(b,f[1]))return m;var
c=l;for(;;){if(c){var
g=c[3],h=c[2];if(0===aF(b,c[1]))return h;var
c=g;continue}throw[0,t]}}throw[0,t]}throw[0,t]}throw[0,t]}function
a(a,b){return cW(a,b[0+1])}var
cm=[0,0];cW(hv,cm);var
hw=2;function
hx(a){var
b=[0,0],d=a.getLen()-1|0,e=0;if(!(d<0)){var
c=e;for(;;){b[1]=(223*b[1]|0)+a.safeGet(c)|0;var
g=c+1|0;if(d!==c){var
c=g;continue}break}}b[1]=b[1]&((1<<31)-1|0);var
f=bd<b[1]?b[1]-(1<<31)|0:b[1];return f}var
$=ci([0,function(a,b){return e2(a,b)}]),aq=ci([0,function(a,b){return e2(a,b)}]),aj=ci([0,function(a,b){return gi(a,b)}]),ec=e3(0,0),hy=[0,0];function
ed(a){return 2<a?ed((a+1|0)/2|0)*2|0:a}function
ee(a){hy[1]++;var
c=a.length-1,d=w((c*2|0)+2|0,ec);l(d,0,c);l(d,1,(x(ed(c),ax)/8|0)-1|0);var
e=c-1|0,f=0;if(!(e<0)){var
b=f;for(;;){l(d,(b*2|0)+3|0,s(a,b));var
g=b+1|0;if(e!==b){var
b=g;continue}break}}return[0,hw,d,aq[1],aj[1],0,0,$[1],0]}function
cn(a,b){var
c=a[2].length-1,g=c<b?1:0;if(g){var
d=w(b,ec),h=a[2],e=0,f=0,j=0<=c?0<=f?(h.length-1-c|0)<f?0:0<=e?(d.length-1-c|0)<e?0:(r5(h,f,d,e,c),1):0:0:0;if(!j)E(gA);a[2]=d;var
i=0}else
var
i=g;return i}var
ef=[0,0],hz=[0,0];function
co(a){var
b=a[2].length-1;cn(a,b+1|0);return b}function
aT(a,b){try{var
d=i(aq[22],b,a[3])}catch(f){if(f[1]===t){var
c=co(a);a[3]=o(aq[4],b,c,a[3]);a[4]=o(aj[4],c,1,a[4]);return c}throw f}return d}function
cq(a){return a===0?0:aO(a)}function
el(a,b){try{var
d=i($[22],b,a[7])}catch(f){if(f[1]===t){var
c=a[1];a[1]=c+1|0;if(y(b,hP))a[7]=o($[4],b,c,a[7]);return c}throw f}return d}function
cs(a){return sc(a,0)?[0]:a}function
en(a,b){if(a)return a;var
c=e3(gR,b[1]);c[0+1]=b[2];var
d=cm[1];c[1+1]=d;cm[1]=d+1|0;return c}function
bu(a){var
b=co(a);if(0===(b%2|0))var
d=0;else
if((2+at(s(a[2],1)*16|0,ax)|0)<b)var
d=0;else{var
c=co(a),d=1}if(!d)var
c=b;l(a[2],c,0);return c}function
eo(a,ap){var
g=[0,0],aq=ap.length-1;for(;;){if(g[1]<aq){var
k=s(ap,g[1]),e=function(a){g[1]++;return s(ap,g[1])},n=e(0);if(typeof
n===m)switch(n){case
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
Y=e(0),_=e(0),f=function(Y,_){return function(a){return i(a[1][Y+1],a,_)}}(Y,_);break;case
17:var
$=e(0),aa=e(0),f=function($,aa){return function(a){return i(a[1][$+1],a,a[aa+1])}}($,aa);break;case
18:var
ab=e(0),ac=e(0),ad=e(0),f=function(ab,ac,ad){return function(a){return i(a[1][ab+1],a,a[ac+1][ad+1])}}(ab,ac,ad);break;case
19:var
ae=e(0),af=e(0),f=function(ae,af){return function(a){var
b=j(a[1][af+1],a);return i(a[1][ae+1],a,b)}}(ae,af);break;case
20:var
ag=e(0),h=e(0);bu(a);var
f=function(ag,h){return function(a){return j(Z(h,ag,0),h)}}(ag,h);break;case
21:var
ah=e(0),ai=e(0);bu(a);var
f=function(ah,ai){return function(a){var
b=a[ai+1];return j(Z(b,ah,0),b)}}(ah,ai);break;case
22:var
ak=e(0),al=e(0),am=e(0);bu(a);var
f=function(ak,al,am){return function(a){var
b=a[al+1][am+1];return j(Z(b,ak,0),b)}}(ak,al,am);break;case
23:var
an=e(0),ao=e(0);bu(a);var
f=function(an,ao){return function(a){var
b=j(a[1][ao+1],a);return j(Z(b,an,0),b)}}(an,ao);break;default:var
o=e(0),f=function(o){return function(a){return o}}(o)}else
var
f=n;hz[1]++;if(i(aj[22],k,a[4])){cn(a,k+1|0);l(a[2],k,f)}else
a[6]=[0,[0,k,f],a[6]];g[1]++;continue}return 0}}function
ct(a,b,c){if(bJ(c,hZ))return b;var
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
cu(a,b,c){if(bJ(c,h0))return b;var
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
cw(a,b){return 47===a.safeGet(b)?1:0}function
ep(a){var
b=a.getLen()<1?1:0,c=b||(47!==a.safeGet(0)?1:0);return c}function
h3(a){var
c=ep(a);if(c){var
e=a.getLen()<2?1:0,d=e||y(u(a,0,2),h5);if(d){var
f=a.getLen()<3?1:0,b=f||y(u(a,0,3),h4)}else
var
b=d}else
var
b=c;return b}function
h6(a,b){var
c=b.getLen()<=a.getLen()?1:0,d=c?bJ(u(a,a.getLen()-b.getLen()|0,b.getLen()),b):c;return d}try{var
rY=bH(rX),cx=rY}catch(f){if(f[1]!==t)throw f;var
cx=h7}function
eq(a){var
d=a.getLen(),b=aQ(d+20|0);G(b,39);var
e=d-1|0,f=0;if(!(e<0)){var
c=f;for(;;){if(39===a.safeGet(c))br(b,h8);else
G(b,a.safeGet(c));var
g=c+1|0;if(e!==c){var
c=g;continue}break}}G(b,39);return aR(b)}function
h9(a){return ct(cw,cv,a)}function
h_(a){return cu(cw,cv,a)}function
az(a,b){var
c=a.safeGet(b),d=47===c?1:0;if(d)var
e=d;else{var
f=92===c?1:0,e=f||(58===c?1:0)}return e}function
cz(a){var
e=a.getLen()<1?1:0,c=e||(47!==a.safeGet(0)?1:0);if(c){var
f=a.getLen()<1?1:0,d=f||(92!==a.safeGet(0)?1:0);if(d){var
g=a.getLen()<2?1:0,b=g||(58!==a.safeGet(1)?1:0)}else
var
b=d}else
var
b=c;return b}function
er(a){var
c=cz(a);if(c){var
g=a.getLen()<2?1:0,d=g||y(u(a,0,2),ie);if(d){var
h=a.getLen()<2?1:0,e=h||y(u(a,0,2),id);if(e){var
i=a.getLen()<3?1:0,f=i||y(u(a,0,3),ic);if(f){var
j=a.getLen()<3?1:0,b=j||y(u(a,0,3),ib)}else
var
b=f}else
var
b=e}else
var
b=d}else
var
b=c;return b}function
es(a,b){var
c=b.getLen()<=a.getLen()?1:0;if(c){var
e=u(a,a.getLen()-b.getLen()|0,b.getLen()),f=dR(b),d=bJ(dR(e),f)}else
var
d=c;return d}try{var
rW=bH(rV),et=rW}catch(f){if(f[1]!==t)throw f;var
et=ig}function
ih(h){var
i=h.getLen(),e=aQ(i+20|0);G(e,34);function
g(a,b){var
c=b;for(;;){if(c===i)return G(e,34);var
f=h.safeGet(c);if(34===f)return a<50?d(1+a,0,c):C(d,[0,0,c]);if(92===f)return a<50?d(1+a,0,c):C(d,[0,0,c]);G(e,f);var
c=c+1|0;continue}}function
d(a,b,c){var
f=b,d=c;for(;;){if(d===i){G(e,34);return a<50?j(1+a,f):C(j,[0,f])}var
l=h.safeGet(d);if(34===l){k((2*f|0)+1|0);G(e,34);return a<50?g(1+a,d+1|0):C(g,[0,d+1|0])}if(92===l){var
f=f+1|0,d=d+1|0;continue}k(f);return a<50?g(1+a,d):C(g,[0,d])}}function
j(a,b){var
d=1;if(!(b<1)){var
c=d;for(;;){G(e,92);var
f=c+1|0;if(b!==c){var
c=f;continue}break}}return 0}function
a(b){return V(g(0,b))}function
b(b,c){return V(d(0,b,c))}function
k(b){return V(j(0,b))}a(0);return aR(e)}function
eu(a){var
c=2<=a.getLen()?1:0;if(c){var
b=a.safeGet(0),g=91<=b?(b+fw|0)<0||25<(b+fw|0)?0:1:65<=b?1:0,d=g?1:0,e=d?58===a.safeGet(1)?1:0:d}else
var
e=c;if(e){var
f=u(a,2,a.getLen()-2|0);return[0,u(a,0,2),f]}return[0,ii,a]}function
ij(a){var
b=eu(a),c=b[1];return h(c,cu(az,cy,b[2]))}function
ik(a){return ct(az,cy,eu(a)[2])}function
io(a){return ct(az,cA,a)}function
ip(a){return cu(az,cA,a)}if(y(cg,iq))if(y(cg,ir)){if(y(cg,is))throw[0,F,it];var
bv=[0,cy,h$,ia,az,cz,er,es,et,ih,ik,ij]}else
var
bv=[0,cv,h1,h2,cw,ep,h3,h6,cx,eq,h9,h_];else
var
bv=[0,cA,il,im,az,cz,er,es,cx,eq,io,ip];var
ev=[0,iw],iu=bv[11],iv=bv[3];a(iz,[0,ev,0,iy,ix]);d_(function(a){if(a[1]===ev){var
c=a[2],d=a[4],e=a[3];if(typeof
c===m)switch(c){case
1:var
b=iC;break;case
2:var
b=iD;break;case
3:var
b=iE;break;case
4:var
b=iF;break;case
5:var
b=iG;break;case
6:var
b=iH;break;case
7:var
b=iI;break;case
8:var
b=iJ;break;case
9:var
b=iK;break;case
10:var
b=iL;break;case
11:var
b=iM;break;case
12:var
b=iN;break;case
13:var
b=iO;break;case
14:var
b=iP;break;case
15:var
b=iQ;break;case
16:var
b=iR;break;case
17:var
b=iS;break;case
18:var
b=iT;break;case
19:var
b=iU;break;case
20:var
b=iV;break;case
21:var
b=iW;break;case
22:var
b=iX;break;case
23:var
b=iY;break;case
24:var
b=iZ;break;case
25:var
b=i0;break;case
26:var
b=i1;break;case
27:var
b=i2;break;case
28:var
b=i3;break;case
29:var
b=i4;break;case
30:var
b=i5;break;case
31:var
b=i6;break;case
32:var
b=i7;break;case
33:var
b=i8;break;case
34:var
b=i9;break;case
35:var
b=i_;break;case
36:var
b=i$;break;case
37:var
b=ja;break;case
38:var
b=jb;break;case
39:var
b=jc;break;case
40:var
b=jd;break;case
41:var
b=je;break;case
42:var
b=jf;break;case
43:var
b=jg;break;case
44:var
b=jh;break;case
45:var
b=ji;break;case
46:var
b=jj;break;case
47:var
b=jk;break;case
48:var
b=jl;break;case
49:var
b=jm;break;case
50:var
b=jn;break;case
51:var
b=jo;break;case
52:var
b=jp;break;case
53:var
b=jq;break;case
54:var
b=jr;break;case
55:var
b=js;break;case
56:var
b=jt;break;case
57:var
b=ju;break;case
58:var
b=jv;break;case
59:var
b=jw;break;case
60:var
b=jx;break;case
61:var
b=jy;break;case
62:var
b=jz;break;case
63:var
b=jA;break;case
64:var
b=jB;break;case
65:var
b=jC;break;case
66:var
b=jD;break;case
67:var
b=jE;break;default:var
b=iA}else{var
f=c[1],b=j(d8(jF),f)}return[0,o(d8(iB),b,e,d)]}return 0});bK(jG);bK(jH);try{bK(rU)}catch(f){if(f[1]!==aM)throw f}try{bK(rT)}catch(f){if(f[1]!==aM)throw f}eb(0,7);function
ew(a){return tZ(a)}ah(32,r);var
jI=6,jJ=0,jO=A(b1),jP=0,jQ=r;if(!(r<0)){var
a7=jP;for(;;){jO.safeSet(a7,dQ(ce(a7)));var
rS=a7+1|0;if(jQ!==a7){var
a7=rS;continue}break}}var
cB=ah(32,0);cB.safeSet(10>>>3,ce(cB.safeGet(10>>>3)|1<<(10&7)));var
jK=A(32),jL=0,jM=31;if(!(31<0)){var
aX=jL;for(;;){jK.safeSet(aX,ce(cB.safeGet(aX)^r));var
jN=aX+1|0;if(jM!==aX){var
aX=jN;continue}break}}var
aA=[0,0],aB=[0,0],ex=[0,0];function
H(a){return aA[1]}function
ey(a){return aB[1]}function
P(a,b,c){return 0===a[2][0]?b?tp(a[1],a,b[1]):tq(a[1],a):b?e4(a[1],b[1]):e4(a[1],0)}var
ez=[3,jI],cC=[0,0];function
aC(e,b,c){cC[1]++;switch(e[0]){case
7:case
8:throw[0,F,jR];case
6:var
g=e[1],m=cC[1],n=e5(0),o=w(ey(0)+1|0,n),p=e6(0),q=w(H(0)+1|0,p),f=[0,-1,[1,[0,td(g,c),g]],q,o,c,0,e,0,0,m,0];break;default:var
h=e[1],i=cC[1],j=e5(0),k=w(ey(0)+1|0,j),l=e6(0),f=[0,-1,[0,r9(h,jJ,[0,c])],w(H(0)+1|0,l),k,c,0,e,0,0,i,0]}if(b){var
d=b[1],a=function(a){{if(0===d[2][0])return 6===e[0]?gq(f,d[1][8],d[1]):gp(f,d[1][8],d[1]);{var
b=d[1],c=H(0);return e7(f,d[1][8]-c|0,b)}}};try{a(0)}catch(f){a9(0);a(0)}f[6]=[0,d]}return f}function
T(a){return a[5]}function
aY(a){return a[6]}function
bw(a){return a[8]}function
bx(a){return a[7]}function
Y(a){return a[2]}function
by(a,b,c){a[1]=b;a[6]=c;return 0}function
cD(a,b,c){return ds<=b?s(a[3],c):s(a[4],c)}function
cE(a,b){var
e=b[3].length-1-2|0,g=0;if(!(e<0)){var
d=g;for(;;){l(b[3],d,s(a[3],d));var
j=d+1|0;if(e!==d){var
d=j;continue}break}}var
f=b[4].length-1-2|0,h=0;if(!(f<0)){var
c=h;for(;;){l(b[4],c,s(a[4],c));var
i=c+1|0;if(f!==c){var
c=i;continue}break}}return 0}function
bz(a,b){b[8]=a[8];return 0}var
ar=[0,jX];a(j6,[0,[0,jS]]);a(j7,[0,[0,jT]]);a(j8,[0,[0,jU]]);a(j9,[0,[0,jV]]);a(j_,[0,[0,jW]]);a(j$,[0,ar]);a(ka,[0,[0,jY]]);a(kb,[0,[0,jZ]]);a(kc,[0,[0,j0]]);a(kd,[0,[0,j2]]);a(ke,[0,[0,j3]]);a(kf,[0,[0,j4]]);a(kg,[0,[0,j5]]);a(kh,[0,[0,j1]]);var
cF=[0,kp];a(kF,[0,[0,ki]]);a(kG,[0,[0,kj]]);a(kH,[0,[0,kk]]);a(kI,[0,[0,kl]]);a(kJ,[0,[0,km]]);a(kK,[0,[0,kn]]);a(kL,[0,[0,ko]]);a(kM,[0,cF]);a(kN,[0,[0,kq]]);a(kO,[0,[0,kr]]);a(kP,[0,[0,ks]]);a(kQ,[0,[0,kt]]);a(kR,[0,[0,ku]]);a(kS,[0,[0,kv]]);a(kT,[0,[0,kw]]);a(kU,[0,[0,kx]]);a(kV,[0,[0,ky]]);a(kW,[0,[0,kz]]);a(kX,[0,[0,kA]]);a(kY,[0,[0,kB]]);a(kZ,[0,[0,kC]]);a(k0,[0,[0,kD]]);a(k1,[0,[0,kE]]);var
bA=1,eA=0;function
aZ(a,b,c){var
d=a[2];if(0===d[0])var
f=r$(d[1],b,c);else{var
e=d[1],f=o(e[2][4],e[1],b,c)}return f}function
a0(a,b){var
c=a[2];if(0===c[0])var
e=r_(c[1],b);else{var
d=c[1],e=i(d[2][3],d[1],b)}return e}function
eB(a,b){P(a,0,0);eG(b,0,0);return P(a,0,0)}function
aa(a,b,c){var
f=a,d=b;for(;;){if(eA)return aZ(f,d,c);var
n=d<0?1:0,o=n||(T(f)<=d?1:0);if(o)throw[0,bm,k2];if(bA){var
i=aY(f);if(typeof
i!==m)eB(i[1],f)}var
j=bw(f);if(j){var
e=j[1];if(1===e[1]){var
k=e[4],g=e[3],l=e[2];return 0===k?aZ(e[5],l+d|0,c):aZ(e[5],(l+x(at(d,g),k+g|0)|0)+aG(d,g)|0,c)}var
h=e[3],f=e[5],d=(e[2]+x(at(d,h),e[4]+h|0)|0)+aG(d,h)|0;continue}return aZ(f,d,c)}}function
ab(a,b){var
e=a,c=b;for(;;){if(eA)return a0(e,c);var
l=c<0?1:0,n=l||(T(e)<=c?1:0);if(n)throw[0,bm,k3];if(bA){var
h=aY(e);if(typeof
h!==m)eB(h[1],e)}var
i=bw(e);if(i){var
d=i[1];if(1===d[1]){var
j=d[4],f=d[3],k=d[2];return 0===j?a0(d[5],k+c|0):a0(d[5],(k+x(at(c,f),j+f|0)|0)+aG(c,f)|0)}var
g=d[3],e=d[5],c=(d[2]+x(at(c,g),d[4]+g|0)|0)+aG(c,g)|0;continue}return a0(e,c)}}function
eC(a){if(a[8]){var
b=aC(a[7],0,a[5]);b[1]=a[1];b[6]=a[6];cE(a,b);var
c=b}else
var
c=a;return c}function
eD(d,b,c){{if(0===c[2][0]){var
a=function(a){return 0===Y(d)[0]?tg(d,c[1][8],c[1],c[3],b):ti(d,c[1][8],c[1],b)};try{var
f=a(0)}catch(f){if(f[1]===ar){try{P(c,0,0);var
g=a(0)}catch(f){a9(0);return a(0)}return g}throw f}return f}var
e=function(a){{if(0===Y(d)[0]){var
e=c[1],f=H(0);return tI(d,c[1][8]-f|0,e,b)}var
g=c[1],h=H(0);return tK(d,c[1][8]-h|0,g,b)}};try{var
i=e(0)}catch(f){try{P(c,0,0);var
h=e(0)}catch(f){a9(0);return e(0)}return h}return i}}function
eE(d,b,c){{if(0===c[2][0]){var
a=function(a){return 0===Y(d)[0]?to(d,c[1][8],c[1],c,b):tj(d,c[1][8],c[1],b)};try{var
f=a(0)}catch(f){if(f[1]===ar){try{P(c,0,0);var
g=a(0)}catch(f){a9(0);return a(0)}return g}throw f}return f}var
e=function(a){{if(0===Y(d)[0]){var
e=c[2],f=c[1],g=H(0);return tO(d,c[1][8]-g|0,f,e,b)}var
h=c[2],i=c[1],j=H(0);return tL(d,c[1][8]-j|0,i,h,b)}};try{var
i=e(0)}catch(f){try{P(c,0,0);var
h=e(0)}catch(f){a9(0);return e(0)}return h}return i}}function
a1(a,b,c,d,e,f,g,h){{if(0===d[2][0])return 0===Y(a)[0]?tx(a,b,d[1][8],d[1],d[3],c,e,f,g,h):tl(a,b,d[1][8],d[1],d[3],c,e,f,g,h);{if(0===Y(a)[0]){var
i=d[3],j=d[1],k=H(0);return tX(a,b,d[1][8]-k|0,j,i,c,e,f,g,h)}var
l=d[3],m=d[1],n=H(0);return tM(a,b,d[1][8]-n|0,m,l,c,e,f,g,h)}}}function
a2(a,b,c,d,e,f,g,h){{if(0===d[2][0])return 0===Y(a)[0]?ty(a,b,d[1][8],d[1],d[3],c,e,f,g,h):tm(a,b,d[1][8],d[1],d[3],c,e,f,g,h);{if(0===Y(a)[0]){var
i=d[3],j=d[1],k=H(0);return tY(a,b,d[1][8]-k|0,j,i,c,e,f,g,h)}var
l=d[3],m=d[1],n=H(0);return tN(a,b,d[1][8]-n|0,m,l,c,e,f,g,h)}}}function
eF(a,b,c){var
q=b;for(;;){var
d=q?q[1]:0,r=aY(a);if(typeof
r===m){by(a,c[1][8],[1,c]);try{cG(a,c)}catch(f){if(f[1]!==ar)f[1]===cF;try{P(c,[0,d],0);cG(a,c)}catch(f){if(f[1]!==ar)if(f[1]!==cF)throw f;P(c,0,0);sg(0);cG(a,c)}}var
z=bw(a);if(z){var
j=z[1];if(1===j[1]){var
k=j[5],s=j[4],f=j[3],l=j[2];if(0===f)a1(k,a,d,c,0,0,l,T(a));else
if(p<f){var
h=0,n=T(a);for(;;){if(f<n){a1(k,a,d,c,x(h,f+s|0),x(h,f),l,f);var
h=h+1|0,n=n-f|0;continue}if(0<n)a1(k,a,d,c,x(h,f+s|0),x(h,f),l,n);break}}else{var
e=0,i=0,g=T(a);for(;;){if(p<g){var
v=aC(bx(a),0,p);bz(a,v);var
A=e+f9|0;if(!(A<e)){var
t=e;for(;;){aa(v,t,ab(a,e));var
H=t+1|0;if(A!==t){var
t=H;continue}break}}a1(k,v,d,c,x(i,p+s|0),i*p|0,l,p);var
e=e+p|0,i=i+1|0,g=g+fL|0;continue}if(0<g){var
w=aC(bx(a),0,g),B=(e+g|0)-1|0;if(!(B<e)){var
u=e;for(;;){aa(w,u,ab(a,e));var
I=u+1|0;if(B!==u){var
u=I;continue}break}}bz(a,w);a1(k,w,d,c,x(i,p+s|0),i*p|0,l,g)}break}}}else{var
y=eC(a),C=T(a)-1|0,J=0;if(!(C<0)){var
o=J;for(;;){aZ(y,o,ab(a,o));var
K=o+1|0;if(C!==o){var
o=K;continue}break}}eD(y,d,c);cE(y,a)}}else
eD(a,d,c);return by(a,c[1][8],[0,c])}else{if(0===r[0]){var
D=r[1],E=cY(D,c);if(E){eG(a,[0,d],0);P(D,0,0);var
q=[0,d];continue}return E}var
F=r[1],G=cY(F,c);if(G){P(F,0,0);var
q=[0,d];continue}return G}}}function
cG(a,b){{if(0===b[2][0])return 0===Y(a)[0]?gp(a,b[1][8],b[1]):gq(a,b[1][8],b[1]);{if(0===Y(a)[0]){var
c=b[1],d=H(0);return e7(a,b[1][8]-d|0,c)}var
e=b[1],f=H(0);return tJ(a,b[1][8]-f|0,e)}}}function
eG(a,b,c){var
w=b;for(;;){var
f=w?w[1]:0,r=aY(a);if(typeof
r===m)return 0;else{if(0===r[0]){var
d=r[1];by(a,d[1][8],[1,d]);var
A=bw(a);if(A){var
k=A[1];if(1===k[1]){var
l=k[5],s=k[4],e=k[3],n=k[2];if(0===e)a2(l,a,f,d,0,0,n,T(a));else
if(p<e){var
i=0,o=T(a);for(;;){if(e<o){a2(l,a,f,d,x(i,e+s|0),x(i,e),n,e);var
i=i+1|0,o=o-e|0;continue}if(0<o)a2(l,a,f,d,x(i,e+s|0),x(i,e),n,o);break}}else{var
j=0,h=T(a),g=0;for(;;){if(p<h){var
y=aC(bx(a),0,p);bz(a,y);var
B=g+f9|0;if(!(B<g)){var
t=g;for(;;){aa(y,t,ab(a,g));var
E=t+1|0;if(B!==t){var
t=E;continue}break}}a2(l,y,f,d,x(j,p+s|0),j*p|0,n,p);var
j=j+1|0,h=h+fL|0;continue}if(0<h){var
z=aC(bx(a),0,h),C=(g+h|0)-1|0;if(!(C<g)){var
u=g;for(;;){aa(z,u,ab(a,g));var
F=u+1|0;if(C!==u){var
u=F;continue}break}}bz(a,z);a2(l,z,f,d,x(j,p+s|0),j*p|0,n,h)}break}}}else{var
v=eC(a);cE(v,a);eE(v,f,d);var
D=T(v)-1|0,G=0;if(!(D<0)){var
q=G;for(;;){aa(a,q,a0(v,q));var
H=q+1|0;if(D!==q){var
q=H;continue}break}}}}else
eE(a,f,d);return by(a,d[1][8],0)}P(r[1],0,0);var
w=[0,f];continue}}}var
k8=[0,k7],k_=[0,k9];function
bB(a,b){var
p=s(gQ,0),q=h(iv,h(a,b)),f=dH(h(iu(p),q));try{var
n=eI,g=eI;a:for(;;){if(1){var
k=function(a,b,c){var
e=b,d=c;for(;;){if(d){var
g=d[1],f=g.getLen(),h=d[2];a8(g,0,a,e-f|0,f);var
e=e-f|0,d=h;continue}return a}},d=0,e=0;for(;;){var
c=sH(f);if(0===c){if(!d)throw[0,bn];var
j=k(A(e),e,d)}else{if(!(0<c)){var
m=A(-c|0);cZ(f,m,0,-c|0);var
d=[0,m,d],e=e-c|0;continue}var
i=A(c-1|0);cZ(f,i,0,c-1|0);sG(f);if(d){var
l=(e+c|0)-1|0,j=k(A(l),l,[0,i,d])}else
var
j=i}var
g=h(g,h(j,k$)),n=g;continue a}}var
o=g;break}}catch(f){if(f[1]!==bn)throw f;var
o=n}dJ(f);return o}var
eJ=[0,la],cH=[],lb=0,lc=0;s$(cH,[0,0,function(f){var
k=el(f,ld),e=cs(k4),d=e.length-1,n=eH.length-1,a=w(d+n|0,0),p=d-1|0,u=0;if(!(p<0)){var
c=u;for(;;){l(a,c,aT(f,s(e,c)));var
y=c+1|0;if(p!==c){var
c=y;continue}break}}var
q=n-1|0,v=0;if(!(q<0)){var
b=v;for(;;){l(a,b+d|0,el(f,s(eH,b)));var
x=b+1|0;if(q!==b){var
b=x;continue}break}}var
r=a[10],m=a[12],h=a[15],i=a[16],j=a[17],g=a[18],z=a[1],A=a[2],B=a[3],C=a[4],D=a[5],E=a[7],F=a[8],G=a[9],H=a[11],I=a[14];function
J(a,b,c,d,e,f){var
h=d?d[1]:d;o(a[1][m+1],a,[0,h],f);var
i=bt(a[g+1],f);return e8(a[1][r+1],a,b,[0,c[1],c[2]],e,f,i)}function
K(a,b,c,d,e){try{var
f=bt(a[g+1],e),h=f}catch(f){if(f[1]!==t)throw f;try{o(a[1][m+1],a,le,e)}catch(f){throw f}var
h=bt(a[g+1],e)}return e8(a[1][r+1],a,b,[0,c[1],c[2]],d,e,h)}function
L(a,b,c){var
y=b?b[1]:b;try{bt(a[g+1],c);var
f=0}catch(f){if(f[1]===t){if(0===c[2][0]){var
z=a[i+1];if(!z)throw[0,eJ,c];var
A=z[1],H=y?tn(A,a[h+1],c[1]):tf(A,a[h+1],c[1]),B=H}else{var
D=a[j+1];if(!D)throw[0,eJ,c];var
E=D[1],I=y?tz(E,a[h+1],c[1]):tH(E,a[h+1],c[1]),B=I}var
d=a[g+1],v=cl(d,c);l(d[2],v,[0,c,B,s(d[2],v)]);d[1]=d[1]+1|0;var
x=d[2].length-1<<1<d[1]?1:0;if(x){var
m=d[2],n=m.length-1,o=n*2|0,p=o<ch?1:0;if(p){var
k=w(o,0);d[2]=k;var
q=function(a){if(a){var
b=a[1],e=a[2];q(a[3]);var
c=cl(d,b);return l(k,c,[0,b,e,s(k,c)])}return 0},r=n-1|0,F=0;if(!(r<0)){var
e=F;for(;;){q(s(m,e));var
G=e+1|0;if(r!==e){var
e=G;continue}break}}var
u=0}else
var
u=p;var
C=u}else
var
C=x;return C}throw f}return f}function
M(a,b){try{var
f=[0,bB(a[k+1],lg),0],c=f}catch(f){var
c=0}a[i+1]=c;try{var
e=[0,bB(a[k+1],lf),0],d=e}catch(f){var
d=0}a[j+1]=d;return 0}function
N(a,b){a[j+1]=[0,b,0];return 0}function
O(a,b){return a[j+1]}function
P(a,b){a[i+1]=[0,b,0];return 0}function
Q(a,b){return a[i+1]}function
R(a,b){var
d=a[g+1];d[1]=0;var
e=d[2].length-1-1|0,f=0;if(!(e<0)){var
c=f;for(;;){l(d[2],c,0);var
h=c+1|0;if(e!==c){var
c=h;continue}break}}return 0}eo(f,[0,G,function(a,b){return a[g+1]},C,R,F,Q,A,P,E,O,z,N,D,M,m,L,B,K,H,J]);return function(a,b,c,d){var
e=en(b,f);e[k+1]=c;e[I+1]=c;e[h+1]=d;try{var
o=[0,bB(c,li),0],l=o}catch(f){var
l=0}e[i+1]=l;try{var
n=[0,bB(c,lh),0],m=n}catch(f){var
m=0}e[j+1]=m;e[g+1]=eb(0,8);return e}},lc,lb]);e9(0);e9(0);function
cI(a){function
e(a,b){var
d=a-1|0,e=0;if(!(d<0)){var
c=e;for(;;){d6(lk);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return j(d6(lj),b)}function
f(a,b){var
c=a,d=b;for(;;)if(typeof
d===m)return 0===d?e(c,ll):e(c,lm);else
switch(d[0]){case
0:e(c,ln);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
1:e(c,lo);var
c=c+1|0,d=d[1];continue;case
2:e(c,lp);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
3:e(c,lq);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
4:e(c,lr);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
5:e(c,ls);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
6:e(c,lt);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
7:e(c,lu);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
8:e(c,lv);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
9:e(c,lw);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
10:e(c,lx);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
11:return e(c,h(ly,d[1]));case
12:return e(c,h(lz,d[1]));case
13:return e(c,h(lA,k(d[1])));case
14:return e(c,h(lB,k(d[1])));case
15:return e(c,h(lC,k(d[1])));case
16:return e(c,h(lD,k(d[1])));case
17:return e(c,h(lE,k(d[1])));case
18:return e(c,lF);case
19:return e(c,lG);case
20:return e(c,lH);case
21:return e(c,lI);case
22:return e(c,lJ);case
23:return e(c,h(lK,k(d[2])));case
24:e(c,lL);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
25:e(c,lM);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
26:e(c,lN);var
c=c+1|0,d=d[1];continue;case
27:e(c,lO);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
28:e(c,lP);var
c=c+1|0,d=d[1];continue;case
29:e(c,lQ);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
30:e(c,lR);f(c+1|0,d[1]);f(c+1|0,d[2]);var
c=c+1|0,d=d[3];continue;case
31:return e(c,lS);case
32:var
g=h(lT,k(d[2]));return e(c,h(lU,h(d[1],g)));case
33:return e(c,h(lV,k(d[1])));case
36:e(c,lX);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
37:e(c,lY);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
38:e(c,lZ);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
39:e(c,l0);f(c+1|0,d[1]);f(c+1|0,d[2]);var
c=c+1|0,d=d[3];continue;case
40:e(c,l1);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
41:e(c,l2);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
42:e(c,l3);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
43:e(c,l4);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
44:e(c,l5);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
45:e(c,l6);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
46:e(c,l7);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
47:e(c,l8);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
48:e(c,l9);f(c+1|0,d[1]);f(c+1|0,d[2]);f(c+1|0,d[3]);var
c=c+1|0,d=d[4];continue;case
49:e(c,l_);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
50:e(c,l$);f(c+1|0,d[1]);var
i=d[2],j=c+1|0;return dL(function(a){return f(j,a)},i);case
51:return e(c,ma);case
52:return e(c,mb);default:return e(c,h(lW,N(d[1])))}}return f(0,a)}function
I(a){return ah(a,32)}var
a3=[0,mc];function
a$(a,b,c){var
d=c;for(;;)if(typeof
d===m)return me;else
switch(d[0]){case
18:case
19:var
T=h(mT,h(e(b,d[2]),mS));return h(mU,h(k(d[1]),T));case
27:case
38:var
ac=d[1],ad=h(nd,h(e(b,d[2]),nc));return h(e(b,ac),ad);case
0:var
g=d[2],B=e(b,d[1]);if(typeof
g===m)var
r=0;else
if(25===g[0]){var
t=e(b,g),r=1}else
var
r=0;if(!r){var
D=h(mf,I(b)),t=h(e(b,g),D)}return h(h(B,t),mg);case
1:var
E=h(e(b,d[1]),mh),G=y(a3[1][1],mi)?h(a3[1][1],mj):ml;return h(mk,h(G,E));case
2:var
H=h(mn,h(U(b,d[2]),mm));return h(mo,h(U(b,d[1]),H));case
3:var
J=h(mq,h(ak(b,d[2]),mp));return h(mr,h(ak(b,d[1]),J));case
4:var
K=h(mt,h(U(b,d[2]),ms));return h(mu,h(U(b,d[1]),K));case
5:var
L=h(mw,h(ak(b,d[2]),mv));return h(mx,h(ak(b,d[1]),L));case
6:var
M=h(mz,h(U(b,d[2]),my));return h(mA,h(U(b,d[1]),M));case
7:var
O=h(mC,h(ak(b,d[2]),mB));return h(mD,h(ak(b,d[1]),O));case
8:var
P=h(mF,h(U(b,d[2]),mE));return h(mG,h(U(b,d[1]),P));case
9:var
Q=h(mI,h(ak(b,d[2]),mH));return h(mJ,h(ak(b,d[1]),Q));case
10:var
R=h(mL,h(U(b,d[2]),mK));return h(mM,h(U(b,d[1]),R));case
13:return h(mN,k(d[1]));case
14:return h(mO,k(d[1]));case
15:throw[0,F,mP];case
16:return h(mQ,k(d[1]));case
17:return h(mR,k(d[1]));case
20:var
V=h(mW,h(e(b,d[2]),mV));return h(mX,h(k(d[1]),V));case
21:var
W=h(mZ,h(e(b,d[2]),mY));return h(m0,h(k(d[1]),W));case
22:var
X=h(m2,h(e(b,d[2]),m1));return h(m3,h(k(d[1]),X));case
23:var
Y=h(m4,k(d[2])),u=d[1];if(typeof
u===m)var
f=0;else
switch(u[0]){case
33:var
o=m6,f=1;break;case
34:var
o=m7,f=1;break;case
35:var
o=m8,f=1;break;default:var
f=0}if(f)return h(o,Y);throw[0,F,m5];case
24:var
i=d[2],v=d[1];if(typeof
i===m){if(0===i){var
d=v;continue}}else
if(24===i[0]){var
Z=h(m_,e(b,i));return h(e(b,v),Z)}return S(m9);case
25:var
_=e(b,d[2]),$=h(m$,h(I(b),_));return h(e(b,d[1]),$);case
26:var
aa=e(b,d[1]),ab=y(a3[1][2],na)?a3[1][2]:nb;return h(ab,aa);case
28:var
d=d[1];continue;case
29:var
ae=h(nf,h(e(b,d[2]),ne));return h(e(b,d[1]),ae);case
30:var
l=d[2],af=e(b,d[3]),ag=h(ng,h(I(b),af));if(typeof
l===m)var
s=0;else
if(31===l[0]){var
w=h(md(l[1]),ni),s=1}else
var
s=0;if(!s)var
w=e(b,l);var
ah=h(nh,h(w,ag));return h(e(b,d[1]),ah);case
31:return a<50?a_(1+a,d[1]):C(a_,[0,d[1]]);case
33:return k(d[1]);case
34:return h(N(d[1]),nj);case
35:return N(d[1]);case
36:var
ai=h(nl,h(e(b,d[2]),nk));return h(e(b,d[1]),ai);case
37:var
aj=h(nn,h(I(b),nm)),al=h(e(b,d[2]),aj),am=h(no,h(I(b),al)),an=h(np,h(e(b,d[1]),am));return h(I(b),an);case
39:var
ao=h(nq,I(b)),ap=h(e(b+2|0,d[3]),ao),aq=h(nr,h(I(b+2|0),ap)),ar=h(ns,h(I(b),aq)),as=h(e(b+2|0,d[2]),ar),at=h(nt,h(I(b+2|0),as));return h(nu,h(e(b,d[1]),at));case
40:var
au=h(nv,I(b)),av=h(e(b+2|0,d[2]),au),aw=h(nw,h(I(b+2|0),av));return h(nx,h(e(b,d[1]),aw));case
41:var
ax=h(ny,e(b,d[2]));return h(e(b,d[1]),ax);case
42:var
ay=h(nz,e(b,d[2]));return h(e(b,d[1]),ay);case
43:var
az=h(nA,e(b,d[2]));return h(e(b,d[1]),az);case
44:var
aA=h(nB,e(b,d[2]));return h(e(b,d[1]),aA);case
45:var
aB=h(nC,e(b,d[2]));return h(e(b,d[1]),aB);case
46:var
aC=h(nD,e(b,d[2]));return h(e(b,d[1]),aC);case
47:var
aD=h(nE,e(b,d[2]));return h(e(b,d[1]),aD);case
48:var
p=e(b,d[1]),aE=e(b,d[2]),aF=e(b,d[3]),aG=h(e(b+2|0,d[4]),nF);return h(nL,h(p,h(nK,h(aE,h(nJ,h(p,h(nI,h(aF,h(nH,h(p,h(nG,h(I(b+2|0),aG))))))))))));case
49:var
aH=e(b,d[1]),aI=h(e(b+2|0,d[2]),nM);return h(nO,h(aH,h(nN,h(I(b+2|0),aI))));case
50:var
x=d[2],n=d[1],z=e(b,n),q=function(a){if(a){var
c=a[2],d=a[1];if(c){var
f=h(nP,q(c));return h(e(b,d),f)}return e(b,d)}throw[0,F,nQ]};if(typeof
n!==m)if(31===n[0]){var
A=n[1];if(!y(A[1],nT))if(!y(A[2],nU))return h(z,h(nW,h(q(aO(x)),nV)))}return h(z,h(nS,h(q(aO(x)),nR)));case
51:return k(j(d[1],0));case
52:return h(N(j(d[1],0)),nX);default:return d[1]}}function
r3(a,b,c){if(typeof
c!==m)switch(c[0]){case
2:case
4:case
6:case
8:case
10:case
50:return a<50?a$(1+a,b,c):C(a$,[0,b,c]);case
32:return c[1];case
33:return k(c[1]);case
36:var
d=h(nZ,h(U(b,c[2]),nY));return h(e(b,c[1]),d);case
51:return k(j(c[1],0));default:}return a<50?c0(1+a,b,c):C(c0,[0,b,c])}function
c0(a,b,c){if(typeof
c!==m)switch(c[0]){case
3:case
5:case
7:case
9:case
29:case
50:return a<50?a$(1+a,b,c):C(a$,[0,b,c]);case
16:return h(n1,k(c[1]));case
31:return a<50?a_(1+a,c[1]):C(a_,[0,c[1]]);case
32:return c[1];case
34:return h(N(c[1]),n2);case
35:return h(n3,N(c[1]));case
36:var
d=h(n5,h(U(b,c[2]),n4));return h(e(b,c[1]),d);case
52:return h(N(j(c[1],0)),n6);default:}cI(c);return S(n0)}function
a_(a,b){return b[1]}function
e(b,c){return V(a$(0,b,c))}function
U(b,c){return V(r3(0,b,c))}function
ak(b,c){return V(c0(0,b,c))}function
md(b){return V(a_(0,b))}function
z(a){return ah(a,32)}var
a4=[0,n7];function
bb(a,b,c){var
d=c;for(;;)if(typeof
d===m)return n9;else
switch(d[0]){case
18:case
19:var
U=h(ou,h(f(b,d[2]),ot));return h(ov,h(k(d[1]),U));case
27:case
38:var
ac=d[1],ad=h(oP,Q(b,d[2]));return h(f(b,ac),ad);case
0:var
g=d[2],D=f(b,d[1]);if(typeof
g===m)var
r=0;else
if(25===g[0]){var
t=f(b,g),r=1}else
var
r=0;if(!r){var
E=h(n_,z(b)),t=h(f(b,g),E)}return h(h(D,t),n$);case
1:var
G=h(f(b,d[1]),oa),H=y(a4[1][1],ob)?h(a4[1][1],oc):oe;return h(od,h(H,G));case
2:var
I=h(of,Q(b,d[2]));return h(Q(b,d[1]),I);case
3:var
J=h(og,al(b,d[2]));return h(al(b,d[1]),J);case
4:var
K=h(oh,Q(b,d[2]));return h(Q(b,d[1]),K);case
5:var
L=h(oi,al(b,d[2]));return h(al(b,d[1]),L);case
6:var
M=h(oj,Q(b,d[2]));return h(Q(b,d[1]),M);case
7:var
O=h(ok,al(b,d[2]));return h(al(b,d[1]),O);case
8:var
P=h(ol,Q(b,d[2]));return h(Q(b,d[1]),P);case
9:var
R=h(om,al(b,d[2]));return h(al(b,d[1]),R);case
10:var
T=h(on,Q(b,d[2]));return h(Q(b,d[1]),T);case
13:return h(oo,k(d[1]));case
14:return h(op,k(d[1]));case
15:throw[0,F,oq];case
16:return h(or,k(d[1]));case
17:return h(os,k(d[1]));case
20:var
V=h(ox,h(f(b,d[2]),ow));return h(oy,h(k(d[1]),V));case
21:var
W=h(oA,h(f(b,d[2]),oz));return h(oB,h(k(d[1]),W));case
22:var
X=h(oD,h(f(b,d[2]),oC));return h(oE,h(k(d[1]),X));case
23:var
Y=h(oF,k(d[2])),u=d[1];if(typeof
u===m)var
e=0;else
switch(u[0]){case
33:var
o=oH,e=1;break;case
34:var
o=oI,e=1;break;case
35:var
o=oJ,e=1;break;default:var
e=0}if(e)return h(o,Y);throw[0,F,oG];case
24:var
i=d[2],v=d[1];if(typeof
i===m){if(0===i){var
d=v;continue}}else
if(24===i[0]){var
Z=h(oL,f(b,i));return h(f(b,v),Z)}return S(oK);case
25:var
_=f(b,d[2]),$=h(oM,h(z(b),_));return h(f(b,d[1]),$);case
26:var
aa=f(b,d[1]),ab=y(a4[1][2],oN)?a4[1][2]:oO;return h(ab,aa);case
28:var
d=d[1];continue;case
29:var
ae=h(oR,h(f(b,d[2]),oQ));return h(f(b,d[1]),ae);case
30:var
l=d[2],af=f(b,d[3]),ag=h(oS,h(z(b),af));if(typeof
l===m)var
s=0;else
if(31===l[0]){var
w=n8(l[1]),s=1}else
var
s=0;if(!s)var
w=f(b,l);var
ah=h(oT,h(w,ag));return h(f(b,d[1]),ah);case
31:return a<50?ba(1+a,d[1]):C(ba,[0,d[1]]);case
33:return k(d[1]);case
34:return h(N(d[1]),oU);case
35:return N(d[1]);case
36:var
ai=h(oW,h(f(b,d[2]),oV));return h(f(b,d[1]),ai);case
37:var
aj=h(oY,h(z(b),oX)),ak=h(f(b,d[2]),aj),am=h(oZ,h(z(b),ak)),an=h(o0,h(f(b,d[1]),am));return h(z(b),an);case
39:var
ao=h(o1,z(b)),ap=h(f(b+2|0,d[3]),ao),aq=h(o2,h(z(b+2|0),ap)),ar=h(o3,h(z(b),aq)),as=h(f(b+2|0,d[2]),ar),at=h(o4,h(z(b+2|0),as));return h(o5,h(f(b,d[1]),at));case
40:var
au=h(o6,z(b)),av=h(o7,h(z(b),au)),aw=h(f(b+2|0,d[2]),av),ax=h(o8,h(z(b+2|0),aw)),ay=h(o9,h(z(b),ax));return h(o_,h(f(b,d[1]),ay));case
41:var
az=h(o$,f(b,d[2]));return h(f(b,d[1]),az);case
42:var
aA=h(pa,f(b,d[2]));return h(f(b,d[1]),aA);case
43:var
aB=h(pb,f(b,d[2]));return h(f(b,d[1]),aB);case
44:var
aC=h(pc,f(b,d[2]));return h(f(b,d[1]),aC);case
45:var
aD=h(pd,f(b,d[2]));return h(f(b,d[1]),aD);case
46:var
aE=h(pe,f(b,d[2]));return h(f(b,d[1]),aE);case
47:var
aF=h(pf,f(b,d[2]));return h(f(b,d[1]),aF);case
48:var
p=f(b,d[1]),aG=f(b,d[2]),aH=f(b,d[3]),aI=h(f(b+2|0,d[4]),pg);return h(pm,h(p,h(pl,h(aG,h(pk,h(p,h(pj,h(aH,h(pi,h(p,h(ph,h(z(b+2|0),aI))))))))))));case
49:var
aJ=f(b,d[1]),aK=h(f(b+2|0,d[2]),pn);return h(pp,h(aJ,h(po,h(z(b+2|0),aK))));case
50:var
x=d[2],n=d[1],A=f(b,n),q=function(a){if(a){var
c=a[2],d=a[1];if(c){var
e=h(pq,q(c));return h(f(b,d),e)}return f(b,d)}throw[0,F,pr]};if(typeof
n!==m)if(31===n[0]){var
B=n[1];if(!y(B[1],pu))if(!y(B[2],pv))return h(A,h(px,h(q(aO(x)),pw)))}return h(A,h(pt,h(q(aO(x)),ps)));case
51:return k(j(d[1],0));case
52:return h(N(j(d[1],0)),py);default:return d[1]}}function
r4(a,b,c){if(typeof
c!==m)switch(c[0]){case
2:case
4:case
6:case
8:case
10:case
50:return a<50?bb(1+a,b,c):C(bb,[0,b,c]);case
32:return c[1];case
33:return k(c[1]);case
36:var
d=h(pA,h(Q(b,c[2]),pz));return h(f(b,c[1]),d);case
51:return k(j(c[1],0));default:}return a<50?c1(1+a,b,c):C(c1,[0,b,c])}function
c1(a,b,c){if(typeof
c!==m)switch(c[0]){case
3:case
5:case
7:case
9:case
50:return a<50?bb(1+a,b,c):C(bb,[0,b,c]);case
16:return h(pC,k(c[1]));case
31:return a<50?ba(1+a,c[1]):C(ba,[0,c[1]]);case
32:return c[1];case
34:return h(N(c[1]),pD);case
35:return h(pE,N(c[1]));case
36:var
d=h(pG,h(Q(b,c[2]),pF));return h(f(b,c[1]),d);case
52:return h(N(j(c[1],0)),pH);default:}cI(c);return S(pB)}function
ba(a,b){return b[2]}function
f(b,c){return V(bb(0,b,c))}function
Q(b,c){return V(r4(0,b,c))}function
al(b,c){return V(c1(0,b,c))}function
n8(b){return V(ba(0,b))}var
pT=h(pS,h(pR,h(pQ,h(pP,h(pO,h(pN,h(pM,h(pL,h(pK,h(pJ,pI)))))))))),p_=h(p9,h(p8,h(p7,h(p6,h(p5,h(p4,h(p3,h(p2,h(p1,h(p0,h(pZ,h(pY,h(pX,h(pW,h(pV,pU))))))))))))))),qg=h(qf,h(qe,h(qd,h(qc,h(qb,h(qa,p$)))))),qo=h(qn,h(qm,h(ql,h(qk,h(qj,h(qi,qh))))));function
v(a){return[32,h(qp,k(a)),a]}function
a5(a,b){return[25,a,b]}function
am(a){return[33,a]}function
as(a,b){return[2,a,b]}function
cJ(a,b){return[6,a,b]}function
cK(a){return[13,a]}function
cL(a,b){return[29,a,b]}function
cM(a,b){return[31,[0,a,b]]}function
cN(a,b){return[37,a,b]}function
cO(a,b){return[27,a,b]}function
cP(a){return[28,a]}function
aD(a,b){return[36,a,b]}function
eK(a){var
e=[0,0];function
b(a){var
c=a;for(;;){if(typeof
c!==m)switch(c[0]){case
0:var
s=b(c[2]);return[0,b(c[1]),s];case
1:return[1,b(c[1])];case
2:var
t=b(c[2]);return[2,b(c[1]),t];case
3:var
h=c[2],i=c[1];if(typeof
i!==m)if(34===i[0])if(typeof
h!==m)if(34===h[0]){e[1]=1;return[34,i[1]+h[1]]}var
u=b(h);return[3,b(i),u];case
4:var
v=b(c[2]);return[4,b(c[1]),v];case
5:var
j=c[2],k=c[1];if(typeof
k!==m)if(34===k[0])if(typeof
j!==m)if(34===j[0]){e[1]=1;return[34,k[1]+j[1]]}var
w=b(j);return[5,b(k),w];case
6:var
x=b(c[2]);return[6,b(c[1]),x];case
7:var
l=c[2],n=c[1];if(typeof
n!==m)if(34===n[0])if(typeof
l!==m)if(34===l[0]){e[1]=1;return[34,n[1]+l[1]]}var
y=b(l);return[7,b(n),y];case
8:var
z=b(c[2]);return[8,b(c[1]),z];case
9:var
o=c[2],p=c[1];if(typeof
p!==m)if(34===p[0])if(typeof
o!==m)if(34===o[0]){e[1]=1;return[34,p[1]+o[1]]}var
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
d!==m)switch(d[0]){case
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
f!==m)switch(f[0]){case
25:var
P=b(f[2]),Q=[29,b(q),P];return[25,f[1],Q];case
39:e[1]=1;var
R=f[3],S=[29,b(q),R],T=b(f[2]),U=[29,b(q),T];return[39,b(f[1]),U,S];default:}var
O=b(f);return[29,b(q),O];case
30:var
V=b(c[3]),W=b(c[2]);return[30,b(c[1]),W,V];case
36:var
g=c[2],r=c[1];if(typeof
g!==m)if(25===g[0]){var
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
ao=aN(b,c[2]);return[50,b(c[1]),ao];default:}return c}}var
c=b(a);for(;;){if(e[1]){e[1]=0;var
c=b(c);continue}return c}}var
qw=[0,qv];function
a6(a,b,c){var
g=c[2],d=c[1],t=a?a[1]:a,u=b?b[1]:2,n=g[3],p=g[2];qw[1]=qx;var
j=n[1];if(typeof
j===m)var
l=1===j?0:1;else
switch(j[0]){case
23:case
29:case
36:var
l=0;break;case
13:case
14:case
17:var
o=h(qF,h(k(j[1]),qE)),l=2;break;default:var
l=1}switch(l){case
1:cI(n[1]);dK(dF);throw[0,F,qy];case
2:break;default:var
o=qz}var
q=[0,e(0,n[1]),o];if(t){a3[1]=q;a4[1]=q}function
r(a){var
q=g[4],r=dM(function(a,b){return 0===b?a:h(qg,a)},qo,q),s=h(r,e(0,eK(p))),j=cU(eZ(qA,gv,438));dG(j,s);cV(j);e0(j);e_(qB);var
m=dH(qC),c=sD(m),n=A(c),o=0;if(0<=0)if(0<=c)if((n.getLen()-c|0)<o)var
f=0;else{var
k=o,b=c;for(;;){if(0<b){var
l=cZ(m,n,k,b);if(0===l)throw[0,bn];var
k=k+l|0,b=b-l|0;continue}var
f=1;break}}else
var
f=0;else
var
f=0;if(!f)E(gx);dJ(m);i(Z(d,723535973,3),d,n);e_(qD);return 0}function
s(a){var
b=g[4],c=dM(function(a,b){return 0===b?a:h(p_,a)},pT,b);return i(Z(d,56985577,4),d,h(c,f(0,eK(p))))}switch(u){case
1:s(0);break;case
2:r(0);s(0);break;default:r(0)}i(Z(d,345714255,5),d,0);return[0,d,g]}var
cQ=d,eL=null,qK=1,qL=1,qM=1,qN=undefined;function
eM(a,b){return a==eL?j(b,0):a}var
eN=Array,qO=true,qP=false;d_(function(a){return a
instanceof
eN?0:[0,new
aw(a.toString())]});function
J(a,b){a.appendChild(b);return 0}function
eO(d){return sA(function(a){if(a){var
e=j(d,a);if(!(e|0))a.preventDefault();return e}var
c=event,b=j(d,c);if(!(b|0))c.returnValue=b;return b})}var
K=cQ.document,qQ="2d";function
bC(a,b){return a?j(b,a[1]):0}function
bD(a,b){return a.createElement(b.toString())}function
bE(a,b){return bD(a,b)}var
eP=[0,fV];function
eQ(a,b,c,d){for(;;){if(0===a)if(0===b)return bD(c,d);var
h=eP[1];if(fV===h){try{var
j=K.createElement('<input name="x">'),k=j.tagName.toLowerCase()===fh?1:0,m=k?j.name===dj?1:0:k,i=m}catch(f){var
i=0}var
l=i?fI:-1003883683;eP[1]=l;continue}if(fI<=h){var
e=new
eN();e.push("<",d.toString());bC(a,function(a){e.push(' type="',e$(a),bZ);return 0});bC(b,function(a){e.push(' name="',e$(a),bZ);return 0});e.push(">");return c.createElement(e.join(g))}var
f=bD(c,d);bC(a,function(a){return f.type=a});bC(b,function(a){return f.name=a});return f}}function
eR(a){return bE(a,qU)}var
qY=[0,qX];cQ.HTMLElement===qN;function
eV(a){return eR(K)}function
eW(a){function
c(a){throw[0,F,q0]}var
b=eM(K.getElementById(gb),c);return j(d7(function(a){J(b,eV(0));J(b,K.createTextNode(a.toString()));return J(b,eV(0))}),a)}function
q1(a){var
k=[0,[4,a]];return function(a,b,c,d){var
h=a[2],i=a[1],l=c[2];if(0===l[0]){var
g=l[1],e=[0,0],f=th(k.length-1),m=g[7][1]<i[1]?1:0;if(m)var
n=m;else{var
s=g[7][2]<i[2]?1:0,n=s||(g[7][3]<i[3]?1:0)}if(n)throw[0,k8];var
o=g[8][1]<h[1]?1:0;if(o)var
p=o;else{var
r=g[8][2]<h[2]?1:0,p=r||(g[8][3]<h[3]?1:0)}if(p)throw[0,k_];b$(function(a,b){function
h(a){if(bA)try{eF(a,0,c);P(c,0,0)}catch(f){if(f[1]===ar)throw[0,ar];throw f}return 11===b[0]?tk(e,f,cD(b[1],ds,c[1][8]),a):tw(e,f,cD(a,ds,c[1][8]),a,c)}switch(b[0]){case
6:case
7:case
8:case
9:case
10:switch(b[0]){case
6:var
d=tv(e,f,b[1]);break;case
7:var
d=tu(e,f,b[1]);break;case
8:var
d=tt(e,f,b[1]);break;case
9:var
d=ts(e,f,b[1]);break;default:var
d=S(k5)}var
g=d;break;case
11:var
g=h(b[1]);break;default:var
g=h(b[1])}return g},k);var
q=tr(e,d,h,i,f,c[1],b)}else{var
j=[0,0];b$(function(a,b){switch(b[0]){case
6:case
7:case
8:case
9:case
10:switch(b[0]){case
6:var
e=tU(j,d,b[1],c[1]);break;case
7:var
e=tV(j,d,b[1],c[1]);break;case
8:var
e=tS(j,d,b[1],c[1]);break;case
9:var
e=tT(j,d,b[1],c[1]);break;default:var
e=S(k6)}var
g=e;break;default:var
f=b[1];if(bA){if(cY(aY(f),[0,c]))eF(f,0,c);P(c,0,0)}var
h=c[1],i=H(0),g=tW(j,d,a,cD(f,-701974253,c[1][8]-i|0),h)}return g},k);var
q=tR(d,h,i,c[1],b)}return q}}if(cR===0)var
c=ee([0]);else{var
aW=ee(aN(hx,cR));b$(function(a,b){var
c=(a*2|0)+2|0;aW[3]=o(aq[4],b,c,aW[3]);aW[4]=o(aj[4],c,1,aW[4]);return 0},cR);var
c=aW}var
cp=aN(function(a){return aT(c,a)},eU),em=cH[2],q2=cp[1],q3=cp[2],q4=cp[3],hR=cH[4],eg=cq(eS),eh=cq(eU),ei=cq(eT),q5=1,cr=ca(function(a){return aT(c,a)},eh),hA=ca(function(a){return aT(c,a)},ei);c[5]=[0,[0,c[3],c[4],c[6],c[7],cr,eg],c[5]];var
hB=$[1],hC=c[7];function
hD(a,b,c){return cd(a,eg)?o($[4],a,b,c):c}c[7]=o($[11],hD,hC,hB);var
aU=[0,aq[1]],aV=[0,aj[1]];dP(function(a,b){aU[1]=o(aq[4],a,b,aU[1]);var
e=aV[1];try{var
f=i(aj[22],b,c[4]),d=f}catch(f){if(f[1]!==t)throw f;var
d=1}aV[1]=o(aj[4],b,d,e);return 0},ei,hA);dP(function(a,b){aU[1]=o(aq[4],a,b,aU[1]);aV[1]=o(aj[4],b,0,aV[1]);return 0},eh,cr);c[3]=aU[1];c[4]=aV[1];var
hE=0,hF=c[6];c[6]=cc(function(a,b){return cd(a[1],cr)?b:[0,a,b]},hF,hE);var
hS=q5?i(em,c,hR):j(em,c),ej=c[5],ay=ej?ej[1]:S(gB),ek=c[5],hG=ay[6],hH=ay[5],hI=ay[4],hJ=ay[3],hK=ay[2],hL=ay[1],hM=ek?ek[2]:S(gC);c[5]=hM;var
cb=hI,bo=hG;for(;;){if(bo){var
dO=bo[1],gD=bo[2],hN=i($[22],dO,c[7]),cb=o($[4],dO,hN,cb),bo=gD;continue}c[7]=cb;c[3]=hL;c[4]=hK;var
hO=c[6];c[6]=cc(function(a,b){return cd(a[1],hH)?b:[0,a,b]},hO,hJ);var
hT=0,hU=cs(eT),hV=[0,aN(function(a){var
e=aT(c,a);try{var
b=c[6];for(;;){if(!b)throw[0,t];var
d=b[1],f=b[2],h=d[2];if(0!==aF(d[1],e)){var
b=f;continue}var
g=h;break}}catch(f){if(f[1]!==t)throw f;var
g=s(c[2],e)}return g},hU),hT],hW=cs(eS),q6=r6([0,[0,hS],[0,aN(function(a){try{var
b=i($[22],a,c[7])}catch(f){if(f[1]===t)throw[0,F,hQ];throw f}return b},hW),hV]])[1],q7=function(a,b){if(1===b.length-1){var
c=b[0+1];if(4===c[0])return c[1]}return S(q8)};eo(c,[0,q3,0,q1,q4,function(a,b){return[0,[4,b]]},q2,q7]);var
q9=function(a,b){var
e=en(b,c);o(q6,e,q$,q_);if(!b){var
f=c[8];if(0!==f){var
d=f;for(;;){if(d){var
g=d[2];j(d[1],e);var
d=g;continue}break}}}return e};ef[1]=(ef[1]+c[1]|0)-1|0;c[8]=dN(c[8]);cn(c,3+at(s(c[2],1)*16|0,ax)|0);var
hX=0,hY=function(a){var
b=a;return q9(hX,b)},rc=v(4),rd=am(2),re=as(v(3),rd),rf=cL(aD(v(0),re),rc),rg=v(4),rh=am(1),ri=as(v(3),rh),rj=a5(cL(aD(v(0),ri),rg),rf),rk=v(4),rl=v(3),rm=a5(cL(aD(v(0),rl),rk),rj),rn=am(3),ro=am(2),rp=as(v(3),ro),rq=aD(v(0),rp),rr=am(1),rs=as(v(3),rr),rt=aD(v(0),rs),ru=v(3),qr=[8,as(as(aD(v(0),ru),rt),rq),rn],rv=a5(cO(v(4),qr),rm),rw=am(4),rx=cJ(v(2),rw),ry=a5(cO(v(3),rx),rv),rz=am(p),rA=cJ(am(p),rz),qs=[39,[45,v(2),rA],1,ry],rD=cM(rC,rB),rG=cJ(cM(rF,rE),rD),rJ=as(cM(rI,rH),rG),qt=[26,a5(cO(v(2),rJ),qs)],rK=cN(cP(cK(4)),qt),rL=cN(cP(cK(3)),rK),ra=[0,0],rb=[0,[13,5],ez],qq=[0,[1,[24,[23,qu,0],0]],cN(cP(cK(2)),rL)],rM=[0,function(a){var
d=qK+(qM*qL|0)|0;if((p*p|0)<d)return 0;var
b=d*4|0,e=ab(a,b+2|0),f=ab(a,b+1|0),c=((ab(a,b)+f|0)+e|0)/3|0;aa(a,b,c);aa(a,b+1|0,c);return aa(a,b+2|0,c)},qq,rb,ra],cS=[0,hY(0),rM],bF=function(a){return eR(K)};cQ.onload=eO(function(a){var
O=eX?eX[1]:2;switch(O){case
1:fa(0);aB[1]=fb(0);break;case
2:fc(0);aA[1]=fd(0);fa(0);aB[1]=fb(0);break;default:fc(0);aA[1]=fd(0)}ex[1]=aA[1]+aB[1]|0;var
y=aA[1]-1|0,x=0,P=0;if(y<0)var
z=x;else{var
g=P,C=x;for(;;){var
D=b_(C,[0,tA(g),0]),Q=g+1|0;if(y!==g){var
g=Q,C=D;continue}var
z=D;break}}var
o=0,d=0,b=z;for(;;){if(o<aB[1]){if(tQ(d)){var
B=d+1|0,A=b_(b,[0,tC(d,d+aA[1]|0),0])}else{var
B=d,A=b}var
o=o+1|0,d=B,b=A;continue}var
n=0,m=b;for(;;){if(m){var
n=n+1|0,m=m[2];continue}ex[1]=n;aB[1]=d;if(b){var
k=0,h=b,L=b[2],M=b[1];for(;;){if(h){var
k=k+1|0,h=h[2];continue}var
v=w(k,M),l=1,f=L;for(;;){if(f){var
N=f[2];v[l+1]=f[1];var
l=l+1|0,f=N;continue}var
q=v;break}break}}else
var
q=[0];var
R=function(a){throw[0,F,rR]},c=eM(K.getElementById(gb),R);J(c,bF(0));var
r=eQ(0,0,K,qS),G=bD(K,qW);J(G,K.createTextNode("Choose a computing device : "));J(c,G);dL(function(a){var
b=bE(K,qR);J(b,K.createTextNode(a[1][1].toString()));return J(r,b)},q);J(c,r);J(c,bF(0));var
e=bE(K,qZ);if(1-(e.getContext==eL?1:0)){e.width=p;e.height=p;var
E=bE(K,qV);E.src="lena.png";var
u=e.getContext(qQ);u.drawImage(E,0,0);J(c,bF(0));J(c,e);var
H=u.getImageData(0,0,p,p),I=H.data;J(c,bF(0));var
S=function(a){var
g=s(q,r.selectedIndex+0|0),w=g[1][1];j(eW(rO),w);var
c=aC(ez,0,(p*p|0)*4|0);d$(hs,e1(0));var
l=T(c)-1|0,x=0;if(!(l<0)){var
e=x;for(;;){aa(c,e,I[e]);var
D=e+1|0;if(l!==e){var
e=D;continue}break}}var
m=g[2];if(0===m[0])var
h=b1;else{var
C=0===m[1][2]?1:b1,h=C}a6(0,rP,cS);var
t=ew(0),f=cS[2],b=cS[1],y=0,z=[0,[0,h,1,1],[0,at(((p*p|0)+h|0)-1|0,h),1,1]],n=0,k=0?n[1]:n;if(0===g[2][0]){if(k)a6(0,qG,[0,b,f]);else
if(!i(Z(b,-723625231,7),b,0))a6(0,qH,[0,b,f])}else
if(k)a6(0,qI,[0,b,f]);else
if(!i(Z(b,649483637,8),b,0))a6(0,qJ,[0,b,f]);(function(a,b,c,d,e,f){return a.length==5?a(b,c,d,e,f):ag(a,[b,c,d,e,f])}(Z(b,5695307,6),b,c,z,y,g));var
v=ew(0)-t;i(eW(rN),rQ,v);var
o=T(c)-1|0,A=0;if(!(o<0)){var
d=A;for(;;){I[d]=ab(c,d);var
B=d+1|0;if(o!==d){var
d=B;continue}break}}u.putImageData(H,0,0);return qO},t=eQ([0,"button"],0,K,qT);t.value="Go";t.onclick=eO(S);J(c,t);return qP}throw[0,qY]}}});dI(0);return}}(this));
