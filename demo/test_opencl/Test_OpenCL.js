// This program was compiled from OCaml by js_of_ocaml 2.00dev+git-97cd021
(function(c){"use strict";var
dt="set_cuda_sources",dL=123,dR=254,b2=";",f6=108,gK="section1",ds="reload_sources",b6="Map.bal",gl=",",ca='"',aa=16777215,dr="get_cuda_sources",gk=0.07,ch=" / ",fQ="Test_OpenCL_js.ml",f5="double spoc_var",dC="args_to_list",b$=" * ",am="(",fP="float spoc_var",dB=65599,fO="jsError",cg="if (",b_="return",gj=" ;\n",dK="exec",bp=115,bn=";}\n",f4=".ptx",t=512,dJ=120,b1="..",gi=-512,O="]",dI=117,b5="; ",dH="compile",fN="Bigarray.blit: dimension mismatch",gJ=" (",_="0",dA="list_to_args",b4=248,gh=126,gI="fd ",dq="get_binaries",f3=" == ",aQ="(float)",dz="Kirc_Cuda.ml",cf=" + ",gg=") ",dG="x",fM=56985577,f2=-97,fL="g",bk=1073741823,gH="parse concat",aA=105,dy="get_opencl_sources",dQ=511,bm=110,gG=-88,ak=" = ",dx="set_opencl_sources",f1=0.21,P="[",b9="'",fK="Unix",b0="int_of_string",dP=262144,gF="(double) ",gf=982028505,bj="){\n",f0=649483637,bo="e",gE="#define __FLOAT64_EXTENSION__ \n",az="-",aP=-48,dw=" : file already exists",b8="(double) spoc_var",fJ="++){\n",fZ="__shared__ float spoc_var",gD="opencl_sources",fY=".cl",dF="reset_binaries",bZ="\n",gC=101,dO=748841679,ce="index out of bounds",fI="spoc_init_opencl_device_vec",dp=125,b7=" - ",gB=";}",q=255,gA="binaries",cd="}",gz=" < ",fH="__shared__ long spoc_var",aO=250,gy=" >= ",fG="input",ge=246,dv=102,gd="Unix.Unix_error",g="",gx=143,fF=" || ",aN=100,dE="Kirc_OpenCL.ml",gw="#ifndef __FLOAT64_EXTENSION__ \n",gc="__shared__ int spoc_var",dN=103,bY=", ",p=" not implemented",gb="./",fX=1e3,fE="for (int ",ga="10px",gv="file_file",gu="spoc_var",ad=".",fW="else{\n",b3="+",f$="(int)",dM="run",aB=65535,dD="#endif\n",aM=";\n",$="f",gt=785140586,fV=127,gs="__shared__ double spoc_var",fU=-32,du=111,f_=" > ",F=" ",gr="int spoc_var",al=")",f9="cuda_sources",cc=256,fT="nan",dn=116,go="../",gp="kernel_name",gq=65520,gn="%.12g",fD=" && ",f8=0.71,bl="/",f7="while (",dm="compile_and_run",cb=114,gm="* spoc_var",bX=" <= ",m="number",fR=-2147483648,fS=" % ",uL=c.spoc_opencl_part_device_to_cpu_b!==undefined?c.spoc_opencl_part_device_to_cpu_b:function(){n("spoc_opencl_part_device_to_cpu_b"+p)},uK=c.spoc_opencl_part_cpu_to_device_b!==undefined?c.spoc_opencl_part_cpu_to_device_b:function(){n("spoc_opencl_part_cpu_to_device_b"+p)},uI=c.spoc_opencl_load_param_int64!==undefined?c.spoc_opencl_load_param_int64:function(){n("spoc_opencl_load_param_int64"+p)},uG=c.spoc_opencl_load_param_float64!==undefined?c.spoc_opencl_load_param_float64:function(){n("spoc_opencl_load_param_float64"+p)},uF=c.spoc_opencl_load_param_float!==undefined?c.spoc_opencl_load_param_float:function(){n("spoc_opencl_load_param_float"+p)},uA=c.spoc_opencl_custom_part_device_to_cpu_b!==undefined?c.spoc_opencl_custom_part_device_to_cpu_b:function(){n("spoc_opencl_custom_part_device_to_cpu_b"+p)},uz=c.spoc_opencl_custom_part_cpu_to_device_b!==undefined?c.spoc_opencl_custom_part_cpu_to_device_b:function(){n("spoc_opencl_custom_part_cpu_to_device_b"+p)},uy=c.spoc_opencl_custom_device_to_cpu!==undefined?c.spoc_opencl_custom_device_to_cpu:function(){n("spoc_opencl_custom_device_to_cpu"+p)},ux=c.spoc_opencl_custom_cpu_to_device!==undefined?c.spoc_opencl_custom_cpu_to_device:function(){n("spoc_opencl_custom_cpu_to_device"+p)},uw=c.spoc_opencl_custom_alloc_vect!==undefined?c.spoc_opencl_custom_alloc_vect:function(){n("spoc_opencl_custom_alloc_vect"+p)},ul=c.spoc_cuda_part_device_to_cpu_b!==undefined?c.spoc_cuda_part_device_to_cpu_b:function(){n("spoc_cuda_part_device_to_cpu_b"+p)},uk=c.spoc_cuda_part_cpu_to_device_b!==undefined?c.spoc_cuda_part_cpu_to_device_b:function(){n("spoc_cuda_part_cpu_to_device_b"+p)},uj=c.spoc_cuda_load_param_vec_b!==undefined?c.spoc_cuda_load_param_vec_b:function(){n("spoc_cuda_load_param_vec_b"+p)},ui=c.spoc_cuda_load_param_int_b!==undefined?c.spoc_cuda_load_param_int_b:function(){n("spoc_cuda_load_param_int_b"+p)},uh=c.spoc_cuda_load_param_int64_b!==undefined?c.spoc_cuda_load_param_int64_b:function(){n("spoc_cuda_load_param_int64_b"+p)},ug=c.spoc_cuda_load_param_float_b!==undefined?c.spoc_cuda_load_param_float_b:function(){n("spoc_cuda_load_param_float_b"+p)},uf=c.spoc_cuda_load_param_float64_b!==undefined?c.spoc_cuda_load_param_float64_b:function(){n("spoc_cuda_load_param_float64_b"+p)},ue=c.spoc_cuda_launch_grid_b!==undefined?c.spoc_cuda_launch_grid_b:function(){n("spoc_cuda_launch_grid_b"+p)},ud=c.spoc_cuda_flush_all!==undefined?c.spoc_cuda_flush_all:function(){n("spoc_cuda_flush_all"+p)},uc=c.spoc_cuda_flush!==undefined?c.spoc_cuda_flush:function(){n("spoc_cuda_flush"+p)},ub=c.spoc_cuda_device_to_cpu!==undefined?c.spoc_cuda_device_to_cpu:function(){n("spoc_cuda_device_to_cpu"+p)},t$=c.spoc_cuda_custom_part_device_to_cpu_b!==undefined?c.spoc_cuda_custom_part_device_to_cpu_b:function(){n("spoc_cuda_custom_part_device_to_cpu_b"+p)},t_=c.spoc_cuda_custom_part_cpu_to_device_b!==undefined?c.spoc_cuda_custom_part_cpu_to_device_b:function(){n("spoc_cuda_custom_part_cpu_to_device_b"+p)},t9=c.spoc_cuda_custom_load_param_vec_b!==undefined?c.spoc_cuda_custom_load_param_vec_b:function(){n("spoc_cuda_custom_load_param_vec_b"+p)},t8=c.spoc_cuda_custom_device_to_cpu!==undefined?c.spoc_cuda_custom_device_to_cpu:function(){n("spoc_cuda_custom_device_to_cpu"+p)},t7=c.spoc_cuda_custom_cpu_to_device!==undefined?c.spoc_cuda_custom_cpu_to_device:function(){n("spoc_cuda_custom_cpu_to_device"+p)},g0=c.spoc_cuda_custom_alloc_vect!==undefined?c.spoc_cuda_custom_alloc_vect:function(){n("spoc_cuda_custom_alloc_vect"+p)},t6=c.spoc_cuda_create_extra!==undefined?c.spoc_cuda_create_extra:function(){n("spoc_cuda_create_extra"+p)},t5=c.spoc_cuda_cpu_to_device!==undefined?c.spoc_cuda_cpu_to_device:function(){n("spoc_cuda_cpu_to_device"+p)},gZ=c.spoc_cuda_alloc_vect!==undefined?c.spoc_cuda_alloc_vect:function(){n("spoc_cuda_alloc_vect"+p)},t2=c.spoc_create_custom!==undefined?c.spoc_create_custom:function(){n("spoc_create_custom"+p)},uO=1;function
gV(a,b){throw[0,a,b]}function
d1(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=c.console;b&&b.error&&b.error(a)}var
v=[0];function
bt(a,b){if(!a)return g;if(a&1)return bt(a-1,b)+b;var
c=bt(a>>1,b);return c+c}function
G(a){if(a!=null){this.bytes=this.fullBytes=a;this.last=this.len=a.length}}function
gY(){gV(v[4],new
G(ce))}G.prototype={string:null,bytes:null,fullBytes:null,array:null,len:null,last:0,toJsString:function(){var
a=this.getFullBytes();try{return this.string=decodeURIComponent(escape(a))}catch(f){d1('MlString.toJsString: wrong encoding for "%s" ',a);return a}},toBytes:function(){if(this.string!=null)try{var
a=unescape(encodeURIComponent(this.string))}catch(f){d1('MlString.toBytes: wrong encoding for "%s" ',this.string);var
a=this.string}else{var
a=g,c=this.array,d=c.length;for(var
b=0;b<d;b++)a+=String.fromCharCode(c[b])}this.bytes=this.fullBytes=a;this.last=this.len=a.length;return a},getBytes:function(){var
a=this.bytes;if(a==null)a=this.toBytes();return a},getFullBytes:function(){var
a=this.fullBytes;if(a!==null)return a;a=this.bytes;if(a==null)a=this.toBytes();if(this.last<this.len){this.bytes=a+=bt(this.len-this.last,"\0");this.last=this.len}this.fullBytes=a;return a},toArray:function(){var
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
b=this.bytes;if(b==null)b=this.toBytes();return a<this.last?b.charCodeAt(a):0},safeGet:function(a){if(this.len==null)this.toBytes();if(a<0||a>=this.len)gY();return this.get(a)},set:function(a,b){var
c=this.array;if(!c){if(this.last==a){this.bytes+=String.fromCharCode(b&q);this.last++;return 0}c=this.toArray()}else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;c[a]=b&q;return 0},safeSet:function(a,b){if(this.len==null)this.toBytes();if(a<0||a>=this.len)gY();this.set(a,b)},fill:function(a,b,c){if(a>=this.last&&this.last&&c==0)return;var
d=this.array;if(!d)d=this.toArray();else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;var
f=a+b;for(var
e=a;e<f;e++)d[e]=c},compare:function(a){if(this.string!=null&&a.string!=null){if(this.string<a.string)return-1;if(this.string>a.string)return 1;return 0}var
b=this.getFullBytes(),c=a.getFullBytes();if(b<c)return-1;if(b>c)return 1;return 0},equal:function(a){if(this.string!=null&&a.string!=null)return this.string==a.string;return this.getFullBytes()==a.getFullBytes()},lessThan:function(a){if(this.string!=null&&a.string!=null)return this.string<a.string;return this.getFullBytes()<a.getFullBytes()},lessEqual:function(a){if(this.string!=null&&a.string!=null)return this.string<=a.string;return this.getFullBytes()<=a.getFullBytes()}};function
ae(a){this.string=a}ae.prototype=new
G();function
sM(a,b,c,d,e){if(d<=b)for(var
f=1;f<=e;f++)c[d+f]=a[b+f];else
for(var
f=e;f>=1;f--)c[d+f]=a[b+f]}function
sN(a){var
c=[0];while(a!==0){var
d=a[1];for(var
b=1;b<d.length;b++)c.push(d[b]);a=a[2]}return c}function
d0(a,b){gV(a,new
ae(b))}function
B(a){d0(v[4],a)}function
af(){B(ce)}function
sO(a,b){if(b<0||b>=a.length-1)af();return a[b+1]}function
sP(a,b,c){if(b<0||b>=a.length-1)af();a[b+1]=c;return 0}function
dT(a){var
d=a.length,c=1;for(var
b=0;b<d;b++){if(a[b]<0)B("Bigarray.create: negative dimension");c=c*a[b]}return c}var
cj;function
sS(){if(!cj){var
a=c;cj=[[a.Float32Array,a.Float64Array,a.Int8Array,a.Uint8Array,a.Int16Array,a.Uint16Array,a.Int32Array,a.Int32Array,a.Int32Array,a.Int32Array,a.Float32Array,a.Float64Array,a.Uint8Array],[0,0,0,0,0,0,0,1,0,0,2,2,0]]}}function
tk(a){return a.slice(1)}function
ci(g,j,c,d,e,f){var
h=f.length,p=dT(f);function
A(a){var
c=0;if(h!=a.length)B("Bigarray.get/set: bad number of dimensions");for(var
b=0;b<h;b++){if(a[b]<0||a[b]>=f[b])af();c=c*f[b]+a[b]}return c}function
C(a){var
c=0;if(h!=a.length)B("Bigarray.get/set: wrong number of indices");for(var
b=h-1;b>=0;b--){if(a[b]<1||a[b]>f[b])af();c=c*f[b]+(a[b]-1)}return c}var
i=e==0?A:C,k=f[0];function
y(a){var
b=i(a),c=g[b];return c}function
x(a){var
d=i(a),c=g[d],b=j[d];return[q,c&aa,c>>>24&q|(b&aB)<<8,b>>>16&aB]}function
w(a){var
b=i(a),d=g[b],c=j[b];return[dR,d,c]}var
b=c==1?x:c==2?w:y;function
u(a){if(a<0||a>=k)af();return g[a]}function
v(a){if(a<1||a>k)af();return g[a-1]}function
t(a){return b([a])}var
s=c==0?e==0?u:v:t;function
o(a,b){g[a]=b}function
n(a,b){g[a]=b[1]|(b[2]&q)<<24;j[a]=b[2]>>>8&aB|b[3]<<16}function
m(a,b){g[a]=b[1];j[a]=b[2]}function
K(a,b){var
c=i(a);return o(c,b)}function
J(a,b){return n(i(a),b)}function
I(a,b){return m(i(a),b)}var
l=c==1?J:c==2?I:K;function
G(a,b){if(a<0||a>=k)af();g[a]=b}function
H(a,b){if(a<1||a>k)af();g[a-1]=b}function
F(a,b){l([a],b)}var
E=c==0?e==0?G:H:F;function
z(a){if(a<0||a>=h)B("Bigarray.dim");return f[a]}function
r(a){if(c==0)for(var
b=0;b<g.length;b++)o(b,a);if(c==1)for(var
b=0;b<g.length;b++)n(b,a);if(c==2)for(var
b=0;b<g.length;b++)m(b,a)}function
a(a){if(h!=a.num_dims)B(fN);for(var
b=0;b<h;b++)if(f[b]!=a.nth_dim(b))B(fN);g.set(a.data);if(c!=0)j.set(a.data2)}function
M(a,b){var
l,k=1;if(e==0){for(var
i=1;i<h;i++)k=k*f[i];l=0}else{for(var
i=0;i<h-1;i++)k=k*f[i];l=h-1;a=a-1}if(a<0||b<0||a+b>f[l])B("Bigarray.sub: bad sub-array");var
n=g.subarray(a*k,(a+b)*k),o=c==0?null:j.subarray(a*k,(a+b)*k),m=[];for(var
i=0;i<h;i++)m[i]=f[i];m[l]=b;return ci(n,o,c,d,e,m)}function
L(a){var
k=a.length,l=[],n=[],m;if(k>=h)B("Bigarray.slice: too many indices");if(e==0){for(var
b=0;b<k;b++)l[b]=a[b];for(;b<h;b++)l[b]=0;m=i(l);n=f.slice(k)}else{for(var
b=0;b<k;b++)l[h-k+b]=a[b];for(var
b=0;b<h-k;b++)l[b]=1;m=i(l);n=f.slice(0,k)}var
o=dT(n),p=g.subarray(m,m+o),q=c==0?null:j.subarray(m,m+o);return ci(p,q,c,d,e,n)}function
D(a){var
f=[],i=a.length;if(i<1)B("Bigarray.reshape: bad number of dimensions");var
h=1;for(var
b=0;b<i;b++){f[b]=a[b];if(f[b]<0)B("Bigarray.reshape: negative dimension");h=h*f[b]}if(h!=p)B("Bigarray.reshape: size mismatch");return ci(g,j,c,d,e,f)}return{data:g,data2:j,data_type:c,num_dims:h,nth_dim:z,kind:d,layout:e,size:p,sub:M,slice:L,blit:a,fill:r,reshape:D,get:b,get1:s,set:l,set1:E}}function
sQ(a,b,c){sS();var
d=tk(c),j=d.length,e=dT(d),f=cj[0][a];if(!f)B("Bigarray.create: unsupported kind");var
i=new
f(e),h=cj[1][a],g=null;if(h!=0)g=new
f(e);return ci(i,g,h,e,b,d)}function
sR(a,b){return a.get1(b)}function
sT(a,b,c){return a.set1(b,c)}function
dU(a,b,c,d,e){if(e===0)return;if(d===c.last&&c.bytes!=null){var
f=a.bytes;if(f==null)f=a.toBytes();if(b>0||a.last>e)f=f.slice(b,b+e);c.bytes+=f;c.last+=f.length;return}var
g=c.array;if(!g)g=c.toArray();else
c.bytes=c.string=null;a.blitToArray(b,g,d,e)}function
an(c,b){if(c.fun)return an(c.fun,b);var
a=c.length,d=a-b.length;if(d==0)return c.apply(null,b);else
if(d<0)return an(c.apply(null,b.slice(0,a)),b.slice(a));else
return function(a){return an(c,b.concat([a]))}}function
sU(a){if(isFinite(a)){if(Math.abs(a)>=2.22507385850720138e-308)return 0;if(a!=0)return 1;return 2}return isNaN(a)?4:3}function
s9(a,b){var
c=a[3]<<16,d=b[3]<<16;if(c>d)return 1;if(c<d)return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
gQ(a,b){if(a<b)return-1;if(a==b)return 0;return 1}function
ck(a,b,c){var
e=[];for(;;){if(!(c&&a===b))if(a
instanceof
G)if(b
instanceof
G){if(a!==b){var
d=a.compare(b);if(d!=0)return d}}else
return 1;else
if(a
instanceof
Array&&a[0]===(a[0]|0)){var
f=a[0];if(f===dR)f=0;if(f===aO){a=a[1];continue}else
if(b
instanceof
Array&&b[0]===(b[0]|0)){var
g=b[0];if(g===dR)g=0;if(g===aO){b=b[1];continue}else
if(f!=g)return f<g?-1:1;else
switch(f){case
b4:var
d=gQ(a[2],b[2]);if(d!=0)return d;break;case
251:B("equal: abstract value");case
q:var
d=s9(a,b);if(d!=0)return d;break;default:if(a.length!=b.length)return a.length<b.length?-1:1;if(a.length>1)e.push(a,b,1)}}else
return 1}else
if(b
instanceof
G||b
instanceof
Array&&b[0]===(b[0]|0))return-1;else{if(a<b)return-1;if(a>b)return 1;if(a!=b){if(!c)return NaN;if(a==a)return 1;if(b==b)return-1}}if(e.length==0)return 0;var
h=e.pop();b=e.pop();a=e.pop();if(h+1<a.length)e.push(a,b,h+1);a=a[h];b=b[h]}}function
sV(a,b){return ck(a,b,true)}function
gL(a){this.bytes=g;this.len=a}gL.prototype=new
G();function
gM(a){if(a<0)B("String.create");return new
gL(a)}function
dZ(a){throw[0,a]}function
gW(){dZ(v[6])}function
sX(a,b){if(b==0)gW();return a/b|0}function
sY(a,b){return+(ck(a,b,false)==0)}function
sZ(a,b,c,d){a.fill(b,c,d)}function
dY(a){a=a.toString();var
e=a.length;if(e>31)B("format_int: format too long");var
b={justify:b3,signstyle:az,filler:F,alternate:false,base:0,signedconv:false,width:0,uppercase:false,sign:1,prec:-1,conv:$};for(var
d=0;d<e;d++){var
c=a.charAt(d);switch(c){case
az:b.justify=az;break;case
b3:case
F:b.signstyle=c;break;case
_:b.filler=_;break;case"#":b.alternate=true;break;case"1":case"2":case"3":case"4":case"5":case"6":case"7":case"8":case"9":b.width=0;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.width=b.width*10+c;d++}d--;break;case
ad:b.prec=0;d++;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.prec=b.prec*10+c;d++}d--;case"d":case"i":b.signedconv=true;case"u":b.base=10;break;case
dG:b.base=16;break;case"X":b.base=16;b.uppercase=true;break;case"o":b.base=8;break;case
bo:case
$:case
fL:b.signedconv=true;b.conv=c;break;case"E":case"F":case"G":b.signedconv=true;b.uppercase=true;b.conv=c.toLowerCase();break}}return b}function
dV(a,b){if(a.uppercase)b=b.toUpperCase();var
e=b.length;if(a.signedconv&&(a.sign<0||a.signstyle!=az))e++;if(a.alternate){if(a.base==8)e+=1;if(a.base==16)e+=2}var
c=g;if(a.justify==b3&&a.filler==F)for(var
d=e;d<a.width;d++)c+=F;if(a.signedconv)if(a.sign<0)c+=az;else
if(a.signstyle!=az)c+=a.signstyle;if(a.alternate&&a.base==8)c+=_;if(a.alternate&&a.base==16)c+="0x";if(a.justify==b3&&a.filler==_)for(var
d=e;d<a.width;d++)c+=_;c+=b;if(a.justify==az)for(var
d=e;d<a.width;d++)c+=F;return new
ae(c)}function
s0(a,b){var
c,f=dY(a),e=f.prec<0?6:f.prec;if(b<0){f.sign=-1;b=-b}if(isNaN(b)){c=fT;f.filler=F}else
if(!isFinite(b)){c="inf";f.filler=F}else
switch(f.conv){case
bo:var
c=b.toExponential(e),d=c.length;if(c.charAt(d-3)==bo)c=c.slice(0,d-1)+_+c.slice(d-1);break;case
$:c=b.toFixed(e);break;case
fL:e=e?e:1;c=b.toExponential(e-1);var
i=c.indexOf(bo),h=+c.slice(i+1);if(h<-4||b.toFixed(0).length>e){var
d=i-1;while(c.charAt(d)==_)d--;if(c.charAt(d)==ad)d--;c=c.slice(0,d+1)+c.slice(i);d=c.length;if(c.charAt(d-3)==bo)c=c.slice(0,d-1)+_+c.slice(d-1);break}else{var
g=e;if(h<0){g-=h+1;c=b.toFixed(g)}else
while(c=b.toFixed(g),c.length>e+1)g--;if(g){var
d=c.length-1;while(c.charAt(d)==_)d--;if(c.charAt(d)==ad)d--;c=c.slice(0,d+1)}}break}return dV(f,c)}function
s1(a,b){if(a.toString()=="%d")return new
ae(g+b);var
c=dY(a);if(b<0)if(c.signedconv){c.sign=-1;b=-b}else
b>>>=0;var
d=b.toString(c.base);if(c.prec>=0){c.filler=F;var
e=c.prec-d.length;if(e>0)d=bt(e,_)+d}return dV(c,d)}function
s3(){return 0}function
s4(){return 0}var
cn=[];function
s5(a,b,c){var
e=a[1],i=cn[c];if(i===null)for(var
h=cn.length;h<c;h++)cn[h]=0;else
if(e[i]===b)return e[i-1];var
d=3,g=e[1]*2+1,f;while(d<g){f=d+g>>1|1;if(b<e[f+1])g=f-2;else
d=f}cn[c]=d+1;return b==e[d+1]?e[d]:0}function
s6(a,b){return+(ck(a,b,false)>=0)}function
gN(a){if(!isFinite(a)){if(isNaN(a))return[q,1,0,gq];return a>0?[q,0,0,32752]:[q,0,0,gq]}var
f=a>=0?0:32768;if(f)a=-a;var
b=Math.floor(Math.LOG2E*Math.log(a))+1023;if(b<=0){b=0;a/=Math.pow(2,-1026)}else{a/=Math.pow(2,b-1027);if(a<16){a*=2;b-=1}if(b==0)a/=2}var
d=Math.pow(2,24),c=a|0;a=(a-c)*d;var
e=a|0;a=(a-e)*d;var
g=a|0;c=c&15|f|b<<4;return[q,g,e,c]}if(!Math.imul)Math.imul=function(a,b){return((a>>16)*b<<16)+(a&aB)*b|0};var
bs=Math.imul,s7=function(){var
p=cc;function
c(a,b){return a<<b|a>>>32-b}function
g(a,b){b=bs(b,3432918353);b=c(b,15);b=bs(b,461845907);a^=b;a=c(a,13);return(a*5|0)+3864292196|0}function
t(a){a^=a>>>16;a=bs(a,2246822507);a^=a>>>13;a=bs(a,3266489909);a^=a>>>16;return a}function
u(a,b){var
d=b[1]|b[2]<<24,c=b[2]>>>8|b[3]<<16;a=g(a,d);a=g(a,c);return a}function
v(a,b){var
d=b[1]|b[2]<<24,c=b[2]>>>8|b[3]<<16;a=g(a,c^d);return a}function
x(a,b){var
e=b.length,c,d;for(c=0;c+4<=e;c+=4){d=b.charCodeAt(c)|b.charCodeAt(c+1)<<8|b.charCodeAt(c+2)<<16|b.charCodeAt(c+3)<<24;a=g(a,d)}d=0;switch(e&3){case
3:d=b.charCodeAt(c+2)<<16;case
2:d|=b.charCodeAt(c+1)<<8;case
1:d|=b.charCodeAt(c);a=g(a,d)}a^=e;return a}function
w(a,b){var
e=b.length,c,d;for(c=0;c+4<=e;c+=4){d=b[c]|b[c+1]<<8|b[c+2]<<16|b[c+3]<<24;a=g(a,d)}d=0;switch(e&3){case
3:d=b[c+2]<<16;case
2:d|=b[c+1]<<8;case
1:d|=b[c];a=g(a,d)}a^=e;return a}return function(a,b,c,d){var
k,l,m,i,h,f,e,j,o;i=b;if(i<0||i>p)i=p;h=a;f=c;k=[d];l=0;m=1;while(l<m&&h>0){e=k[l++];if(e
instanceof
Array&&e[0]===(e[0]|0))switch(e[0]){case
b4:f=g(f,e[2]);h--;break;case
aO:k[--l]=e[1];break;case
q:f=v(f,e);h--;break;default:var
s=e.length-1<<10|e[0];f=g(f,s);for(j=1,o=e.length;j<o;j++){if(m>=i)break;k[m++]=e[j]}break}else
if(e
instanceof
G){var
n=e.array;if(n)f=w(f,n);else{var
r=e.getFullBytes();f=x(f,r)}h--;break}else
if(e===(e|0)){f=g(f,e+e+1);h--}else
if(e===+e){f=u(f,gN(e));h--;break}}f=t(f);return f&bk}}();function
tf(a){return[a[3]>>8,a[3]&q,a[2]>>16,a[2]>>8&q,a[2]&q,a[1]>>16,a[1]>>8&q,a[1]&q]}function
s8(e,b,c){var
d=0;function
f(a){b--;if(e<0||b<0)return;if(a
instanceof
Array&&a[0]===(a[0]|0))switch(a[0]){case
b4:e--;d=d*dB+a[2]|0;break;case
aO:b++;f(a);break;case
q:e--;d=d*dB+a[1]+(a[2]<<24)|0;break;default:e--;d=d*19+a[0]|0;for(var
c=a.length-1;c>0;c--)f(a[c])}else
if(a
instanceof
G){e--;var
g=a.array,h=a.getLen();if(g)for(var
c=0;c<h;c++)d=d*19+g[c]|0;else{var
i=a.getFullBytes();for(var
c=0;c<h;c++)d=d*19+i.charCodeAt(c)|0}}else
if(a===(a|0)){e--;d=d*dB+a|0}else
if(a===+a){e--;var
j=tf(gN(a));for(var
c=7;c>=0;c--)d=d*19+j[c]|0}}f(c);return d&bk}function
ta(a){return(a[3]|a[2]|a[1])==0}function
td(a){return[q,a&aa,a>>24&aa,a>>31&aB]}function
te(a,b){var
c=a[1]-b[1],d=a[2]-b[2]+(c>>24),e=a[3]-b[3]+(d>>24);return[q,c&aa,d&aa,e&aB]}function
gP(a,b){if(a[3]>b[3])return 1;if(a[3]<b[3])return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
gO(a){a[3]=a[3]<<1|a[2]>>23;a[2]=(a[2]<<1|a[1]>>23)&aa;a[1]=a[1]<<1&aa}function
tb(a){a[1]=(a[1]>>>1|a[2]<<23)&aa;a[2]=(a[2]>>>1|a[3]<<23)&aa;a[3]=a[3]>>>1}function
th(a,b){var
e=0,d=a.slice(),c=b.slice(),f=[q,0,0,0];while(gP(d,c)>0){e++;gO(c)}while(e>=0){e--;gO(f);if(gP(d,c)>=0){f[1]++;d=te(d,c)}tb(c)}return[0,f,d]}function
tg(a){return a[1]|a[2]<<24}function
s$(a){return a[3]<<16<0}function
tc(a){var
b=-a[1],c=-a[2]+(b>>24),d=-a[3]+(c>>24);return[q,b&aa,c&aa,d&aB]}function
s_(a,b){var
c=dY(a);if(c.signedconv&&s$(b)){c.sign=-1;b=tc(b)}var
d=g,i=td(c.base),h="0123456789abcdef";do{var
f=th(b,i);b=f[1];d=h.charAt(tg(f[2]))+d}while(!ta(b));if(c.prec>=0){c.filler=F;var
e=c.prec-d.length;if(e>0)d=bt(e,_)+d}return dV(c,d)}function
tE(a){var
b=0,c=10,d=a.get(0)==45?(b++,-1):1;if(a.get(b)==48)switch(a.get(b+1)){case
dJ:case
88:c=16;b+=2;break;case
du:case
79:c=8;b+=2;break;case
98:case
66:c=2;b+=2;break}return[b,d,c]}function
gT(a){if(a>=48&&a<=57)return a-48;if(a>=65&&a<=90)return a-55;if(a>=97&&a<=122)return a-87;return-1}function
n(a){d0(v[3],a)}function
ti(a){var
g=tE(a),f=g[0],h=g[1],d=g[2],i=-1>>>0,e=a.get(f),c=gT(e);if(c<0||c>=d)n(b0);var
b=c;for(;;){f++;e=a.get(f);if(e==95)continue;c=gT(e);if(c<0||c>=d)break;b=d*b+c;if(b>i)n(b0)}if(f!=a.getLen())n(b0);b=h*b;if(d==10&&(b|0)!=b)n(b0);return b|0}function
tj(a){return+(a>31&&a<fV)}var
cl={amp:/&/g,lt:/</g,quot:/\"/g,all:/[&<\"]/};function
tl(a){if(!cl.all.test(a))return a;return a.replace(cl.amp,"&amp;").replace(cl.lt,"&lt;").replace(cl.quot,"&quot;")}function
tm(a){var
c=Array.prototype.slice;return function(){var
b=arguments.length>0?c.call(arguments):[undefined];return an(a,b)}}function
tn(a,b){var
d=[0];for(var
c=1;c<=a;c++)d[c]=b;return d}function
dS(a){var
b=a.length;this.array=a;this.len=this.last=b}dS.prototype=new
G();var
to=function(){function
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
n=0;n<4;n++)o[g*4+n]=l[g]>>8*n&q;return o}return function(a,b,c){var
h=[];if(a.array){var
f=a.array;for(var
d=0;d<c;d+=4){var
e=d+b;h[d>>2]=f[e]|f[e+1]<<8|f[e+2]<<16|f[e+3]<<24}for(;d<c;d++)h[d>>2]|=f[d+b]<<8*(d&3)}else{var
g=a.getFullBytes();for(var
d=0;d<c;d+=4){var
e=d+b;h[d>>2]=g.charCodeAt(e)|g.charCodeAt(e+1)<<8|g.charCodeAt(e+2)<<16|g.charCodeAt(e+3)<<24}for(;d<c;d++)h[d>>2]|=g.charCodeAt(d+b)<<8*(d&3)}return new
dS(n(h,c))}}();function
tp(a){return a.data.array.length}function
ab(a){d0(v[2],a)}function
dX(a){if(!a.opened)ab("Cannot flush a closed channel");if(a.buffer==g)return 0;if(a.output)switch(a.output.length){case
2:a.output(a,a.buffer);break;default:a.output(a.buffer)}a.buffer=g}var
br=new
Array();function
tq(a){dX(a);a.opened=false;delete
br[a.fd];return 0}function
tr(a,b,c,d){var
e=a.data.array.length-a.data.offset;if(e<d)d=e;dU(new
dS(a.data.array),a.data.offset,b,c,d);a.data.offset+=d;return d}function
tF(){dZ(v[5])}function
ts(a){if(a.data.offset>=a.data.array.length)tF();if(a.data.offset<0||a.data.offset>a.data.array.length)af();var
b=a.data.array[a.data.offset];a.data.offset++;return b}function
tt(a){var
b=a.data.offset,c=a.data.array.length;if(b>=c)return 0;while(true){if(b>=c)return-(b-a.data.offset);if(b<0||b>a.data.array.length)af();if(a.data.array[b]==10)return b-a.data.offset+1;b++}}function
gU(a){a=a
instanceof
G?a.toString():a;ab(a+": No such file or directory")}var
sW=bl;function
cm(a){a=a
instanceof
G?a.toString():a;if(a.charCodeAt(0)!=47)a=sW+a;var
d=a.split(bl),b=[];for(var
c=0;c<d.length;c++)switch(d[c]){case
b1:if(b.length>1)b.pop();break;case
ad:case
g:if(b.length==0)b.push(g);break;default:b.push(d[c]);break}b.orig=a;return b}function
aC(){this.content={}}aC.prototype={exists:function(a){return this.content[a]?1:0},mk:function(a,b){this.content[a]=b},get:function(a){return this.content[a]},list:function(){var
a=[];for(var
b
in
this.content)a.push(b);return a},remove:function(a){delete
this.content[a]}};var
co=new
aC();co.mk(g,new
aC());function
dW(a){var
b=co;for(var
c=0;c<a.length;c++){if(!(b.exists&&b.exists(a[c])))gU(a.orig);b=b.get(a[c])}return b}function
tR(a){var
c=cm(a),b=dW(c);return b
instanceof
aC?1:0}function
bq(a){this.data=a}bq.prototype={content:function(){return this.data},truncate:function(){this.data.length=0}};function
s2(a,b){var
e=cm(a),c=co;for(var
f=0;f<e.length-1;f++){var
d=e[f];if(!c.exists(d))c.mk(d,new
aC());c=c.get(d);if(!(c
instanceof
aC))ab(e.orig+dw)}var
d=e[e.length-1];if(c.exists(d))ab(e.orig+dw);if(b
instanceof
aC)c.mk(d,b);else
if(b
instanceof
bq)c.mk(d,b);else
if(b
instanceof
G)c.mk(d,new
bq(b.getArray()));else
if(b
instanceof
Array)c.mk(d,new
bq(b));else
if(b.toString)c.mk(d,new
bq(new
G(b.toString()).getArray()));else
B("caml_fs_register")}function
tN(a){var
b=co,d=cm(a),e;for(var
c=0;c<d.length;c++){if(b.auto)e=b.auto;if(!(b.exists&&b.exists(d[c])))return e?e(d.join(bl)):0;b=b.get(d[c])}return 1}function
bu(a,b,c){if(v.fds===undefined)v.fds=new
Array();c=c?c:{};var
d={};d.array=b;d.offset=c.append?d.array.length:0;d.flags=c;v.fds[a]=d;v.fd_last_idx=a;return a}function
tS(a,b,c){var
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
f=a.toString(),h=cm(a);if(d.rdonly&&d.wronly)ab(f+" : flags Open_rdonly and Open_wronly are not compatible");if(d.text&&d.binary)ab(f+" : flags Open_text and Open_binary are not compatible");if(tN(a)){if(tR(a))ab(f+" : is a directory");if(d.create&&d.excl)ab(f+dw);var
g=v.fd_last_idx?v.fd_last_idx:0,e=dW(h);if(d.truncate)e.truncate();return bu(g+1,e.content(),d)}else
if(d.create){var
g=v.fd_last_idx?v.fd_last_idx:0;s2(a,[]);var
e=dW(h);return bu(g+1,e.content(),d)}else
gU(a)}bu(0,[]);bu(1,[]);bu(2,[]);function
tu(a){var
b=v.fds[a];if(b.flags.wronly)ab(gI+a+" is writeonly");return{data:b,fd:a,opened:true}}function
t0(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=c.console;b&&b.log&&b.log(a)}function
tJ(a,b){var
e=new
G(b),d=e.getLen();for(var
c=0;c<d;c++)a.data.array[a.data.offset+c]=e.get(c);a.data.offset+=d;return 0}function
tv(a){var
b;switch(a){case
1:b=t0;break;case
2:b=d1;break;default:b=tJ}var
d=v.fds[a];if(d.flags.rdonly)ab(gI+a+" is readonly");var
c={data:d,fd:a,opened:true,buffer:g,output:b};br[c.fd]=c;return c}function
tw(){var
a=0;for(var
b
in
br)if(br[b].opened)a=[0,br[b],a];return a}function
gR(a,b,c,d){if(!a.opened)ab("Cannot output to a closed channel");var
f;if(c==0&&b.getLen()==d)f=b;else{f=gM(d);dU(b,c,f,0,d)}var
e=f.toString(),g=e.lastIndexOf("\n");if(g<0)a.buffer+=e;else{a.buffer+=e.substr(0,g+1);dX(a);a.buffer+=e.substr(g+1)}}function
U(a){return new
G(a)}function
tx(a,b){var
c=U(String.fromCharCode(b));gR(a,c,0,1)}function
ty(a,b){if(b==0)gW();return a%b}function
tA(a,b){return+(ck(a,b,false)!=0)}function
tB(a,b){var
d=[a];for(var
c=1;c<=b;c++)d[c]=0;return d}function
tC(a,b){a[0]=b;return 0}function
tD(a){return a
instanceof
Array?a[0]:fX}function
tH(a,b){v[a+1]=b}var
gS={};function
tI(a,b){gS[a.toString()]=b;return 0}function
tK(a,b){return a.compare(b)}function
gX(a,b){var
c=a.fullBytes,d=b.fullBytes;if(c!=null&&d!=null)return c==d?1:0;return a.getFullBytes()==b.getFullBytes()?1:0}function
tL(a,b){return 1-gX(a,b)}function
tM(){return 32}function
tO(){var
a=new
ae("a.out");return[0,a,[0,a]]}function
tP(){return[0,new
ae(fK),32,0]}function
tG(){dZ(v[7])}function
tQ(){tG()}function
tT(){var
a=new
Date()^4294967295*Math.random();return{valueOf:function(){return a},0:0,1:a,length:2}}function
tU(){console.log("caml_sys_system_command");return 0}function
tV(a){var
b=1;while(a&&a.joo_tramp){a=a.joo_tramp.apply(null,a.joo_args);b++}return a}function
tW(a,b){return{joo_tramp:a,joo_args:b}}function
tX(a,b){if(typeof
b==="function"){a.fun=b;return 0}if(b.fun){a.fun=b.fun;return 0}var
c=b.length;while(c--)a[c]=b[c];return 0}function
tz(a){return gS[a]}function
tY(a){if(a
instanceof
Array)return a;if(c.RangeError&&a
instanceof
c.RangeError&&a.message&&a.message.match(/maximum call stack/i))return[0,v[9]];if(c.InternalError&&a
instanceof
c.InternalError&&a.message&&a.message.match(/too much recursion/i))return[0,v[9]];if(a
instanceof
c.Error)return[0,tz(fO),a];return[0,v[3],new
ae(String(a))]}function
tZ(){return 0}var
d2=0;function
t1(){if(window.webcl==undefined){alert("Unfortunately your system does not support WebCL. "+"Make sure that you have both the OpenCL driver "+"and the WebCL browser extension installed.");d2=1}else{console.log("INIT OPENCL");d2=0}return 0}function
t3(){console.log(" spoc_cuInit");return 0}function
t4(){console.log(" spoc_cuda_compile");return 0}function
ua(){console.log(" spoc_cuda_debug_compile");return 0}function
um(a,b,c){console.log(" spoc_debug_opencl_compile");console.log(a.bytes);var
e=c[9],f=e[0],d=f.createProgram(a.bytes),g=d.getInfo(WebCL.PROGRAM_DEVICES);d.build(g);var
h=d.createKernel(b.bytes);e[0]=f;c[9]=e;return h}function
un(a){console.log("spoc_getCudaDevice");return 0}function
uo(){console.log(" spoc_getCudaDevicesCount");return 0}function
up(a,b){console.log(" spoc_getOpenCLDevice");var
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
f=k[o],j=f.getDevices(),m=j.length;console.log("there "+g+F+m+F+a);if(g+m>=a)for(var
q
in
j){var
c=j[q];if(g==a){console.log("current ----------"+g);e[1]=U(c.getInfo(WebCL.DEVICE_NAME));console.log(e[1]);e[2]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_SIZE);e[3]=c.getInfo(WebCL.DEVICE_LOCAL_MEM_SIZE);e[4]=c.getInfo(WebCL.DEVICE_MAX_CLOCK_FREQUENCY);e[5]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE);e[6]=c.getInfo(WebCL.DEVICE_MAX_COMPUTE_UNITS);e[7]=c.getInfo(WebCL.DEVICE_ERROR_CORRECTION_SUPPORT);e[8]=b;var
i=new
Array(3);i[0]=webcl.createContext(c);i[1]=i[0].createCommandQueue();i[2]=i[0].createCommandQueue();e[9]=i;h[1]=U(f.getInfo(WebCL.PLATFORM_PROFILE));h[2]=U(f.getInfo(WebCL.PLATFORM_VERSION));h[3]=U(f.getInfo(WebCL.PLATFORM_NAME));h[4]=U(f.getInfo(WebCL.PLATFORM_VENDOR));h[5]=U(f.getInfo(WebCL.PLATFORM_EXTENSIONS));h[6]=m;var
l=c.getInfo(WebCL.DEVICE_TYPE),v=0;if(l&WebCL.DEVICE_TYPE_CPU)d[2]=0;if(l&WebCL.DEVICE_TYPE_GPU)d[2]=1;if(l&WebCL.DEVICE_TYPE_ACCELERATOR)d[2]=2;if(l&WebCL.DEVICE_TYPE_DEFAULT)d[2]=3;d[3]=U(c.getInfo(WebCL.DEVICE_PROFILE));d[4]=U(c.getInfo(WebCL.DEVICE_VERSION));d[5]=U(c.getInfo(WebCL.DEVICE_VENDOR));var
r=c.getInfo(WebCL.DEVICE_EXTENSIONS);d[6]=U(r);d[7]=c.getInfo(WebCL.DEVICE_VENDOR_ID);d[8]=c.getInfo(WebCL.DEVICE_MAX_WORK_ITEM_DIMENSIONS);d[9]=c.getInfo(WebCL.DEVICE_ADDRESS_BITS);d[10]=c.getInfo(WebCL.DEVICE_MAX_MEM_ALLOC_SIZE);d[11]=c.getInfo(WebCL.DEVICE_IMAGE_SUPPORT);d[12]=c.getInfo(WebCL.DEVICE_MAX_READ_IMAGE_ARGS);d[13]=c.getInfo(WebCL.DEVICE_MAX_WRITE_IMAGE_ARGS);d[14]=c.getInfo(WebCL.DEVICE_MAX_SAMPLERS);d[15]=c.getInfo(WebCL.DEVICE_MEM_BASE_ADDR_ALIGN);d[17]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHELINE_SIZE);d[18]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHE_SIZE);d[19]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_ARGS);d[20]=c.getInfo(WebCL.DEVICE_ENDIAN_LITTLE);d[21]=c.getInfo(WebCL.DEVICE_AVAILABLE);d[22]=c.getInfo(WebCL.DEVICE_COMPILER_AVAILABLE);d[23]=c.getInfo(WebCL.DEVICE_SINGLE_FP_CONFIG);d[24]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHE_TYPE);d[25]=c.getInfo(WebCL.DEVICE_QUEUE_PROPERTIES);d[26]=c.getInfo(WebCL.DEVICE_LOCAL_MEM_TYPE);d[28]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE);d[29]=c.getInfo(WebCL.DEVICE_EXECUTION_CAPABILITIES);d[31]=c.getInfo(WebCL.DEVICE_MAX_WORK_GROUP_SIZE);d[32]=c.getInfo(WebCL.DEVICE_IMAGE2D_MAX_HEIGHT);d[33]=c.getInfo(WebCL.DEVICE_IMAGE2D_MAX_WIDTH);d[34]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_DEPTH);d[35]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_HEIGHT);d[36]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_WIDTH);d[37]=c.getInfo(WebCL.DEVICE_MAX_PARAMETER_SIZE);d[38]=[0];var
n=c.getInfo(WebCL.DEVICE_MAX_WORK_ITEM_SIZES);d[38][1]=n[0];d[38][2]=n[1];d[38][3]=n[2];d[39]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);d[40]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);d[41]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_INT);d[42]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_LONG);d[43]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);d[45]=c.getInfo(WebCL.DEVICE_PROFILING_TIMER_RESOLUTION);d[46]=U(c.getInfo(WebCL.DRIVER_VERSION));g++;break}else
g++}else
g+=m}var
c=[0];d[1]=h;p[1]=d;c[1]=e;c[2]=p;return c}function
uq(){console.log(" spoc_getOpenCLDevicesCount");var
a=0,b=webcl.getPlatforms();for(var
d
in
b){var
e=b[d],c=e.getDevices();a+=c.length}return a}function
ur(){console.log(fI);return 0}function
us(){console.log(fI);var
a=new
Array(3);a[0]=0;return a}function
d3(a){if(a.data
instanceof
Float32Array||a.data.constructor.name=="Float32Array")return 4;if(a.data
instanceof
Int32Array||a.data.constructor.name=="Int32Array")return 4;console.log("unimplemented vector type");console.log(a.data.constructor.name);return 4}function
ut(a,b,c){console.log("spoc_opencl_alloc_vect");var
f=a[2][1],i=a[4],h=i[b+1],j=a[5],k=d3(f),d=c[9],e=d[0],d=c[9],e=d[0],g=e.createBuffer(WebCL.MEM_READ_WRITE,j*k);h[2]=g;d[0]=e;c[9]=d;return 0}function
uu(){console.log(" spoc_opencl_compile");return 0}function
uv(a,b,c,d){console.log("spoc_opencl_cpu_to_device");var
f=a[2][1],k=a[4],j=k[b+1],l=a[5],m=d3(f),e=c[9],h=e[0],g=e[d+1],i=j[2];g.enqueueWriteBuffer(i,false,0,l*m,f.data);e[d+1]=g;e[0]=h;c[9]=e;return 0}function
uB(a,b,c,d,e){console.log("spoc_opencl_device_to_cpu");var
g=a[2][1],l=a[4],k=l[b+1],n=a[5],o=d3(g),f=c[9],j=f[0],i=f[e+1],h=k[2],m=g.data;i.enqueueReadBuffer(h,true,0,n*o,m);console.log("release buffer after transfer to CPU");h.release();f[e+1]=i;f[0]=j;c[9]=f;return 0}function
uC(a,b){console.log("spoc_opencl_flush");var
c=a[9][b+1];c.flush();a[9][b+1]=c;return 0}function
uD(){console.log(" spoc_opencl_is_available");return!d2}function
uE(a,b,c,d,e){console.log("spoc_opencl_launch_grid");var
m=b[1],n=b[2],o=b[3],h=c[1],i=c[2],j=c[3],g=new
Array(3);g[0]=m*h;g[1]=n*i;g[2]=o*j;var
f=new
Array(3);f[0]=h;f[1]=i;f[2]=j;var
l=d[9],k=l[e+1];if(h==1&&i==1&&j==1)k.enqueueNDRangeKernel(a,f.length,null,g);else
k.enqueueNDRangeKernel(a,f.length,null,g,f);return 0}function
uH(a,b,c,d){console.log("spoc_opencl_load_param_int");b.setArg(a[1],new
Uint32Array([c]));a[1]=a[1]+1;return 0}function
uJ(a,b,c,d,e){console.log("spoc_opencl_load_param_vec");var
f=d[2];b.setArg(a[1],f);a[1]=a[1]+1;return 0}function
uM(){return new
Date().getTime()/fX}function
uN(){return 0}var
r=sO,l=sP,bd=dU,aK=sV,E=gM,ay=sX,dd=s0,bS=s1,be=s4,Y=s5,dh=tj,fy=tl,y=tn,fn=tq,df=dX,dj=tr,fl=tu,de=tv,aL=ty,z=bs,b=U,di=tA,fq=tB,aj=tH,dg=tI,fp=tK,bV=gX,A=tL,bT=tQ,fm=tS,fo=tT,fx=tU,Z=tV,I=tW,s=tY,fw=tZ,fz=t1,fB=t3,fC=uo,fA=uq,ft=ur,fs=us,fu=ut,fr=uC,bW=uN;function
j(a,b){return a.length==1?a(b):an(a,[b])}function
i(a,b,c){return a.length==2?a(b,c):an(a,[b,c])}function
o(a,b,c,d){return a.length==3?a(b,c,d):an(a,[b,c,d])}function
fv(a,b,c,d,e,f,g){return a.length==6?a(b,c,d,e,f,g):an(a,[b,c,d,e,f,g])}var
aR=[0,b("Failure")],bv=[0,b("Invalid_argument")],bw=[0,b("End_of_file")],u=[0,b("Not_found")],K=[0,b("Assert_failure")],cO=b(ad),cR=b(ad),cT=b(ad),e6=b(g),e5=[0,b(gv),b(gp),b(f9),b(gD),b(gA)],fk=[0,1],fe=[0,b(gD),b(gp),b(gv),b(f9),b(gA)],ff=[0,b(dH),b(dm),b(dq),b(dr),b(dy),b(ds),b(dF),b(dM),b(dt),b(dx)],fg=[0,b(dA),b(dK),b(dC)],dc=[0,b(dK),b(dq),b(dr),b(dC),b(dA),b(dm),b(dM),b(dx),b(dH),b(ds),b(dF),b(dy),b(dt)];aj(11,[0,b("Undefined_recursive_module")]);aj(8,[0,b("Stack_overflow")]);aj(7,[0,b("Match_failure")]);aj(6,u);aj(5,[0,b("Division_by_zero")]);aj(4,bw);aj(3,bv);aj(2,aR);aj(1,[0,b("Sys_error")]);var
g7=b("really_input"),g6=[0,0,[0,7,0]],g5=[0,1,[0,3,[0,4,[0,7,0]]]],g4=b(gn),g3=b(ad),g1=b("true"),g2=b("false"),g8=b("Pervasives.do_at_exit"),g_=b("Array.blit"),hc=b("List.iter2"),ha=b("tl"),g$=b("hd"),hg=b("\\b"),hh=b("\\t"),hi=b("\\n"),hj=b("\\r"),hf=b("\\\\"),he=b("\\'"),hd=b("Char.chr"),hm=b("String.contains_from"),hl=b("String.blit"),hk=b("String.sub"),ht=b("Map.remove_min_elt"),hu=[0,0,0,0],hv=[0,b("map.ml"),270,10],hw=[0,0,0],hp=b(b6),hq=b(b6),hr=b(b6),hs=b(b6),hx=b("CamlinternalLazy.Undefined"),hA=b("Buffer.add: cannot grow buffer"),hQ=b(g),hR=b(g),hU=b(gn),hV=b(ca),hW=b(ca),hS=b(b9),hT=b(b9),hP=b(fT),hN=b("neg_infinity"),hO=b("infinity"),hM=b(ad),hL=b("printf: bad positional specification (0)."),hK=b("%_"),hJ=[0,b("printf.ml"),gx,8],hH=b(b9),hI=b("Printf: premature end of format string '"),hD=b(b9),hE=b(" in format string '"),hF=b(", at char number "),hG=b("Printf: bad conversion %"),hB=b("Sformat.index_of_int: negative argument "),hY=b(dG),hZ=[0,987910699,495797812,364182224,414272206,318284740,990407751,383018966,270373319,840823159,24560019,536292337,512266505,189156120,730249596,143776328,51606627,140166561,366354223,1003410265,700563762,981890670,913149062,526082594,1021425055,784300257,667753350,630144451,949649812,48546892,415514493,258888527,511570777,89983870,283659902,308386020,242688715,482270760,865188196,1027664170,207196989,193777847,619708188,671350186,149669678,257044018,87658204,558145612,183450813,28133145,901332182,710253903,510646120,652377910,409934019,801085050],sI=b("OCAMLRUNPARAM"),sG=b("CAMLRUNPARAM"),h1=b(g),io=[0,b("camlinternalOO.ml"),287,50],im=b(g),h3=b("CamlinternalOO.last_id"),iQ=b(g),iN=b(gb),iM=b(".\\"),iL=b(go),iK=b("..\\"),iC=b(gb),iB=b(go),ix=b(g),iw=b(g),iy=b(b1),iz=b(bl),sE=b("TMPDIR"),iE=b("/tmp"),iF=b("'\\''"),iI=b(b1),iJ=b("\\"),sC=b("TEMP"),iO=b(ad),iT=b(b1),iU=b(bl),iX=b("Cygwin"),iY=b(fK),iZ=b("Win32"),i0=[0,b("filename.ml"),189,9],i7=b("E2BIG"),i9=b("EACCES"),i_=b("EAGAIN"),i$=b("EBADF"),ja=b("EBUSY"),jb=b("ECHILD"),jc=b("EDEADLK"),jd=b("EDOM"),je=b("EEXIST"),jf=b("EFAULT"),jg=b("EFBIG"),jh=b("EINTR"),ji=b("EINVAL"),jj=b("EIO"),jk=b("EISDIR"),jl=b("EMFILE"),jm=b("EMLINK"),jn=b("ENAMETOOLONG"),jo=b("ENFILE"),jp=b("ENODEV"),jq=b("ENOENT"),jr=b("ENOEXEC"),js=b("ENOLCK"),jt=b("ENOMEM"),ju=b("ENOSPC"),jv=b("ENOSYS"),jw=b("ENOTDIR"),jx=b("ENOTEMPTY"),jy=b("ENOTTY"),jz=b("ENXIO"),jA=b("EPERM"),jB=b("EPIPE"),jC=b("ERANGE"),jD=b("EROFS"),jE=b("ESPIPE"),jF=b("ESRCH"),jG=b("EXDEV"),jH=b("EWOULDBLOCK"),jI=b("EINPROGRESS"),jJ=b("EALREADY"),jK=b("ENOTSOCK"),jL=b("EDESTADDRREQ"),jM=b("EMSGSIZE"),jN=b("EPROTOTYPE"),jO=b("ENOPROTOOPT"),jP=b("EPROTONOSUPPORT"),jQ=b("ESOCKTNOSUPPORT"),jR=b("EOPNOTSUPP"),jS=b("EPFNOSUPPORT"),jT=b("EAFNOSUPPORT"),jU=b("EADDRINUSE"),jV=b("EADDRNOTAVAIL"),jW=b("ENETDOWN"),jX=b("ENETUNREACH"),jY=b("ENETRESET"),jZ=b("ECONNABORTED"),j0=b("ECONNRESET"),j1=b("ENOBUFS"),j2=b("EISCONN"),j3=b("ENOTCONN"),j4=b("ESHUTDOWN"),j5=b("ETOOMANYREFS"),j6=b("ETIMEDOUT"),j7=b("ECONNREFUSED"),j8=b("EHOSTDOWN"),j9=b("EHOSTUNREACH"),j_=b("ELOOP"),j$=b("EOVERFLOW"),ka=b("EUNKNOWNERR %d"),i8=b("Unix.Unix_error(Unix.%s, %S, %S)"),i3=b(gd),i4=b(g),i5=b(g),i6=b(gd),kb=b("0.0.0.0"),kc=b("127.0.0.1"),sB=b("::"),sA=b("::1"),kk=[0,b("Vector.ml"),gh,25],kl=b("Cuda.No_Cuda_Device"),km=b("Cuda.ERROR_DEINITIALIZED"),kn=b("Cuda.ERROR_NOT_INITIALIZED"),ko=b("Cuda.ERROR_INVALID_CONTEXT"),kp=b("Cuda.ERROR_INVALID_VALUE"),kq=b("Cuda.ERROR_OUT_OF_MEMORY"),kr=b("Cuda.ERROR_INVALID_DEVICE"),ks=b("Cuda.ERROR_NOT_FOUND"),kt=b("Cuda.ERROR_FILE_NOT_FOUND"),ku=b("Cuda.ERROR_UNKNOWN"),kv=b("Cuda.ERROR_LAUNCH_FAILED"),kw=b("Cuda.ERROR_LAUNCH_OUT_OF_RESOURCES"),kx=b("Cuda.ERROR_LAUNCH_TIMEOUT"),ky=b("Cuda.ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"),kz=b("no_cuda_device"),kA=b("cuda_error_deinitialized"),kB=b("cuda_error_not_initialized"),kC=b("cuda_error_invalid_context"),kD=b("cuda_error_invalid_value"),kE=b("cuda_error_out_of_memory"),kF=b("cuda_error_invalid_device"),kG=b("cuda_error_not_found"),kH=b("cuda_error_file_not_found"),kI=b("cuda_error_launch_failed"),kJ=b("cuda_error_launch_out_of_resources"),kK=b("cuda_error_launch_timeout"),kL=b("cuda_error_launch_incompatible_texturing"),kM=b("cuda_error_unknown"),kN=b("OpenCL.No_OpenCL_Device"),kO=b("OpenCL.OPENCL_ERROR_UNKNOWN"),kP=b("OpenCL.INVALID_CONTEXT"),kQ=b("OpenCL.INVALID_DEVICE"),kR=b("OpenCL.INVALID_VALUE"),kS=b("OpenCL.INVALID_QUEUE_PROPERTIES"),kT=b("OpenCL.OUT_OF_RESOURCES"),kU=b("OpenCL.MEM_OBJECT_ALLOCATION_FAILURE"),kV=b("OpenCL.OUT_OF_HOST_MEMORY"),kW=b("OpenCL.FILE_NOT_FOUND"),kX=b("OpenCL.INVALID_PROGRAM"),kY=b("OpenCL.INVALID_BINARY"),kZ=b("OpenCL.INVALID_BUILD_OPTIONS"),k0=b("OpenCL.INVALID_OPERATION"),k1=b("OpenCL.COMPILER_NOT_AVAILABLE"),k2=b("OpenCL.BUILD_PROGRAM_FAILURE"),k3=b("OpenCL.INVALID_KERNEL"),k4=b("OpenCL.INVALID_ARG_INDEX"),k5=b("OpenCL.INVALID_ARG_VALUE"),k6=b("OpenCL.INVALID_MEM_OBJECT"),k7=b("OpenCL.INVALID_SAMPLER"),k8=b("OpenCL.INVALID_ARG_SIZE"),k9=b("OpenCL.INVALID_COMMAND_QUEUE"),k_=b("no_opencl_device"),k$=b("opencl_error_unknown"),la=b("opencl_invalid_context"),lb=b("opencl_invalid_device"),lc=b("opencl_invalid_value"),ld=b("opencl_invalid_queue_properties"),le=b("opencl_out_of_resources"),lf=b("opencl_mem_object_allocation_failure"),lg=b("opencl_out_of_host_memory"),lh=b("opencl_file_not_found"),li=b("opencl_invalid_program"),lj=b("opencl_invalid_binary"),lk=b("opencl_invalid_build_options"),ll=b("opencl_invalid_operation"),lm=b("opencl_compiler_not_available"),ln=b("opencl_build_program_failure"),lo=b("opencl_invalid_kernel"),lp=b("opencl_invalid_arg_index"),lq=b("opencl_invalid_arg_value"),lr=b("opencl_invalid_mem_object"),ls=b("opencl_invalid_sampler"),lt=b("opencl_invalid_arg_size"),lu=b("opencl_invalid_command_queue"),lv=b(ce),lw=b(ce),lN=b(f4),lM=b(fY),lL=b(f4),lK=b(fY),lJ=[0,1],lI=b(g),lE=b(bZ),lz=b("Cl LOAD ARG Type Not Implemented\n"),ly=b("CU LOAD ARG Type Not Implemented\n"),lx=[0,b(dx),b(dt),b(dM),b(dF),b(ds),b(dA),b(dy),b(dr),b(dq),b(dK),b(dm),b(dH),b(dC)],lA=b("Kernel.ERROR_BLOCK_SIZE"),lC=b("Kernel.ERROR_GRID_SIZE"),lF=b("Kernel.No_source_for_device"),lQ=b("Empty"),lR=b("Unit"),lS=b("Kern"),lT=b("Params"),lU=b("Plus"),lV=b("Plusf"),lW=b("Min"),lX=b("Minf"),lY=b("Mul"),lZ=b("Mulf"),l0=b("Div"),l1=b("Divf"),l2=b("Mod"),l3=b("Id "),l4=b("IdName "),l5=b("IntVar "),l6=b("FloatVar "),l7=b("UnitVar "),l8=b("CastDoubleVar "),l9=b("DoubleVar "),l_=b("IntArr"),l$=b("Int32Arr"),ma=b("Int64Arr"),mb=b("Float32Arr"),mc=b("Float64Arr"),md=b("VecVar "),me=b("Concat"),mf=b("Seq"),mg=b("Return"),mh=b("Set"),mi=b("Decl"),mj=b("SetV"),mk=b("SetLocalVar"),ml=b("Intrinsics"),mm=b(F),mn=b("IntId "),mo=b("Int "),mq=b("IntVecAcc"),mr=b("Local"),ms=b("Acc"),mt=b("Ife"),mu=b("If"),mv=b("Or"),mw=b("And"),mx=b("EqBool"),my=b("LtBool"),mz=b("GtBool"),mA=b("LtEBool"),mB=b("GtEBool"),mC=b("DoLoop"),mD=b("While"),mE=b("App"),mF=b("GInt"),mG=b("GFloat"),mp=b("Float "),lP=b("  "),lO=b("%s\n"),oi=b(gl),oj=[0,b(dz),166,14],mJ=b(g),mK=b(bZ),mL=b("\n}\n#ifdef __cplusplus\n}\n#endif"),mM=b(" ) {\n"),mN=b(g),mO=b(bY),mQ=b(g),mP=b('#ifdef __cplusplus\nextern "C" {\n#endif\n\n__global__ void spoc_dummy ( '),mR=b(al),mS=b(cf),mT=b(am),mU=b(al),mV=b(cf),mW=b(am),mX=b(al),mY=b(b7),mZ=b(am),m0=b(al),m1=b(b7),m2=b(am),m3=b(al),m4=b(b$),m5=b(am),m6=b(al),m7=b(b$),m8=b(am),m9=b(al),m_=b(ch),m$=b(am),na=b(al),nb=b(ch),nc=b(am),nd=b(al),ne=b(fS),nf=b(am),ng=b(gr),nh=b(fP),ni=[0,b(dz),65,17],nj=b(b8),nk=b(f5),nl=b(O),nm=b(P),nn=b(gc),no=b(O),np=b(P),nq=b(fH),nr=b(O),ns=b(P),nt=b(fZ),nu=b(O),nv=b(P),nw=b(gs),nx=b(gm),nz=b("int"),nA=b("float"),nB=b("double"),ny=[0,b(dz),60,12],nD=b(bY),nC=b(gH),nE=b(gj),nF=b(g),nG=b(g),nJ=b(b2),nK=b(ak),nL=b(aM),nN=b(b2),nM=b(ak),nO=b($),nP=b(O),nQ=b(P),nR=b("}\n"),nS=b(aM),nT=b(aM),nU=b("{"),nV=b(bn),nW=b(fW),nX=b(bn),nY=b(bj),nZ=b(cg),n0=b(bn),n1=b(bj),n2=b(cg),n3=b(fF),n4=b(fD),n5=b(f3),n6=b(gz),n7=b(f_),n8=b(bX),n9=b(gy),n_=b(cd),n$=b(fJ),oa=b(b5),ob=b(bX),oc=b(b5),od=b(ak),oe=b(fE),of=b(cd),og=b(bj),oh=b(f7),om=b(b_),on=b(b_),oo=b(F),op=b(F),ok=b(gg),ol=b(gJ),oq=b($),nH=b(b2),nI=b(ak),or=b(O),os=b(P),ou=b(b8),ov=b($),ow=b(gF),ox=b(O),oy=b(P),oz=b($),ot=b("cuda error parse_float"),mH=[0,b(g),b(g)],pV=b(gl),pW=[0,b(dE),162,14],oC=b(g),oD=b(bZ),oE=b(cd),oF=b(" ) \n{\n"),oG=b(g),oH=b(bY),oJ=b(g),oI=b("__kernel void spoc_dummy ( "),oK=b(cf),oL=b(cf),oM=b(b7),oN=b(b7),oO=b(b$),oP=b(b$),oQ=b(ch),oR=b(ch),oS=b(fS),oT=b(gr),oU=b(fP),oV=[0,b(dE),65,17],oW=b(b8),oX=b(f5),oY=b(O),oZ=b(P),o0=b(gc),o1=b(O),o2=b(P),o3=b(fH),o4=b(O),o5=b(P),o6=b(fZ),o7=b(O),o8=b(P),o9=b(gs),o_=b(gm),pa=b("__global int"),pb=b("__global float"),pc=b("__global double"),o$=[0,b(dE),60,12],pe=b(bY),pd=b(gH),pf=b(gj),pg=b(g),ph=b(g),pj=b(b2),pk=b(ak),pl=b(aM),pm=b(ak),pn=b($),po=b(O),pp=b(P),pq=b(g),pr=b(bZ),ps=b(aM),pt=b(g),pu=b(bn),pv=b(fW),pw=b(bn),px=b(bj),py=b(cg),pz=b(cd),pA=b(aM),pB=b("{\n"),pC=b(")\n"),pD=b(cg),pE=b(fF),pF=b(fD),pG=b(f3),pH=b(gz),pI=b(f_),pJ=b(bX),pK=b(gy),pL=b(gB),pM=b(fJ),pN=b(b5),pO=b(bX),pP=b(b5),pQ=b(ak),pR=b(fE),pS=b(gB),pT=b(bj),pU=b(f7),pZ=b(b_),p0=b(b_),p1=b(F),p2=b(F),pX=b(gg),pY=b(gJ),p3=b($),pi=b(ak),p4=b(O),p5=b(P),p7=b(b8),p8=b($),p9=b(gF),p_=b(O),p$=b(P),qa=b($),p6=b("opencl error parse_float"),oA=[0,b(g),b(g)],q_=[0,0],q$=[0,0],ra=[0,1],rb=[0,1],q4=b("kirc_kernel.cu"),q5=b("nvcc -m64 -arch=sm_10 -O3 -ptx kirc_kernel.cu -o kirc_kernel.ptx"),q6=b("kirc_kernel.ptx"),q7=b("rm kirc_kernel.cu kirc_kernel.ptx"),q1=[0,b(g),b(g)],q3=b(g),q2=[0,b("Kirc.ml"),411,81],q8=b(ak),q9=b(gu),qY=[33,0],qU=b(gu),qb=b("int spoc_xor (int a, int b ) { return (a^b);}\n"),qc=b("int spoc_powint (int a, int b ) { return ((int) pow (((float) a), ((float) b)));}\n"),qd=b("int logical_and (int a, int b ) { return (a & b);}\n"),qe=b("float spoc_fdiv ( float a, float b ) { return (a / b);}\n"),qf=b("float spoc_fmul ( float a, float b ) { return (a * b);}\n"),qg=b("float spoc_fminus ( float a, float b ) { return (a - b);}\n"),qh=b("float spoc_fadd ( float a, float b ) { return (a + b);}\n"),qi=b("float spoc_fdiv ( float a, float b );\n"),qj=b("float spoc_fmul ( float a, float b );\n"),qk=b("float spoc_fminus ( float a, float b );\n"),ql=b("float spoc_fadd ( float a, float b );\n"),qn=b(dD),qo=b("double spoc_ddiv ( double a, double b ) { return (a / b);}\n"),qp=b("double spoc_dmul ( double a, double b ) { return (a * b);}\n"),qq=b("double spoc_dminus ( double a, double b ) { return (a - b);}\n"),qr=b("double spoc_dadd ( double a, double b ) { return (a + b);}\n"),qs=b("double spoc_ddiv ( double a, double b );\n"),qt=b("double spoc_dmul ( double a, double b );\n"),qu=b("double spoc_dminus ( double a, double b );\n"),qv=b("double spoc_dadd ( double a, double b );\n"),qw=b(dD),qx=b("#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"),qy=b("#elif defined(cl_amd_fp64)  // AMD extension available?\n"),qz=b("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"),qA=b("#if defined(cl_khr_fp64)  // Khronos extension available?\n"),qB=b(gE),qC=b(gw),qE=b(dD),qF=b("__device__ double spoc_ddiv ( double a, double b ) { return (a / b);}\n"),qG=b("__device__ double spoc_dmul ( double a, double b ) { return (a * b);}\n"),qH=b("__device__ double spoc_dminus ( double a, double b ) { return (a - b);}\n"),qI=b("__device__ double spoc_dadd ( double a, double b ) { return (a + b);}\n"),qJ=b(gE),qK=b(gw),qM=b("__device__ int spoc_xor (int a, int b ) { return (a^b);}\n"),qN=b("__device__ int spoc_powint (int a, int b ) { return ((int) pow (((double) a), ((double) b)));}\n"),qO=b("__device__ int logical_and (int a, int b ) { return (a & b);}\n"),qP=b("__device__ float spoc_fdiv ( float a, float b ) { return (a / b);}\n"),qQ=b("__device__ float spoc_fmul ( float a, float b ) { return (a * b);}\n"),qR=b("__device__ float spoc_fminus ( float a, float b ) { return (a - b);}\n"),qS=b("__device__ float spoc_fadd ( float a, float b ) { return (a + b);}\n"),qZ=[0,b(g),b(g)],rd=b("Js.Error"),re=b(fO),rp=b("canvas"),rm=b("span"),rl=b("img"),rk=b("br"),rj=b("textarea"),ri=b(fG),rh=b("select"),rg=b("option"),rn=b("Dom_html.Canvas_not_available"),sw=[0,1],sx=b("GO"),sy=b("Reset picture"),sv=[0,b(fQ),134,17],ss=b("Will use device : %s!"),st=b("%s"),su=b(g),sr=b("Time %s : %Fs\n%!"),rz=b("spoc_dummy"),rA=b("kirc_kernel"),ry=b("spoc_kernel_extension error"),rq=[0,b(fQ),12,15],rR=b(aQ),rS=b(aQ),rY=b(aQ),rZ=b(aQ),r4=b(aQ),r5=b(aQ),r8=b(f$),r9=b(f$),sf=b("(get_group_id (0))"),sg=b("blockIdx.x"),si=b("(get_local_size (0))"),sj=b("blockDim.x"),sl=b("(get_local_id (0))"),sm=b("threadIdx.x");function
V(a){throw[0,aR,a]}function
J(a){throw[0,bv,a]}function
h(a,b){var
c=a.getLen(),e=b.getLen(),d=E(c+e|0);bd(a,0,d,0,c);bd(b,0,d,c,e);return d}function
k(a){return b(g+a)}function
Q(a){var
c=dd(g4,a),b=0,e=c.getLen();for(;;){if(e<=b)return h(c,g3);var
d=c.safeGet(b),f=48<=d?58<=d?0:1:45===d?1:0;if(f){var
b=b+1|0;continue}return c}}function
cp(a,b){if(a){var
c=a[1];return[0,c,cp(a[2],b)]}return b}fl(0);var
d4=de(1);de(2);function
d5(a,b){return gR(a,b,0,b.getLen())}function
d6(a){return fl(fm(a,g6,0))}function
d7(a){var
b=tw(0);for(;;){if(b){var
c=b[2],d=b[1];try{df(d)}catch(f){}var
b=c;continue}return 0}}dg(g8,d7);function
d8(a){return fn(a)}function
g9(a,b){return tx(a,b)}function
d9(a){return df(a)}function
d_(a,b){var
d=b.length-1-1|0,e=0;if(!(d<0)){var
c=e;for(;;){j(a,b[c+1]);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return 0}function
aS(a,b){var
d=b.length-1;if(0===d)return[0];var
e=y(d,j(a,b[0+1])),f=d-1|0,g=1;if(!(f<1)){var
c=g;for(;;){e[c+1]=j(a,b[c+1]);var
h=c+1|0;if(f!==c){var
c=h;continue}break}}return e}function
cq(a,b){var
d=b.length-1-1|0,e=0;if(!(d<0)){var
c=e;for(;;){i(a,c,b[c+1]);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return 0}function
aT(a){var
b=a.length-1-1|0,c=0;for(;;){if(0<=b){var
d=[0,a[b+1],c],b=b-1|0,c=d;continue}return c}}function
d$(a,b,c){var
e=[0,b],f=c.length-1-1|0,g=0;if(!(f<0)){var
d=g;for(;;){e[1]=i(a,e[1],c[d+1]);var
h=d+1|0;if(f!==d){var
d=h;continue}break}}return e[1]}function
ea(a){return a?a[1]:V(g$)}function
eb(a){var
b=a,c=0;for(;;){if(b){var
d=[0,b[1],c],b=b[2],c=d;continue}return c}}function
cr(a,b){if(b){var
c=b[2],d=j(a,b[1]);return[0,d,cr(a,c)]}return 0}function
ct(a,b,c){if(b){var
d=b[1];return i(a,d,ct(a,b[2],c))}return c}function
ed(a,b,c){var
e=b,d=c;for(;;){if(e){if(d){var
f=d[2],g=e[2];i(a,e[1],d[1]);var
e=g,d=f;continue}}else
if(!d)return 0;return J(hc)}}function
cu(a,b){var
c=b;for(;;){if(c){var
e=c[2],d=0===aK(c[1],a)?1:0;if(d)return d;var
c=e;continue}return 0}}function
cv(a){if(0<=a)if(!(q<a))return a;return J(hd)}function
ee(a){var
b=65<=a?90<a?0:1:0;if(!b){var
c=192<=a?214<a?0:1:0;if(!c){var
d=216<=a?222<a?1:0:1;if(d)return a}}return a+32|0}function
ao(a,b){var
c=E(a);sZ(c,0,a,b);return c}function
w(a,b,c){if(0<=b)if(0<=c)if(!((a.getLen()-c|0)<b)){var
d=E(c);bd(a,b,d,0,c);return d}return J(hk)}function
by(a,b,c,d,e){if(0<=e)if(0<=b)if(!((a.getLen()-e|0)<b))if(0<=d)if(!((c.getLen()-e|0)<d))return bd(a,b,c,d,e);return J(hl)}function
ef(a){var
c=a.getLen();if(0===c)return a;var
d=E(c),e=c-1|0,f=0;if(!(e<0)){var
b=f;for(;;){d.safeSet(b,ee(a.safeGet(b)));var
g=b+1|0;if(e!==b){var
b=g;continue}break}}return d}var
cw=tP(0)[1],aD=tM(0),cx=(1<<(aD-10|0))-1|0,aU=z(aD/8|0,cx)-1|0,ho=tO(0)[2];function
cy(k){function
h(a){return a?a[5]:0}function
e(a,b,c,d){var
e=h(a),f=h(d),g=f<=e?e+1|0:f+1|0;return[0,a,b,c,d,g]}function
q(a,b){return[0,0,a,b,0,1]}function
f(a,b,c,d){var
i=a?a[5]:0,j=d?d[5]:0;if((j+2|0)<i){if(a){var
f=a[4],m=a[3],n=a[2],k=a[1],q=h(f);if(q<=h(k))return e(k,n,m,e(f,b,c,d));if(f){var
r=f[3],s=f[2],t=f[1],u=e(f[4],b,c,d);return e(e(k,n,m,t),s,r,u)}return J(hp)}return J(hq)}if((i+2|0)<j){if(d){var
l=d[4],o=d[3],p=d[2],g=d[1],v=h(g);if(v<=h(l))return e(e(a,b,c,g),p,o,l);if(g){var
w=g[3],x=g[2],y=g[1],z=e(g[4],p,o,l);return e(e(a,b,c,y),x,w,z)}return J(hr)}return J(hs)}var
A=j<=i?i+1|0:j+1|0;return[0,a,b,c,d,A]}var
a=0;function
G(a){return a?0:1}function
r(a,b,c){if(c){var
d=c[4],h=c[3],e=c[2],g=c[1],l=c[5],j=i(k[1],a,e);return 0===j?[0,g,a,b,d,l]:0<=j?f(g,e,h,r(a,b,d)):f(r(a,b,g),e,h,d)}return[0,0,a,b,0,1]}function
H(a,b){var
c=b;for(;;){if(c){var
e=c[4],f=c[3],g=c[1],d=i(k[1],a,c[2]);if(0===d)return f;var
h=0<=d?e:g,c=h;continue}throw[0,u]}}function
I(a,b){var
c=b;for(;;){if(c){var
f=c[4],g=c[1],d=i(k[1],a,c[2]),e=0===d?1:0;if(e)return e;var
h=0<=d?f:g,c=h;continue}return 0}}function
n(a){var
b=a;for(;;){if(b){var
c=b[1];if(c){var
b=c;continue}return[0,b[2],b[3]]}throw[0,u]}}function
L(a){var
b=a;for(;;){if(b){var
c=b[4],d=b[3],e=b[2];if(c){var
b=c;continue}return[0,e,d]}throw[0,u]}}function
s(a){if(a){var
b=a[1];if(b){var
c=a[4],d=a[3],e=a[2];return f(s(b),e,d,c)}return a[4]}return J(ht)}function
t(a,b){if(b){var
c=b[4],h=b[3],e=b[2],d=b[1],j=i(k[1],a,e);if(0===j){if(d){if(c){var
g=n(c),l=g[2],m=g[1];return f(d,m,l,s(c))}return d}return c}return 0<=j?f(d,e,h,t(a,c)):f(t(a,d),e,h,c)}return 0}function
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
E(a,b,c,d){return c?g(a,b,c[1],d):p(a,d)}function
l(a,b){if(b){var
c=b[4],d=b[3],e=b[2],f=b[1],m=i(k[1],a,e);if(0===m)return[0,f,[0,d],c];if(0<=m){var
h=l(a,c),n=h[3],o=h[2];return[0,g(f,e,d,h[1]),o,n]}var
j=l(a,f),p=j[2],q=j[1];return[0,q,p,g(j[3],e,d,c)]}return hu}function
m(a,b,c){if(b){var
d=b[2],i=b[5],j=b[4],k=b[3],n=b[1];if(h(c)<=i){var
e=l(d,c),p=e[2],q=e[1],r=m(a,j,e[3]),s=o(a,d,[0,k],p);return E(m(a,n,q),d,s,r)}}else
if(!c)return 0;if(c){var
f=c[2],t=c[4],u=c[3],v=c[1],g=l(f,b),w=g[2],x=g[1],y=m(a,g[3],t),z=o(a,f,w,[0,u]);return E(m(a,x,v),f,z,y)}throw[0,K,hv]}function
w(a,b){if(b){var
c=b[3],d=b[2],h=b[4],e=w(a,b[1]),j=i(a,d,c),f=w(a,h);return j?g(e,d,c,f):p(e,f)}return 0}function
x(a,b){if(b){var
c=b[3],d=b[2],m=b[4],e=x(a,b[1]),f=e[2],h=e[1],n=i(a,d,c),j=x(a,m),k=j[2],l=j[1];if(n){var
o=p(f,k);return[0,g(h,d,c,l),o]}var
q=g(f,d,c,k);return[0,p(h,l),q]}return hw}function
d(a,b){var
c=a,d=b;for(;;){if(c){var
e=[0,c[2],c[3],c[4],d],c=c[1],d=e;continue}return d}}function
M(a,b,c){var
r=d(c,0),f=d(b,0),e=r;for(;;){if(f){if(e){var
j=e[4],l=e[3],m=e[2],n=f[4],o=f[3],p=f[2],g=i(k[1],f[1],e[1]);if(0===g){var
h=i(a,p,m);if(0===h){var
q=d(l,j),f=d(o,n),e=q;continue}return h}return g}return 1}return e?-1:0}}function
N(a,b,c){var
s=d(c,0),f=d(b,0),e=s;for(;;){if(f){if(e){var
l=e[4],m=e[3],n=e[2],o=f[4],p=f[3],q=f[2],g=0===i(k[1],f[1],e[1])?1:0;if(g){var
h=i(a,q,n);if(h){var
r=d(m,l),f=d(p,o),e=r;continue}var
j=h}else
var
j=g;return j}return 0}return e?0:1}}function
b(a){if(a){var
c=a[1],d=b(a[4]);return(b(c)+1|0)+d|0}return 0}function
F(a,b){var
d=a,c=b;for(;;){if(c){var
e=c[3],f=c[2],g=c[1],d=[0,[0,f,e],F(d,c[4])],c=g;continue}return d}}return[0,a,G,I,r,q,t,m,M,N,y,z,A,B,w,x,b,function(a){return F(0,a)},n,L,n,l,H,c,v]}var
hy=[0,hx];function
hz(a){throw[0,hy]}function
aV(a){var
b=1<=a?a:1,c=aU<b?aU:b,d=E(c);return[0,d,0,c,d]}function
aW(a){return w(a[1],0,a[2])}function
ei(a,b){var
c=[0,a[3]];for(;;){if(c[1]<(a[2]+b|0)){c[1]=2*c[1]|0;continue}if(aU<c[1])if((a[2]+b|0)<=aU)c[1]=aU;else
V(hA);var
d=E(c[1]);by(a[1],0,d,0,a[2]);a[1]=d;a[3]=c[1];return 0}}function
L(a,b){var
c=a[2];if(a[3]<=c)ei(a,1);a[1].safeSet(c,b);a[2]=c+1|0;return 0}function
bA(a,b){var
c=b.getLen(),d=a[2]+c|0;if(a[3]<d)ei(a,c);by(b,0,a[1],a[2],c);a[2]=d;return 0}function
cz(a){return 0<=a?a:V(h(hB,k(a)))}function
ej(a,b){return cz(a+b|0)}var
hC=1;function
ek(a){return ej(hC,a)}function
el(a){return w(a,0,a.getLen())}function
em(a,b,c){var
d=h(hE,h(a,hD)),e=h(hF,h(k(b),d));return J(h(hG,h(ao(1,c),e)))}function
aX(a,b,c){return em(el(a),b,c)}function
bB(a){return J(h(hI,h(el(a),hH)))}function
at(e,b,c,d){function
h(a){if((e.safeGet(a)+aP|0)<0||9<(e.safeGet(a)+aP|0))return a;var
b=a+1|0;for(;;){var
c=e.safeGet(b);if(48<=c){if(!(58<=c)){var
b=b+1|0;continue}}else
if(36===c)return b+1|0;return a}}var
i=h(b+1|0),f=aV((c-i|0)+10|0);L(f,37);var
a=i,g=eb(d);for(;;){if(a<=c){var
j=e.safeGet(a);if(42===j){if(g){var
l=g[2];bA(f,k(g[1]));var
a=h(a+1|0),g=l;continue}throw[0,K,hJ]}L(f,j);var
a=a+1|0;continue}return aW(f)}}function
en(a,b,c,d,e){var
f=at(b,c,d,e);if(78!==a)if(bm!==a)return f;f.safeSet(f.getLen()-1|0,dI);return f}function
eo(a){return function(d,b){var
k=d.getLen();function
l(a,b){var
m=40===a?41:dp,c=b;for(;;){if(k<=c)return bB(d);if(37===d.safeGet(c)){var
e=c+1|0;if(k<=e)return bB(d);var
f=d.safeGet(e),g=f-40|0;if(g<0||1<g){var
i=g-83|0;if(i<0||2<i)var
h=1;else
switch(i){case
1:var
h=1;break;case
2:var
j=1,h=0;break;default:var
j=0,h=0}if(h){var
c=e+1|0;continue}}else
var
j=0===g?0:1;if(j)return f===m?e+1|0:aX(d,b,f);var
c=l(f,e+1|0)+1|0;continue}var
c=c+1|0;continue}}return l(a,b)}}function
ep(j,b,c){var
m=j.getLen()-1|0;function
s(a){var
l=a;a:for(;;){if(l<m){if(37===j.safeGet(l)){var
f=0,h=l+1|0;for(;;){if(m<h)var
e=bB(j);else{var
n=j.safeGet(h);if(58<=n){if(95===n){var
f=1,h=h+1|0;continue}}else
if(32<=n)switch(n+fU|0){case
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
h=o(b,f,h,aA);continue;default:var
h=h+1|0;continue}var
d=h;b:for(;;){if(m<d)var
e=bB(j);else{var
k=j.safeGet(d);if(gh<=k)var
g=0;else
switch(k){case
78:case
88:case
aN:case
aA:case
du:case
dI:case
dJ:var
e=o(b,f,d,aA),g=1;break;case
69:case
70:case
71:case
gC:case
dv:case
dN:var
e=o(b,f,d,dv),g=1;break;case
33:case
37:case
44:case
64:var
e=d+1|0,g=1;break;case
83:case
91:case
bp:var
e=o(b,f,d,bp),g=1;break;case
97:case
cb:case
dn:var
e=o(b,f,d,k),g=1;break;case
76:case
f6:case
bm:var
t=d+1|0;if(m<t)var
e=o(b,f,d,aA),g=1;else{var
q=j.safeGet(t)+gG|0;if(q<0||32<q)var
r=1;else
switch(q){case
0:case
12:case
17:case
23:case
29:case
32:var
e=i(c,o(b,f,d,k),aA),g=1,r=0;break;default:var
r=1}if(r)var
e=o(b,f,d,aA),g=1}break;case
67:case
99:var
e=o(b,f,d,99),g=1;break;case
66:case
98:var
e=o(b,f,d,66),g=1;break;case
41:case
dp:var
e=o(b,f,d,k),g=1;break;case
40:var
e=s(o(b,f,d,k)),g=1;break;case
dL:var
u=o(b,f,d,k),v=i(eo(k),j,u),p=u;for(;;){if(p<(v-2|0)){var
p=i(c,p,j.safeGet(p));continue}var
d=v-1|0;continue b}default:var
g=0}if(!g)var
e=aX(j,d,k)}break}}var
l=e;continue a}}var
l=l+1|0;continue}return l}}s(0);return 0}function
eq(a){var
d=[0,0,0,0];function
b(a,b,c){var
f=41!==c?1:0,g=f?dp!==c?1:0:f;if(g){var
e=97===c?2:1;if(cb===c)d[3]=d[3]+1|0;if(a)d[2]=d[2]+e|0;else
d[1]=d[1]+e|0}return b+1|0}ep(a,b,function(a,b){return a+1|0});return d[1]}function
er(a,b,c){var
g=a.safeGet(c);if((g+aP|0)<0||9<(g+aP|0))return i(b,0,c);var
e=g+aP|0,d=c+1|0;for(;;){var
f=a.safeGet(d);if(48<=f){if(!(58<=f)){var
e=(10*e|0)+(f+aP|0)|0,d=d+1|0;continue}}else
if(36===f)return 0===e?V(hL):i(b,[0,cz(e-1|0)],d+1|0);return i(b,0,c)}}function
R(a,b){return a?b:ek(b)}function
es(a,b){return a?a[1]:b}function
et(aG,b,c,d,e,f,g){var
B=j(b,g);function
ae(a){return i(d,B,a)}function
aH(a,b,k,aI){var
l=k.getLen();function
C(o,b){var
n=b;for(;;){if(l<=n)return j(a,B);var
d=k.safeGet(n);if(37===d){var
m=function(a,b){return r(aI,es(a,b))},ar=function(g,f,c,d){var
a=d;for(;;){var
$=k.safeGet(a)+fU|0;if(!($<0||25<$))switch($){case
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
10:return er(k,function(a,b){var
d=[0,m(a,f),c];return ar(g,R(a,f),d,b)},a+1|0);default:var
a=a+1|0;continue}var
o=k.safeGet(a);if(!(124<=o))switch(o){case
78:case
88:case
aN:case
aA:case
du:case
dI:case
dJ:var
a6=m(g,f),a7=bS(en(o,k,n,a,c),a6);return p(R(g,f),a7,a+1|0);case
69:case
71:case
gC:case
dv:case
dN:var
aZ=m(g,f),a0=dd(at(k,n,a,c),aZ);return p(R(g,f),a0,a+1|0);case
76:case
f6:case
bm:var
ac=k.safeGet(a+1|0)+gG|0;if(!(ac<0||32<ac))switch(ac){case
0:case
12:case
17:case
23:case
29:case
32:var
T=a+1|0,ad=o-108|0;if(ad<0||2<ad)var
af=0;else{switch(ad){case
1:var
af=0,ag=0;break;case
2:var
a5=m(g,f),ay=bS(at(k,n,T,c),a5),ag=1;break;default:var
a4=m(g,f),ay=bS(at(k,n,T,c),a4),ag=1}if(ag)var
ax=ay,af=1}if(!af)var
a3=m(g,f),ax=s_(at(k,n,T,c),a3);return p(R(g,f),ax,T+1|0)}var
a1=m(g,f),a2=bS(en(bm,k,n,a,c),a1);return p(R(g,f),a2,a+1|0);case
37:case
64:return p(f,ao(1,o),a+1|0);case
83:case
bp:var
x=m(g,f);if(bp===o)var
y=x;else{var
b=[0,0],ak=x.getLen()-1|0,aJ=0;if(!(ak<0)){var
K=aJ;for(;;){var
v=x.safeGet(K),bb=14<=v?34===v?1:92===v?1:0:11<=v?13<=v?1:0:8<=v?1:0,aM=bb?2:dh(v)?1:4;b[1]=b[1]+aM|0;var
aO=K+1|0;if(ak!==K){var
K=aO;continue}break}}if(b[1]===x.getLen())var
aB=x;else{var
l=E(b[1]);b[1]=0;var
al=x.getLen()-1|0,aK=0;if(!(al<0)){var
J=aK;for(;;){var
u=x.safeGet(J),z=u-34|0;if(z<0||58<z)if(-20<=z)var
U=1;else{switch(z+34|0){case
8:l.safeSet(b[1],92);b[1]++;l.safeSet(b[1],98);var
I=1;break;case
9:l.safeSet(b[1],92);b[1]++;l.safeSet(b[1],dn);var
I=1;break;case
10:l.safeSet(b[1],92);b[1]++;l.safeSet(b[1],bm);var
I=1;break;case
13:l.safeSet(b[1],92);b[1]++;l.safeSet(b[1],cb);var
I=1;break;default:var
U=1,I=0}if(I)var
U=0}else
var
U=(z-1|0)<0||56<(z-1|0)?(l.safeSet(b[1],92),b[1]++,l.safeSet(b[1],u),0):1;if(U)if(dh(u))l.safeSet(b[1],u);else{l.safeSet(b[1],92);b[1]++;l.safeSet(b[1],48+(u/aN|0)|0);b[1]++;l.safeSet(b[1],48+((u/10|0)%10|0)|0);b[1]++;l.safeSet(b[1],48+(u%10|0)|0)}b[1]++;var
aL=J+1|0;if(al!==J){var
J=aL;continue}break}}var
aB=l}var
y=h(hW,h(aB,hV))}if(a===(n+1|0))var
az=y;else{var
H=at(k,n,a,c);try{var
V=0,r=1;for(;;){if(H.getLen()<=r)var
am=[0,0,V];else{var
W=H.safeGet(r);if(49<=W)if(58<=W)var
ah=0;else
var
am=[0,ti(w(H,r,(H.getLen()-r|0)-1|0)),V],ah=1;else{if(45===W){var
V=1,r=r+1|0;continue}var
ah=0}if(!ah){var
r=r+1|0;continue}}var
Y=am;break}}catch(f){f=s(f);if(f[1]!==aR)throw f;var
Y=em(H,0,bp)}var
M=Y[1],A=y.getLen(),aP=Y[2],N=0,aQ=32;if(M===A)if(0===N)var
Z=y,ai=1;else
var
ai=0;else
var
ai=0;if(!ai)if(M<=A)var
Z=w(y,N,A);else{var
X=ao(M,aQ);if(aP)by(y,N,X,0,A);else
by(y,N,X,M-A|0,A);var
Z=X}var
az=Z}return p(R(g,f),az,a+1|0);case
67:case
99:var
q=m(g,f);if(99===o)var
av=ao(1,q);else{if(39===q)var
t=he;else
if(92===q)var
t=hf;else{if(14<=q)var
D=0;else
switch(q){case
8:var
t=hg,D=1;break;case
9:var
t=hh,D=1;break;case
10:var
t=hi,D=1;break;case
13:var
t=hj,D=1;break;default:var
D=0}if(!D)if(dh(q)){var
aj=E(1);aj.safeSet(0,q);var
t=aj}else{var
F=E(4);F.safeSet(0,92);F.safeSet(1,48+(q/aN|0)|0);F.safeSet(2,48+((q/10|0)%10|0)|0);F.safeSet(3,48+(q%10|0)|0);var
t=F}}var
av=h(hT,h(t,hS))}return p(R(g,f),av,a+1|0);case
66:case
98:var
aU=a+1|0,aY=m(g,f)?g1:g2;return p(R(g,f),aY,aU);case
40:case
dL:var
S=m(g,f),as=i(eo(o),k,a+1|0);if(dL===o){var
O=aV(S.getLen()),an=function(a,b){L(O,b);return a+1|0};ep(S,function(a,b,c){if(a)bA(O,hK);else
L(O,37);return an(b,c)},an);var
aS=aW(O);return p(R(g,f),aS,as)}var
au=R(g,f),ba=ej(eq(S),au);return aH(function(a){return C(ba,as)},au,S,aI);case
33:j(e,B);return C(f,a+1|0);case
41:return p(f,hQ,a+1|0);case
44:return p(f,hR,a+1|0);case
70:var
aa=m(g,f);if(0===c)var
aw=hU;else{var
_=at(k,n,a,c);if(70===o)_.safeSet(_.getLen()-1|0,dN);var
aw=_}var
aq=sU(aa);if(3===aq)var
ab=aa<0?hN:hO;else
if(4<=aq)var
ab=hP;else{var
Q=dd(aw,aa),P=0,aT=Q.getLen();for(;;){if(aT<=P)var
ap=h(Q,hM);else{var
G=Q.safeGet(P)-46|0,bc=G<0||23<G?55===G?1:0:(G-1|0)<0||21<(G-1|0)?1:0;if(!bc){var
P=P+1|0;continue}var
ap=Q}var
ab=ap;break}}return p(R(g,f),ab,a+1|0);case
91:return aX(k,a,o);case
97:var
aC=m(g,f),aD=ek(es(g,f)),aE=m(0,aD),a8=a+1|0,a9=R(g,aD);if(aG)ae(i(aC,0,aE));else
i(aC,B,aE);return C(a9,a8);case
cb:return aX(k,a,o);case
dn:var
aF=m(g,f),a_=a+1|0,a$=R(g,f);if(aG)ae(j(aF,0));else
j(aF,B);return C(a$,a_)}return aX(k,a,o)}},f=n+1|0,g=0;return er(k,function(a,b){return ar(a,o,g,b)},f)}i(c,B,d);var
n=n+1|0;continue}}function
p(a,b,c){ae(b);return C(a,c)}return C(b,0)}var
o=cz(0);function
k(a,b){return aH(f,o,a,b)}var
m=eq(g);if(m<0||6<m){var
n=function(f,b){if(m<=f){var
h=y(m,0),i=function(a,b){return l(h,(m-a|0)-1|0,b)},c=0,a=b;for(;;){if(a){var
d=a[2],e=a[1];if(d){i(c,e);var
c=c+1|0,a=d;continue}i(c,e)}return k(g,h)}}return function(a){return n(f+1|0,[0,a,b])}};return n(0,0)}switch(m){case
1:return function(a){var
b=y(1,0);l(b,0,a);return k(g,b)};case
2:return function(a,b){var
c=y(2,0);l(c,0,a);l(c,1,b);return k(g,c)};case
3:return function(a,b,c){var
d=y(3,0);l(d,0,a);l(d,1,b);l(d,2,c);return k(g,d)};case
4:return function(a,b,c,d){var
e=y(4,0);l(e,0,a);l(e,1,b);l(e,2,c);l(e,3,d);return k(g,e)};case
5:return function(a,b,c,d,e){var
f=y(5,0);l(f,0,a);l(f,1,b);l(f,2,c);l(f,3,d);l(f,4,e);return k(g,f)};case
6:return function(a,b,c,d,e,f){var
h=y(6,0);l(h,0,a);l(h,1,b);l(h,2,c);l(h,3,d);l(h,4,e);l(h,5,f);return k(g,h)};default:return k(g,[0])}}function
eu(a){function
b(a){return 0}return et(0,function(a){return d4},g9,d5,d9,b,a)}function
hX(a){return aV(2*a.getLen()|0)}function
ev(c){function
b(a){var
b=aW(a);a[2]=0;return j(c,b)}function
d(a){return 0}var
e=1;return function(a){return et(e,hX,L,bA,d,b,a)}}function
cA(a){return j(ev(function(a){return a}),a)}var
ew=[0,0];function
cB(a){ew[1]=[0,a,ew[1]];return 0}function
ex(a,b){var
j=0===b.length-1?[0,0]:b,f=j.length-1,o=0;if(!0){var
d=o;for(;;){l(a[1],d,d);var
v=d+1|0;if(54!==d){var
d=v;continue}break}}var
g=[0,hY],p=0,q=55,s=s6(55,f)?q:f,m=54+s|0;if(!(m<0)){var
c=p;for(;;){var
n=c%55|0,t=g[1],i=h(t,k(r(j,aL(c,f))));g[1]=to(i,0,i.getLen());var
e=g[1];l(a[1],n,(r(a[1],n)^(((e.safeGet(0)+(e.safeGet(1)<<8)|0)+(e.safeGet(2)<<16)|0)+(e.safeGet(3)<<24)|0))&bk);var
u=c+1|0;if(m!==c){var
c=u;continue}break}}a[2]=0;return 0}32===aD;var
h0=[0,hZ.slice(),0];try{var
sJ=bT(sI),cC=sJ}catch(f){f=s(f);if(f[1]!==u)throw f;try{var
sH=bT(sG),ey=sH}catch(f){f=s(f);if(f[1]!==u)throw f;var
ey=h1}var
cC=ey}var
eg=cC.getLen(),h2=82,eh=0;if(0<=0)if(eg<eh)var
bU=0;else
try{var
bz=eh;for(;;){if(eg<=bz)throw[0,u];if(cC.safeGet(bz)!==h2){var
bz=bz+1|0;continue}var
hn=1,cD=hn,bU=1;break}}catch(f){f=s(f);if(f[1]!==u)throw f;var
cD=0,bU=1}else
var
bU=0;if(!bU)var
cD=J(hm);var
ap=[ge,function(a){var
b=[0,y(55,0),0];ex(b,fo(0));return b}];function
ez(a,b){var
m=a?a[1]:cD,d=16;for(;;){if(!(b<=d))if(!(cx<(d*2|0))){var
d=d*2|0;continue}if(m){var
h=tD(ap);if(aO===h)var
c=ap[1];else
if(ge===h){var
k=ap[0+1];ap[0+1]=hz;try{var
e=j(k,0);ap[0+1]=e;tC(ap,aO)}catch(f){f=s(f);ap[0+1]=function(a){throw f};throw f}var
c=e}else
var
c=ap;c[2]=(c[2]+1|0)%55|0;var
f=r(c[1],c[2]),g=(r(c[1],(c[2]+24|0)%55|0)+(f^f>>>25&31)|0)&bk;l(c[1],c[2],g);var
i=g}else
var
i=0;return[0,0,y(d,0),i,d]}}function
cE(a,b){return 3<=a.length-1?s7(10,aN,a[3],b)&(a[2].length-1-1|0):aL(s8(10,aN,b),a[2].length-1)}function
bC(a,b){var
i=cE(a,b),d=r(a[2],i);if(d){var
e=d[3],j=d[2];if(0===aK(b,d[1]))return j;if(e){var
f=e[3],k=e[2];if(0===aK(b,e[1]))return k;if(f){var
l=f[3],m=f[2];if(0===aK(b,f[1]))return m;var
c=l;for(;;){if(c){var
g=c[3],h=c[2];if(0===aK(b,c[1]))return h;var
c=g;continue}throw[0,u]}}throw[0,u]}throw[0,u]}throw[0,u]}function
a(a,b){return dg(a,b[0+1])}var
cF=[0,0];dg(h3,cF);var
h4=2;function
h5(a){var
b=[0,0],d=a.getLen()-1|0,e=0;if(!(d<0)){var
c=e;for(;;){b[1]=(223*b[1]|0)+a.safeGet(c)|0;var
g=c+1|0;if(d!==c){var
c=g;continue}break}}b[1]=b[1]&(fR-1|0);var
f=bk<b[1]?b[1]-fR|0:b[1];return f}var
ag=cy([0,function(a,b){return fp(a,b)}]),au=cy([0,function(a,b){return fp(a,b)}]),aq=cy([0,function(a,b){return gQ(a,b)}]),eA=fq(0,0),h6=[0,0];function
eB(a){return 2<a?eB((a+1|0)/2|0)*2|0:a}function
eC(a){h6[1]++;var
c=a.length-1,d=y((c*2|0)+2|0,eA);l(d,0,c);l(d,1,(z(eB(c),aD)/8|0)-1|0);var
e=c-1|0,f=0;if(!(e<0)){var
b=f;for(;;){l(d,(b*2|0)+3|0,r(a,b));var
g=b+1|0;if(e!==b){var
b=g;continue}break}}return[0,h4,d,au[1],aq[1],0,0,ag[1],0]}function
cG(a,b){var
c=a[2].length-1,g=c<b?1:0;if(g){var
d=y(b,eA),h=a[2],e=0,f=0,j=0<=c?0<=f?(h.length-1-c|0)<f?0:0<=e?(d.length-1-c|0)<e?0:(sM(h,f,d,e,c),1):0:0:0;if(!j)J(g_);a[2]=d;var
i=0}else
var
i=g;return i}var
eD=[0,0],h7=[0,0];function
cH(a){var
b=a[2].length-1;cG(a,b+1|0);return b}function
aY(a,b){try{var
d=i(au[22],b,a[3])}catch(f){f=s(f);if(f[1]===u){var
c=cH(a);a[3]=o(au[4],b,c,a[3]);a[4]=o(aq[4],c,1,a[4]);return c}throw f}return d}function
cJ(a){return a===0?0:aT(a)}function
eI(a,b){try{var
d=i(ag[22],b,a[7])}catch(f){f=s(f);if(f[1]===u){var
c=a[1];a[1]=c+1|0;if(A(b,im))a[7]=o(ag[4],b,c,a[7]);return c}throw f}return d}function
cL(a){return sY(a,0)?[0]:a}function
eK(a,b){if(a)return a;var
c=fq(b4,b[1]);c[0+1]=b[2];var
d=cF[1];c[1+1]=d;cF[1]=d+1|0;return c}function
bE(a){var
b=cH(a);if(0===(b%2|0))var
d=0;else
if((2+ay(r(a[2],1)*16|0,aD)|0)<b)var
d=0;else
var
c=cH(a),d=1;if(!d)var
c=b;l(a[2],c,0);return c}function
eM(a,ao){var
g=[0,0],ap=ao.length-1;for(;;){if(g[1]<ap){var
k=r(ao,g[1]),e=function(a){g[1]++;return r(ao,g[1])},n=e(0);if(typeof
n===m)switch(n){case
1:var
p=e(0),f=function(p){return function(a){return a[p+1]}}(p);break;case
2:var
q=e(0),b=e(0),f=function(q,b){return function(a){return a[q+1][b+1]}}(q,b);break;case
3:var
s=e(0),f=function(s){return function(a){return j(a[1][s+1],a)}}(s);break;case
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
Z=e(0),_=e(0),f=function(Z,_){return function(a){return i(a[1][Z+1],a,_)}}(Z,_);break;case
17:var
$=e(0),aa=e(0),f=function($,aa){return function(a){return i(a[1][$+1],a,a[aa+1])}}($,aa);break;case
18:var
ab=e(0),ac=e(0),ad=e(0),f=function(ab,ac,ad){return function(a){return i(a[1][ab+1],a,a[ac+1][ad+1])}}(ab,ac,ad);break;case
19:var
ae=e(0),af=e(0),f=function(ae,af){return function(a){var
b=j(a[1][af+1],a);return i(a[1][ae+1],a,b)}}(ae,af);break;case
20:var
ag=e(0),h=e(0);bE(a);var
f=function(ag,h){return function(a){return j(Y(h,ag,0),h)}}(ag,h);break;case
21:var
ah=e(0),ai=e(0);bE(a);var
f=function(ah,ai){return function(a){var
b=a[ai+1];return j(Y(b,ah,0),b)}}(ah,ai);break;case
22:var
aj=e(0),ak=e(0),al=e(0);bE(a);var
f=function(aj,ak,al){return function(a){var
b=a[ak+1][al+1];return j(Y(b,aj,0),b)}}(aj,ak,al);break;case
23:var
am=e(0),an=e(0);bE(a);var
f=function(am,an){return function(a){var
b=j(a[1][an+1],a);return j(Y(b,am,0),b)}}(am,an);break;default:var
o=e(0),f=function(o){return function(a){return o}}(o)}else
var
f=n;h7[1]++;if(i(aq[22],k,a[4])){cG(a,k+1|0);l(a[2],k,f)}else
a[6]=[0,[0,k,f],a[6]];g[1]++;continue}return 0}}function
cM(a,b,c){if(bV(c,iw))return b;var
d=c.getLen()-1|0;for(;;){if(0<=d){if(i(a,c,d)){var
d=d-1|0;continue}var
f=d+1|0,e=d;for(;;){if(0<=e){if(i(a,c,e))return w(c,e+1|0,(f-e|0)-1|0);var
e=e-1|0;continue}return w(c,0,f)}}return w(c,0,1)}}function
cN(a,b,c){if(bV(c,ix))return b;var
d=c.getLen()-1|0;for(;;){if(0<=d){if(i(a,c,d)){var
d=d-1|0;continue}var
e=d;for(;;){if(0<=e){if(i(a,c,e)){var
f=e;for(;;){if(0<=f){if(i(a,c,f)){var
f=f-1|0;continue}return w(c,0,f+1|0)}return w(c,0,1)}}var
e=e-1|0;continue}return b}}return w(c,0,1)}}function
cP(a,b){return 47===a.safeGet(b)?1:0}function
eN(a){var
b=a.getLen()<1?1:0,c=b||(47!==a.safeGet(0)?1:0);return c}function
iA(a){var
c=eN(a);if(c){var
e=a.getLen()<2?1:0,d=e||A(w(a,0,2),iC);if(d)var
f=a.getLen()<3?1:0,b=f||A(w(a,0,3),iB);else
var
b=d}else
var
b=c;return b}function
iD(a,b){var
c=b.getLen()<=a.getLen()?1:0,d=c?bV(w(a,a.getLen()-b.getLen()|0,b.getLen()),b):c;return d}try{var
sF=bT(sE),cQ=sF}catch(f){f=s(f);if(f[1]!==u)throw f;var
cQ=iE}function
eO(a){var
d=a.getLen(),b=aV(d+20|0);L(b,39);var
e=d-1|0,f=0;if(!(e<0)){var
c=f;for(;;){if(39===a.safeGet(c))bA(b,iF);else
L(b,a.safeGet(c));var
g=c+1|0;if(e!==c){var
c=g;continue}break}}L(b,39);return aW(b)}function
iG(a){return cM(cP,cO,a)}function
iH(a){return cN(cP,cO,a)}function
aF(a,b){var
c=a.safeGet(b),d=47===c?1:0;if(d)var
e=d;else
var
f=92===c?1:0,e=f||(58===c?1:0);return e}function
cS(a){var
e=a.getLen()<1?1:0,c=e||(47!==a.safeGet(0)?1:0);if(c){var
f=a.getLen()<1?1:0,d=f||(92!==a.safeGet(0)?1:0);if(d)var
g=a.getLen()<2?1:0,b=g||(58!==a.safeGet(1)?1:0);else
var
b=d}else
var
b=c;return b}function
eP(a){var
c=cS(a);if(c){var
g=a.getLen()<2?1:0,d=g||A(w(a,0,2),iN);if(d){var
h=a.getLen()<2?1:0,e=h||A(w(a,0,2),iM);if(e){var
i=a.getLen()<3?1:0,f=i||A(w(a,0,3),iL);if(f)var
j=a.getLen()<3?1:0,b=j||A(w(a,0,3),iK);else
var
b=f}else
var
b=e}else
var
b=d}else
var
b=c;return b}function
eQ(a,b){var
c=b.getLen()<=a.getLen()?1:0;if(c)var
e=w(a,a.getLen()-b.getLen()|0,b.getLen()),f=ef(b),d=bV(ef(e),f);else
var
d=c;return d}try{var
sD=bT(sC),eR=sD}catch(f){f=s(f);if(f[1]!==u)throw f;var
eR=iO}function
iP(h){var
i=h.getLen(),e=aV(i+20|0);L(e,34);function
g(a,b){var
c=b;for(;;){if(c===i)return L(e,34);var
f=h.safeGet(c);if(34===f)return a<50?d(1+a,0,c):I(d,[0,0,c]);if(92===f)return a<50?d(1+a,0,c):I(d,[0,0,c]);L(e,f);var
c=c+1|0;continue}}function
d(a,b,c){var
f=b,d=c;for(;;){if(d===i){L(e,34);return a<50?j(1+a,f):I(j,[0,f])}var
l=h.safeGet(d);if(34===l){k((2*f|0)+1|0);L(e,34);return a<50?g(1+a,d+1|0):I(g,[0,d+1|0])}if(92===l){var
f=f+1|0,d=d+1|0;continue}k(f);return a<50?g(1+a,d):I(g,[0,d])}}function
j(a,b){var
d=1;if(!(b<1)){var
c=d;for(;;){L(e,92);var
f=c+1|0;if(b!==c){var
c=f;continue}break}}return 0}function
a(b){return Z(g(0,b))}function
b(b,c){return Z(d(0,b,c))}function
k(b){return Z(j(0,b))}a(0);return aW(e)}function
eS(a){var
c=2<=a.getLen()?1:0;if(c)var
b=a.safeGet(0),g=91<=b?(b+f2|0)<0||25<(b+f2|0)?0:1:65<=b?1:0,d=g?1:0,e=d?58===a.safeGet(1)?1:0:d;else
var
e=c;if(e){var
f=w(a,2,a.getLen()-2|0);return[0,w(a,0,2),f]}return[0,iQ,a]}function
iR(a){var
b=eS(a),c=b[1];return h(c,cN(aF,cR,b[2]))}function
iS(a){return cM(aF,cR,eS(a)[2])}function
iV(a){return cM(aF,cT,a)}function
iW(a){return cN(aF,cT,a)}if(A(cw,iX))if(A(cw,iY)){if(A(cw,iZ))throw[0,K,i0];var
bF=[0,cR,iI,iJ,aF,cS,eP,eQ,eR,iP,iS,iR]}else
var
bF=[0,cO,iy,iz,cP,eN,iA,iD,cQ,eO,iG,iH];else
var
bF=[0,cT,iT,iU,aF,cS,eP,eQ,cQ,eO,iV,iW];var
eT=[0,i3],i1=bF[11],i2=bF[3];a(i6,[0,eT,0,i5,i4]);cB(function(a){if(a[1]===eT){var
c=a[2],d=a[4],e=a[3];if(typeof
c===m)switch(c){case
1:var
b=i9;break;case
2:var
b=i_;break;case
3:var
b=i$;break;case
4:var
b=ja;break;case
5:var
b=jb;break;case
6:var
b=jc;break;case
7:var
b=jd;break;case
8:var
b=je;break;case
9:var
b=jf;break;case
10:var
b=jg;break;case
11:var
b=jh;break;case
12:var
b=ji;break;case
13:var
b=jj;break;case
14:var
b=jk;break;case
15:var
b=jl;break;case
16:var
b=jm;break;case
17:var
b=jn;break;case
18:var
b=jo;break;case
19:var
b=jp;break;case
20:var
b=jq;break;case
21:var
b=jr;break;case
22:var
b=js;break;case
23:var
b=jt;break;case
24:var
b=ju;break;case
25:var
b=jv;break;case
26:var
b=jw;break;case
27:var
b=jx;break;case
28:var
b=jy;break;case
29:var
b=jz;break;case
30:var
b=jA;break;case
31:var
b=jB;break;case
32:var
b=jC;break;case
33:var
b=jD;break;case
34:var
b=jE;break;case
35:var
b=jF;break;case
36:var
b=jG;break;case
37:var
b=jH;break;case
38:var
b=jI;break;case
39:var
b=jJ;break;case
40:var
b=jK;break;case
41:var
b=jL;break;case
42:var
b=jM;break;case
43:var
b=jN;break;case
44:var
b=jO;break;case
45:var
b=jP;break;case
46:var
b=jQ;break;case
47:var
b=jR;break;case
48:var
b=jS;break;case
49:var
b=jT;break;case
50:var
b=jU;break;case
51:var
b=jV;break;case
52:var
b=jW;break;case
53:var
b=jX;break;case
54:var
b=jY;break;case
55:var
b=jZ;break;case
56:var
b=j0;break;case
57:var
b=j1;break;case
58:var
b=j2;break;case
59:var
b=j3;break;case
60:var
b=j4;break;case
61:var
b=j5;break;case
62:var
b=j6;break;case
63:var
b=j7;break;case
64:var
b=j8;break;case
65:var
b=j9;break;case
66:var
b=j_;break;case
67:var
b=j$;break;default:var
b=i7}else
var
f=c[1],b=j(cA(ka),f);return[0,o(cA(i8),b,e,d)]}return 0});bW(kb);bW(kc);try{bW(sB)}catch(f){f=s(f);if(f[1]!==aR)throw f}try{bW(sA)}catch(f){f=s(f);if(f[1]!==aR)throw f}ez(0,7);function
eU(a){return uM(a)}ao(32,q);var
kd=6,ke=0,ki=E(cc),kj=0;if(!0){var
bc=kj;for(;;){ki.safeSet(bc,ee(cv(bc)));var
sz=bc+1|0;if(q!==bc){var
bc=sz;continue}break}}var
cU=ao(32,0);cU.safeSet(10>>>3,cv(cU.safeGet(10>>>3)|1<<(10&7)));var
kf=E(32),kg=0;if(!0){var
a2=kg;for(;;){kf.safeSet(a2,cv(cU.safeGet(a2)^q));var
kh=a2+1|0;if(31!==a2){var
a2=kh;continue}break}}var
aG=[0,0],aH=[0,0],eV=[0,0];function
M(a){return aG[1]}function
eW(a){return aH[1]}function
S(a,b,c){return 0===a[2][0]?b?uc(a[1],a,b[1]):ud(a[1],a):b?fr(a[1],b[1]):fr(a[1],0)}var
eX=[3,kd],cV=[0,0];function
aI(e,b,c){cV[1]++;switch(e[0]){case
7:case
8:throw[0,K,kk];case
6:var
g=e[1],n=cV[1],o=fs(0),p=y(eW(0)+1|0,o),q=ft(0),r=y(M(0)+1|0,q),f=[0,-1,[1,[0,t2(g,c),g]],r,p,c,0,e,0,0,n,0];break;default:var
h=e[1],i=cV[1],j=fs(0),k=y(eW(0)+1|0,j),l=ft(0),m=y(M(0)+1|0,l),f=[0,-1,[0,sQ(h,ke,[0,c])],m,k,c,0,e,0,0,i,0]}if(b){var
d=b[1],a=function(a){if(0===d[2][0])return 6===e[0]?g0(f,d[1][8],d[1]):gZ(f,d[1][8],d[1]);var
b=d[1],c=M(0);return fu(f,d[1][8]-c|0,b)};try{a(0)}catch(f){be(0);a(0)}f[6]=[0,d]}return f}function
W(a){return a[5]}function
a3(a){return a[6]}function
bG(a){return a[8]}function
bH(a){return a[7]}function
ac(a){return a[2]}function
bI(a,b,c){a[1]=b;a[6]=c;return 0}function
cW(a,b,c){return dO<=b?r(a[3],c):r(a[4],c)}function
cX(a,b){var
e=b[3].length-1-2|0,g=0;if(!(e<0)){var
d=g;for(;;){l(b[3],d,r(a[3],d));var
j=d+1|0;if(e!==d){var
d=j;continue}break}}var
f=b[4].length-1-2|0,h=0;if(!(f<0)){var
c=h;for(;;){l(b[4],c,r(a[4],c));var
i=c+1|0;if(f!==c){var
c=i;continue}break}}return 0}function
bJ(a,b){b[8]=a[8];return 0}var
av=[0,kq];a(kz,[0,[0,kl]]);a(kA,[0,[0,km]]);a(kB,[0,[0,kn]]);a(kC,[0,[0,ko]]);a(kD,[0,[0,kp]]);a(kE,[0,av]);a(kF,[0,[0,kr]]);a(kG,[0,[0,ks]]);a(kH,[0,[0,kt]]);a(kI,[0,[0,kv]]);a(kJ,[0,[0,kw]]);a(kK,[0,[0,kx]]);a(kL,[0,[0,ky]]);a(kM,[0,[0,ku]]);var
cY=[0,kU];a(k_,[0,[0,kN]]);a(k$,[0,[0,kO]]);a(la,[0,[0,kP]]);a(lb,[0,[0,kQ]]);a(lc,[0,[0,kR]]);a(ld,[0,[0,kS]]);a(le,[0,[0,kT]]);a(lf,[0,cY]);a(lg,[0,[0,kV]]);a(lh,[0,[0,kW]]);a(li,[0,[0,kX]]);a(lj,[0,[0,kY]]);a(lk,[0,[0,kZ]]);a(ll,[0,[0,k0]]);a(lm,[0,[0,k1]]);a(ln,[0,[0,k2]]);a(lo,[0,[0,k3]]);a(lp,[0,[0,k4]]);a(lq,[0,[0,k5]]);a(lr,[0,[0,k6]]);a(ls,[0,[0,k7]]);a(lt,[0,[0,k8]]);a(lu,[0,[0,k9]]);var
bK=1,eY=0;function
a4(a,b,c){var
d=a[2];if(0===d[0])return sT(d[1],b,c);var
e=d[1];return o(e[2][4],e[1],b,c)}function
a5(a,b){var
c=a[2];if(0===c[0])return sR(c[1],b);var
d=c[1];return i(d[2][3],d[1],b)}function
eZ(a,b){S(a,0,0);e4(b,0,0);return S(a,0,0)}function
ah(a,b,c){var
f=a,d=b;for(;;){if(eY)return a4(f,d,c);var
n=d<0?1:0,o=n||(W(f)<=d?1:0);if(o)throw[0,bv,lv];if(bK){var
i=a3(f);if(typeof
i!==m)eZ(i[1],f)}var
j=bG(f);if(j){var
e=j[1];if(1===e[1]){var
k=e[4],g=e[3],l=e[2];return 0===k?a4(e[5],l+d|0,c):a4(e[5],(l+z(ay(d,g),k+g|0)|0)+aL(d,g)|0,c)}var
h=e[3],f=e[5],d=(e[2]+z(ay(d,h),e[4]+h|0)|0)+aL(d,h)|0;continue}return a4(f,d,c)}}function
ai(a,b){var
e=a,c=b;for(;;){if(eY)return a5(e,c);var
l=c<0?1:0,n=l||(W(e)<=c?1:0);if(n)throw[0,bv,lw];if(bK){var
h=a3(e);if(typeof
h!==m)eZ(h[1],e)}var
i=bG(e);if(i){var
d=i[1];if(1===d[1]){var
j=d[4],f=d[3],k=d[2];return 0===j?a5(d[5],k+c|0):a5(d[5],(k+z(ay(c,f),j+f|0)|0)+aL(c,f)|0)}var
g=d[3],e=d[5],c=(d[2]+z(ay(c,g),d[4]+g|0)|0)+aL(c,g)|0;continue}return a5(e,c)}}function
e0(a){if(a[8]){var
b=aI(a[7],0,a[5]);b[1]=a[1];b[6]=a[6];cX(a,b);return b}return a}function
e1(d,b,c){if(0===c[2][0]){var
a=function(a){return 0===ac(d)[0]?t5(d,c[1][8],c[1],c[3],b):t7(d,c[1][8],c[1],b)};try{var
f=a(0)}catch(f){f=s(f);if(f[1]===av){try{S(c,0,0);var
g=a(0)}catch(f){be(0);return a(0)}return g}throw f}return f}var
e=function(a){if(0===ac(d)[0]){var
e=c[1],f=M(0);return uv(d,c[1][8]-f|0,e,b)}var
g=c[1],h=M(0);return ux(d,c[1][8]-h|0,g,b)};try{var
i=e(0)}catch(f){try{S(c,0,0);var
h=e(0)}catch(f){be(0);return e(0)}return h}return i}function
e2(d,b,c){if(0===c[2][0]){var
a=function(a){return 0===ac(d)[0]?ub(d,c[1][8],c[1],c,b):t8(d,c[1][8],c[1],b)};try{var
f=a(0)}catch(f){f=s(f);if(f[1]===av){try{S(c,0,0);var
g=a(0)}catch(f){be(0);return a(0)}return g}throw f}return f}var
e=function(a){if(0===ac(d)[0]){var
e=c[2],f=c[1],g=M(0);return uB(d,c[1][8]-g|0,f,e,b)}var
h=c[2],i=c[1],j=M(0);return uy(d,c[1][8]-j|0,i,h,b)};try{var
i=e(0)}catch(f){try{S(c,0,0);var
h=e(0)}catch(f){be(0);return e(0)}return h}return i}function
a6(a,b,c,d,e,f,g,h){if(0===d[2][0])return 0===ac(a)[0]?uk(a,b,d[1][8],d[1],d[3],c,e,f,g,h):t_(a,b,d[1][8],d[1],d[3],c,e,f,g,h);if(0===ac(a)[0]){var
i=d[3],j=d[1],k=M(0);return uK(a,b,d[1][8]-k|0,j,i,c,e,f,g,h)}var
l=d[3],m=d[1],n=M(0);return uz(a,b,d[1][8]-n|0,m,l,c,e,f,g,h)}function
a7(a,b,c,d,e,f,g,h){if(0===d[2][0])return 0===ac(a)[0]?ul(a,b,d[1][8],d[1],d[3],c,e,f,g,h):t$(a,b,d[1][8],d[1],d[3],c,e,f,g,h);if(0===ac(a)[0]){var
i=d[3],j=d[1],k=M(0);return uL(a,b,d[1][8]-k|0,j,i,c,e,f,g,h)}var
l=d[3],m=d[1],n=M(0);return uA(a,b,d[1][8]-n|0,m,l,c,e,f,g,h)}function
e3(a,b,c){var
p=b;for(;;){var
d=p?p[1]:0,q=a3(a);if(typeof
q===m){bI(a,c[1][8],[1,c]);try{cZ(a,c)}catch(f){f=s(f);if(f[1]!==av)f[1]===cY;try{S(c,[0,d],0);cZ(a,c)}catch(f){f=s(f);if(f[1]!==av)if(f[1]!==cY)throw f;S(c,0,0);s3(0);cZ(a,c)}}var
A=bG(a);if(A){var
j=A[1];if(1===j[1]){var
k=j[5],r=j[4],f=j[3],l=j[2];if(0===f)a6(k,a,d,c,0,0,l,W(a));else
if(t<f){var
h=0,n=W(a);for(;;){if(f<n){a6(k,a,d,c,z(h,f+r|0),z(h,f),l,f);var
h=h+1|0,n=n-f|0;continue}if(0<n)a6(k,a,d,c,z(h,f+r|0),z(h,f),l,n);break}}else{var
e=0,i=0,g=W(a);for(;;){if(t<g){var
w=aI(bH(a),0,t);bJ(a,w);var
B=e+dQ|0;if(!(B<e)){var
u=e;for(;;){ah(w,u,ai(a,e));var
I=u+1|0;if(B!==u){var
u=I;continue}break}}a6(k,w,d,c,z(i,t+r|0),i*t|0,l,t);var
e=e+t|0,i=i+1|0,g=g+gi|0;continue}if(0<g){var
x=aI(bH(a),0,g),C=(e+g|0)-1|0;if(!(C<e)){var
v=e;for(;;){ah(x,v,ai(a,e));var
J=v+1|0;if(C!==v){var
v=J;continue}break}}bJ(a,x);a6(k,x,d,c,z(i,t+r|0),i*t|0,l,g)}break}}}else{var
y=e0(a),D=W(a)-1|0,K=0;if(!(D<0)){var
o=K;for(;;){a4(y,o,ai(a,o));var
L=o+1|0;if(D!==o){var
o=L;continue}break}}e1(y,d,c);cX(y,a)}}else
e1(a,d,c);return bI(a,c[1][8],[0,c])}else{if(0===q[0]){var
E=q[1],F=di(E,c);if(F){e4(a,[0,d],0);S(E,0,0);var
p=[0,d];continue}return F}var
G=q[1],H=di(G,c);if(H){S(G,0,0);var
p=[0,d];continue}return H}}}function
cZ(a,b){if(0===b[2][0])return 0===ac(a)[0]?gZ(a,b[1][8],b[1]):g0(a,b[1][8],b[1]);if(0===ac(a)[0]){var
c=b[1],d=M(0);return fu(a,b[1][8]-d|0,c)}var
e=b[1],f=M(0);return uw(a,b[1][8]-f|0,e)}function
e4(a,b,c){var
w=b;for(;;){var
f=w?w[1]:0,p=a3(a);if(typeof
p===m)return 0;else{if(0===p[0]){var
d=p[1];bI(a,d[1][8],[1,d]);var
A=bG(a);if(A){var
j=A[1];if(1===j[1]){var
k=j[5],q=j[4],e=j[3],l=j[2];if(0===e)a7(k,a,f,d,0,0,l,W(a));else
if(t<e){var
h=0,n=W(a);for(;;){if(e<n){a7(k,a,f,d,z(h,e+q|0),z(h,e),l,e);var
h=h+1|0,n=n-e|0;continue}if(0<n)a7(k,a,f,d,z(h,e+q|0),z(h,e),l,n);break}}else{var
i=0,g=W(a),r=0;for(;;){if(t<g){var
x=aI(bH(a),0,t);bJ(a,x);var
D=dQ;if(!(dQ<0)){var
s=r;for(;;){ah(x,s,ai(a,r));var
E=s+1|0;if(D!==s){var
s=E;continue}break}}a7(k,x,f,d,z(i,t+q|0),i*t|0,l,t);var
i=i+1|0,g=g+gi|0;continue}if(0<g){var
y=aI(bH(a),0,g),B=(0+g|0)-1|0;if(!(B<0)){var
u=r;for(;;){ah(y,u,ai(a,r));var
F=u+1|0;if(B!==u){var
u=F;continue}break}}bJ(a,y);a7(k,y,f,d,z(i,t+q|0),i*t|0,l,g)}break}}}else{var
v=e0(a);cX(v,a);e2(v,f,d);var
C=W(v)-1|0,G=0;if(!(C<0)){var
o=G;for(;;){ah(a,o,a5(v,o));var
H=o+1|0;if(C!==o){var
o=H;continue}break}}}}else
e2(a,f,d);return bI(a,d[1][8],0)}S(p[1],0,0);var
w=[0,f];continue}}}var
lB=[0,lA],lD=[0,lC];function
bL(a,b){var
o=r(ho,0),p=h(i2,h(a,b)),g=d6(h(i1(o),p));try{var
n=e6,i=e6;a:for(;;){var
k=function(a,b,c){var
e=b,d=c;for(;;){if(d){var
g=d[1],f=g.getLen(),h=d[2];bd(g,0,a,e-f|0,f);var
e=e-f|0,d=h;continue}return a}},d=0,e=0;for(;;){var
c=tt(g);if(0===c){if(!d)throw[0,bw];var
j=k(E(e),e,d)}else{if(!(0<c)){var
m=E(-c|0);dj(g,m,0,-c|0);var
d=[0,m,d],e=e-c|0;continue}var
f=E(c-1|0);dj(g,f,0,c-1|0);ts(g);if(d)var
l=(e+c|0)-1|0,j=k(E(l),l,[0,f,d]);else
var
j=f}var
i=h(i,h(j,lE)),n=i;continue a}}}catch(f){f=s(f);if(f[1]===bw){d8(g);return n}throw f}}var
e7=[0,lF],c0=[],lG=0,lH=0;tX(c0,[0,0,function(f){var
k=eI(f,lI),e=cL(lx),d=e.length-1,n=e5.length-1,a=y(d+n|0,0),p=d-1|0,v=0;if(!(p<0)){var
c=v;for(;;){l(a,c,aY(f,r(e,c)));var
z=c+1|0;if(p!==c){var
c=z;continue}break}}var
q=n-1|0,w=0;if(!(q<0)){var
b=w;for(;;){l(a,b+d|0,eI(f,r(e5,b)));var
x=b+1|0;if(q!==b){var
b=x;continue}break}}var
t=a[10],m=a[12],h=a[15],i=a[16],j=a[17],g=a[18],A=a[1],B=a[2],C=a[3],D=a[4],E=a[5],F=a[7],G=a[8],H=a[9],I=a[11],J=a[14];function
K(a,b,c,d,e,f){var
h=d?d[1]:d;o(a[1][m+1],a,[0,h],f);var
i=bC(a[g+1],f);return fv(a[1][t+1],a,b,[0,c[1],c[2]],e,f,i)}function
L(a,b,c,d,e){try{var
f=bC(a[g+1],e),h=f}catch(f){f=s(f);if(f[1]!==u)throw f;try{o(a[1][m+1],a,lJ,e)}catch(f){f=s(f);throw f}var
h=bC(a[g+1],e)}return fv(a[1][t+1],a,b,[0,c[1],c[2]],d,e,h)}function
M(a,b,c){var
z=b?b[1]:b;try{bC(a[g+1],c);var
f=0}catch(f){f=s(f);if(f[1]===u){if(0===c[2][0]){var
A=a[i+1];if(!A)throw[0,e7,c];var
B=A[1],H=z?ua(B,a[h+1],c[1]):t4(B,a[h+1],c[1]),C=H}else{var
D=a[j+1];if(!D)throw[0,e7,c];var
E=D[1],I=z?um(E,a[h+1],c[1]):uu(E,a[h+1],c[1]),C=I}var
d=a[g+1],w=cE(d,c);l(d[2],w,[0,c,C,r(d[2],w)]);d[1]=d[1]+1|0;var
x=d[2].length-1<<1<d[1]?1:0;if(x){var
m=d[2],n=m.length-1,o=n*2|0,p=o<cx?1:0;if(p){var
k=y(o,0);d[2]=k;var
q=function(a){if(a){var
b=a[1],e=a[2];q(a[3]);var
c=cE(d,b);return l(k,c,[0,b,e,r(k,c)])}return 0},t=n-1|0,F=0;if(!(t<0)){var
e=F;for(;;){q(r(m,e));var
G=e+1|0;if(t!==e){var
e=G;continue}break}}var
v=0}else
var
v=p;return v}return x}throw f}return f}function
N(a,b){try{var
f=[0,bL(a[k+1],lL),0],c=f}catch(f){var
c=0}a[i+1]=c;try{var
e=[0,bL(a[k+1],lK),0],d=e}catch(f){var
d=0}a[j+1]=d;return 0}function
O(a,b){a[j+1]=[0,b,0];return 0}function
P(a,b){return a[j+1]}function
Q(a,b){a[i+1]=[0,b,0];return 0}function
R(a,b){return a[i+1]}function
S(a,b){var
d=a[g+1];d[1]=0;var
e=d[2].length-1-1|0,f=0;if(!(e<0)){var
c=f;for(;;){l(d[2],c,0);var
h=c+1|0;if(e!==c){var
c=h;continue}break}}return 0}eM(f,[0,H,function(a,b){return a[g+1]},D,S,G,R,B,Q,F,P,A,O,E,N,m,M,C,L,I,K]);return function(a,b,c,d){var
e=eK(b,f);e[k+1]=c;e[J+1]=c;e[h+1]=d;try{var
o=[0,bL(c,lN),0],l=o}catch(f){var
l=0}e[i+1]=l;try{var
n=[0,bL(c,lM),0],m=n}catch(f){var
m=0}e[j+1]=m;e[g+1]=ez(0,8);return e}},lH,lG]);fw(0);fw(0);function
c1(a){function
e(a,b){var
d=a-1|0,e=0;if(!(d<0)){var
c=e;for(;;){eu(lP);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return j(eu(lO),b)}function
f(a,b){var
c=a,d=b;for(;;)if(typeof
d===m)return 0===d?e(c,lQ):e(c,lR);else
switch(d[0]){case
0:e(c,lS);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
1:e(c,lT);var
c=c+1|0,d=d[1];continue;case
2:e(c,lU);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
3:e(c,lV);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
4:e(c,lW);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
5:e(c,lX);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
6:e(c,lY);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
7:e(c,lZ);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
8:e(c,l0);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
9:e(c,l1);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
10:e(c,l2);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
11:return e(c,h(l3,d[1]));case
12:return e(c,h(l4,d[1]));case
13:return e(c,h(l5,k(d[1])));case
14:return e(c,h(l6,k(d[1])));case
15:return e(c,h(l7,k(d[1])));case
16:return e(c,h(l8,k(d[1])));case
17:return e(c,h(l9,k(d[1])));case
18:return e(c,l_);case
19:return e(c,l$);case
20:return e(c,ma);case
21:return e(c,mb);case
22:return e(c,mc);case
23:return e(c,h(md,k(d[2])));case
24:e(c,me);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
25:e(c,mf);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
26:e(c,mg);var
c=c+1|0,d=d[1];continue;case
27:e(c,mh);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
28:e(c,mi);var
c=c+1|0,d=d[1];continue;case
29:e(c,mj);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
30:e(c,mk);f(c+1|0,d[1]);f(c+1|0,d[2]);var
c=c+1|0,d=d[3];continue;case
31:return e(c,ml);case
32:var
g=h(mm,k(d[2]));return e(c,h(mn,h(d[1],g)));case
33:return e(c,h(mo,k(d[1])));case
36:e(c,mq);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
37:e(c,mr);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
38:e(c,ms);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
39:e(c,mt);f(c+1|0,d[1]);f(c+1|0,d[2]);var
c=c+1|0,d=d[3];continue;case
40:e(c,mu);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
41:e(c,mv);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
42:e(c,mw);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
43:e(c,mx);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
44:e(c,my);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
45:e(c,mz);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
46:e(c,mA);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
47:e(c,mB);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
48:e(c,mC);f(c+1|0,d[1]);f(c+1|0,d[2]);f(c+1|0,d[3]);var
c=c+1|0,d=d[4];continue;case
49:e(c,mD);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
50:e(c,mE);f(c+1|0,d[1]);var
i=d[2],j=c+1|0;return d_(function(a){return f(j,a)},i);case
51:return e(c,mF);case
52:return e(c,mG);default:return e(c,h(mp,Q(d[1])))}}return f(0,a)}function
N(a){return ao(a,32)}var
a8=[0,mH];function
bg(a,b,c){var
d=c;for(;;)if(typeof
d===m)return mJ;else
switch(d[0]){case
18:case
19:var
S=h(nm,h(e(b,d[2]),nl));return h(nn,h(k(d[1]),S));case
27:case
38:var
ac=d[1],ad=h(nI,h(e(b,d[2]),nH));return h(e(b,ac),ad);case
0:var
g=d[2],B=e(b,d[1]);if(typeof
g===m)var
r=0;else
if(25===g[0])var
t=e(b,g),r=1;else
var
r=0;if(!r)var
C=h(mK,N(b)),t=h(e(b,g),C);return h(h(B,t),mL);case
1:var
D=h(e(b,d[1]),mM),E=A(a8[1][1],mN)?h(a8[1][1],mO):mQ;return h(mP,h(E,D));case
2:var
F=h(mS,h(X(b,d[2]),mR));return h(mT,h(X(b,d[1]),F));case
3:var
G=h(mV,h(ar(b,d[2]),mU));return h(mW,h(ar(b,d[1]),G));case
4:var
H=h(mY,h(X(b,d[2]),mX));return h(mZ,h(X(b,d[1]),H));case
5:var
J=h(m1,h(ar(b,d[2]),m0));return h(m2,h(ar(b,d[1]),J));case
6:var
L=h(m4,h(X(b,d[2]),m3));return h(m5,h(X(b,d[1]),L));case
7:var
M=h(m7,h(ar(b,d[2]),m6));return h(m8,h(ar(b,d[1]),M));case
8:var
O=h(m_,h(X(b,d[2]),m9));return h(m$,h(X(b,d[1]),O));case
9:var
P=h(nb,h(ar(b,d[2]),na));return h(nc,h(ar(b,d[1]),P));case
10:var
R=h(ne,h(X(b,d[2]),nd));return h(nf,h(X(b,d[1]),R));case
13:return h(ng,k(d[1]));case
14:return h(nh,k(d[1]));case
15:throw[0,K,ni];case
16:return h(nj,k(d[1]));case
17:return h(nk,k(d[1]));case
20:var
T=h(np,h(e(b,d[2]),no));return h(nq,h(k(d[1]),T));case
21:var
U=h(ns,h(e(b,d[2]),nr));return h(nt,h(k(d[1]),U));case
22:var
W=h(nv,h(e(b,d[2]),nu));return h(nw,h(k(d[1]),W));case
23:var
Y=h(nx,k(d[2])),u=d[1];if(typeof
u===m)var
f=0;else
switch(u[0]){case
33:var
o=nz,f=1;break;case
34:var
o=nA,f=1;break;case
35:var
o=nB,f=1;break;default:var
f=0}if(f)return h(o,Y);throw[0,K,ny];case
24:var
i=d[2],v=d[1];if(typeof
i===m){if(0===i){var
d=v;continue}}else
if(24===i[0]){var
Z=h(nD,e(b,i));return h(e(b,v),Z)}return V(nC);case
25:var
_=e(b,d[2]),$=h(nE,h(N(b),_));return h(e(b,d[1]),$);case
26:var
aa=e(b,d[1]),ab=A(a8[1][2],nF)?a8[1][2]:nG;return h(ab,aa);case
28:var
d=d[1];continue;case
29:var
ae=h(nK,h(e(b,d[2]),nJ));return h(e(b,d[1]),ae);case
30:var
l=d[2],af=e(b,d[3]),ag=h(nL,h(N(b),af));if(typeof
l===m)var
s=0;else
if(31===l[0])var
w=h(mI(l[1]),nN),s=1;else
var
s=0;if(!s)var
w=e(b,l);var
ah=h(nM,h(w,ag));return h(e(b,d[1]),ah);case
31:return a<50?bf(1+a,d[1]):I(bf,[0,d[1]]);case
33:return k(d[1]);case
34:return h(Q(d[1]),nO);case
35:return Q(d[1]);case
36:var
ai=h(nQ,h(e(b,d[2]),nP));return h(e(b,d[1]),ai);case
37:var
aj=h(nS,h(N(b),nR)),ak=h(e(b,d[2]),aj),al=h(nT,h(N(b),ak)),am=h(nU,h(e(b,d[1]),al));return h(N(b),am);case
39:var
an=h(nV,N(b)),ao=h(e(b+2|0,d[3]),an),ap=h(nW,h(N(b+2|0),ao)),aq=h(nX,h(N(b),ap)),as=h(e(b+2|0,d[2]),aq),at=h(nY,h(N(b+2|0),as));return h(nZ,h(e(b,d[1]),at));case
40:var
au=h(n0,N(b)),av=h(e(b+2|0,d[2]),au),aw=h(n1,h(N(b+2|0),av));return h(n2,h(e(b,d[1]),aw));case
41:var
ax=h(n3,e(b,d[2]));return h(e(b,d[1]),ax);case
42:var
ay=h(n4,e(b,d[2]));return h(e(b,d[1]),ay);case
43:var
az=h(n5,e(b,d[2]));return h(e(b,d[1]),az);case
44:var
aA=h(n6,e(b,d[2]));return h(e(b,d[1]),aA);case
45:var
aB=h(n7,e(b,d[2]));return h(e(b,d[1]),aB);case
46:var
aC=h(n8,e(b,d[2]));return h(e(b,d[1]),aC);case
47:var
aD=h(n9,e(b,d[2]));return h(e(b,d[1]),aD);case
48:var
p=e(b,d[1]),aE=e(b,d[2]),aF=e(b,d[3]),aG=h(e(b+2|0,d[4]),n_);return h(oe,h(p,h(od,h(aE,h(oc,h(p,h(ob,h(aF,h(oa,h(p,h(n$,h(N(b+2|0),aG))))))))))));case
49:var
aH=e(b,d[1]),aI=h(e(b+2|0,d[2]),of);return h(oh,h(aH,h(og,h(N(b+2|0),aI))));case
50:var
x=d[2],n=d[1],y=e(b,n),q=function(a){if(a){var
c=a[2],d=a[1];if(c){var
f=h(oi,q(c));return h(e(b,d),f)}return e(b,d)}throw[0,K,oj]};if(typeof
n!==m)if(31===n[0]){var
z=n[1];if(!A(z[1],om))if(!A(z[2],on))return h(y,h(op,h(q(aT(x)),oo)))}return h(y,h(ol,h(q(aT(x)),ok)));case
51:return k(j(d[1],0));case
52:return h(Q(j(d[1],0)),oq);default:return d[1]}}function
sK(a,b,c){if(typeof
c!==m)switch(c[0]){case
2:case
4:case
6:case
8:case
10:case
50:return a<50?bg(1+a,b,c):I(bg,[0,b,c]);case
32:return c[1];case
33:return k(c[1]);case
36:var
d=h(os,h(X(b,c[2]),or));return h(e(b,c[1]),d);case
51:return k(j(c[1],0))}return a<50?dk(1+a,b,c):I(dk,[0,b,c])}function
dk(a,b,c){if(typeof
c!==m)switch(c[0]){case
3:case
5:case
7:case
9:case
29:case
50:return a<50?bg(1+a,b,c):I(bg,[0,b,c]);case
16:return h(ou,k(c[1]));case
31:return a<50?bf(1+a,c[1]):I(bf,[0,c[1]]);case
32:return c[1];case
34:return h(Q(c[1]),ov);case
35:return h(ow,Q(c[1]));case
36:var
d=h(oy,h(X(b,c[2]),ox));return h(e(b,c[1]),d);case
52:return h(Q(j(c[1],0)),oz)}c1(c);return V(ot)}function
bf(a,b){return b[1]}function
e(b,c){return Z(bg(0,b,c))}function
X(b,c){return Z(sK(0,b,c))}function
ar(b,c){return Z(dk(0,b,c))}function
mI(b){return Z(bf(0,b))}function
C(a){return ao(a,32)}var
a9=[0,oA];function
bi(a,b,c){var
d=c;for(;;)if(typeof
d===m)return oC;else
switch(d[0]){case
18:case
19:var
S=h(oZ,h(f(b,d[2]),oY));return h(o0,h(k(d[1]),S));case
27:case
38:var
ac=d[1],ad=h(pi,T(b,d[2]));return h(f(b,ac),ad);case
0:var
g=d[2],B=f(b,d[1]);if(typeof
g===m)var
r=0;else
if(25===g[0])var
t=f(b,g),r=1;else
var
r=0;if(!r)var
D=h(oD,C(b)),t=h(f(b,g),D);return h(h(B,t),oE);case
1:var
E=h(f(b,d[1]),oF),F=A(a9[1][1],oG)?h(a9[1][1],oH):oJ;return h(oI,h(F,E));case
2:var
G=h(oK,T(b,d[2]));return h(T(b,d[1]),G);case
3:var
H=h(oL,as(b,d[2]));return h(as(b,d[1]),H);case
4:var
J=h(oM,T(b,d[2]));return h(T(b,d[1]),J);case
5:var
L=h(oN,as(b,d[2]));return h(as(b,d[1]),L);case
6:var
M=h(oO,T(b,d[2]));return h(T(b,d[1]),M);case
7:var
N=h(oP,as(b,d[2]));return h(as(b,d[1]),N);case
8:var
O=h(oQ,T(b,d[2]));return h(T(b,d[1]),O);case
9:var
P=h(oR,as(b,d[2]));return h(as(b,d[1]),P);case
10:var
R=h(oS,T(b,d[2]));return h(T(b,d[1]),R);case
13:return h(oT,k(d[1]));case
14:return h(oU,k(d[1]));case
15:throw[0,K,oV];case
16:return h(oW,k(d[1]));case
17:return h(oX,k(d[1]));case
20:var
U=h(o2,h(f(b,d[2]),o1));return h(o3,h(k(d[1]),U));case
21:var
W=h(o5,h(f(b,d[2]),o4));return h(o6,h(k(d[1]),W));case
22:var
X=h(o8,h(f(b,d[2]),o7));return h(o9,h(k(d[1]),X));case
23:var
Y=h(o_,k(d[2])),u=d[1];if(typeof
u===m)var
e=0;else
switch(u[0]){case
33:var
o=pa,e=1;break;case
34:var
o=pb,e=1;break;case
35:var
o=pc,e=1;break;default:var
e=0}if(e)return h(o,Y);throw[0,K,o$];case
24:var
i=d[2],v=d[1];if(typeof
i===m){if(0===i){var
d=v;continue}}else
if(24===i[0]){var
Z=h(pe,f(b,i));return h(f(b,v),Z)}return V(pd);case
25:var
_=f(b,d[2]),$=h(pf,h(C(b),_));return h(f(b,d[1]),$);case
26:var
aa=f(b,d[1]),ab=A(a9[1][2],pg)?a9[1][2]:ph;return h(ab,aa);case
28:var
d=d[1];continue;case
29:var
ae=h(pk,h(f(b,d[2]),pj));return h(f(b,d[1]),ae);case
30:var
l=d[2],af=f(b,d[3]),ag=h(pl,h(C(b),af));if(typeof
l===m)var
s=0;else
if(31===l[0])var
w=oB(l[1]),s=1;else
var
s=0;if(!s)var
w=f(b,l);var
ah=h(pm,h(w,ag));return h(f(b,d[1]),ah);case
31:return a<50?bh(1+a,d[1]):I(bh,[0,d[1]]);case
33:return k(d[1]);case
34:return h(Q(d[1]),pn);case
35:return Q(d[1]);case
36:var
ai=h(pp,h(f(b,d[2]),po));return h(f(b,d[1]),ai);case
37:var
aj=h(pr,h(C(b),pq)),ak=h(f(b,d[2]),aj),al=h(ps,h(C(b),ak)),am=h(pt,h(f(b,d[1]),al));return h(C(b),am);case
39:var
an=h(pu,C(b)),ao=h(f(b+2|0,d[3]),an),ap=h(pv,h(C(b+2|0),ao)),aq=h(pw,h(C(b),ap)),ar=h(f(b+2|0,d[2]),aq),at=h(px,h(C(b+2|0),ar));return h(py,h(f(b,d[1]),at));case
40:var
au=h(pz,C(b)),av=h(pA,h(C(b),au)),aw=h(f(b+2|0,d[2]),av),ax=h(pB,h(C(b+2|0),aw)),ay=h(pC,h(C(b),ax));return h(pD,h(f(b,d[1]),ay));case
41:var
az=h(pE,f(b,d[2]));return h(f(b,d[1]),az);case
42:var
aA=h(pF,f(b,d[2]));return h(f(b,d[1]),aA);case
43:var
aB=h(pG,f(b,d[2]));return h(f(b,d[1]),aB);case
44:var
aC=h(pH,f(b,d[2]));return h(f(b,d[1]),aC);case
45:var
aD=h(pI,f(b,d[2]));return h(f(b,d[1]),aD);case
46:var
aE=h(pJ,f(b,d[2]));return h(f(b,d[1]),aE);case
47:var
aF=h(pK,f(b,d[2]));return h(f(b,d[1]),aF);case
48:var
p=f(b,d[1]),aG=f(b,d[2]),aH=f(b,d[3]),aI=h(f(b+2|0,d[4]),pL);return h(pR,h(p,h(pQ,h(aG,h(pP,h(p,h(pO,h(aH,h(pN,h(p,h(pM,h(C(b+2|0),aI))))))))))));case
49:var
aJ=f(b,d[1]),aK=h(f(b+2|0,d[2]),pS);return h(pU,h(aJ,h(pT,h(C(b+2|0),aK))));case
50:var
x=d[2],n=d[1],y=f(b,n),q=function(a){if(a){var
c=a[2],d=a[1];if(c){var
e=h(pV,q(c));return h(f(b,d),e)}return f(b,d)}throw[0,K,pW]};if(typeof
n!==m)if(31===n[0]){var
z=n[1];if(!A(z[1],pZ))if(!A(z[2],p0))return h(y,h(p2,h(q(aT(x)),p1)))}return h(y,h(pY,h(q(aT(x)),pX)));case
51:return k(j(d[1],0));case
52:return h(Q(j(d[1],0)),p3);default:return d[1]}}function
sL(a,b,c){if(typeof
c!==m)switch(c[0]){case
2:case
4:case
6:case
8:case
10:case
50:return a<50?bi(1+a,b,c):I(bi,[0,b,c]);case
32:return c[1];case
33:return k(c[1]);case
36:var
d=h(p5,h(T(b,c[2]),p4));return h(f(b,c[1]),d);case
51:return k(j(c[1],0))}return a<50?dl(1+a,b,c):I(dl,[0,b,c])}function
dl(a,b,c){if(typeof
c!==m)switch(c[0]){case
3:case
5:case
7:case
9:case
50:return a<50?bi(1+a,b,c):I(bi,[0,b,c]);case
16:return h(p7,k(c[1]));case
31:return a<50?bh(1+a,c[1]):I(bh,[0,c[1]]);case
32:return c[1];case
34:return h(Q(c[1]),p8);case
35:return h(p9,Q(c[1]));case
36:var
d=h(p$,h(T(b,c[2]),p_));return h(f(b,c[1]),d);case
52:return h(Q(j(c[1],0)),qa)}c1(c);return V(p6)}function
bh(a,b){return b[2]}function
f(b,c){return Z(bi(0,b,c))}function
T(b,c){return Z(sL(0,b,c))}function
as(b,c){return Z(dl(0,b,c))}function
oB(b){return Z(bh(0,b))}var
qm=h(ql,h(qk,h(qj,h(qi,h(qh,h(qg,h(qf,h(qe,h(qd,h(qc,qb)))))))))),qD=h(qC,h(qB,h(qA,h(qz,h(qy,h(qx,h(qw,h(qv,h(qu,h(qt,h(qs,h(qr,h(qq,h(qp,h(qo,qn))))))))))))))),qL=h(qK,h(qJ,h(qI,h(qH,h(qG,h(qF,qE)))))),qT=h(qS,h(qR,h(qQ,h(qP,h(qO,h(qN,qM))))));function
x(a){return[32,h(qU,k(a)),a]}function
a_(a,b){return[25,a,b]}function
bM(a,b){return[50,a,b]}function
aw(a){return[33,a]}function
c2(a){return[34,a]}function
a$(a,b){return[2,a,b]}function
e8(a,b){return[3,a,b]}function
c3(a,b){return[6,a,b]}function
c4(a,b){return[7,a,b]}function
c5(a){return[13,a]}function
c6(a,b){return[29,a,b]}function
ax(a,b){return[31,[0,a,b]]}function
c7(a,b){return[37,a,b]}function
c8(a,b){return[27,a,b]}function
c9(a){return[28,a]}function
aJ(a,b){return[36,a,b]}function
e9(a){var
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
M=b([26,d[2]]);return[49,b(d[1]),M]}return[26,b(d)];case
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
R=f[3],S=[29,b(q),R],T=b(f[2]),U=[29,b(q),T];return[39,b(f[1]),U,S]}var
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
ao=aS(b,c[2]);return[50,b(c[1]),ao]}return c}}var
c=b(a);for(;;){if(e[1]){e[1]=0;var
c=b(c);continue}return c}}var
q0=[0,qZ];function
ba(a,b,c){var
t=a?a[1]:a,u=b?b[1]:2,g=c[2],d=c[1],n=g[3],p=g[2];q0[1]=q1;var
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
o=h(q9,h(k(j[1]),q8)),l=2;break;default:var
l=1}switch(l){case
1:c1(n[1]);d9(d4);throw[0,K,q2];case
2:break;default:var
o=q3}var
q=[0,e(0,n[1]),o];if(t){a8[1]=q;a9[1]=q}function
r(a){var
q=g[4],r=d$(function(a,b){return 0===b?a:h(qL,a)},qT,q),s=h(r,e(0,e9(p))),j=de(fm(q4,g5,438));d5(j,s);df(j);fn(j);fx(q5);var
m=d6(q6),c=tp(m),n=E(c),o=0;if(0<=0)if(0<=c)if((n.getLen()-c|0)<o)var
f=0;else{var
k=o,b=c;for(;;){if(0<b){var
l=dj(m,n,k,b);if(0===l)throw[0,bw];var
k=k+l|0,b=b-l|0;continue}var
f=1;break}}else
var
f=0;else
var
f=0;if(!f)J(g7);d8(m);i(Y(d,723535973,3),d,n);fx(q7);return 0}function
s(a){var
b=g[4],c=d$(function(a,b){return 0===b?a:h(qD,a)},qm,b);return i(Y(d,fM,4),d,h(c,f(0,e9(p))))}switch(u){case
1:s(0);break;case
2:r(0);s(0);break;default:r(0)}i(Y(d,345714255,5),d,0);return[0,d,g]}var
c_=c,e_=null,rc=undefined;function
e$(a,b){return a==e_?j(b,0):a}var
bN=false,fa=Array,fb=[0,rd];a(re,[0,fb,{}]);cB(function(a){return a[1]===fb?[0,new
ae(a[2].toString())]:0});cB(function(a){return a
instanceof
fa?0:[0,new
ae(a.toString())]});function
D(a,b){a.appendChild(b);return 0}function
c$(d){return tm(function(a){if(a){var
e=j(d,a);if(!(e|0))a.preventDefault();return e}var
c=event,b=j(d,c);if(!(b|0))c.returnValue=b;return b})}var
H=c_.document,rf="2d";function
bO(a,b){return a?j(b,a[1]):0}function
bP(a,b){return a.createElement(b.toString())}function
bQ(a,b){return bP(a,b)}var
fc=[0,gt];function
da(a,b,c,d){for(;;){if(0===a)if(0===b)return bP(c,d);var
h=fc[1];if(gt===h){try{var
j=H.createElement('<input name="x">'),k=j.tagName.toLowerCase()===fG?1:0,m=k?j.name===dG?1:0:k,i=m}catch(f){var
i=0}var
l=i?gf:-1003883683;fc[1]=l;continue}if(gf<=h){var
e=new
fa();e.push("<",d.toString());bO(a,function(a){e.push(' type="',fy(a),ca);return 0});bO(b,function(a){e.push(' name="',fy(a),ca);return 0});e.push(">");return c.createElement(e.join(g))}var
f=bP(c,d);bO(a,function(a){return f.type=a});bO(b,function(a){return f.name=a});return f}}function
fd(a){return bQ(a,rk)}var
ro=[0,rn];c_.HTMLElement===rc;function
fh(a){return fd(H)}function
fi(a){function
c(a){throw[0,K,rq]}var
b=e$(H.getElementById(gK),c);return j(ev(function(a){D(b,fh(0));D(b,H.createTextNode(a.toString()));return D(b,fh(0))}),a)}function
rr(a){var
k=[0,[4,a]];return function(a,b,c,d){var
h=a[2],i=a[1],l=c[2];if(0===l[0]){var
g=l[1],e=[0,0],f=t6(k.length-1),m=g[7][1]<i[1]?1:0;if(m)var
n=m;else
var
r=g[7][2]<i[2]?1:0,n=r||(g[7][3]<i[3]?1:0);if(n)throw[0,lB];var
o=g[8][1]<h[1]?1:0;if(o)var
p=o;else
var
q=g[8][2]<h[2]?1:0,p=q||(g[8][3]<h[3]?1:0);if(p)throw[0,lD];cq(function(a,b){function
d(a){if(bK)try{e3(a,0,c);S(c,0,0)}catch(f){f=s(f);if(f[1]===av)throw[0,av];throw f}return 11===b[0]?t9(e,f,cW(b[1],dO,c[1][8]),a):uj(e,f,cW(a,dO,c[1][8]),a,c)}switch(b[0]){case
6:case
7:case
8:case
9:case
10:switch(b[0]){case
6:return ui(e,f,b[1]);case
7:return uh(e,f,b[1]);case
8:return ug(e,f,b[1]);case
9:return uf(e,f,b[1]);default:return V(ly)}case
11:return d(b[1]);default:return d(b[1])}},k);return ue(e,d,h,i,f,c[1],b)}var
j=[0,0];cq(function(a,b){switch(b[0]){case
6:case
7:case
8:case
9:case
10:switch(b[0]){case
6:return uH(j,d,b[1],c[1]);case
7:return uI(j,d,b[1],c[1]);case
8:return uF(j,d,b[1],c[1]);case
9:return uG(j,d,b[1],c[1]);default:return V(lz)}default:var
e=b[1];if(bK){if(di(a3(e),[0,c]))e3(e,0,c);S(c,0,0)}var
f=c[1],g=M(0);return uJ(j,d,a,cW(e,-701974253,c[1][8]-g|0),f)}},k);return uE(d,h,i,c[1],b)}}if(dc===0)var
d=eC([0]);else{var
a1=eC(aS(h5,dc));cq(function(a,b){var
c=(a*2|0)+2|0;a1[3]=o(au[4],b,c,a1[3]);a1[4]=o(aq[4],c,1,a1[4]);return 0},dc);var
d=a1}var
cI=aS(function(a){return aY(d,a)},fg),eJ=c0[2],rs=cI[1],rt=cI[2],ru=cI[3],ip=c0[4],eE=cJ(fe),eF=cJ(fg),eG=cJ(ff),rv=1,cK=cr(function(a){return aY(d,a)},eF),h8=cr(function(a){return aY(d,a)},eG);d[5]=[0,[0,d[3],d[4],d[6],d[7],cK,eE],d[5]];var
h9=ag[1],h_=d[7];function
h$(a,b,c){return cu(a,eE)?o(ag[4],a,b,c):c}d[7]=o(ag[11],h$,h_,h9);var
aZ=[0,au[1]],a0=[0,aq[1]];ed(function(a,b){aZ[1]=o(au[4],a,b,aZ[1]);var
e=a0[1];try{var
f=i(aq[22],b,d[4]),c=f}catch(f){f=s(f);if(f[1]!==u)throw f;var
c=1}a0[1]=o(aq[4],b,c,e);return 0},eG,h8);ed(function(a,b){aZ[1]=o(au[4],a,b,aZ[1]);a0[1]=o(aq[4],b,0,a0[1]);return 0},eF,cK);d[3]=aZ[1];d[4]=a0[1];var
ia=0,ib=d[6];d[6]=ct(function(a,b){return cu(a[1],cK)?b:[0,a,b]},ib,ia);var
iq=rv?i(eJ,d,ip):j(eJ,d),aE=ea(d[5]),eH=d[5],ic=aE[6],id=aE[5],ie=aE[4],ig=aE[3],ih=aE[2],ii=aE[1],ij=eH?eH[2]:V(ha);d[5]=ij;var
cs=ie,bx=ic;for(;;){if(bx){var
ec=bx[1],hb=bx[2],ik=i(ag[22],ec,d[7]),cs=o(ag[4],ec,ik,cs),bx=hb;continue}d[7]=cs;d[3]=ii;d[4]=ih;var
il=d[6];d[6]=ct(function(a,b){return cu(a[1],id)?b:[0,a,b]},il,ig);var
ir=0,is=cL(ff),it=[0,aS(function(a){var
e=aY(d,a);try{var
b=d[6];for(;;){if(!b)throw[0,u];var
c=b[1],f=b[2],g=c[2];if(0!==aK(c[1],e)){var
b=f;continue}break}}catch(f){f=s(f);if(f[1]===u)return r(d[2],e);throw f}return g},is),ir],iu=cL(fe),rw=sN([0,[0,iq],[0,aS(function(a){try{var
b=i(ag[22],a,d[7])}catch(f){f=s(f);if(f[1]===u)throw[0,K,io];throw f}return b},iu),it]])[1],rx=function(a,b){if(1===b.length-1){var
c=b[0+1];if(4===c[0])return c[1]}return V(ry)};eM(d,[0,rt,0,rr,ru,function(a,b){return[0,[4,b]]},rs,rx]);eD[1]=(eD[1]+d[1]|0)-1|0;d[8]=eb(d[8]);cG(d,3+ay(r(d[2],1)*16|0,aD)|0);var
rD=x(4),rE=aw(2),rF=a$(x(3),rE),rG=c6(aJ(x(0),rF),rD),rH=x(4),rI=aw(1),rJ=a$(x(3),rI),rK=a_(c6(aJ(x(0),rJ),rH),rG),rL=x(4),rM=x(3),rN=a_(c6(aJ(x(0),rM),rL),rK),rO=aw(2),rP=a$(x(3),rO),rQ=[0,aJ(x(0),rP)],rT=bM(ax(rS,rR),rQ),rU=c4(c2(gk),rT),rV=aw(1),rW=a$(x(3),rV),rX=[0,aJ(x(0),rW)],r0=bM(ax(rZ,rY),rX),r1=c4(c2(f8),r0),r2=x(3),r3=[0,aJ(x(0),r2)],r6=bM(ax(r5,r4),r3),r7=[0,e8(e8(c4(c2(f1),r6),r1),rU)],r_=bM(ax(r9,r8),r7),r$=a_(c8(x(4),r_),rN),sa=aw(4),sb=c3(x(2),sa),sc=a_(c8(x(3),sb),r$),sd=aw(t),se=c3(aw(t),sd),qW=[40,[46,x(2),se],sc],sh=ax(sg,sf),sk=c3(ax(sj,si),sh),sn=a$(ax(sm,sl),sk),qX=[26,a_(c8(x(2),sn),qW)],so=c7(c9(c5(4)),qX),sp=c7(c9(c5(3)),so),rB=[0,0],rC=[0,[13,5],eX],qV=[0,[1,[24,[23,qY,0],0]],c7(c9(c5(2)),sp)],sq=[0,function(a){var
d=1+1|0,e=d<=dP?1:0;if(e){var
b=d*4|0,f=gk*ai(a,b+2|0),g=f8*ai(a,b+1|0),c=f1*ai(a,b)+g+f|0;ah(a,b,c);ah(a,b+1|0,c);return ah(a,b+2|0,c)}return e},qV,rC,rB],db=eK(0,d);o(rw,db,rA,rz);if(!0){var
eL=d[8];if(0!==eL){var
bD=eL;for(;;){if(bD){var
iv=bD[2];j(bD[1],db);var
bD=iv;continue}break}}}var
bb=[0,db,sq],fj=function(a,b){var
c=da([0,"button"],0,H,ri);c.value=a.toString();c.onclick=c$(b);c.style.margin=ga;return c},bR=function(a){return fd(H)};c_.onload=c$(function(a){function
e(a){throw[0,K,sv]}var
c=e$(H.getElementById(gK),e);D(c,bR(0));var
v=da(0,0,H,rh),d=bP(H,rm);D(d,H.createTextNode("Choose a computing device : "));D(c,d);D(c,v);D(c,bR(0));var
f=bQ(H,rp);if(1-(f.getContext==e_?1:0)){f[b("width")]=t;f[b("height")]=t;f.style.margin=ga;var
l=bQ(H,rl);l.src="lena.png";var
m=f.getContext(rf);l.onload=c$(function(a){m.drawImage(l,0,0);D(c,bR(0));D(c,f);var
O=fk?fk[1]:2;switch(O){case
1:fz(0);aH[1]=fA(0);break;case
2:fB(0);aG[1]=fC(0);fz(0);aH[1]=fA(0);break;default:fB(0);aG[1]=fC(0)}eV[1]=aG[1]+aH[1]|0;var
A=aG[1]-1|0,z=0,P=0;if(A<0)var
B=z;else{var
k=P,F=z;for(;;){var
G=cp(F,[0,un(k),0]),Q=k+1|0;if(A!==k){var
k=Q,F=G;continue}var
B=G;break}}var
u=0,e=0,d=B;for(;;){if(u<aH[1]){if(uD(e))var
E=e+1|0,C=cp(d,[0,up(e,e+aG[1]|0),0]);else
var
E=e,C=d;var
u=u+1|0,e=E,d=C;continue}var
s=0,q=d;for(;;){if(q){var
s=s+1|0,q=q[2];continue}eV[1]=s;aH[1]=e;if(d){var
o=0,n=d,L=d[2],M=d[1];for(;;){if(n){var
o=o+1|0,n=n[2];continue}var
x=y(o,M),p=1,h=L;for(;;){if(h){var
N=h[2];x[p+1]=h[1];var
p=p+1|0,h=N;continue}var
w=x;break}break}}else
var
w=[0];var
I=m.getImageData(0,0,t,t),J=I.data;ba(0,sw,bb);var
g=da(0,0,H,rj);D(c,g);var
K=bb[1];g.value=ea(i(Y(K,f0,gx),K,0)).toString();D(c,bR(0));g[b("rows")]=33;g[b("cols")]=80;d_(function(a){var
b=bQ(H,rg);D(b,H.createTextNode(a[1][1].toString()));return D(v,b)},w);D(c,fj(sx,function(a){var
h=r(w,v.selectedIndex+0|0),x=h[1][1];j(fi(ss),x);var
c=aI(eX,0,dP*4|0),y=new
ae(g.value),z=j(cA(st),y),n=bb[1];i(Y(n,fM,fV),n,z);ex(h0,fo(0));var
o=W(c)-1|0,A=0;if(!(o<0)){var
e=A;for(;;){ah(c,e,J[e]);var
G=e+1|0;if(o!==e){var
e=G;continue}break}}var
p=h[2];if(0===p[0])var
k=cc;else
var
F=0===p[1][2]?1:cc,k=F;var
t=eU(0),B=0,C=[0,[0,k,1,1],[0,ay((dP+k|0)-1|0,k),1,1]],q=0,l=0?q[1]:q,f=bb[2],b=bb[1];if(0===h[2][0]){if(l)ba(0,q_,[0,b,f]);else
if(!i(Y(b,-723625231,7),b,0))ba(0,q$,[0,b,f])}else
if(l)ba(0,ra,[0,b,f]);else
if(!i(Y(b,f0,8),b,0))ba(0,rb,[0,b,f]);(function(a,b,c,d,e,f){return a.length==5?a(b,c,d,e,f):an(a,[b,c,d,e,f])}(Y(b,5695307,6),b,c,C,B,h));var
u=eU(0)-t;i(fi(sr),su,u);var
s=W(c)-1|0,D=0;if(!(s<0)){var
d=D;for(;;){J[d]=ai(c,d);var
E=d+1|0;if(s!==d){var
d=E;continue}break}}m.putImageData(I,0,0);return bN}));D(c,fj(sy,function(a){m.drawImage(l,0,0);return bN}));return bN}}});return bN}throw[0,ro]});d7(0);return}}(this));
