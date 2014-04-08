// This program was compiled from OCaml by js_of_ocaml 1.99dev
(function(d){"use strict";var
dg="set_cuda_sources",dy=123,b2=";",fN=108,gr="section1",df="reload_sources",b6="Map.bal",f1=",",ca='"',ag=16777215,de="get_cuda_sources",ch=" / ",fM="double spoc_var",dp="args_to_list",b$=" * ",ak="(",gq=0.5,fz="float spoc_var",dn=65599,cg="if (",b_="return",f0=" ;\n",dx="exec",br=115,bp=";}\n",fL=".ptx",z=512,dw=120,dd="..",fZ=-512,M="]",dv=117,b5="; ",du="compile",gp=" (",_="0",dm="list_to_args",b4=248,fY=126,go="fd ",dc="get_binaries",fK=" == ",au="(float)",dl="Kirc_Cuda.ml",cf=" + ",fX=") ",dt="x",fJ=-97,fy="g",bn=1073741823,gn="parse concat",aC=105,dk="get_opencl_sources",gm=511,bo=110,gl=-88,ai=" = ",dj="set_opencl_sources",fW=200,N="[",b9="'",fx="Unix",b1="int_of_string",gk="(double) ",fV=982028505,bm="){\n",bq="e",gj="#define __FLOAT64_EXTENSION__ \n",aB="-",aU=-48,b8="(double) spoc_var",fw="++){\n",fI="__shared__ float spoc_var",gi="opencl_sources",fH=".cl",ds="reset_binaries",b0="\n",gh=101,dB=748841679,ce="index out of bounds",fv="spoc_init_opencl_device_vec",db=125,b7=" - ",gg=";}",r=255,gf="binaries",cd="}",ge=" < ",fu="__shared__ long spoc_var",aT=250,gd=" >= ",gc=1024,ft="input",fU=246,di=102,fT="Unix.Unix_error",g="",fs=" || ",aS=100,dr="Kirc_OpenCL.ml",gb="#ifndef __FLOAT64_EXTENSION__ \n",fS="__shared__ int spoc_var",dA=103,bZ=", ",fR="./",fG=1e3,fr="for (int ",ga="file_file",f$="spoc_var",al=".",fF="else{\n",b3="+",dz="run",cc=65535,dq="#endif\n",aR=";\n",$="f",fE="Mandelbrot_js_js.ml",f_=785140586,f9="__shared__ double spoc_var",fD=-32,dh=111,fQ=" > ",C=" ",f8="int spoc_var",aj=")",fP="cuda_sources",f7=256,fC="nan",da=116,f4="../",f5="kernel_name",f6=65520,f3="%.12g",fq=" && ",fB="/",fO="while (",c$="compile_and_run",cb=114,f2="* spoc_var",bY=" <= ",n="number",fA=" % ",vl=d.spoc_opencl_part_device_to_cpu_b!==undefined?d.spoc_opencl_part_device_to_cpu_b:function(){o("spoc_opencl_part_device_to_cpu_b not implemented")},vk=d.spoc_opencl_part_cpu_to_device_b!==undefined?d.spoc_opencl_part_cpu_to_device_b:function(){o("spoc_opencl_part_cpu_to_device_b not implemented")},vi=d.spoc_opencl_load_param_int64!==undefined?d.spoc_opencl_load_param_int64:function(){o("spoc_opencl_load_param_int64 not implemented")},vg=d.spoc_opencl_load_param_float64!==undefined?d.spoc_opencl_load_param_float64:function(){o("spoc_opencl_load_param_float64 not implemented")},vf=d.spoc_opencl_load_param_float!==undefined?d.spoc_opencl_load_param_float:function(){o("spoc_opencl_load_param_float not implemented")},va=d.spoc_opencl_custom_part_device_to_cpu_b!==undefined?d.spoc_opencl_custom_part_device_to_cpu_b:function(){o("spoc_opencl_custom_part_device_to_cpu_b not implemented")},u$=d.spoc_opencl_custom_part_cpu_to_device_b!==undefined?d.spoc_opencl_custom_part_cpu_to_device_b:function(){o("spoc_opencl_custom_part_cpu_to_device_b not implemented")},u_=d.spoc_opencl_custom_device_to_cpu!==undefined?d.spoc_opencl_custom_device_to_cpu:function(){o("spoc_opencl_custom_device_to_cpu not implemented")},u9=d.spoc_opencl_custom_cpu_to_device!==undefined?d.spoc_opencl_custom_cpu_to_device:function(){o("spoc_opencl_custom_cpu_to_device not implemented")},u8=d.spoc_opencl_custom_alloc_vect!==undefined?d.spoc_opencl_custom_alloc_vect:function(){o("spoc_opencl_custom_alloc_vect not implemented")},uX=d.spoc_cuda_part_device_to_cpu_b!==undefined?d.spoc_cuda_part_device_to_cpu_b:function(){o("spoc_cuda_part_device_to_cpu_b not implemented")},uW=d.spoc_cuda_part_cpu_to_device_b!==undefined?d.spoc_cuda_part_cpu_to_device_b:function(){o("spoc_cuda_part_cpu_to_device_b not implemented")},uV=d.spoc_cuda_load_param_vec_b!==undefined?d.spoc_cuda_load_param_vec_b:function(){o("spoc_cuda_load_param_vec_b not implemented")},uU=d.spoc_cuda_load_param_int_b!==undefined?d.spoc_cuda_load_param_int_b:function(){o("spoc_cuda_load_param_int_b not implemented")},uT=d.spoc_cuda_load_param_int64_b!==undefined?d.spoc_cuda_load_param_int64_b:function(){o("spoc_cuda_load_param_int64_b not implemented")},uS=d.spoc_cuda_load_param_float_b!==undefined?d.spoc_cuda_load_param_float_b:function(){o("spoc_cuda_load_param_float_b not implemented")},uR=d.spoc_cuda_load_param_float64_b!==undefined?d.spoc_cuda_load_param_float64_b:function(){o("spoc_cuda_load_param_float64_b not implemented")},uQ=d.spoc_cuda_launch_grid_b!==undefined?d.spoc_cuda_launch_grid_b:function(){o("spoc_cuda_launch_grid_b not implemented")},uP=d.spoc_cuda_flush_all!==undefined?d.spoc_cuda_flush_all:function(){o("spoc_cuda_flush_all not implemented")},uO=d.spoc_cuda_flush!==undefined?d.spoc_cuda_flush:function(){o("spoc_cuda_flush not implemented")},uN=d.spoc_cuda_device_to_cpu!==undefined?d.spoc_cuda_device_to_cpu:function(){o("spoc_cuda_device_to_cpu not implemented")},uL=d.spoc_cuda_custom_part_device_to_cpu_b!==undefined?d.spoc_cuda_custom_part_device_to_cpu_b:function(){o("spoc_cuda_custom_part_device_to_cpu_b not implemented")},uK=d.spoc_cuda_custom_part_cpu_to_device_b!==undefined?d.spoc_cuda_custom_part_cpu_to_device_b:function(){o("spoc_cuda_custom_part_cpu_to_device_b not implemented")},uJ=d.spoc_cuda_custom_load_param_vec_b!==undefined?d.spoc_cuda_custom_load_param_vec_b:function(){o("spoc_cuda_custom_load_param_vec_b not implemented")},uI=d.spoc_cuda_custom_device_to_cpu!==undefined?d.spoc_cuda_custom_device_to_cpu:function(){o("spoc_cuda_custom_device_to_cpu not implemented")},uH=d.spoc_cuda_custom_cpu_to_device!==undefined?d.spoc_cuda_custom_cpu_to_device:function(){o("spoc_cuda_custom_cpu_to_device not implemented")},gG=d.spoc_cuda_custom_alloc_vect!==undefined?d.spoc_cuda_custom_alloc_vect:function(){o("spoc_cuda_custom_alloc_vect not implemented")},uG=d.spoc_cuda_create_extra!==undefined?d.spoc_cuda_create_extra:function(){o("spoc_cuda_create_extra not implemented")},uF=d.spoc_cuda_cpu_to_device!==undefined?d.spoc_cuda_cpu_to_device:function(){o("spoc_cuda_cpu_to_device not implemented")},gF=d.spoc_cuda_alloc_vect!==undefined?d.spoc_cuda_alloc_vect:function(){o("spoc_cuda_alloc_vect not implemented")},uC=d.spoc_create_custom!==undefined?d.spoc_create_custom:function(){o("spoc_create_custom not implemented")},vo=1;function
gB(a,b){throw[0,a,b]}function
dL(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=d.console;b&&b.error&&b.error(a)}var
q=[0];function
bu(a,b){if(!a)return g;if(a&1)return bu(a-1,b)+b;var
c=bu(a>>1,b);return c+c}function
F(a){if(a!=null){this.bytes=this.fullBytes=a;this.last=this.len=a.length}}function
gE(){gB(q[4],new
F(ce))}F.prototype={string:null,bytes:null,fullBytes:null,array:null,len:null,last:0,toJsString:function(){var
a=this.getFullBytes();try{return this.string=decodeURIComponent(escape(a))}catch(f){dL('MlString.toJsString: wrong encoding for \"%s\" ',a);return a}},toBytes:function(){if(this.string!=null)try{var
a=unescape(encodeURIComponent(this.string))}catch(f){dL('MlString.toBytes: wrong encoding for \"%s\" ',this.string);var
a=this.string}else{var
a=g,c=this.array,d=c.length;for(var
b=0;b<d;b++)a+=String.fromCharCode(c[b])}this.bytes=this.fullBytes=a;this.last=this.len=a.length;return a},getBytes:function(){var
a=this.bytes;if(a==null)a=this.toBytes();return a},getFullBytes:function(){var
a=this.fullBytes;if(a!==null)return a;a=this.bytes;if(a==null)a=this.toBytes();if(this.last<this.len){this.bytes=a+=bu(this.len-this.last,"\0");this.last=this.len}this.fullBytes=a;return a},toArray:function(){var
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
b=this.bytes;if(b==null)b=this.toBytes();return a<this.last?b.charCodeAt(a):0},safeGet:function(a){if(this.len==null)this.toBytes();if(a<0||a>=this.len)gE();return this.get(a)},set:function(a,b){var
c=this.array;if(!c){if(this.last==a){this.bytes+=String.fromCharCode(b&r);this.last++;return 0}c=this.toArray()}else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;c[a]=b&r;return 0},safeSet:function(a,b){if(this.len==null)this.toBytes();if(a<0||a>=this.len)gE();this.set(a,b)},fill:function(a,b,c){if(a>=this.last&&this.last&&c==0)return;var
d=this.array;if(!d)d=this.toArray();else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;var
f=a+b;for(var
e=a;e<f;e++)d[e]=c},compare:function(a){if(this.string!=null&&a.string!=null){if(this.string<a.string)return-1;if(this.string>a.string)return 1;return 0}var
b=this.getFullBytes(),c=a.getFullBytes();if(b<c)return-1;if(b>c)return 1;return 0},equal:function(a){if(this.string!=null&&a.string!=null)return this.string==a.string;return this.getFullBytes()==a.getFullBytes()},lessThan:function(a){if(this.string!=null&&a.string!=null)return this.string<a.string;return this.getFullBytes()<a.getFullBytes()},lessEqual:function(a){if(this.string!=null&&a.string!=null)return this.string<=a.string;return this.getFullBytes()<=a.getFullBytes()}};function
aD(a){this.string=a}aD.prototype=new
F();function
ts(a,b,c,d,e){if(d<=b)for(var
f=1;f<=e;f++)c[d+f]=a[b+f];else
for(var
f=e;f>=1;f--)c[d+f]=a[b+f]}function
tt(a){var
c=[0];while(a!==0){var
d=a[1];for(var
b=1;b<d.length;b++)c.push(d[b]);a=a[2]}return c}function
dK(a,b){gB(a,new
aD(b))}function
av(a){dK(q[4],a)}function
aV(){av(ce)}function
tu(a,b){if(b<0||b>=a.length-1)aV();return a[b+1]}function
tv(a,b,c){if(b<0||b>=a.length-1)aV();a[b+1]=c;return 0}var
dD;function
tw(a,b,c){if(c.length!=2)av("Bigarray.create: bad number of dimensions");if(b!=0)av("Bigarray.create: unsupported layout");if(c[1]<0)av("Bigarray.create: negative dimension");if(!dD){var
e=d;dD=[e.Float32Array,e.Float64Array,e.Int8Array,e.Uint8Array,e.Int16Array,e.Uint16Array,e.Int32Array,null,e.Int32Array,e.Int32Array,null,null,e.Uint8Array]}var
f=dD[a];if(!f)av("Bigarray.create: unsupported kind");return new
f(c[1])}function
tx(a,b){if(b<0||b>=a.length)aV();return a[b]}function
ty(a,b,c){if(b<0||b>=a.length)aV();a[b]=c;return 0}function
dE(a,b,c,d,e){if(e===0)return;if(d===c.last&&c.bytes!=null){var
f=a.bytes;if(f==null)f=a.toBytes();if(b>0||a.last>e)f=f.slice(b,b+e);c.bytes+=f;c.last+=f.length;return}var
g=c.array;if(!g)g=c.toArray();else
c.bytes=c.string=null;a.blitToArray(b,g,d,e)}function
am(c,b){if(c.fun)return am(c.fun,b);var
a=c.length,d=a-b.length;if(d==0)return c.apply(null,b);else
if(d<0)return am(c.apply(null,b.slice(0,a)),b.slice(a));else
return function(a){return am(c,b.concat([a]))}}function
tz(a){if(isFinite(a)){if(Math.abs(a)>=2.22507385850720138e-308)return 0;if(a!=0)return 1;return 2}return isNaN(a)?4:3}function
tL(a,b){var
c=a[3]<<16,d=b[3]<<16;if(c>d)return 1;if(c<d)return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
gy(a,b){if(a<b)return-1;if(a==b)return 0;return 1}function
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
b4:{var
d=gy(a[2],b[2]);if(d!=0)return d;break}case
251:av("equal: abstract value");case
r:{var
d=tL(a,b);if(d!=0)return d;break}default:if(a.length!=b.length)return a.length<b.length?-1:1;if(a.length>1)e.push(a,b,1)}}else
return 1}else
if(b
instanceof
F||b
instanceof
Array&&b[0]===(b[0]|0))return-1;else{if(a<b)return-1;if(a>b)return 1;if(c&&a!=b){if(a==a)return 1;if(b==b)return-1}}if(e.length==0)return 0;var
f=e.pop();b=e.pop();a=e.pop();if(f+1<a.length)e.push(a,b,f+1);a=a[f];b=b[f]}}function
gt(a,b){return dF(a,b,true)}function
gs(a){this.bytes=g;this.len=a}gs.prototype=new
F();function
gu(a){if(a<0)av("String.create");return new
gs(a)}function
dJ(a){throw[0,a]}function
gC(){dJ(q[6])}function
tA(a,b){if(b==0)gC();return a/b|0}function
tB(a,b){return+(dF(a,b,false)==0)}function
tC(a,b,c,d){a.fill(b,c,d)}function
dI(a){a=a.toString();var
e=a.length;if(e>31)av("format_int: format too long");var
b={justify:b3,signstyle:aB,filler:C,alternate:false,base:0,signedconv:false,width:0,uppercase:false,sign:1,prec:-1,conv:$};for(var
d=0;d<e;d++){var
c=a.charAt(d);switch(c){case
aB:b.justify=aB;break;case
b3:case
C:b.signstyle=c;break;case
_:b.filler=_;break;case"#":b.alternate=true;break;case"1":case"2":case"3":case"4":case"5":case"6":case"7":case"8":case"9":b.width=0;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.width=b.width*10+c;d++}d--;break;case
al:b.prec=0;d++;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.prec=b.prec*10+c;d++}d--;case"d":case"i":b.signedconv=true;case"u":b.base=10;break;case
dt:b.base=16;break;case"X":b.base=16;b.uppercase=true;break;case"o":b.base=8;break;case
bq:case
$:case
fy:b.signedconv=true;b.conv=c;break;case"E":case"F":case"G":b.signedconv=true;b.uppercase=true;b.conv=c.toLowerCase();break}}return b}function
dG(a,b){if(a.uppercase)b=b.toUpperCase();var
e=b.length;if(a.signedconv&&(a.sign<0||a.signstyle!=aB))e++;if(a.alternate){if(a.base==8)e+=1;if(a.base==16)e+=2}var
c=g;if(a.justify==b3&&a.filler==C)for(var
d=e;d<a.width;d++)c+=C;if(a.signedconv)if(a.sign<0)c+=aB;else
if(a.signstyle!=aB)c+=a.signstyle;if(a.alternate&&a.base==8)c+=_;if(a.alternate&&a.base==16)c+="0x";if(a.justify==b3&&a.filler==_)for(var
d=e;d<a.width;d++)c+=_;c+=b;if(a.justify==aB)for(var
d=e;d<a.width;d++)c+=C;return new
aD(c)}function
tD(a,b){var
c,f=dI(a),e=f.prec<0?6:f.prec;if(b<0){f.sign=-1;b=-b}if(isNaN(b)){c=fC;f.filler=C}else
if(!isFinite(b)){c="inf";f.filler=C}else
switch(f.conv){case
bq:var
c=b.toExponential(e),d=c.length;if(c.charAt(d-3)==bq)c=c.slice(0,d-1)+_+c.slice(d-1);break;case
$:c=b.toFixed(e);break;case
fy:e=e?e:1;c=b.toExponential(e-1);var
i=c.indexOf(bq),h=+c.slice(i+1);if(h<-4||b.toFixed(0).length>e){var
d=i-1;while(c.charAt(d)==_)d--;if(c.charAt(d)==al)d--;c=c.slice(0,d+1)+c.slice(i);d=c.length;if(c.charAt(d-3)==bq)c=c.slice(0,d-1)+_+c.slice(d-1);break}else{var
g=e;if(h<0){g-=h+1;c=b.toFixed(g)}else
while(c=b.toFixed(g),c.length>e+1)g--;if(g){var
d=c.length-1;while(c.charAt(d)==_)d--;if(c.charAt(d)==al)d--;c=c.slice(0,d+1)}}break}return dG(f,c)}function
tE(a,b){if(a.toString()=="%d")return new
aD(g+b);var
c=dI(a);if(b<0)if(c.signedconv){c.sign=-1;b=-b}else
b>>>=0;var
d=b.toString(c.base);if(c.prec>=0){c.filler=C;var
e=c.prec-d.length;if(e>0)d=bu(e,_)+d}return dG(c,d)}function
tF(){return 0}function
tG(){return 0}var
cj=[];function
tH(a,b,c){var
e=a[1],i=cj[c];if(i===null)for(var
h=cj.length;h<c;h++)cj[h]=0;else
if(e[i]===b)return e[i-1];var
d=3,g=e[1]*2+1,f;while(d<g){f=d+g>>1|1;if(b<e[f+1])g=f-2;else
d=f}cj[c]=d+1;return b==e[d+1]?e[d]:0}function
tI(a,b){return+(gt(a,b,false)>=0)}function
gv(a){if(!isFinite(a)){if(isNaN(a))return[r,1,0,f6];return a>0?[r,0,0,32752]:[r,0,0,f6]}var
f=a>=0?0:32768;if(f)a=-a;var
b=Math.floor(Math.LOG2E*Math.log(a))+1023;if(b<=0){b=0;a/=Math.pow(2,-1026)}else{a/=Math.pow(2,b-1027);if(a<16){a*=2;b-=1}if(b==0)a/=2}var
d=Math.pow(2,24),c=a|0;a=(a-c)*d;var
e=a|0;a=(a-e)*d;var
g=a|0;c=c&15|f|b<<4;return[r,g,e,c]}function
bt(a,b){return((a>>16)*b<<16)+(a&cc)*b|0}var
tJ=function(){var
p=f7;function
c(a,b){return a<<b|a>>>32-b}function
g(a,b){b=bt(b,3432918353);b=c(b,15);b=bt(b,461845907);a^=b;a=c(a,13);return(a*5|0)+3864292196|0}function
t(a){a^=a>>>16;a=bt(a,2246822507);a^=a>>>13;a=bt(a,3266489909);a^=a>>>16;return a}function
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
b4:f=g(f,e[2]);h--;break;case
aT:k[--l]=e[1];break;case
r:f=v(f,e);h--;break;default:var
s=e.length-1<<10|e[0];f=g(f,s);for(j=1,o=e.length;j<o;j++){if(m>=i)break;k[m++]=e[j]}break}else
if(e
instanceof
F){var
n=e.array;if(n)f=w(f,n);else{var
q=e.getFullBytes();f=x(f,q)}h--;break}else
if(e===(e|0)){f=g(f,e+e+1);h--}else
if(e===+e){f=u(f,gv(e));h--;break}}f=t(f);return f&bn}}();function
tT(a){return[a[3]>>8,a[3]&r,a[2]>>16,a[2]>>8&r,a[2]&r,a[1]>>16,a[1]>>8&r,a[1]&r]}function
tK(e,b,c){var
d=0;function
f(a){b--;if(e<0||b<0)return;if(a
instanceof
Array&&a[0]===(a[0]|0))switch(a[0]){case
b4:e--;d=d*dn+a[2]|0;break;case
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
j=tT(gv(a));for(var
c=7;c>=0;c--)d=d*19+j[c]|0}}f(c);return d&bn}function
tO(a){return(a[3]|a[2]|a[1])==0}function
tR(a){return[r,a&ag,a>>24&ag,a>>31&cc]}function
tS(a,b){var
c=a[1]-b[1],d=a[2]-b[2]+(c>>24),e=a[3]-b[3]+(d>>24);return[r,c&ag,d&ag,e&cc]}function
gx(a,b){if(a[3]>b[3])return 1;if(a[3]<b[3])return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
gw(a){a[3]=a[3]<<1|a[2]>>23;a[2]=(a[2]<<1|a[1]>>23)&ag;a[1]=a[1]<<1&ag}function
tP(a){a[1]=(a[1]>>>1|a[2]<<23)&ag;a[2]=(a[2]>>>1|a[3]<<23)&ag;a[3]=a[3]>>>1}function
tV(a,b){var
e=0,d=a.slice(),c=b.slice(),f=[r,0,0,0];while(gx(d,c)>0){e++;gw(c)}while(e>=0){e--;gw(f);if(gx(d,c)>=0){f[1]++;d=tS(d,c)}tP(c)}return[0,f,d]}function
tU(a){return a[1]|a[2]<<24}function
tN(a){return a[3]<<16<0}function
tQ(a){var
b=-a[1],c=-a[2]+(b>>24),d=-a[3]+(c>>24);return[r,b&ag,c&ag,d&cc]}function
tM(a,b){var
c=dI(a);if(c.signedconv&&tN(b)){c.sign=-1;b=tQ(b)}var
d=g,i=tR(c.base),h="0123456789abcdef";do{var
f=tV(b,i);b=f[1];d=h.charAt(tU(f[2]))+d}while(!tO(b));if(c.prec>=0){c.filler=C;var
e=c.prec-d.length;if(e>0)d=bu(e,_)+d}return dG(c,d)}function
uf(a){var
b=0,c=10,d=a.get(0)==45?(b++,-1):1;if(a.get(b)==48)switch(a.get(b+1)){case
dw:case
88:c=16;b+=2;break;case
dh:case
79:c=8;b+=2;break;case
98:case
66:c=2;b+=2;break}return[b,d,c]}function
gA(a){if(a>=48&&a<=57)return a-48;if(a>=65&&a<=90)return a-55;if(a>=97&&a<=122)return a-87;return-1}function
o(a){dK(q[3],a)}function
tW(a){var
g=uf(a),e=g[0],h=g[1],f=g[2],i=-1>>>0,d=a.get(e),c=gA(d);if(c<0||c>=f)o(b1);var
b=c;for(;;){e++;d=a.get(e);if(d==95)continue;c=gA(d);if(c<0||c>=f)break;b=f*b+c;if(b>i)o(b1)}if(e!=a.getLen())o(b1);b=h*b;if((b|0)!=b)o(b1);return b}function
tX(a){return+(a>31&&a<127)}var
ci={amp:/&/g,lt:/</g,quot:/\"/g,all:/[&<\"]/};function
tY(a){if(!ci.all.test(a))return a;return a.replace(ci.amp,"&amp;").replace(ci.lt,"&lt;").replace(ci.quot,"&quot;")}function
tZ(a){var
c=Array.prototype.slice;return function(){var
b=arguments.length>0?c.call(arguments):[undefined];return am(a,b)}}function
t0(a,b){var
d=[0];for(var
c=1;c<=a;c++)d[c]=b;return d}function
dC(a){var
b=a.length;this.array=a;this.len=this.last=b}dC.prototype=new
F();var
t1=function(){function
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
t2(a){return a.data.array.length}function
aw(a){dK(q[2],a)}function
dH(a){if(!a.opened)aw("Cannot flush a closed channel");if(a.buffer==g)return 0;if(a.output){switch(a.output.length){case
2:a.output(a,a.buffer);break;default:a.output(a.buffer)}}a.buffer=g}var
bs=new
Array();function
t3(a){dH(a);a.opened=false;delete
bs[a.fd];return 0}function
t4(a,b,c,d){var
e=a.data.array.length-a.data.offset;if(e<d)d=e;dE(new
dC(a.data.array),a.data.offset,b,c,d);a.data.offset+=d;return d}function
ug(){dJ(q[5])}function
t5(a){if(a.data.offset>=a.data.array.length)ug();if(a.data.offset<0||a.data.offset>a.data.array.length)aV();var
b=a.data.array[a.data.offset];a.data.offset++;return b}function
t6(a){var
b=a.data.offset,c=a.data.array.length;if(b>=c)return 0;while(true){if(b>=c)return-(b-a.data.offset);if(b<0||b>a.data.array.length)aV();if(a.data.array[b]==10)return b-a.data.offset+1;b++}}function
ui(a,b){if(!q.files)q.files={};if(b
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
up(a){return q.files&&q.files[a.toString()]?1:q.auto_register_file===undefined?0:q.auto_register_file(a)}function
bv(a,b,c){if(q.fds===undefined)q.fds=new
Array();c=c?c:{};var
d={};d.array=b;d.offset=c.append?d.array.length:0;d.flags=c;q.fds[a]=d;q.fd_last_idx=a;return a}function
ut(a,b,c){var
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
e=a.toString();if(d.rdonly&&d.wronly)aw(e+" : flags Open_rdonly and Open_wronly are not compatible");if(d.text&&d.binary)aw(e+" : flags Open_text and Open_binary are not compatible");if(up(a)){if(d.create&&d.excl)aw(e+" : file already exists");var
f=q.fd_last_idx?q.fd_last_idx:0;if(d.truncate)q.files[e]=g;return bv(f+1,q.files[e],d)}else
if(d.create){var
f=q.fd_last_idx?q.fd_last_idx:0;ui(e,[]);return bv(f+1,q.files[e],d)}else
aw(e+": no such file or directory")}bv(0,[]);bv(1,[]);bv(2,[]);function
t7(a){var
b=q.fds[a];if(b.flags.wronly)aw(go+a+" is writeonly");return{data:b,fd:a,opened:true}}function
uA(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=d.console;b&&b.log&&b.log(a)}function
ul(a,b){var
e=new
F(b),d=e.getLen();for(var
c=0;c<d;c++)a.data.array[a.data.offset+c]=e.get(c);a.data.offset+=d;return 0}function
t8(a){var
b;switch(a){case
1:b=uA;break;case
2:b=dL;break;default:b=ul}var
d=q.fds[a];if(d.flags.rdonly)aw(go+a+" is readonly");var
c={data:d,fd:a,opened:true,buffer:g,output:b};bs[c.fd]=c;return c}function
t9(){var
a=0;for(var
b
in
bs)if(bs[b].opened)a=[0,bs[b],a];return a}function
gz(a,b,c,d){if(!a.opened)aw("Cannot output to a closed channel");var
f;if(c==0&&b.getLen()==d)f=b;else{f=gu(d);dE(b,c,f,0,d)}var
e=f.toString(),g=e.lastIndexOf("\n");if(g<0)a.buffer+=e;else{a.buffer+=e.substr(0,g+1);dH(a);a.buffer+=e.substr(g+1)}}function
S(a){return new
F(a)}function
t_(a,b){var
c=S(String.fromCharCode(b));gz(a,c,0,1)}function
t$(a,b){if(b==0)gC();return a%b}function
ub(a,b){return+(dF(a,b,false)!=0)}function
uc(a,b){var
d=[a];for(var
c=1;c<=b;c++)d[c]=0;return d}function
ud(a,b){a[0]=b;return 0}function
ue(a){return a
instanceof
Array?a[0]:fG}function
uj(a,b){q[a+1]=b}var
ua={};function
uk(a,b){ua[a]=b;return 0}function
um(a,b){return a.compare(b)}function
gD(a,b){var
c=a.fullBytes,d=b.fullBytes;if(c!=null&&d!=null)return c==d?1:0;return a.getFullBytes()==b.getFullBytes()?1:0}function
un(a,b){return 1-gD(a,b)}function
uo(){return 32}function
uq(){var
a=new
aD("a.out");return[0,a,[0,a]]}function
ur(){return[0,new
aD(fx),32,0]}function
uh(){dJ(q[7])}function
us(){uh()}function
uu(){var
a=new
Date()^4294967295*Math.random();return{valueOf:function(){return a},0:0,1:a,length:2}}function
uv(){console.log("caml_sys_system_command");return 0}function
uw(a){var
b=1;while(a&&a.joo_tramp){a=a.joo_tramp.apply(null,a.joo_args);b++}return a}function
ux(a,b){return{joo_tramp:a,joo_args:b}}function
uy(a,b){if(typeof
b==="function"){a.fun=b;return 0}if(b.fun){a.fun=b.fun;return 0}var
c=b.length;while(c--)a[c]=b[c];return 0}function
uz(){return 0}var
dM=0;function
uB(){if(window.webcl==undefined){alert("Unfortunately your system does not support WebCL. "+"Make sure that you have both the OpenCL driver "+"and the WebCL browser extension installed.");dM=1}else{console.log("INIT OPENCL");dM=0}return 0}function
uD(){console.log(" spoc_cuInit");return 0}function
uE(){console.log(" spoc_cuda_compile");return 0}function
uM(){console.log(" spoc_cuda_debug_compile");return 0}function
uY(a,b,c){console.log(" spoc_debug_opencl_compile");console.log(a.bytes);var
e=c[9],f=e[0],d=f.createProgram(a.bytes),g=d.getInfo(WebCL.PROGRAM_DEVICES);d.build(g);var
h=d.createKernel(b.bytes);e[0]=f;c[9]=e;return h}function
uZ(a){console.log("spoc_getCudaDevice");return 0}function
u0(){console.log(" spoc_getCudaDevicesCount");return 0}function
u1(a,b){console.log(" spoc_getOpenCLDevice");var
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
u2(){console.log(" spoc_getOpenCLDevicesCount");var
a=0,b=webcl.getPlatforms();for(var
d
in
b){var
e=b[d],c=e.getDevices();a+=c.length}return a}function
u3(){console.log(fv);return 0}function
u4(){console.log(fv);var
a=new
Array(3);a[0]=0;return a}function
dN(a){if(a[1]instanceof
Float32Array||a[1].constructor.name=="Float32Array")return 4;if(a[1]instanceof
Int32Array||a[1].constructor.name=="Int32Array")return 4;{console.log("unimplemented vector type");console.log(a[1].constructor.name);return 4}}function
u5(a,b,c){console.log("spoc_opencl_alloc_vect");var
f=a[2],i=a[4],h=i[b+1],j=a[5],k=dN(f),d=c[9],e=d[0],d=c[9],e=d[0],g=e.createBuffer(WebCL.MEM_READ_WRITE,j*k);h[2]=g;d[0]=e;c[9]=d;return 0}function
u6(){console.log(" spoc_opencl_compile");return 0}function
u7(a,b,c,d){console.log("spoc_opencl_cpu_to_device");var
f=a[2],k=a[4],j=k[b+1],l=a[5],m=dN(f),e=c[9],h=e[0],g=e[d+1],i=j[2];g.enqueueWriteBuffer(i,false,0,l*m,f[1]);e[d+1]=g;e[0]=h;c[9]=e;return 0}function
vb(a,b,c,d,e){console.log("spoc_opencl_device_to_cpu");var
g=a[2],l=a[4],k=l[b+1],n=a[5],o=dN(g),f=c[9],i=f[0],h=f[e+1],j=k[2],m=g[1];h.enqueueReadBuffer(j,false,0,n*o,m);f[e+1]=h;f[0]=i;c[9]=f;return 0}function
vc(a,b){console.log("spoc_opencl_flush");var
c=a[9][b+1];c.flush();a[9][b+1]=c;return 0}function
vd(){console.log(" spoc_opencl_is_available");return!dM}function
ve(a,b,c,d,e){console.log("spoc_opencl_launch_grid");var
k=b[1],l=b[2],m=b[3],h=c[1],i=c[2],j=c[3],g=new
Array(3);g[0]=k*h;g[1]=l*i;g[2]=m*j;var
f=new
Array(3);f[0]=h;f[1]=i;f[2]=j;var
o=d[9],n=o[e+1];if(h==1&&i==1&&j==1)n.enqueueNDRangeKernel(a,f.length,null,[k,l,m]);else
n.enqueueNDRangeKernel(a,f.length,null,g,f);return 0}function
vh(a,b,c,d){console.log("spoc_opencl_load_param_int");b.setArg(a[1],new
Uint32Array([c]));a[1]=a[1]+1;return 0}function
vj(a,b,c,d,e){console.log("spoc_opencl_load_param_vec");var
f=d[2];b.setArg(a[1],f);a[1]=a[1]+1;return 0}function
vm(){return new
Date().getTime()/fG}function
vn(){return 0}var
s=tu,m=tv,bg=dE,aP=gt,B=gu,at=tA,c2=tD,bT=tE,bh=tG,af=tH,c6=tX,fl=tY,v=t0,fa=t3,c4=dH,c8=t4,e_=t7,c3=t8,aQ=t$,x=bt,b=S,c7=ub,fd=uc,aO=uj,c5=uk,fc=um,bW=gD,y=un,bU=us,e$=ut,fb=uu,fk=uv,Z=uw,E=ux,fj=uz,fm=uB,fo=uD,fp=u0,fn=u2,fg=u3,ff=u4,fh=u5,fe=vc,bX=vn;function
j(a,b){return a.length==1?a(b):am(a,[b])}function
i(a,b,c){return a.length==2?a(b,c):am(a,[b,c])}function
p(a,b,c,d){return a.length==3?a(b,c,d):am(a,[b,c,d])}function
fi(a,b,c,d,e,f,g){return a.length==6?a(b,c,d,e,f,g):am(a,[b,c,d,e,f,g])}var
aW=[0,b("Failure")],bw=[0,b("Invalid_argument")],bx=[0,b("End_of_file")],t=[0,b("Not_found")],H=[0,b("Assert_failure")],cH=b(al),cK=b(al),cM=b(al),eR=b(g),eQ=[0,b(ga),b(f5),b(fP),b(gi),b(gf)],e9=[0,1],e4=[0,b(gi),b(f5),b(ga),b(fP),b(gf)],e5=[0,b(du),b(c$),b(dc),b(de),b(dk),b(df),b(ds),b(dz),b(dg),b(dj)],e6=[0,b(dm),b(dx),b(dp)],c0=[0,b(dx),b(dc),b(de),b(dp),b(dm),b(c$),b(dz),b(dj),b(du),b(df),b(ds),b(dk),b(dg)];aO(6,t);aO(5,[0,b("Division_by_zero")]);aO(4,bx);aO(3,bw);aO(2,aW);aO(1,[0,b("Sys_error")]);var
gN=b("really_input"),gM=[0,0,[0,7,0]],gL=[0,1,[0,3,[0,4,[0,7,0]]]],gK=b(f3),gJ=b(al),gH=b("true"),gI=b("false"),gO=b("Pervasives.do_at_exit"),gQ=b("Array.blit"),gU=b("List.iter2"),gS=b("tl"),gR=b("hd"),gY=b("\\b"),gZ=b("\\t"),g0=b("\\n"),g1=b("\\r"),gX=b("\\\\"),gW=b("\\'"),gV=b("Char.chr"),g4=b("String.contains_from"),g3=b("String.blit"),g2=b("String.sub"),hb=b("Map.remove_min_elt"),hc=[0,0,0,0],hd=[0,b("map.ml"),270,10],he=[0,0,0],g9=b(b6),g_=b(b6),g$=b(b6),ha=b(b6),hf=b("CamlinternalLazy.Undefined"),hi=b("Buffer.add: cannot grow buffer"),hy=b(g),hz=b(g),hC=b(f3),hD=b(ca),hE=b(ca),hA=b(b9),hB=b(b9),hx=b(fC),hv=b("neg_infinity"),hw=b("infinity"),hu=b(al),ht=b("printf: bad positional specification (0)."),hs=b("%_"),hr=[0,b("printf.ml"),143,8],hp=b(b9),hq=b("Printf: premature end of format string '"),hl=b(b9),hm=b(" in format string '"),hn=b(", at char number "),ho=b("Printf: bad conversion %"),hj=b("Sformat.index_of_int: negative argument "),hG=b(dt),hH=[0,987910699,495797812,364182224,414272206,318284740,990407751,383018966,270373319,840823159,24560019,536292337,512266505,189156120,730249596,143776328,51606627,140166561,366354223,1003410265,700563762,981890670,913149062,526082594,1021425055,784300257,667753350,630144451,949649812,48546892,415514493,258888527,511570777,89983870,283659902,308386020,242688715,482270760,865188196,1027664170,207196989,193777847,619708188,671350186,149669678,257044018,87658204,558145612,183450813,28133145,901332182,710253903,510646120,652377910,409934019,801085050],to=b("OCAMLRUNPARAM"),tm=b("CAMLRUNPARAM"),hJ=b(g),h6=[0,b("camlinternalOO.ml"),287,50],h5=b(g),hL=b("CamlinternalOO.last_id"),iz=b(g),iw=b(fR),iv=b(".\\"),iu=b(f4),it=b("..\\"),ik=b(fR),ij=b(f4),ie=b(g),id=b(g),ig=b(dd),ih=b(fB),tk=b("TMPDIR"),im=b("/tmp"),io=b("'\\''"),ir=b(dd),is=b("\\"),ti=b("TEMP"),ix=b(al),iC=b(dd),iD=b(fB),iG=b("Cygwin"),iH=b(fx),iI=b("Win32"),iJ=[0,b("filename.ml"),189,9],iQ=b("E2BIG"),iS=b("EACCES"),iT=b("EAGAIN"),iU=b("EBADF"),iV=b("EBUSY"),iW=b("ECHILD"),iX=b("EDEADLK"),iY=b("EDOM"),iZ=b("EEXIST"),i0=b("EFAULT"),i1=b("EFBIG"),i2=b("EINTR"),i3=b("EINVAL"),i4=b("EIO"),i5=b("EISDIR"),i6=b("EMFILE"),i7=b("EMLINK"),i8=b("ENAMETOOLONG"),i9=b("ENFILE"),i_=b("ENODEV"),i$=b("ENOENT"),ja=b("ENOEXEC"),jb=b("ENOLCK"),jc=b("ENOMEM"),jd=b("ENOSPC"),je=b("ENOSYS"),jf=b("ENOTDIR"),jg=b("ENOTEMPTY"),jh=b("ENOTTY"),ji=b("ENXIO"),jj=b("EPERM"),jk=b("EPIPE"),jl=b("ERANGE"),jm=b("EROFS"),jn=b("ESPIPE"),jo=b("ESRCH"),jp=b("EXDEV"),jq=b("EWOULDBLOCK"),jr=b("EINPROGRESS"),js=b("EALREADY"),jt=b("ENOTSOCK"),ju=b("EDESTADDRREQ"),jv=b("EMSGSIZE"),jw=b("EPROTOTYPE"),jx=b("ENOPROTOOPT"),jy=b("EPROTONOSUPPORT"),jz=b("ESOCKTNOSUPPORT"),jA=b("EOPNOTSUPP"),jB=b("EPFNOSUPPORT"),jC=b("EAFNOSUPPORT"),jD=b("EADDRINUSE"),jE=b("EADDRNOTAVAIL"),jF=b("ENETDOWN"),jG=b("ENETUNREACH"),jH=b("ENETRESET"),jI=b("ECONNABORTED"),jJ=b("ECONNRESET"),jK=b("ENOBUFS"),jL=b("EISCONN"),jM=b("ENOTCONN"),jN=b("ESHUTDOWN"),jO=b("ETOOMANYREFS"),jP=b("ETIMEDOUT"),jQ=b("ECONNREFUSED"),jR=b("EHOSTDOWN"),jS=b("EHOSTUNREACH"),jT=b("ELOOP"),jU=b("EOVERFLOW"),jV=b("EUNKNOWNERR %d"),iR=b("Unix.Unix_error(Unix.%s, %S, %S)"),iM=b(fT),iN=b(g),iO=b(g),iP=b(fT),jW=b("0.0.0.0"),jX=b("127.0.0.1"),th=b("::"),tg=b("::1"),j7=[0,b("Vector.ml"),fY,25],j8=b("Cuda.No_Cuda_Device"),j9=b("Cuda.ERROR_DEINITIALIZED"),j_=b("Cuda.ERROR_NOT_INITIALIZED"),j$=b("Cuda.ERROR_INVALID_CONTEXT"),ka=b("Cuda.ERROR_INVALID_VALUE"),kb=b("Cuda.ERROR_OUT_OF_MEMORY"),kc=b("Cuda.ERROR_INVALID_DEVICE"),kd=b("Cuda.ERROR_NOT_FOUND"),ke=b("Cuda.ERROR_FILE_NOT_FOUND"),kf=b("Cuda.ERROR_UNKNOWN"),kg=b("Cuda.ERROR_LAUNCH_FAILED"),kh=b("Cuda.ERROR_LAUNCH_OUT_OF_RESOURCES"),ki=b("Cuda.ERROR_LAUNCH_TIMEOUT"),kj=b("Cuda.ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"),kk=b("no_cuda_device"),kl=b("cuda_error_deinitialized"),km=b("cuda_error_not_initialized"),kn=b("cuda_error_invalid_context"),ko=b("cuda_error_invalid_value"),kp=b("cuda_error_out_of_memory"),kq=b("cuda_error_invalid_device"),kr=b("cuda_error_not_found"),ks=b("cuda_error_file_not_found"),kt=b("cuda_error_launch_failed"),ku=b("cuda_error_launch_out_of_resources"),kv=b("cuda_error_launch_timeout"),kw=b("cuda_error_launch_incompatible_texturing"),kx=b("cuda_error_unknown"),ky=b("OpenCL.No_OpenCL_Device"),kz=b("OpenCL.OPENCL_ERROR_UNKNOWN"),kA=b("OpenCL.INVALID_CONTEXT"),kB=b("OpenCL.INVALID_DEVICE"),kC=b("OpenCL.INVALID_VALUE"),kD=b("OpenCL.INVALID_QUEUE_PROPERTIES"),kE=b("OpenCL.OUT_OF_RESOURCES"),kF=b("OpenCL.MEM_OBJECT_ALLOCATION_FAILURE"),kG=b("OpenCL.OUT_OF_HOST_MEMORY"),kH=b("OpenCL.FILE_NOT_FOUND"),kI=b("OpenCL.INVALID_PROGRAM"),kJ=b("OpenCL.INVALID_BINARY"),kK=b("OpenCL.INVALID_BUILD_OPTIONS"),kL=b("OpenCL.INVALID_OPERATION"),kM=b("OpenCL.COMPILER_NOT_AVAILABLE"),kN=b("OpenCL.BUILD_PROGRAM_FAILURE"),kO=b("OpenCL.INVALID_KERNEL"),kP=b("OpenCL.INVALID_ARG_INDEX"),kQ=b("OpenCL.INVALID_ARG_VALUE"),kR=b("OpenCL.INVALID_MEM_OBJECT"),kS=b("OpenCL.INVALID_SAMPLER"),kT=b("OpenCL.INVALID_ARG_SIZE"),kU=b("OpenCL.INVALID_COMMAND_QUEUE"),kV=b("no_opencl_device"),kW=b("opencl_error_unknown"),kX=b("opencl_invalid_context"),kY=b("opencl_invalid_device"),kZ=b("opencl_invalid_value"),k0=b("opencl_invalid_queue_properties"),k1=b("opencl_out_of_resources"),k2=b("opencl_mem_object_allocation_failure"),k3=b("opencl_out_of_host_memory"),k4=b("opencl_file_not_found"),k5=b("opencl_invalid_program"),k6=b("opencl_invalid_binary"),k7=b("opencl_invalid_build_options"),k8=b("opencl_invalid_operation"),k9=b("opencl_compiler_not_available"),k_=b("opencl_build_program_failure"),k$=b("opencl_invalid_kernel"),la=b("opencl_invalid_arg_index"),lb=b("opencl_invalid_arg_value"),lc=b("opencl_invalid_mem_object"),ld=b("opencl_invalid_sampler"),le=b("opencl_invalid_arg_size"),lf=b("opencl_invalid_command_queue"),lg=b(ce),lh=b(ce),ly=b(fL),lx=b(fH),lw=b(fL),lv=b(fH),lu=[0,1],lt=b(g),lp=b(b0),lk=b("Cl LOAD ARG Type Not Implemented\n"),lj=b("CU LOAD ARG Type Not Implemented\n"),li=[0,b(dj),b(dg),b(dz),b(ds),b(df),b(dm),b(dk),b(de),b(dc),b(dx),b(c$),b(du),b(dp)],ll=b("Kernel.ERROR_BLOCK_SIZE"),ln=b("Kernel.ERROR_GRID_SIZE"),lq=b("Kernel.No_source_for_device"),lB=b("Empty"),lC=b("Unit"),lD=b("Kern"),lE=b("Params"),lF=b("Plus"),lG=b("Plusf"),lH=b("Min"),lI=b("Minf"),lJ=b("Mul"),lK=b("Mulf"),lL=b("Div"),lM=b("Divf"),lN=b("Mod"),lO=b("Id "),lP=b("IdName "),lQ=b("IntVar "),lR=b("FloatVar "),lS=b("UnitVar "),lT=b("CastDoubleVar "),lU=b("DoubleVar "),lV=b("IntArr"),lW=b("Int32Arr"),lX=b("Int64Arr"),lY=b("Float32Arr"),lZ=b("Float64Arr"),l0=b("VecVar "),l1=b("Concat"),l2=b("Seq"),l3=b("Return"),l4=b("Set"),l5=b("Decl"),l6=b("SetV"),l7=b("SetLocalVar"),l8=b("Intrinsics"),l9=b(C),l_=b("IntId "),l$=b("Int "),mb=b("IntVecAcc"),mc=b("Local"),md=b("Acc"),me=b("Ife"),mf=b("If"),mg=b("Or"),mh=b("And"),mi=b("EqBool"),mj=b("LtBool"),mk=b("GtBool"),ml=b("LtEBool"),mm=b("GtEBool"),mn=b("DoLoop"),mo=b("While"),mp=b("App"),mq=b("GInt"),mr=b("GFloat"),ma=b("Float "),lA=b("  "),lz=b("%s\n"),n5=b(f1),n6=[0,b(dl),166,14],mu=b(g),mv=b(b0),mw=b("\n}\n#ifdef __cplusplus\n}\n#endif"),mx=b(" ) {\n"),my=b(g),mz=b(bZ),mB=b(g),mA=b('#ifdef __cplusplus\nextern "C" {\n#endif\n\n__global__ void spoc_dummy ( '),mC=b(aj),mD=b(cf),mE=b(ak),mF=b(aj),mG=b(cf),mH=b(ak),mI=b(aj),mJ=b(b7),mK=b(ak),mL=b(aj),mM=b(b7),mN=b(ak),mO=b(aj),mP=b(b$),mQ=b(ak),mR=b(aj),mS=b(b$),mT=b(ak),mU=b(aj),mV=b(ch),mW=b(ak),mX=b(aj),mY=b(ch),mZ=b(ak),m0=b(aj),m1=b(fA),m2=b(ak),m3=b(f8),m4=b(fz),m5=[0,b(dl),65,17],m6=b(b8),m7=b(fM),m8=b(M),m9=b(N),m_=b(fS),m$=b(M),na=b(N),nb=b(fu),nc=b(M),nd=b(N),ne=b(fI),nf=b(M),ng=b(N),nh=b(f9),ni=b(f2),nk=b("int"),nl=b("float"),nm=b("double"),nj=[0,b(dl),60,12],no=b(bZ),nn=b(gn),np=b(f0),nq=b(g),nr=b(g),nu=b(b2),nv=b(ai),nw=b(aR),ny=b(b2),nx=b(ai),nz=b($),nA=b(M),nB=b(N),nC=b("}\n"),nD=b(aR),nE=b(aR),nF=b("{"),nG=b(bp),nH=b(fF),nI=b(bp),nJ=b(bm),nK=b(cg),nL=b(bp),nM=b(bm),nN=b(cg),nO=b(fs),nP=b(fq),nQ=b(fK),nR=b(ge),nS=b(fQ),nT=b(bY),nU=b(gd),nV=b(cd),nW=b(fw),nX=b(b5),nY=b(bY),nZ=b(b5),n0=b(ai),n1=b(fr),n2=b(cd),n3=b(bm),n4=b(fO),n9=b(b_),n_=b(b_),n$=b(C),oa=b(C),n7=b(fX),n8=b(gp),ob=b($),ns=b(b2),nt=b(ai),oc=b(M),od=b(N),of=b(b8),og=b($),oh=b(gk),oi=b(M),oj=b(N),ok=b($),oe=b("cuda error parse_float"),ms=[0,b(g),b(g)],pG=b(f1),pH=[0,b(dr),162,14],on=b(g),oo=b(b0),op=b(cd),oq=b(" ) \n{\n"),or=b(g),os=b(bZ),ou=b(g),ot=b("__kernel void spoc_dummy ( "),ov=b(cf),ow=b(cf),ox=b(b7),oy=b(b7),oz=b(b$),oA=b(b$),oB=b(ch),oC=b(ch),oD=b(fA),oE=b(f8),oF=b(fz),oG=[0,b(dr),65,17],oH=b(b8),oI=b(fM),oJ=b(M),oK=b(N),oL=b(fS),oM=b(M),oN=b(N),oO=b(fu),oP=b(M),oQ=b(N),oR=b(fI),oS=b(M),oT=b(N),oU=b(f9),oV=b(f2),oX=b("__global int"),oY=b("__global float"),oZ=b("__global double"),oW=[0,b(dr),60,12],o1=b(bZ),o0=b(gn),o2=b(f0),o3=b(g),o4=b(g),o6=b(b2),o7=b(ai),o8=b(aR),o9=b(ai),o_=b($),o$=b(M),pa=b(N),pb=b(g),pc=b(b0),pd=b(aR),pe=b(g),pf=b(bp),pg=b(fF),ph=b(bp),pi=b(bm),pj=b(cg),pk=b(cd),pl=b(aR),pm=b("{\n"),pn=b(")\n"),po=b(cg),pp=b(fs),pq=b(fq),pr=b(fK),ps=b(ge),pt=b(fQ),pu=b(bY),pv=b(gd),pw=b(gg),px=b(fw),py=b(b5),pz=b(bY),pA=b(b5),pB=b(ai),pC=b(fr),pD=b(gg),pE=b(bm),pF=b(fO),pK=b(b_),pL=b(b_),pM=b(C),pN=b(C),pI=b(fX),pJ=b(gp),pO=b($),o5=b(ai),pP=b(M),pQ=b(N),pS=b(b8),pT=b($),pU=b(gk),pV=b(M),pW=b(N),pX=b($),pR=b("opencl error parse_float"),ol=[0,b(g),b(g)],qX=[0,0],qY=[0,0],qZ=[0,1],q0=[0,1],qR=b("kirc_kernel.cu"),qS=b("nvcc -m64 -arch=sm_10 -O3 -ptx kirc_kernel.cu -o kirc_kernel.ptx"),qT=b("kirc_kernel.ptx"),qU=b("rm kirc_kernel.cu kirc_kernel.ptx"),qO=[0,b(g),b(g)],qQ=b(g),qP=[0,b("Kirc.ml"),407,81],qV=b(ai),qW=b(f$),qJ=[33,0],qF=b(f$),pY=b("int spoc_xor (int a, int b ) { return (a^b);}\n"),pZ=b("int spoc_powint (int a, int b ) { return ((int) pow (((float) a), ((float) b)));}\n"),p0=b("int logical_and (int a, int b ) { return (a & b);}\n"),p1=b("float spoc_fdiv ( float a, float b ) { return (a / b);}\n"),p2=b("float spoc_fmul ( float a, float b ) { return (a * b);}\n"),p3=b("float spoc_fminus ( float a, float b ) { return (a - b);}\n"),p4=b("float spoc_fadd ( float a, float b ) { return (a + b);}\n"),p5=b("float spoc_fdiv ( float a, float b );\n"),p6=b("float spoc_fmul ( float a, float b );\n"),p7=b("float spoc_fminus ( float a, float b );\n"),p8=b("float spoc_fadd ( float a, float b );\n"),p_=b(dq),p$=b("double spoc_ddiv ( double a, double b ) { return (a / b);}\n"),qa=b("double spoc_dmul ( double a, double b ) { return (a * b);}\n"),qb=b("double spoc_dminus ( double a, double b ) { return (a - b);}\n"),qc=b("double spoc_dadd ( double a, double b ) { return (a + b);}\n"),qd=b("double spoc_ddiv ( double a, double b );\n"),qe=b("double spoc_dmul ( double a, double b );\n"),qf=b("double spoc_dminus ( double a, double b );\n"),qg=b("double spoc_dadd ( double a, double b );\n"),qh=b(dq),qi=b("#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"),qj=b("#elif defined(cl_amd_fp64)  // AMD extension available?\n"),qk=b("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"),ql=b("#if defined(cl_khr_fp64)  // Khronos extension available?\n"),qm=b(gj),qn=b(gb),qp=b(dq),qq=b("__device__ double spoc_ddiv ( double a, double b ) { return (a / b);}\n"),qr=b("__device__ double spoc_dmul ( double a, double b ) { return (a * b);}\n"),qs=b("__device__ double spoc_dminus ( double a, double b ) { return (a - b);}\n"),qt=b("__device__ double spoc_dadd ( double a, double b ) { return (a + b);}\n"),qu=b(gj),qv=b(gb),qx=b("__device__ int spoc_xor (int a, int b ) { return (a^b);}\n"),qy=b("__device__ int spoc_powint (int a, int b ) { return ((int) pow (((double) a), ((double) b)));}\n"),qz=b("__device__ int logical_and (int a, int b ) { return (a & b);}\n"),qA=b("__device__ float spoc_fdiv ( float a, float b ) { return (a / b);}\n"),qB=b("__device__ float spoc_fmul ( float a, float b ) { return (a * b);}\n"),qC=b("__device__ float spoc_fminus ( float a, float b ) { return (a - b);}\n"),qD=b("__device__ float spoc_fadd ( float a, float b ) { return (a + b);}\n"),qM=[0,b(g),b(g)],rh=b("canvas"),re=b("span"),rd=b("a"),rc=b("br"),rb=b(ft),ra=b("select"),q$=b("option"),rf=b("Dom_html.Canvas_not_available"),td=[0,b(fE),167,17],te=[0,1],tb=b("Will use device : %s!"),tc=b(g),ta=[0,196,fW,fW],s$=b("Time %s : %Fs\n%!"),rs=b("spoc_dummy"),rt=b("kirc_kernel"),rq=b("spoc_kernel_extension error"),ri=[0,b(fE),12,15],r7=b(au),r8=b(au),r$=b(au),sa=b(au),sg=b(au),sh=b(au),sk=b(au),sl=b(au),sG=b("(get_local_size (0))"),sH=b("blockDim.x"),sJ=b("(get_group_id (0))"),sK=b("blockIdx.x"),sM=b("(get_local_id (0))"),sN=b("threadIdx.x"),sQ=b("(get_local_size (1))"),sR=b("blockDim.y"),sT=b("(get_group_id (1))"),sU=b("blockIdx.y"),sW=b("(get_local_id (1))"),sX=b("threadIdx.y");function
T(a){throw[0,aW,a]}function
G(a){throw[0,bw,a]}function
h(a,b){var
c=a.getLen(),e=b.getLen(),d=B(c+e|0);bg(a,0,d,0,c);bg(b,0,d,c,e);return d}function
k(a){return b(g+a)}function
O(a){var
c=c2(gK,a),b=0,f=c.getLen();for(;;){if(f<=b)var
e=h(c,gJ);else{var
d=c.safeGet(b),g=48<=d?58<=d?0:1:45===d?1:0;if(g){var
b=b+1|0;continue}var
e=c}return e}}function
ck(a,b){if(a){var
c=a[1];return[0,c,ck(a[2],b)]}return b}e_(0);var
dO=c3(1);c3(2);function
dP(a,b){return gz(a,b,0,b.getLen())}function
dQ(a){return e_(e$(a,gM,0))}function
dR(a){var
b=t9(0);for(;;){if(b){var
c=b[2],d=b[1];try{c4(d)}catch(f){}var
b=c;continue}return 0}}c5(gO,dR);function
dS(a){return fa(a)}function
gP(a,b){return t_(a,b)}function
dT(a){return c4(a)}function
dU(a,b){var
d=b.length-1-1|0,e=0;if(!(d<0)){var
c=e;for(;;){j(a,b[c+1]);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return 0}function
aX(a,b){var
d=b.length-1;if(0===d)return[0];var
e=v(d,j(a,b[0+1])),f=d-1|0,g=1;if(!(f<1)){var
c=g;for(;;){e[c+1]=j(a,b[c+1]);var
h=c+1|0;if(f!==c){var
c=h;continue}break}}return e}function
cl(a,b){var
d=b.length-1-1|0,e=0;if(!(d<0)){var
c=e;for(;;){i(a,c,b[c+1]);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return 0}function
aY(a){var
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
cm(a,b){if(b){var
c=b[2],d=j(a,b[1]);return[0,d,cm(a,c)]}return 0}function
co(a,b,c){if(b){var
d=b[1];return i(a,d,co(a,b[2],c))}return c}function
dY(a,b,c){var
e=b,d=c;for(;;){if(e){if(d){var
f=d[2],g=e[2];i(a,e[1],d[1]);var
e=g,d=f;continue}}else
if(!d)return 0;return G(gU)}}function
cp(a,b){var
c=b;for(;;){if(c){var
e=c[2],d=0===aP(c[1],a)?1:0;if(d)return d;var
c=e;continue}return 0}}function
cq(a){if(0<=a)if(!(r<a))return a;return G(gV)}function
dZ(a){var
b=65<=a?90<a?0:1:0;if(!b){var
c=192<=a?214<a?0:1:0;if(!c){var
d=216<=a?222<a?1:0:1;if(d)return a}}return a+32|0}function
an(a,b){var
c=B(a);tC(c,0,a,b);return c}function
u(a,b,c){if(0<=b)if(0<=c)if(!((a.getLen()-c|0)<b)){var
d=B(c);bg(a,b,d,0,c);return d}return G(g2)}function
bz(a,b,c,d,e){if(0<=e)if(0<=b)if(!((a.getLen()-e|0)<b))if(0<=d)if(!((c.getLen()-e|0)<d))return bg(a,b,c,d,e);return G(g3)}function
d0(a){var
c=a.getLen();if(0===c)var
f=a;else{var
d=B(c),e=c-1|0,g=0;if(!(e<0)){var
b=g;for(;;){d.safeSet(b,dZ(a.safeGet(b)));var
h=b+1|0;if(e!==b){var
b=h;continue}break}}var
f=d}return f}var
cs=ur(0)[1],aE=uo(0),ct=(1<<(aE-10|0))-1|0,aZ=x(aE/8|0,ct)-1|0,g6=uq(0)[2],g7=b4,g8=aT;function
cu(k){function
h(a){return a?a[5]:0}function
e(a,b,c,d){var
e=h(a),f=h(d),g=f<=e?e+1|0:f+1|0;return[0,a,b,c,d,g]}function
q(a,b){return[0,0,a,b,0,1]}function
f(a,b,c,d){var
i=a?a[5]:0,j=d?d[5]:0;if((j+2|0)<i){if(a){var
f=a[4],m=a[3],n=a[2],k=a[1],q=h(f);if(q<=h(k))return e(k,n,m,e(f,b,c,d));if(f){var
r=f[3],s=f[2],t=f[1],u=e(f[4],b,c,d);return e(e(k,n,m,t),s,r,u)}return G(g9)}return G(g_)}if((i+2|0)<j){if(d){var
l=d[4],o=d[3],p=d[2],g=d[1],v=h(g);if(v<=h(l))return e(e(a,b,c,g),p,o,l);if(g){var
w=g[3],x=g[2],y=g[1],z=e(g[4],p,o,l);return e(e(a,b,c,y),x,w,z)}return G(g$)}return G(ha)}var
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
c=a[4],d=a[3],e=a[2];return f(s(b),e,d,c)}return a[4]}return G(hb)}function
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
j=l(a,f),p=j[2],q=j[1];return[0,q,p,g(j[3],e,d,c)]}return hc}function
m(a,b,c){if(b){var
d=b[2],i=b[5],j=b[4],k=b[3],n=b[1];if(h(c)<=i){var
e=l(d,c),o=e[2],q=e[1],r=m(a,j,e[3]),s=p(a,d,[0,k],o);return E(m(a,n,q),d,s,r)}}else
if(!c)return 0;if(c){var
f=c[2],t=c[4],u=c[3],v=c[1],g=l(f,b),w=g[2],x=g[1],y=m(a,g[3],t),z=p(a,f,w,[0,u]);return E(m(a,x,v),f,z,y)}throw[0,H,hd]}function
w(a,b){if(b){var
c=b[3],d=b[2],h=b[4],e=w(a,b[1]),j=i(a,d,c),f=w(a,h);return j?g(e,d,c,f):o(e,f)}return 0}function
x(a,b){if(b){var
c=b[3],d=b[2],m=b[4],e=x(a,b[1]),f=e[2],h=e[1],n=i(a,d,c),j=x(a,m),k=j[2],l=j[1];if(n){var
p=o(f,k);return[0,g(h,d,c,l),p]}var
q=g(f,d,c,k);return[0,o(h,l),q]}return he}function
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
hg=[0,hf];function
hh(a){throw[0,hg]}function
a0(a){var
b=1<=a?a:1,c=aZ<b?aZ:b,d=B(c);return[0,d,0,c,d]}function
a1(a){return u(a[1],0,a[2])}function
d3(a,b){var
c=[0,a[3]];for(;;){if(c[1]<(a[2]+b|0)){c[1]=2*c[1]|0;continue}if(aZ<c[1])if((a[2]+b|0)<=aZ)c[1]=aZ;else
T(hi);var
d=B(c[1]);bz(a[1],0,d,0,a[2]);a[1]=d;a[3]=c[1];return 0}}function
I(a,b){var
c=a[2];if(a[3]<=c)d3(a,1);a[1].safeSet(c,b);a[2]=c+1|0;return 0}function
bB(a,b){var
c=b.getLen(),d=a[2]+c|0;if(a[3]<d)d3(a,c);bz(b,0,a[1],a[2],c);a[2]=d;return 0}function
cv(a){return 0<=a?a:T(h(hj,k(a)))}function
d4(a,b){return cv(a+b|0)}var
hk=1;function
d5(a){return d4(hk,a)}function
d6(a){return u(a,0,a.getLen())}function
d7(a,b,c){var
d=h(hm,h(a,hl)),e=h(hn,h(k(b),d));return G(h(ho,h(an(1,c),e)))}function
a2(a,b,c){return d7(d6(a),b,c)}function
bC(a){return G(h(hq,h(d6(a),hp)))}function
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
i=h(b+1|0),f=a0((c-i|0)+10|0);I(f,37);var
a=i,g=dW(d);for(;;){if(a<=c){var
j=e.safeGet(a);if(42===j){if(g){var
l=g[2];bB(f,k(g[1]));var
a=h(a+1|0),g=l;continue}throw[0,H,hr]}I(f,j);var
a=a+1|0;continue}return a1(f)}}function
d8(a,b,c,d,e){var
f=ax(b,c,d,e);if(78!==a)if(bo!==a)return f;f.safeSet(f.getLen()-1|0,dv);return f}function
d9(a){return function(c,b){var
m=c.getLen();function
n(a,b){var
o=40===a?41:db;function
k(a){var
d=a;for(;;){if(m<=d)return bC(c);if(37===c.safeGet(d)){var
e=d+1|0;if(m<=e)var
f=bC(c);else{var
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
f=g===o?e+1|0:a2(c,b,g);break;case
2:break;default:var
f=k(n(g,e+1|0)+1|0)}}return f}var
d=d+1|0;continue}}return k(b)}return n(a,b)}}function
d_(j,b,c){var
m=j.getLen()-1|0;function
s(a){var
l=a;a:for(;;){if(l<m){if(37===j.safeGet(l)){var
e=0,h=l+1|0;for(;;){if(m<h)var
w=bC(j);else{var
n=j.safeGet(h);if(58<=n){if(95===n){var
e=1,h=h+1|0;continue}}else
if(32<=n)switch(n+fD|0){case
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
f=bC(j);else{var
k=j.safeGet(d);if(fY<=k)var
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
gh:case
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
br:var
f=p(b,e,d,br),g=1;break;case
97:case
cb:case
da:var
f=p(b,e,d,k),g=1;break;case
76:case
fN:case
bo:var
t=d+1|0;if(m<t){var
f=p(b,e,d,aC),g=1}else{var
q=j.safeGet(t)+gl|0;if(q<0||32<q)var
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
f=a2(j,d,k)}var
w=f;break}}var
l=w;continue a}}var
l=l+1|0;continue}return l}}s(0);return 0}function
d$(a){var
d=[0,0,0,0];function
b(a,b,c){var
f=41!==c?1:0,g=f?db!==c?1:0:f;if(g){var
e=97===c?2:1;if(cb===c)d[3]=d[3]+1|0;if(a)d[2]=d[2]+e|0;else
d[1]=d[1]+e|0}return b+1|0}d_(a,b,function(a,b){return a+1|0});return d[1]}function
ea(a,b,c){var
h=a.safeGet(c);if((h+aU|0)<0||9<(h+aU|0))return i(b,0,c);var
e=h+aU|0,d=c+1|0;for(;;){var
f=a.safeGet(d);if(48<=f){if(!(58<=f)){var
e=(10*e|0)+(f+aU|0)|0,d=d+1|0;continue}var
g=0}else
if(36===f)if(0===e){var
j=T(ht),g=1}else{var
j=i(b,[0,cv(e-1|0)],d+1|0),g=1}else
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
aa=m.safeGet(a)+fD|0;if(!(aa<0||25<aa))switch(aa){case
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
a8=o(g,f),a9=bT(d8(q,m,p,a,c),a8),l=r(P(g,f),a9,a+1|0),k=1;break;case
69:case
71:case
gh:case
di:case
dA:var
aY=o(g,f),aZ=c2(ax(m,p,a,c),aY),l=r(P(g,f),aZ,a+1|0),k=1;break;case
76:case
fN:case
bo:var
ad=m.safeGet(a+1|0)+gl|0;if(ad<0||32<ad)var
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
a7=o(g,f),aA=bT(ax(m,p,U,c),a7),ai=1;break;default:var
a6=o(g,f),aA=bT(ax(m,p,U,c),a6),ai=1}if(ai){var
az=aA,ah=1}}if(!ah){var
a5=o(g,f),az=tM(ax(m,p,U,c),a5)}var
l=r(P(g,f),az,U+1|0),k=1,ag=0;break;default:var
ag=1}if(ag){var
a3=o(g,f),a4=bT(d8(bo,m,p,a,c),a3),l=r(P(g,f),a4,a+1|0),k=1}break;case
37:case
64:var
l=r(f,an(1,q),a+1|0),k=1;break;case
83:case
br:var
y=o(g,f);if(br===q)var
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
10:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],bo);var
K=1;break;case
13:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],cb);var
K=1;break;default:var
V=1,K=0}if(K)var
V=0}else
var
V=(A-1|0)<0||56<(A-1|0)?(n.safeSet(b[1],92),b[1]++,n.safeSet(b[1],w),0):1;if(V)if(c6(w))n.safeSet(b[1],w);else{n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],48+(w/aS|0)|0);b[1]++;n.safeSet(b[1],48+((w/10|0)%10|0)|0);b[1]++;n.safeSet(b[1],48+(w%10|0)|0)}b[1]++;var
aN=L+1|0;if(ao!==L){var
L=aN;continue}break}}var
aD=n}var
z=h(hE,h(aD,hD))}if(a===(p+1|0))var
aB=z;else{var
J=ax(m,p,a,c);try{var
W=0,t=1;for(;;){if(J.getLen()<=t)var
ap=[0,0,W];else{var
X=J.safeGet(t);if(49<=X)if(58<=X)var
aj=0;else{var
ap=[0,tW(u(J,t,(J.getLen()-t|0)-1|0)),W],aj=1}else{if(45===X){var
W=1,t=t+1|0;continue}var
aj=0}if(!aj){var
t=t+1|0;continue}}var
Z=ap;break}}catch(f){if(f[1]!==aW)throw f;var
Z=d7(J,0,br)}var
N=Z[1],C=z.getLen(),aQ=Z[2],O=0,aR=32;if(N===C)if(0===O){var
_=z,ak=1}else
var
ak=0;else
var
ak=0;if(!ak)if(N<=C)var
_=u(z,O,C);else{var
Y=an(N,aR);if(aQ)bz(z,O,Y,0,C);else
bz(z,O,Y,N-C|0,C);var
_=Y}var
aB=_}var
l=r(P(g,f),aB,a+1|0),k=1;break;case
67:case
99:var
s=o(g,f);if(99===q)var
aw=an(1,s);else{if(39===s)var
v=gW;else
if(92===s)var
v=gX;else{if(14<=s)var
F=0;else
switch(s){case
8:var
v=gY,F=1;break;case
9:var
v=gZ,F=1;break;case
10:var
v=g0,F=1;break;case
13:var
v=g1,F=1;break;default:var
F=0}if(!F)if(c6(s)){var
al=B(1);al.safeSet(0,s);var
v=al}else{var
G=B(4);G.safeSet(0,92);G.safeSet(1,48+(s/aS|0)|0);G.safeSet(2,48+((s/10|0)%10|0)|0);G.safeSet(3,48+(s%10|0)|0);var
v=G}}var
aw=h(hB,h(v,hA))}var
l=r(P(g,f),aw,a+1|0),k=1;break;case
66:case
98:var
aV=a+1|0,aX=o(g,f)?gH:gI,l=r(P(g,f),aX,aV),k=1;break;case
40:case
dy:var
T=o(g,f),au=i(d9(q),m,a+1|0);if(dy===q){var
Q=a0(T.getLen()),aq=function(a,b){I(Q,b);return a+1|0};d_(T,function(a,b,c){if(a)bB(Q,hs);else
I(Q,37);return aq(b,c)},aq);var
aT=a1(Q),l=r(P(g,f),aT,au),k=1}else{var
av=P(g,f),bc=d4(d$(T),av),l=aJ(function(a){return E(bc,au)},av,T,aK),k=1}break;case
33:j(e,D);var
l=E(f,a+1|0),k=1;break;case
41:var
l=r(f,hy,a+1|0),k=1;break;case
44:var
l=r(f,hz,a+1|0),k=1;break;case
70:var
ab=o(g,f);if(0===c)var
ay=hC;else{var
$=ax(m,p,a,c);if(70===q)$.safeSet($.getLen()-1|0,dA);var
ay=$}var
as=tz(ab);if(3===as)var
ac=ab<0?hv:hw;else
if(4<=as)var
ac=hx;else{var
S=c2(ay,ab),R=0,aU=S.getLen();for(;;){if(aU<=R)var
ar=h(S,hu);else{var
H=S.safeGet(R)-46|0,be=H<0||23<H?55===H?1:0:(H-1|0)<0||21<(H-1|0)?1:0;if(!be){var
R=R+1|0;continue}var
ar=S}var
ac=ar;break}}var
l=r(P(g,f),ac,a+1|0),k=1;break;case
91:var
l=a2(m,a,q),k=1;break;case
97:var
aE=o(g,f),aF=d5(eb(g,f)),aG=o(0,aF),a_=a+1|0,a$=P(g,aF);if(aI)af(i(aE,0,aG));else
i(aE,D,aG);var
l=E(a$,a_),k=1;break;case
cb:var
l=a2(m,a,q),k=1;break;case
da:var
aH=o(g,f),ba=a+1|0,bb=P(g,f);if(aI)af(j(aH,0));else
j(aH,D);var
l=E(bb,ba),k=1;break;default:var
k=0}if(!k)var
l=a2(m,a,q);return l}},f=p+1|0,g=0;return ea(m,function(a,b){return at(a,l,g,b)},f)}i(c,D,d);var
p=p+1|0;continue}}function
r(a,b,c){af(b);return E(a,c)}return E(b,0)}var
o=cv(0);function
k(a,b){return aJ(f,o,a,b)}var
l=d$(g);if(l<0||6<l){var
n=function(f,b){if(l<=f){var
h=v(l,0),i=function(a,b){return m(h,(l-a|0)-1|0,b)},c=0,a=b;for(;;){if(a){var
d=a[2],e=a[1];if(d){i(c,e);var
c=c+1|0,a=d;continue}i(c,e)}return k(g,h)}}return function(a){return n(f+1|0,[0,a,b])}},a=n(0,0)}else
switch(l){case
1:var
a=function(a){var
b=v(1,0);m(b,0,a);return k(g,b)};break;case
2:var
a=function(a,b){var
c=v(2,0);m(c,0,a);m(c,1,b);return k(g,c)};break;case
3:var
a=function(a,b,c){var
d=v(3,0);m(d,0,a);m(d,1,b);m(d,2,c);return k(g,d)};break;case
4:var
a=function(a,b,c,d){var
e=v(4,0);m(e,0,a);m(e,1,b);m(e,2,c);m(e,3,d);return k(g,e)};break;case
5:var
a=function(a,b,c,d,e){var
f=v(5,0);m(f,0,a);m(f,1,b);m(f,2,c);m(f,3,d);m(f,4,e);return k(g,f)};break;case
6:var
a=function(a,b,c,d,e,f){var
h=v(6,0);m(h,0,a);m(h,1,b);m(h,2,c);m(h,3,d);m(h,4,e);m(h,5,f);return k(g,h)};break;default:var
a=k(g,[0])}return a}function
ed(a){function
b(a){return 0}return ec(0,function(a){return dO},gP,dP,dT,b,a)}function
hF(a){return a0(2*a.getLen()|0)}function
ee(c){function
b(a){var
b=a1(a);a[2]=0;return j(c,b)}function
d(a){return 0}var
e=1;return function(a){return ec(e,hF,I,bB,d,b,a)}}function
ef(a){return j(ee(function(a){return a}),a)}var
eg=[0,0];function
eh(a){eg[1]=[0,a,eg[1]];return 0}function
ei(a,b){var
j=0===b.length-1?[0,0]:b,f=j.length-1,p=0,q=54;if(!(54<0)){var
d=p;for(;;){m(a[1],d,d);var
w=d+1|0;if(q!==d){var
d=w;continue}break}}var
g=[0,hG],l=0,r=55,t=tI(55,f)?r:f,n=54+t|0;if(!(n<l)){var
c=l;for(;;){var
o=c%55|0,u=g[1],i=h(u,k(s(j,aQ(c,f))));g[1]=t1(i,0,i.getLen());var
e=g[1];m(a[1],o,(s(a[1],o)^(((e.safeGet(0)+(e.safeGet(1)<<8)|0)+(e.safeGet(2)<<16)|0)+(e.safeGet(3)<<24)|0))&bn);var
v=c+1|0;if(n!==c){var
c=v;continue}break}}a[2]=0;return 0}32===aE;var
hI=[0,hH.slice(),0];try{var
tp=bU(to),cw=tp}catch(f){if(f[1]!==t)throw f;try{var
tn=bU(tm),ej=tn}catch(f){if(f[1]!==t)throw f;var
ej=hJ}var
cw=ej}var
d1=cw.getLen(),hK=82,d2=0;if(0<=0)if(d1<d2)var
bV=0;else
try{var
bA=d2;for(;;){if(d1<=bA)throw[0,t];if(cw.safeGet(bA)!==hK){var
bA=bA+1|0;continue}var
g5=1,cr=g5,bV=1;break}}catch(f){if(f[1]!==t)throw f;var
cr=0,bV=1}else
var
bV=0;if(!bV)var
cr=G(g4);var
ao=[fU,function(a){var
b=[0,v(55,0),0];ei(b,fb(0));return b}];function
ek(a,b){var
l=a?a[1]:cr,d=16;for(;;){if(!(b<=d))if(!(ct<(d*2|0))){var
d=d*2|0;continue}if(l){var
h=ue(ao);if(aT===h)var
c=ao[1];else
if(fU===h){var
k=ao[0+1];ao[0+1]=hh;try{var
e=j(k,0);ao[0+1]=e;ud(ao,g8)}catch(f){ao[0+1]=function(a){throw f};throw f}var
c=e}else
var
c=ao;c[2]=(c[2]+1|0)%55|0;var
f=s(c[1],c[2]),g=(s(c[1],(c[2]+24|0)%55|0)+(f^f>>>25&31)|0)&bn;m(c[1],c[2],g);var
i=g}else
var
i=0;return[0,0,v(d,0),i,d]}}function
cx(a,b){return 3<=a.length-1?tJ(10,aS,a[3],b)&(a[2].length-1-1|0):aQ(tK(10,aS,b),a[2].length-1)}function
bD(a,b){var
i=cx(a,b),d=s(a[2],i);if(d){var
e=d[3],j=d[2];if(0===aP(b,d[1]))return j;if(e){var
f=e[3],k=e[2];if(0===aP(b,e[1]))return k;if(f){var
l=f[3],m=f[2];if(0===aP(b,f[1]))return m;var
c=l;for(;;){if(c){var
g=c[3],h=c[2];if(0===aP(b,c[1]))return h;var
c=g;continue}throw[0,t]}}throw[0,t]}throw[0,t]}throw[0,t]}function
l(a,b){return c5(a,b[0+1])}var
cy=[0,0];c5(hL,cy);var
hM=2;function
hN(a){var
b=[0,0],d=a.getLen()-1|0,e=0;if(!(d<0)){var
c=e;for(;;){b[1]=(223*b[1]|0)+a.safeGet(c)|0;var
g=c+1|0;if(d!==c){var
c=g;continue}break}}b[1]=b[1]&((1<<31)-1|0);var
f=bn<b[1]?b[1]-(1<<31)|0:b[1];return f}var
ah=cu([0,function(a,b){return fc(a,b)}]),ay=cu([0,function(a,b){return fc(a,b)}]),ap=cu([0,function(a,b){return gy(a,b)}]),el=fd(0,0),hO=[0,0];function
em(a){return 2<a?em((a+1|0)/2|0)*2|0:a}function
en(a){hO[1]++;var
c=a.length-1,d=v((c*2|0)+2|0,el);m(d,0,c);m(d,1,(x(em(c),aE)/8|0)-1|0);var
e=c-1|0,f=0;if(!(e<0)){var
b=f;for(;;){m(d,(b*2|0)+3|0,s(a,b));var
g=b+1|0;if(e!==b){var
b=g;continue}break}}return[0,hM,d,ay[1],ap[1],0,0,ah[1],0]}function
cz(a,b){var
c=a[2].length-1,g=c<b?1:0;if(g){var
d=v(b,el),h=a[2],e=0,f=0,j=0<=c?0<=f?(h.length-1-c|0)<f?0:0<=e?(d.length-1-c|0)<e?0:(ts(h,f,d,e,c),1):0:0:0;if(!j)G(gQ);a[2]=d;var
i=0}else
var
i=g;return i}var
eo=[0,0],hP=[0,0];function
cA(a){var
b=a[2].length-1;cz(a,b+1|0);return b}function
a3(a,b){try{var
d=i(ay[22],b,a[3])}catch(f){if(f[1]===t){var
c=cA(a);a[3]=p(ay[4],b,c,a[3]);a[4]=p(ap[4],c,1,a[4]);return c}throw f}return d}function
cC(a){return a===0?0:aY(a)}function
eu(a,b){try{var
d=i(ah[22],b,a[7])}catch(f){if(f[1]===t){var
c=a[1];a[1]=c+1|0;if(y(b,h5))a[7]=p(ah[4],b,c,a[7]);return c}throw f}return d}function
cE(a){return tB(a,0)?[0]:a}function
ew(a,b){if(a)return a;var
c=fd(g7,b[1]);c[0+1]=b[2];var
d=cy[1];c[1+1]=d;cy[1]=d+1|0;return c}function
bE(a){var
b=cA(a);if(0===(b%2|0))var
d=0;else
if((2+at(s(a[2],1)*16|0,aE)|0)<b)var
d=0;else{var
c=cA(a),d=1}if(!d)var
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
ag=e(0),h=e(0);bE(a);var
f=function(ag,h){return function(a){return j(af(h,ag,0),h)}}(ag,h);break;case
21:var
ah=e(0),ai=e(0);bE(a);var
f=function(ah,ai){return function(a){var
b=a[ai+1];return j(af(b,ah,0),b)}}(ah,ai);break;case
22:var
aj=e(0),ak=e(0),al=e(0);bE(a);var
f=function(aj,ak,al){return function(a){var
b=a[ak+1][al+1];return j(af(b,aj,0),b)}}(aj,ak,al);break;case
23:var
am=e(0),an=e(0);bE(a);var
f=function(am,an){return function(a){var
b=j(a[1][an+1],a);return j(af(b,am,0),b)}}(am,an);break;default:var
o=e(0),f=function(o){return function(a){return o}}(o)}else
var
f=l;hP[1]++;if(i(ap[22],k,a[4])){cz(a,k+1|0);m(a[2],k,f)}else
a[6]=[0,[0,k,f],a[6]];g[1]++;continue}return 0}}function
cF(a,b,c){if(bW(c,id))return b;var
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
cG(a,b,c){if(bW(c,ie))return b;var
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
cI(a,b){return 47===a.safeGet(b)?1:0}function
ey(a){var
b=a.getLen()<1?1:0,c=b||(47!==a.safeGet(0)?1:0);return c}function
ii(a){var
c=ey(a);if(c){var
e=a.getLen()<2?1:0,d=e||y(u(a,0,2),ik);if(d){var
f=a.getLen()<3?1:0,b=f||y(u(a,0,3),ij)}else
var
b=d}else
var
b=c;return b}function
il(a,b){var
c=b.getLen()<=a.getLen()?1:0,d=c?bW(u(a,a.getLen()-b.getLen()|0,b.getLen()),b):c;return d}try{var
tl=bU(tk),cJ=tl}catch(f){if(f[1]!==t)throw f;var
cJ=im}function
ez(a){var
d=a.getLen(),b=a0(d+20|0);I(b,39);var
e=d-1|0,f=0;if(!(e<0)){var
c=f;for(;;){if(39===a.safeGet(c))bB(b,io);else
I(b,a.safeGet(c));var
g=c+1|0;if(e!==c){var
c=g;continue}break}}I(b,39);return a1(b)}function
ip(a){return cF(cI,cH,a)}function
iq(a){return cG(cI,cH,a)}function
aG(a,b){var
c=a.safeGet(b),d=47===c?1:0;if(d)var
e=d;else{var
f=92===c?1:0,e=f||(58===c?1:0)}return e}function
cL(a){var
e=a.getLen()<1?1:0,c=e||(47!==a.safeGet(0)?1:0);if(c){var
f=a.getLen()<1?1:0,d=f||(92!==a.safeGet(0)?1:0);if(d){var
g=a.getLen()<2?1:0,b=g||(58!==a.safeGet(1)?1:0)}else
var
b=d}else
var
b=c;return b}function
eA(a){var
c=cL(a);if(c){var
g=a.getLen()<2?1:0,d=g||y(u(a,0,2),iw);if(d){var
h=a.getLen()<2?1:0,e=h||y(u(a,0,2),iv);if(e){var
i=a.getLen()<3?1:0,f=i||y(u(a,0,3),iu);if(f){var
j=a.getLen()<3?1:0,b=j||y(u(a,0,3),it)}else
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
e=u(a,a.getLen()-b.getLen()|0,b.getLen()),f=d0(b),d=bW(d0(e),f)}else
var
d=c;return d}try{var
tj=bU(ti),eC=tj}catch(f){if(f[1]!==t)throw f;var
eC=ix}function
iy(h){var
i=h.getLen(),e=a0(i+20|0);I(e,34);function
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
a(b){return Z(g(0,b))}function
b(b,c){return Z(d(0,b,c))}function
k(b){return Z(j(0,b))}a(0);return a1(e)}function
eD(a){var
c=2<=a.getLen()?1:0;if(c){var
b=a.safeGet(0),g=91<=b?(b+fJ|0)<0||25<(b+fJ|0)?0:1:65<=b?1:0,d=g?1:0,e=d?58===a.safeGet(1)?1:0:d}else
var
e=c;if(e){var
f=u(a,2,a.getLen()-2|0);return[0,u(a,0,2),f]}return[0,iz,a]}function
iA(a){var
b=eD(a),c=b[1];return h(c,cG(aG,cK,b[2]))}function
iB(a){return cF(aG,cK,eD(a)[2])}function
iE(a){return cF(aG,cM,a)}function
iF(a){return cG(aG,cM,a)}if(y(cs,iG))if(y(cs,iH)){if(y(cs,iI))throw[0,H,iJ];var
bF=[0,cK,ir,is,aG,cL,eA,eB,eC,iy,iB,iA]}else
var
bF=[0,cH,ig,ih,cI,ey,ii,il,cJ,ez,ip,iq];else
var
bF=[0,cM,iC,iD,aG,cL,eA,eB,cJ,ez,iE,iF];var
eE=[0,iM],iK=bF[11],iL=bF[3];l(iP,[0,eE,0,iO,iN]);eh(function(a){if(a[1]===eE){var
c=a[2],d=a[4],e=a[3];if(typeof
c===n)switch(c){case
1:var
b=iS;break;case
2:var
b=iT;break;case
3:var
b=iU;break;case
4:var
b=iV;break;case
5:var
b=iW;break;case
6:var
b=iX;break;case
7:var
b=iY;break;case
8:var
b=iZ;break;case
9:var
b=i0;break;case
10:var
b=i1;break;case
11:var
b=i2;break;case
12:var
b=i3;break;case
13:var
b=i4;break;case
14:var
b=i5;break;case
15:var
b=i6;break;case
16:var
b=i7;break;case
17:var
b=i8;break;case
18:var
b=i9;break;case
19:var
b=i_;break;case
20:var
b=i$;break;case
21:var
b=ja;break;case
22:var
b=jb;break;case
23:var
b=jc;break;case
24:var
b=jd;break;case
25:var
b=je;break;case
26:var
b=jf;break;case
27:var
b=jg;break;case
28:var
b=jh;break;case
29:var
b=ji;break;case
30:var
b=jj;break;case
31:var
b=jk;break;case
32:var
b=jl;break;case
33:var
b=jm;break;case
34:var
b=jn;break;case
35:var
b=jo;break;case
36:var
b=jp;break;case
37:var
b=jq;break;case
38:var
b=jr;break;case
39:var
b=js;break;case
40:var
b=jt;break;case
41:var
b=ju;break;case
42:var
b=jv;break;case
43:var
b=jw;break;case
44:var
b=jx;break;case
45:var
b=jy;break;case
46:var
b=jz;break;case
47:var
b=jA;break;case
48:var
b=jB;break;case
49:var
b=jC;break;case
50:var
b=jD;break;case
51:var
b=jE;break;case
52:var
b=jF;break;case
53:var
b=jG;break;case
54:var
b=jH;break;case
55:var
b=jI;break;case
56:var
b=jJ;break;case
57:var
b=jK;break;case
58:var
b=jL;break;case
59:var
b=jM;break;case
60:var
b=jN;break;case
61:var
b=jO;break;case
62:var
b=jP;break;case
63:var
b=jQ;break;case
64:var
b=jR;break;case
65:var
b=jS;break;case
66:var
b=jT;break;case
67:var
b=jU;break;default:var
b=iQ}else{var
f=c[1],b=j(ef(jV),f)}return[0,p(ef(iR),b,e,d)]}return 0});bX(jW);bX(jX);try{bX(th)}catch(f){if(f[1]!==aW)throw f}try{bX(tg)}catch(f){if(f[1]!==aW)throw f}ek(0,7);function
eF(a){return vm(a)}an(32,r);var
jY=6,jZ=0,j4=B(f7),j5=0,j6=r;if(!(r<0)){var
bf=j5;for(;;){j4.safeSet(bf,dZ(cq(bf)));var
tf=bf+1|0;if(j6!==bf){var
bf=tf;continue}break}}var
cN=an(32,0);cN.safeSet(10>>>3,cq(cN.safeGet(10>>>3)|1<<(10&7)));var
j0=B(32),j1=0,j2=31;if(!(31<0)){var
a7=j1;for(;;){j0.safeSet(a7,cq(cN.safeGet(a7)^r));var
j3=a7+1|0;if(j2!==a7){var
a7=j3;continue}break}}var
aH=[0,0],aI=[0,0],eG=[0,0];function
J(a){return aH[1]}function
eH(a){return aI[1]}function
Q(a,b,c){return 0===a[2][0]?b?uO(a[1],a,b[1]):uP(a[1],a):b?fe(a[1],b[1]):fe(a[1],0)}var
eI=[3,jY],cO=[0,0];function
aJ(e,b,c){cO[1]++;switch(e[0]){case
7:case
8:throw[0,H,j7];case
6:var
g=e[1],m=cO[1],n=ff(0),o=v(eH(0)+1|0,n),p=fg(0),q=v(J(0)+1|0,p),f=[0,-1,[1,[0,uC(g,c),g]],q,o,c,0,e,0,0,m,0];break;default:var
h=e[1],i=cO[1],j=ff(0),k=v(eH(0)+1|0,j),l=fg(0),f=[0,-1,[0,tw(h,jZ,[0,c])],v(J(0)+1|0,l),k,c,0,e,0,0,i,0]}if(b){var
d=b[1],a=function(a){{if(0===d[2][0])return 6===e[0]?gG(f,d[1][8],d[1]):gF(f,d[1][8],d[1]);{var
b=d[1],c=J(0);return fh(f,d[1][8]-c|0,b)}}};try{a(0)}catch(f){bh(0);a(0)}f[6]=[0,d]}return f}function
Y(a){return a[5]}function
a8(a){return a[6]}function
bG(a){return a[8]}function
bH(a){return a[7]}function
aa(a){return a[2]}function
bI(a,b,c){a[1]=b;a[6]=c;return 0}function
cP(a,b,c){return dB<=b?s(a[3],c):s(a[4],c)}function
cQ(a,b){var
e=b[3].length-1-2|0,g=0;if(!(e<0)){var
d=g;for(;;){m(b[3],d,s(a[3],d));var
j=d+1|0;if(e!==d){var
d=j;continue}break}}var
f=b[4].length-1-2|0,h=0;if(!(f<0)){var
c=h;for(;;){m(b[4],c,s(a[4],c));var
i=c+1|0;if(f!==c){var
c=i;continue}break}}return 0}function
bJ(a,b){b[8]=a[8];return 0}var
az=[0,kb];l(kk,[0,[0,j8]]);l(kl,[0,[0,j9]]);l(km,[0,[0,j_]]);l(kn,[0,[0,j$]]);l(ko,[0,[0,ka]]);l(kp,[0,az]);l(kq,[0,[0,kc]]);l(kr,[0,[0,kd]]);l(ks,[0,[0,ke]]);l(kt,[0,[0,kg]]);l(ku,[0,[0,kh]]);l(kv,[0,[0,ki]]);l(kw,[0,[0,kj]]);l(kx,[0,[0,kf]]);var
cR=[0,kF];l(kV,[0,[0,ky]]);l(kW,[0,[0,kz]]);l(kX,[0,[0,kA]]);l(kY,[0,[0,kB]]);l(kZ,[0,[0,kC]]);l(k0,[0,[0,kD]]);l(k1,[0,[0,kE]]);l(k2,[0,cR]);l(k3,[0,[0,kG]]);l(k4,[0,[0,kH]]);l(k5,[0,[0,kI]]);l(k6,[0,[0,kJ]]);l(k7,[0,[0,kK]]);l(k8,[0,[0,kL]]);l(k9,[0,[0,kM]]);l(k_,[0,[0,kN]]);l(k$,[0,[0,kO]]);l(la,[0,[0,kP]]);l(lb,[0,[0,kQ]]);l(lc,[0,[0,kR]]);l(ld,[0,[0,kS]]);l(le,[0,[0,kT]]);l(lf,[0,[0,kU]]);var
bK=1,eJ=0;function
a9(a,b,c){var
d=a[2];if(0===d[0])var
f=ty(d[1],b,c);else{var
e=d[1],f=p(e[2][4],e[1],b,c)}return f}function
a_(a,b){var
c=a[2];if(0===c[0])var
e=tx(c[1],b);else{var
d=c[1],e=i(d[2][3],d[1],b)}return e}function
eK(a,b){Q(a,0,0);eP(b,0,0);return Q(a,0,0)}function
aK(a,b,c){var
f=a,d=b;for(;;){if(eJ)return a9(f,d,c);var
m=d<0?1:0,o=m||(Y(f)<=d?1:0);if(o)throw[0,bw,lg];if(bK){var
i=a8(f);if(typeof
i!==n)eK(i[1],f)}var
j=bG(f);if(j){var
e=j[1];if(1===e[1]){var
k=e[4],g=e[3],l=e[2];return 0===k?a9(e[5],l+d|0,c):a9(e[5],(l+x(at(d,g),k+g|0)|0)+aQ(d,g)|0,c)}var
h=e[3],f=e[5],d=(e[2]+x(at(d,h),e[4]+h|0)|0)+aQ(d,h)|0;continue}return a9(f,d,c)}}function
aL(a,b){var
e=a,c=b;for(;;){if(eJ)return a_(e,c);var
l=c<0?1:0,m=l||(Y(e)<=c?1:0);if(m)throw[0,bw,lh];if(bK){var
h=a8(e);if(typeof
h!==n)eK(h[1],e)}var
i=bG(e);if(i){var
d=i[1];if(1===d[1]){var
j=d[4],f=d[3],k=d[2];return 0===j?a_(d[5],k+c|0):a_(d[5],(k+x(at(c,f),j+f|0)|0)+aQ(c,f)|0)}var
g=d[3],e=d[5],c=(d[2]+x(at(c,g),d[4]+g|0)|0)+aQ(c,g)|0;continue}return a_(e,c)}}function
eL(a){if(a[8]){var
b=aJ(a[7],0,a[5]);b[1]=a[1];b[6]=a[6];cQ(a,b);var
c=b}else
var
c=a;return c}function
eM(d,b,c){{if(0===c[2][0]){var
a=function(a){return 0===aa(d)[0]?uF(d,c[1][8],c[1],c[3],b):uH(d,c[1][8],c[1],b)};try{var
f=a(0)}catch(f){if(f[1]===az){try{Q(c,0,0);var
g=a(0)}catch(f){bh(0);return a(0)}return g}throw f}return f}var
e=function(a){{if(0===aa(d)[0]){var
e=c[1],f=J(0);return u7(d,c[1][8]-f|0,e,b)}var
g=c[1],h=J(0);return u9(d,c[1][8]-h|0,g,b)}};try{var
i=e(0)}catch(f){try{Q(c,0,0);var
h=e(0)}catch(f){bh(0);return e(0)}return h}return i}}function
eN(d,b,c){{if(0===c[2][0]){var
a=function(a){return 0===aa(d)[0]?uN(d,c[1][8],c[1],c,b):uI(d,c[1][8],c[1],b)};try{var
f=a(0)}catch(f){if(f[1]===az){try{Q(c,0,0);var
g=a(0)}catch(f){bh(0);return a(0)}return g}throw f}return f}var
e=function(a){{if(0===aa(d)[0]){var
e=c[2],f=c[1],g=J(0);return vb(d,c[1][8]-g|0,f,e,b)}var
h=c[2],i=c[1],j=J(0);return u_(d,c[1][8]-j|0,i,h,b)}};try{var
i=e(0)}catch(f){try{Q(c,0,0);var
h=e(0)}catch(f){bh(0);return e(0)}return h}return i}}function
a$(a,b,c,d,e,f,g,h){{if(0===d[2][0])return 0===aa(a)[0]?uW(a,b,d[1][8],d[1],d[3],c,e,f,g,h):uK(a,b,d[1][8],d[1],d[3],c,e,f,g,h);{if(0===aa(a)[0]){var
i=d[3],j=d[1],k=J(0);return vk(a,b,d[1][8]-k|0,j,i,c,e,f,g,h)}var
l=d[3],m=d[1],n=J(0);return u$(a,b,d[1][8]-n|0,m,l,c,e,f,g,h)}}}function
ba(a,b,c,d,e,f,g,h){{if(0===d[2][0])return 0===aa(a)[0]?uX(a,b,d[1][8],d[1],d[3],c,e,f,g,h):uL(a,b,d[1][8],d[1],d[3],c,e,f,g,h);{if(0===aa(a)[0]){var
i=d[3],j=d[1],k=J(0);return vl(a,b,d[1][8]-k|0,j,i,c,e,f,g,h)}var
l=d[3],m=d[1],n=J(0);return va(a,b,d[1][8]-n|0,m,l,c,e,f,g,h)}}}function
eO(a,b,c){var
p=b;for(;;){var
d=p?p[1]:0,q=a8(a);if(typeof
q===n){bI(a,c[1][8],[1,c]);try{cS(a,c)}catch(f){if(f[1]!==az)f[1]===cR;try{Q(c,[0,d],0);cS(a,c)}catch(f){if(f[1]!==az)if(f[1]!==cR)throw f;Q(c,0,0);tF(0);cS(a,c)}}var
y=bG(a);if(y){var
j=y[1];if(1===j[1]){var
k=j[5],r=j[4],f=j[3],l=j[2];if(0===f)a$(k,a,d,c,0,0,l,Y(a));else
if(z<f){var
h=0,m=Y(a);for(;;){if(f<m){a$(k,a,d,c,x(h,f+r|0),x(h,f),l,f);var
h=h+1|0,m=m-f|0;continue}if(0<m)a$(k,a,d,c,x(h,f+r|0),x(h,f),l,m);break}}else{var
e=0,i=0,g=Y(a);for(;;){if(z<g){var
u=aJ(bH(a),0,z);bJ(a,u);var
A=e+gm|0;if(!(A<e)){var
s=e;for(;;){aK(u,s,aL(a,e));var
H=s+1|0;if(A!==s){var
s=H;continue}break}}a$(k,u,d,c,x(i,z+r|0),i*z|0,l,z);var
e=e+z|0,i=i+1|0,g=g+fZ|0;continue}if(0<g){var
v=aJ(bH(a),0,g),B=(e+g|0)-1|0;if(!(B<e)){var
t=e;for(;;){aK(v,t,aL(a,e));var
I=t+1|0;if(B!==t){var
t=I;continue}break}}bJ(a,v);a$(k,v,d,c,x(i,z+r|0),i*z|0,l,g)}break}}}else{var
w=eL(a),C=Y(a)-1|0,J=0;if(!(C<0)){var
o=J;for(;;){a9(w,o,aL(a,o));var
K=o+1|0;if(C!==o){var
o=K;continue}break}}eM(w,d,c);cQ(w,a)}}else
eM(a,d,c);return bI(a,c[1][8],[0,c])}else{if(0===q[0]){var
D=q[1],E=c7(D,c);if(E){eP(a,[0,d],0);Q(D,0,0);var
p=[0,d];continue}return E}var
F=q[1],G=c7(F,c);if(G){Q(F,0,0);var
p=[0,d];continue}return G}}}function
cS(a,b){{if(0===b[2][0])return 0===aa(a)[0]?gF(a,b[1][8],b[1]):gG(a,b[1][8],b[1]);{if(0===aa(a)[0]){var
c=b[1],d=J(0);return fh(a,b[1][8]-d|0,c)}var
e=b[1],f=J(0);return u8(a,b[1][8]-f|0,e)}}}function
eP(a,b,c){var
v=b;for(;;){var
f=v?v[1]:0,q=a8(a);if(typeof
q===n)return 0;else{if(0===q[0]){var
d=q[1];bI(a,d[1][8],[1,d]);var
A=bG(a);if(A){var
k=A[1];if(1===k[1]){var
l=k[5],r=k[4],e=k[3],m=k[2];if(0===e)ba(l,a,f,d,0,0,m,Y(a));else
if(z<e){var
i=0,o=Y(a);for(;;){if(e<o){ba(l,a,f,d,x(i,e+r|0),x(i,e),m,e);var
i=i+1|0,o=o-e|0;continue}if(0<o)ba(l,a,f,d,x(i,e+r|0),x(i,e),m,o);break}}else{var
j=0,h=Y(a),g=0;for(;;){if(z<h){var
w=aJ(bH(a),0,z);bJ(a,w);var
B=g+gm|0;if(!(B<g)){var
s=g;for(;;){aK(w,s,aL(a,g));var
E=s+1|0;if(B!==s){var
s=E;continue}break}}ba(l,w,f,d,x(j,z+r|0),j*z|0,m,z);var
j=j+1|0,h=h+fZ|0;continue}if(0<h){var
y=aJ(bH(a),0,h),C=(g+h|0)-1|0;if(!(C<g)){var
t=g;for(;;){aK(y,t,aL(a,g));var
F=t+1|0;if(C!==t){var
t=F;continue}break}}bJ(a,y);ba(l,y,f,d,x(j,z+r|0),j*z|0,m,h)}break}}}else{var
u=eL(a);cQ(u,a);eN(u,f,d);var
D=Y(u)-1|0,G=0;if(!(D<0)){var
p=G;for(;;){aK(a,p,a_(u,p));var
H=p+1|0;if(D!==p){var
p=H;continue}break}}}}else
eN(a,f,d);return bI(a,d[1][8],0)}Q(q[1],0,0);var
v=[0,f];continue}}}var
lm=[0,ll],lo=[0,ln];function
bL(a,b){var
p=s(g6,0),q=h(iL,h(a,b)),f=dQ(h(iK(p),q));try{var
n=eR,g=eR;a:for(;;){if(1){var
k=function(a,b,c){var
e=b,d=c;for(;;){if(d){var
g=d[1],f=g.getLen(),h=d[2];bg(g,0,a,e-f|0,f);var
e=e-f|0,d=h;continue}return a}},d=0,e=0;for(;;){var
c=t6(f);if(0===c){if(!d)throw[0,bx];var
j=k(B(e),e,d)}else{if(!(0<c)){var
m=B(-c|0);c8(f,m,0,-c|0);var
d=[0,m,d],e=e-c|0;continue}var
i=B(c-1|0);c8(f,i,0,c-1|0);t5(f);if(d){var
l=(e+c|0)-1|0,j=k(B(l),l,[0,i,d])}else
var
j=i}var
g=h(g,h(j,lp)),n=g;continue a}}var
o=g;break}}catch(f){if(f[1]!==bx)throw f;var
o=n}dS(f);return o}var
eS=[0,lq],cT=[],lr=0,ls=0;uy(cT,[0,0,function(f){var
k=eu(f,lt),e=cE(li),d=e.length-1,n=eQ.length-1,a=v(d+n|0,0),o=d-1|0,u=0;if(!(o<0)){var
c=u;for(;;){m(a,c,a3(f,s(e,c)));var
y=c+1|0;if(o!==c){var
c=y;continue}break}}var
q=n-1|0,w=0;if(!(q<0)){var
b=w;for(;;){m(a,b+d|0,eu(f,s(eQ,b)));var
x=b+1|0;if(q!==b){var
b=x;continue}break}}var
r=a[10],l=a[12],h=a[15],i=a[16],j=a[17],g=a[18],z=a[1],A=a[2],B=a[3],C=a[4],D=a[5],E=a[7],F=a[8],G=a[9],H=a[11],I=a[14];function
J(a,b,c,d,e,f){var
h=d?d[1]:d;p(a[1][l+1],a,[0,h],f);var
i=bD(a[g+1],f);return fi(a[1][r+1],a,b,[0,c[1],c[2]],e,f,i)}function
K(a,b,c,d,e){try{var
f=bD(a[g+1],e),h=f}catch(f){if(f[1]!==t)throw f;try{p(a[1][l+1],a,lu,e)}catch(f){throw f}var
h=bD(a[g+1],e)}return fi(a[1][r+1],a,b,[0,c[1],c[2]],d,e,h)}function
L(a,b,c){var
y=b?b[1]:b;try{bD(a[g+1],c);var
f=0}catch(f){if(f[1]===t){if(0===c[2][0]){var
z=a[i+1];if(!z)throw[0,eS,c];var
A=z[1],H=y?uM(A,a[h+1],c[1]):uE(A,a[h+1],c[1]),B=H}else{var
D=a[j+1];if(!D)throw[0,eS,c];var
E=D[1],I=y?uY(E,a[h+1],c[1]):u6(E,a[h+1],c[1]),B=I}var
d=a[g+1],w=cx(d,c);m(d[2],w,[0,c,B,s(d[2],w)]);d[1]=d[1]+1|0;var
x=d[2].length-1<<1<d[1]?1:0;if(x){var
l=d[2],n=l.length-1,o=n*2|0,p=o<ct?1:0;if(p){var
k=v(o,0);d[2]=k;var
q=function(a){if(a){var
b=a[1],e=a[2];q(a[3]);var
c=cx(d,b);return m(k,c,[0,b,e,s(k,c)])}return 0},r=n-1|0,F=0;if(!(r<0)){var
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
f=[0,bL(a[k+1],lw),0],c=f}catch(f){var
c=0}a[i+1]=c;try{var
e=[0,bL(a[k+1],lv),0],d=e}catch(f){var
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
o=[0,bL(c,ly),0],l=o}catch(f){var
l=0}e[i+1]=l;try{var
n=[0,bL(c,lx),0],m=n}catch(f){var
m=0}e[j+1]=m;e[g+1]=ek(0,8);return e}},ls,lr]);fj(0);fj(0);function
cU(a){function
e(a,b){var
d=a-1|0,e=0;if(!(d<0)){var
c=e;for(;;){ed(lA);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return j(ed(lz),b)}function
f(a,b){var
c=a,d=b;for(;;)if(typeof
d===n)return 0===d?e(c,lB):e(c,lC);else
switch(d[0]){case
0:e(c,lD);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
1:e(c,lE);var
c=c+1|0,d=d[1];continue;case
2:e(c,lF);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
3:e(c,lG);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
4:e(c,lH);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
5:e(c,lI);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
6:e(c,lJ);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
7:e(c,lK);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
8:e(c,lL);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
9:e(c,lM);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
10:e(c,lN);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
11:return e(c,h(lO,d[1]));case
12:return e(c,h(lP,d[1]));case
13:return e(c,h(lQ,k(d[1])));case
14:return e(c,h(lR,k(d[1])));case
15:return e(c,h(lS,k(d[1])));case
16:return e(c,h(lT,k(d[1])));case
17:return e(c,h(lU,k(d[1])));case
18:return e(c,lV);case
19:return e(c,lW);case
20:return e(c,lX);case
21:return e(c,lY);case
22:return e(c,lZ);case
23:return e(c,h(l0,k(d[2])));case
24:e(c,l1);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
25:e(c,l2);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
26:e(c,l3);var
c=c+1|0,d=d[1];continue;case
27:e(c,l4);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
28:e(c,l5);var
c=c+1|0,d=d[1];continue;case
29:e(c,l6);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
30:e(c,l7);f(c+1|0,d[1]);f(c+1|0,d[2]);var
c=c+1|0,d=d[3];continue;case
31:return e(c,l8);case
32:var
g=h(l9,k(d[2]));return e(c,h(l_,h(d[1],g)));case
33:return e(c,h(l$,k(d[1])));case
36:e(c,mb);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
37:e(c,mc);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
38:e(c,md);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
39:e(c,me);f(c+1|0,d[1]);f(c+1|0,d[2]);var
c=c+1|0,d=d[3];continue;case
40:e(c,mf);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
41:e(c,mg);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
42:e(c,mh);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
43:e(c,mi);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
44:e(c,mj);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
45:e(c,mk);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
46:e(c,ml);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
47:e(c,mm);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
48:e(c,mn);f(c+1|0,d[1]);f(c+1|0,d[2]);f(c+1|0,d[3]);var
c=c+1|0,d=d[4];continue;case
49:e(c,mo);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
50:e(c,mp);f(c+1|0,d[1]);var
i=d[2],j=c+1|0;return dU(function(a){return f(j,a)},i);case
51:return e(c,mq);case
52:return e(c,mr);default:return e(c,h(ma,O(d[1])))}}return f(0,a)}function
K(a){return an(a,32)}var
bb=[0,ms];function
bj(a,b,c){var
d=c;for(;;)if(typeof
d===n)return mu;else
switch(d[0]){case
18:case
19:var
S=h(m9,h(e(b,d[2]),m8));return h(m_,h(k(d[1]),S));case
27:case
38:var
ac=d[1],ad=h(nt,h(e(b,d[2]),ns));return h(e(b,ac),ad);case
0:var
g=d[2],B=e(b,d[1]);if(typeof
g===n)var
r=0;else
if(25===g[0]){var
t=e(b,g),r=1}else
var
r=0;if(!r){var
C=h(mv,K(b)),t=h(e(b,g),C)}return h(h(B,t),mw);case
1:var
D=h(e(b,d[1]),mx),F=y(bb[1][1],my)?h(bb[1][1],mz):mB;return h(mA,h(F,D));case
2:var
G=h(mD,h(U(b,d[2]),mC));return h(mE,h(U(b,d[1]),G));case
3:var
I=h(mG,h(aq(b,d[2]),mF));return h(mH,h(aq(b,d[1]),I));case
4:var
J=h(mJ,h(U(b,d[2]),mI));return h(mK,h(U(b,d[1]),J));case
5:var
L=h(mM,h(aq(b,d[2]),mL));return h(mN,h(aq(b,d[1]),L));case
6:var
M=h(mP,h(U(b,d[2]),mO));return h(mQ,h(U(b,d[1]),M));case
7:var
N=h(mS,h(aq(b,d[2]),mR));return h(mT,h(aq(b,d[1]),N));case
8:var
P=h(mV,h(U(b,d[2]),mU));return h(mW,h(U(b,d[1]),P));case
9:var
Q=h(mY,h(aq(b,d[2]),mX));return h(mZ,h(aq(b,d[1]),Q));case
10:var
R=h(m1,h(U(b,d[2]),m0));return h(m2,h(U(b,d[1]),R));case
13:return h(m3,k(d[1]));case
14:return h(m4,k(d[1]));case
15:throw[0,H,m5];case
16:return h(m6,k(d[1]));case
17:return h(m7,k(d[1]));case
20:var
V=h(na,h(e(b,d[2]),m$));return h(nb,h(k(d[1]),V));case
21:var
W=h(nd,h(e(b,d[2]),nc));return h(ne,h(k(d[1]),W));case
22:var
X=h(ng,h(e(b,d[2]),nf));return h(nh,h(k(d[1]),X));case
23:var
Y=h(ni,k(d[2])),u=d[1];if(typeof
u===n)var
f=0;else
switch(u[0]){case
33:var
o=nk,f=1;break;case
34:var
o=nl,f=1;break;case
35:var
o=nm,f=1;break;default:var
f=0}if(f)return h(o,Y);throw[0,H,nj];case
24:var
i=d[2],v=d[1];if(typeof
i===n){if(0===i){var
d=v;continue}}else
if(24===i[0]){var
Z=h(no,e(b,i));return h(e(b,v),Z)}return T(nn);case
25:var
_=e(b,d[2]),$=h(np,h(K(b),_));return h(e(b,d[1]),$);case
26:var
aa=e(b,d[1]),ab=y(bb[1][2],nq)?bb[1][2]:nr;return h(ab,aa);case
28:var
d=d[1];continue;case
29:var
ae=h(nv,h(e(b,d[2]),nu));return h(e(b,d[1]),ae);case
30:var
l=d[2],af=e(b,d[3]),ag=h(nw,h(K(b),af));if(typeof
l===n)var
s=0;else
if(31===l[0]){var
w=h(mt(l[1]),ny),s=1}else
var
s=0;if(!s)var
w=e(b,l);var
ah=h(nx,h(w,ag));return h(e(b,d[1]),ah);case
31:return a<50?bi(1+a,d[1]):E(bi,[0,d[1]]);case
33:return k(d[1]);case
34:return h(O(d[1]),nz);case
35:return O(d[1]);case
36:var
ai=h(nB,h(e(b,d[2]),nA));return h(e(b,d[1]),ai);case
37:var
aj=h(nD,h(K(b),nC)),ak=h(e(b,d[2]),aj),al=h(nE,h(K(b),ak)),am=h(nF,h(e(b,d[1]),al));return h(K(b),am);case
39:var
an=h(nG,K(b)),ao=h(e(b+2|0,d[3]),an),ap=h(nH,h(K(b+2|0),ao)),ar=h(nI,h(K(b),ap)),as=h(e(b+2|0,d[2]),ar),at=h(nJ,h(K(b+2|0),as));return h(nK,h(e(b,d[1]),at));case
40:var
au=h(nL,K(b)),av=h(e(b+2|0,d[2]),au),aw=h(nM,h(K(b+2|0),av));return h(nN,h(e(b,d[1]),aw));case
41:var
ax=h(nO,e(b,d[2]));return h(e(b,d[1]),ax);case
42:var
ay=h(nP,e(b,d[2]));return h(e(b,d[1]),ay);case
43:var
az=h(nQ,e(b,d[2]));return h(e(b,d[1]),az);case
44:var
aA=h(nR,e(b,d[2]));return h(e(b,d[1]),aA);case
45:var
aB=h(nS,e(b,d[2]));return h(e(b,d[1]),aB);case
46:var
aC=h(nT,e(b,d[2]));return h(e(b,d[1]),aC);case
47:var
aD=h(nU,e(b,d[2]));return h(e(b,d[1]),aD);case
48:var
p=e(b,d[1]),aE=e(b,d[2]),aF=e(b,d[3]),aG=h(e(b+2|0,d[4]),nV);return h(n1,h(p,h(n0,h(aE,h(nZ,h(p,h(nY,h(aF,h(nX,h(p,h(nW,h(K(b+2|0),aG))))))))))));case
49:var
aH=e(b,d[1]),aI=h(e(b+2|0,d[2]),n2);return h(n4,h(aH,h(n3,h(K(b+2|0),aI))));case
50:var
x=d[2],m=d[1],z=e(b,m),q=function(a){if(a){var
c=a[2],d=a[1];if(c){var
f=h(n5,q(c));return h(e(b,d),f)}return e(b,d)}throw[0,H,n6]};if(typeof
m!==n)if(31===m[0]){var
A=m[1];if(!y(A[1],n9))if(!y(A[2],n_))return h(z,h(oa,h(q(aY(x)),n$)))}return h(z,h(n8,h(q(aY(x)),n7)));case
51:return k(j(d[1],0));case
52:return h(O(j(d[1],0)),ob);default:return d[1]}}function
tq(a,b,c){if(typeof
c!==n)switch(c[0]){case
2:case
4:case
6:case
8:case
10:case
50:return a<50?bj(1+a,b,c):E(bj,[0,b,c]);case
32:return c[1];case
33:return k(c[1]);case
36:var
d=h(od,h(U(b,c[2]),oc));return h(e(b,c[1]),d);case
51:return k(j(c[1],0));default:}return a<50?c9(1+a,b,c):E(c9,[0,b,c])}function
c9(a,b,c){if(typeof
c!==n)switch(c[0]){case
3:case
5:case
7:case
9:case
29:case
50:return a<50?bj(1+a,b,c):E(bj,[0,b,c]);case
16:return h(of,k(c[1]));case
31:return a<50?bi(1+a,c[1]):E(bi,[0,c[1]]);case
32:return c[1];case
34:return h(O(c[1]),og);case
35:return h(oh,O(c[1]));case
36:var
d=h(oj,h(U(b,c[2]),oi));return h(e(b,c[1]),d);case
52:return h(O(j(c[1],0)),ok);default:}cU(c);return T(oe)}function
bi(a,b){return b[1]}function
e(b,c){return Z(bj(0,b,c))}function
U(b,c){return Z(tq(0,b,c))}function
aq(b,c){return Z(c9(0,b,c))}function
mt(b){return Z(bi(0,b))}function
A(a){return an(a,32)}var
bc=[0,ol];function
bl(a,b,c){var
d=c;for(;;)if(typeof
d===n)return on;else
switch(d[0]){case
18:case
19:var
U=h(oK,h(f(b,d[2]),oJ));return h(oL,h(k(d[1]),U));case
27:case
38:var
ac=d[1],ad=h(o5,R(b,d[2]));return h(f(b,ac),ad);case
0:var
g=d[2],C=f(b,d[1]);if(typeof
g===n)var
r=0;else
if(25===g[0]){var
t=f(b,g),r=1}else
var
r=0;if(!r){var
D=h(oo,A(b)),t=h(f(b,g),D)}return h(h(C,t),op);case
1:var
F=h(f(b,d[1]),oq),G=y(bc[1][1],or)?h(bc[1][1],os):ou;return h(ot,h(G,F));case
2:var
I=h(ov,R(b,d[2]));return h(R(b,d[1]),I);case
3:var
J=h(ow,ar(b,d[2]));return h(ar(b,d[1]),J);case
4:var
K=h(ox,R(b,d[2]));return h(R(b,d[1]),K);case
5:var
L=h(oy,ar(b,d[2]));return h(ar(b,d[1]),L);case
6:var
M=h(oz,R(b,d[2]));return h(R(b,d[1]),M);case
7:var
N=h(oA,ar(b,d[2]));return h(ar(b,d[1]),N);case
8:var
P=h(oB,R(b,d[2]));return h(R(b,d[1]),P);case
9:var
Q=h(oC,ar(b,d[2]));return h(ar(b,d[1]),Q);case
10:var
S=h(oD,R(b,d[2]));return h(R(b,d[1]),S);case
13:return h(oE,k(d[1]));case
14:return h(oF,k(d[1]));case
15:throw[0,H,oG];case
16:return h(oH,k(d[1]));case
17:return h(oI,k(d[1]));case
20:var
V=h(oN,h(f(b,d[2]),oM));return h(oO,h(k(d[1]),V));case
21:var
W=h(oQ,h(f(b,d[2]),oP));return h(oR,h(k(d[1]),W));case
22:var
X=h(oT,h(f(b,d[2]),oS));return h(oU,h(k(d[1]),X));case
23:var
Y=h(oV,k(d[2])),u=d[1];if(typeof
u===n)var
e=0;else
switch(u[0]){case
33:var
o=oX,e=1;break;case
34:var
o=oY,e=1;break;case
35:var
o=oZ,e=1;break;default:var
e=0}if(e)return h(o,Y);throw[0,H,oW];case
24:var
i=d[2],v=d[1];if(typeof
i===n){if(0===i){var
d=v;continue}}else
if(24===i[0]){var
Z=h(o1,f(b,i));return h(f(b,v),Z)}return T(o0);case
25:var
_=f(b,d[2]),$=h(o2,h(A(b),_));return h(f(b,d[1]),$);case
26:var
aa=f(b,d[1]),ab=y(bc[1][2],o3)?bc[1][2]:o4;return h(ab,aa);case
28:var
d=d[1];continue;case
29:var
ae=h(o7,h(f(b,d[2]),o6));return h(f(b,d[1]),ae);case
30:var
l=d[2],af=f(b,d[3]),ag=h(o8,h(A(b),af));if(typeof
l===n)var
s=0;else
if(31===l[0]){var
w=om(l[1]),s=1}else
var
s=0;if(!s)var
w=f(b,l);var
ah=h(o9,h(w,ag));return h(f(b,d[1]),ah);case
31:return a<50?bk(1+a,d[1]):E(bk,[0,d[1]]);case
33:return k(d[1]);case
34:return h(O(d[1]),o_);case
35:return O(d[1]);case
36:var
ai=h(pa,h(f(b,d[2]),o$));return h(f(b,d[1]),ai);case
37:var
aj=h(pc,h(A(b),pb)),ak=h(f(b,d[2]),aj),al=h(pd,h(A(b),ak)),am=h(pe,h(f(b,d[1]),al));return h(A(b),am);case
39:var
an=h(pf,A(b)),ao=h(f(b+2|0,d[3]),an),ap=h(pg,h(A(b+2|0),ao)),aq=h(ph,h(A(b),ap)),as=h(f(b+2|0,d[2]),aq),at=h(pi,h(A(b+2|0),as));return h(pj,h(f(b,d[1]),at));case
40:var
au=h(pk,A(b)),av=h(pl,h(A(b),au)),aw=h(f(b+2|0,d[2]),av),ax=h(pm,h(A(b+2|0),aw)),ay=h(pn,h(A(b),ax));return h(po,h(f(b,d[1]),ay));case
41:var
az=h(pp,f(b,d[2]));return h(f(b,d[1]),az);case
42:var
aA=h(pq,f(b,d[2]));return h(f(b,d[1]),aA);case
43:var
aB=h(pr,f(b,d[2]));return h(f(b,d[1]),aB);case
44:var
aC=h(ps,f(b,d[2]));return h(f(b,d[1]),aC);case
45:var
aD=h(pt,f(b,d[2]));return h(f(b,d[1]),aD);case
46:var
aE=h(pu,f(b,d[2]));return h(f(b,d[1]),aE);case
47:var
aF=h(pv,f(b,d[2]));return h(f(b,d[1]),aF);case
48:var
p=f(b,d[1]),aG=f(b,d[2]),aH=f(b,d[3]),aI=h(f(b+2|0,d[4]),pw);return h(pC,h(p,h(pB,h(aG,h(pA,h(p,h(pz,h(aH,h(py,h(p,h(px,h(A(b+2|0),aI))))))))))));case
49:var
aJ=f(b,d[1]),aK=h(f(b+2|0,d[2]),pD);return h(pF,h(aJ,h(pE,h(A(b+2|0),aK))));case
50:var
x=d[2],m=d[1],z=f(b,m),q=function(a){if(a){var
c=a[2],d=a[1];if(c){var
e=h(pG,q(c));return h(f(b,d),e)}return f(b,d)}throw[0,H,pH]};if(typeof
m!==n)if(31===m[0]){var
B=m[1];if(!y(B[1],pK))if(!y(B[2],pL))return h(z,h(pN,h(q(aY(x)),pM)))}return h(z,h(pJ,h(q(aY(x)),pI)));case
51:return k(j(d[1],0));case
52:return h(O(j(d[1],0)),pO);default:return d[1]}}function
tr(a,b,c){if(typeof
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
d=h(pQ,h(R(b,c[2]),pP));return h(f(b,c[1]),d);case
51:return k(j(c[1],0));default:}return a<50?c_(1+a,b,c):E(c_,[0,b,c])}function
c_(a,b,c){if(typeof
c!==n)switch(c[0]){case
3:case
5:case
7:case
9:case
50:return a<50?bl(1+a,b,c):E(bl,[0,b,c]);case
16:return h(pS,k(c[1]));case
31:return a<50?bk(1+a,c[1]):E(bk,[0,c[1]]);case
32:return c[1];case
34:return h(O(c[1]),pT);case
35:return h(pU,O(c[1]));case
36:var
d=h(pW,h(R(b,c[2]),pV));return h(f(b,c[1]),d);case
52:return h(O(j(c[1],0)),pX);default:}cU(c);return T(pR)}function
bk(a,b){return b[2]}function
f(b,c){return Z(bl(0,b,c))}function
R(b,c){return Z(tr(0,b,c))}function
ar(b,c){return Z(c_(0,b,c))}function
om(b){return Z(bk(0,b))}var
p9=h(p8,h(p7,h(p6,h(p5,h(p4,h(p3,h(p2,h(p1,h(p0,h(pZ,pY)))))))))),qo=h(qn,h(qm,h(ql,h(qk,h(qj,h(qi,h(qh,h(qg,h(qf,h(qe,h(qd,h(qc,h(qb,h(qa,h(p$,p_))))))))))))))),qw=h(qv,h(qu,h(qt,h(qs,h(qr,h(qq,qp)))))),qE=h(qD,h(qC,h(qB,h(qA,h(qz,h(qy,qx))))));function
a(a){return[32,h(qF,k(a)),a]}function
w(a,b){return[25,a,b]}function
bM(a,b){return[50,a,b]}function
eT(a){return[33,a]}function
aM(a){return[51,a]}function
ab(a){return[34,a]}function
bN(a,b){return[2,a,b]}function
bO(a,b){return[3,a,b]}function
cV(a,b){return[5,a,b]}function
cW(a,b){return[6,a,b]}function
ac(a,b){return[7,a,b]}function
eU(a,b){return[9,a,b]}function
bd(a){return[13,a]}function
aA(a){return[14,a]}function
ad(a,b){return[31,[0,a,b]]}function
V(a,b){return[37,a,b]}function
W(a,b){return[27,a,b]}function
X(a){return[28,a]}function
aN(a,b){return[38,a,b]}function
eV(a,b){return[42,a,b]}function
cX(a,b){return[44,a,b]}function
eW(a){var
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
ao=aX(b,c[2]);return[50,b(c[1]),ao];default:}return c}}var
c=b(a);for(;;){if(e[1]){e[1]=0;var
c=b(c);continue}return c}}var
qN=[0,qM];function
be(a,b,c){var
g=c[2],d=c[1],t=a?a[1]:a,u=b?b[1]:2,m=g[3],p=g[2];qN[1]=qO;var
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
o=h(qW,h(k(j[1]),qV)),l=2;break;default:var
l=1}switch(l){case
1:cU(m[1]);dT(dO);throw[0,H,qP];case
2:break;default:var
o=qQ}var
q=[0,e(0,m[1]),o];if(t){bb[1]=q;bc[1]=q}function
r(a){var
q=g[4],r=dV(function(a,b){return 0===b?a:h(qw,a)},qE,q),s=h(r,e(0,eW(p))),j=c3(e$(qR,gL,438));dP(j,s);c4(j);fa(j);fk(qS);var
m=dQ(qT),c=t2(m),n=B(c),o=0;if(0<=0)if(0<=c)if((n.getLen()-c|0)<o)var
f=0;else{var
k=o,b=c;for(;;){if(0<b){var
l=c8(m,n,k,b);if(0===l)throw[0,bx];var
k=k+l|0,b=b-l|0;continue}var
f=1;break}}else
var
f=0;else
var
f=0;if(!f)G(gN);dS(m);i(af(d,723535973,3),d,n);fk(qU);return 0}function
s(a){var
b=g[4],c=dV(function(a,b){return 0===b?a:h(qo,a)},p9,b);return i(af(d,56985577,4),d,h(c,f(0,eW(p))))}switch(u){case
1:s(0);break;case
2:r(0);s(0);break;default:r(0)}i(af(d,345714255,5),d,0);return[0,d,g]}var
cY=d,eX=null,q1=1,q2=1,q3=1,q4=1,q5=1,q6=1,q7=undefined;function
eY(a,b){return a==eX?j(b,0):a}var
eZ=Array,q8=true,q9=false;eh(function(a){return a
instanceof
eZ?0:[0,new
aD(a.toString())]});function
D(a,b){a.appendChild(b);return 0}function
e0(d){return tZ(function(a){if(a){var
e=j(d,a);if(!(e|0))a.preventDefault();return e}var
c=event,b=j(d,c);if(!(b|0))c.returnValue=b;return b})}var
L=cY.document,q_="2d";function
bP(a,b){return a?j(b,a[1]):0}function
bQ(a,b){return a.createElement(b.toString())}function
bR(a,b){return bQ(a,b)}var
e1=[0,f_];function
e2(a,b,c,d){for(;;){if(0===a)if(0===b)return bQ(c,d);var
h=e1[1];if(f_===h){try{var
j=L.createElement('<input name="x">'),k=j.tagName.toLowerCase()===ft?1:0,m=k?j.name===dt?1:0:k,i=m}catch(f){var
i=0}var
l=i?fV:-1003883683;e1[1]=l;continue}if(fV<=h){var
e=new
eZ();e.push("<",d.toString());bP(a,function(a){e.push(' type="',fl(a),ca);return 0});bP(b,function(a){e.push(' name="',fl(a),ca);return 0});e.push(">");return c.createElement(e.join(g))}var
f=bQ(c,d);bP(a,function(a){return f.type=a});bP(b,function(a){return f.name=a});return f}}function
e3(a){return bR(a,rc)}var
rg=[0,rf];cY.HTMLElement===q7;function
e7(a){return e3(L)}function
e8(a){function
c(a){throw[0,H,ri]}var
b=eY(L.getElementById(gr),c);return j(ee(function(a){D(b,e7(0));D(b,L.createTextNode(a.toString()));return D(b,e7(0))}),a)}var
ae=gc,as=gc,cZ=50;function
rj(a){var
k=[0,[4,a]];return function(a,b,c,d){var
h=a[2],i=a[1],l=c[2];if(0===l[0]){var
g=l[1],e=[0,0],f=uG(k.length-1),m=g[7][1]<i[1]?1:0;if(m)var
n=m;else{var
s=g[7][2]<i[2]?1:0,n=s||(g[7][3]<i[3]?1:0)}if(n)throw[0,lm];var
o=g[8][1]<h[1]?1:0;if(o)var
p=o;else{var
r=g[8][2]<h[2]?1:0,p=r||(g[8][3]<h[3]?1:0)}if(p)throw[0,lo];cl(function(a,b){function
h(a){if(bK)try{eO(a,0,c);Q(c,0,0)}catch(f){if(f[1]===az)throw[0,az];throw f}return 11===b[0]?uJ(e,f,cP(b[1],dB,c[1][8]),a):uV(e,f,cP(a,dB,c[1][8]),a,c)}switch(b[0]){case
6:case
7:case
8:case
9:case
10:switch(b[0]){case
6:var
d=uU(e,f,b[1]);break;case
7:var
d=uT(e,f,b[1]);break;case
8:var
d=uS(e,f,b[1]);break;case
9:var
d=uR(e,f,b[1]);break;default:var
d=T(lj)}var
g=d;break;case
11:var
g=h(b[1]);break;default:var
g=h(b[1])}return g},k);var
q=uQ(e,d,h,i,f,c[1],b)}else{var
j=[0,0];cl(function(a,b){switch(b[0]){case
6:case
7:case
8:case
9:case
10:switch(b[0]){case
6:var
e=vh(j,d,b[1],c[1]);break;case
7:var
e=vi(j,d,b[1],c[1]);break;case
8:var
e=vf(j,d,b[1],c[1]);break;case
9:var
e=vg(j,d,b[1],c[1]);break;default:var
e=T(lk)}var
g=e;break;default:var
f=b[1];if(bK){if(c7(a8(f),[0,c]))eO(f,0,c);Q(c,0,0)}var
h=c[1],i=J(0),g=vj(j,d,a,cP(f,-701974253,c[1][8]-i|0),h)}return g},k);var
q=ve(d,h,i,c[1],b)}return q}}if(c0===0)var
c=en([0]);else{var
a6=en(aX(hN,c0));cl(function(a,b){var
c=(a*2|0)+2|0;a6[3]=p(ay[4],b,c,a6[3]);a6[4]=p(ap[4],c,1,a6[4]);return 0},c0);var
c=a6}var
cB=aX(function(a){return a3(c,a)},e6),ev=cT[2],rk=cB[1],rl=cB[2],rm=cB[3],h7=cT[4],ep=cC(e4),eq=cC(e6),er=cC(e5),rn=1,cD=cm(function(a){return a3(c,a)},eq),hQ=cm(function(a){return a3(c,a)},er);c[5]=[0,[0,c[3],c[4],c[6],c[7],cD,ep],c[5]];var
hR=ah[1],hS=c[7];function
hT(a,b,c){return cp(a,ep)?p(ah[4],a,b,c):c}c[7]=p(ah[11],hT,hS,hR);var
a4=[0,ay[1]],a5=[0,ap[1]];dY(function(a,b){a4[1]=p(ay[4],a,b,a4[1]);var
e=a5[1];try{var
f=i(ap[22],b,c[4]),d=f}catch(f){if(f[1]!==t)throw f;var
d=1}a5[1]=p(ap[4],b,d,e);return 0},er,hQ);dY(function(a,b){a4[1]=p(ay[4],a,b,a4[1]);a5[1]=p(ap[4],b,0,a5[1]);return 0},eq,cD);c[3]=a4[1];c[4]=a5[1];var
hU=0,hV=c[6];c[6]=co(function(a,b){return cp(a[1],cD)?b:[0,a,b]},hV,hU);var
h8=rn?i(ev,c,h7):j(ev,c),es=c[5],aF=es?es[1]:T(gR),et=c[5],hW=aF[6],hX=aF[5],hY=aF[4],hZ=aF[3],h0=aF[2],h1=aF[1],h2=et?et[2]:T(gS);c[5]=h2;var
cn=hY,by=hW;for(;;){if(by){var
dX=by[1],gT=by[2],h3=i(ah[22],dX,c[7]),cn=p(ah[4],dX,h3,cn),by=gT;continue}c[7]=cn;c[3]=h1;c[4]=h0;var
h4=c[6];c[6]=co(function(a,b){return cp(a[1],hX)?b:[0,a,b]},h4,hZ);var
h9=0,h_=cE(e5),h$=[0,aX(function(a){var
e=a3(c,a);try{var
b=c[6];for(;;){if(!b)throw[0,t];var
d=b[1],f=b[2],h=d[2];if(0!==aP(d[1],e)){var
b=f;continue}var
g=h;break}}catch(f){if(f[1]!==t)throw f;var
g=s(c[2],e)}return g},h_),h9],ia=cE(e4),ro=tt([0,[0,h8],[0,aX(function(a){try{var
b=i(ah[22],a,c[7])}catch(f){if(f[1]===t)throw[0,H,h6];throw f}return b},ia),h$]])[1],rp=function(a,b){if(1===b.length-1){var
c=b[0+1];if(4===c[0])return c[1]}return T(rq)};ex(c,[0,rl,0,rj,rm,function(a,b){return[0,[4,b]]},rk,rp]);var
rr=function(a,b){var
e=ew(b,c);p(ro,e,rt,rs);if(!b){var
f=c[8];if(0!==f){var
d=f;for(;;){if(d){var
g=d[2];j(d[1],e);var
d=g;continue}break}}}return e};eo[1]=(eo[1]+c[1]|0)-1|0;c[8]=dW(c[8]);cz(c,3+at(s(c[2],1)*16|0,aE)|0);var
ib=0,ic=function(a){var
b=a;return rr(ib,b)},rw=a(6),ru=[0,0],rv=[0,[13,14],eI],rx=a(3),ry=aM(function(a){return ae}),rz=bN(cW(a(2),ry),rx),qK=[29,[36,a(0),rz],rw],rA=a(8),rB=ac(a(8),rA),rC=a(7),rD=bO(ac(a(7),rC),rB),rE=aN(a(13),rD),rF=a(10),rG=w(aN(a(8),rF),rE),rH=a(9),rI=w(aN(a(7),rH),rG),rJ=a(12),rK=a(8),rL=a(7),rM=bO(ac(ac(ab(2),rL),rK),rJ),rN=w(aN(a(10),rM),rI),rO=a(11),rP=a(8),rQ=ac(a(8),rP),rR=a(7),rS=bO(cV(ac(a(7),rR),rQ),rO),rT=w(aN(a(9),rS),rN),rU=eT(1),rV=bN(a(6),rU),rW=w(aN(a(6),rV),rT),rX=ab(4),qL=[46,a(13),rX],rY=aM(function(a){return cZ}),rZ=w([49,eV(cX(a(6),rY),qL),rW],qK),r0=a(8),r1=ac(a(8),r0),r2=a(7),r3=bO(ac(a(7),r2),r1),r4=w(W(a(13),r3),rZ),r5=ab(2),r6=[0,aM(function(a){return as})],r9=bM(ad(r8,r7),r6),r_=[0,a(5)],sb=eU(bM(ad(sa,r$),r_),r9),sc=cV(ac(ab(4),sb),r5),sd=w(W(a(12),sc),r4),se=ab(2),sf=[0,aM(function(a){return ae})],si=bM(ad(sh,sg),sf),sj=[0,a(4)],sm=eU(bM(ad(sl,sk),sj),si),sn=cV(ac(ab(4),sm),se),so=w(W(a(11),sn),sd),sp=ab(0),sq=w(W(a(10),sp),so),sr=ab(0),ss=w(W(a(9),sr),sq),st=ab(0),su=w(W(a(8),st),ss),sv=ab(0),sw=w(W(a(7),sv),su),sx=eT(0),sy=w(W(a(6),sx),sw),sz=a(2),sA=w(W(a(5),sz),sy),sB=a(3),sC=w(W(a(4),sB),sA),sD=aM(function(a){return ae}),sE=cX(a(3),sD),sF=aM(function(a){return as}),qH=[40,eV(cX(a(2),sF),sE),sC],sI=ad(sH,sG),sL=cW(ad(sK,sJ),sI),sO=bN(ad(sN,sM),sL),sP=w(W(a(3),sO),qH),sS=ad(sR,sQ),sV=cW(ad(sU,sT),sS),sY=bN(ad(sX,sW),sV),qI=[26,w(W(a(2),sY),sP)],sZ=V(X(aA(13)),qI),s0=V(X(aA(12)),sZ),s1=V(X(aA(11)),s0),s2=V(X(aA(10)),s1),s3=V(X(aA(9)),s2),s4=V(X(aA(8)),s3),s5=V(X(aA(7)),s4),s6=V(X(bd(6)),s5),s7=V(X(bd(5)),s6),s8=V(X(bd(4)),s7),s9=V(X(bd(3)),s8),qG=[0,[1,[24,[23,qJ,0],0]],V(X(bd(2)),s9)],s_=[0,function(a){var
e=q2+(q4*q6|0)|0,f=q1+(q3*q5|0)|0,i=e<as?1:0,j=i?f<ae?1:0:i;if(j){var
k=0*0+0*0,d=0,c=0,b=0,l=4*(f/ae)-2,m=4*(e/as)-2;for(;;){if(b<cZ)if(k<=4){var
g=c*c-d*d+l,h=2*c*d+m,k=g*g+h*h,d=h,c=g,b=b+1|0;continue}return aK(a,(e*ae|0)+f|0,b)}}return j},qG,rv,ru],c1=[0,ic(0),s_],bS=function(a){return e3(L)};cY.onload=e0(function(a){function
R(a){throw[0,H,td]}var
c=eY(L.getElementById(gr),R);D(c,bS(0));var
h=e2(0,0,L,ra),G=bQ(L,re);D(G,L.createTextNode("Choose a computing device : "));D(c,G);h.style.margin="10px";D(c,h);var
F=bR(L,rd);D(c,F);D(c,bS(0));var
e=bR(L,rh);if(1-(e.getContext==eX?1:0)){e.width=ae;e.height=as;var
I=e.getContext(q_);D(c,bS(0));D(c,e);var
O=e9?e9[1]:2;switch(O){case
1:fm(0);aI[1]=fn(0);break;case
2:fo(0);aH[1]=fp(0);fm(0);aI[1]=fn(0);break;default:fo(0);aH[1]=fp(0)}eG[1]=aH[1]+aI[1]|0;var
y=aH[1]-1|0,x=0,P=0;if(y<0)var
z=x;else{var
g=P,C=x;for(;;){var
E=ck(C,[0,uZ(g),0]),Q=g+1|0;if(y!==g){var
g=Q,C=E;continue}var
z=E;break}}var
q=0,d=0,b=z;for(;;){if(q<aI[1]){if(vd(d)){var
B=d+1|0,A=ck(b,[0,u1(d,d+aH[1]|0),0])}else{var
B=d,A=b}var
q=q+1|0,d=B,b=A;continue}var
p=0,o=b;for(;;){if(o){var
p=p+1|0,o=o[2];continue}eG[1]=p;aI[1]=d;if(b){var
m=0,k=b,K=b[2],M=b[1];for(;;){if(k){var
m=m+1|0,k=k[2];continue}var
w=v(m,M),n=1,f=K;for(;;){if(f){var
N=f[2];w[n+1]=f[1];var
n=n+1|0,f=N;continue}var
t=w;break}break}}else
var
t=[0];be(0,te,c1);var
J=I.getImageData(0,0,ae,as),l=J.data;D(c,bS(0));dU(function(a){var
b=bR(L,q$);D(b,L.createTextNode(a[1][1].toString()));return D(h,b)},t);var
S=function(a){var
f=s(t,h.selectedIndex+0|0),z=f[1][1];j(e8(tb),z);var
n=aJ(eI,0,ae*as|0);ei(hI,fb(0));var
p=f[2];if(0===p[0])var
d=16;else{var
E=0===p[1][2]?1:16,d=E}var
v=eF(0),e=c1[2],b=c1[1],A=0,B=[0,[0,d,d,1],[0,at((ae+d|0)-1|0,d),at((as+d|0)-1|0,d),1]],q=0,o=0?q[1]:q;if(0===f[2][0]){if(o)be(0,qX,[0,b,e]);else
if(!i(af(b,-723625231,7),b,0))be(0,qY,[0,b,e])}else
if(o)be(0,qZ,[0,b,e]);else
if(!i(af(b,649483637,8),b,0))be(0,q0,[0,b,e]);(function(a,b,c,d,e,f){return a.length==5?a(b,c,d,e,f):am(a,[b,c,d,e,f])}(af(b,5695307,6),b,n,B,A,f));var
u=Y(n)-1|0,C=0;if(!(u<0)){var
c=C;for(;;){var
g=aL(n,c);if(g===cZ)var
k=ta;else{var
m=function(a){return r*(gq+gq*Math.sin(a*0.1))|0},x=m(g),y=m(g+16|0),k=[0,m(g+32|0),y,x]}l[c*4|0]=k[1];l[(c*4|0)+1|0]=k[2];l[(c*4|0)+2|0]=k[3];l[(c*4|0)+3|0]=r;var
D=c+1|0;if(u!==c){var
c=D;continue}break}}var
w=eF(0)-v;i(e8(s$),tc,w);I.putImageData(J,0,0);return q8},u=e2([0,"button"],0,L,rb);u.value="Go";u.onclick=e0(S);D(F,u);return q9}}}throw[0,rg]});dR(0);return}}(this));
