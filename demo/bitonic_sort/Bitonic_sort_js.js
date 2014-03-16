// This program was compiled from OCaml by js_of_ocaml 1.99dev
(function(d){"use strict";var
dc="set_cuda_sources",du=123,bQ=";",fN=108,gs="section1",fz="spoc_xor",db="reload_sources",bU="Map.bal",f1=",",b0='"',ab=16777215,da="get_cuda_sources",b8=" / ",fM="double spoc_var",dk="args_to_list",bZ=" * ",af="(",fy="float spoc_var",dj=65599,b7="if (",bY="return",f0=" ;\n",dt="exec",bg=115,be=";}\n",fL=".ptx",y=512,gr="logical_and",ds=120,c$="..",fZ=-512,L="]",dr=117,bT="; ",dq="compile",gq=" (",Z="0",gp=1026,di="list_to_args",bS=248,fK="Bitonic_sort_js_js.ml",fY=126,go="fd ",c_="get_binaries",fJ=" == ",dh="Kirc_Cuda.ml",b6=" + ",fX=") ",dp="x",fI=-97,fx="g",bc=1073741823,gn="parse concat",ay=105,dg="get_opencl_sources",gm=511,bd=110,gl=-88,ad=" = ",df="set_opencl_sources",M="[",bX="'",fw="Unix",bP="int_of_string",gk="(double) ",fW=982028505,bb="){\n",bf="e",gj="#define __FLOAT64_EXTENSION__ \n",ax="-",aJ=-48,bW="(double) spoc_var",fv="++){\n",fH="__shared__ float spoc_var",gi="opencl_sources",fG=".cl",dn="reset_binaries",bO="\n",gh=101,c9=32768,dx=748841679,b5="index out of bounds",fu="spoc_init_opencl_device_vec",c8=125,bV=" - ",gg=";}",s=255,gf="binaries",b4="}",ge=" < ",ft="__shared__ long spoc_var",gd=1027,aI=250,gc=" >= ",fs="input",fV=246,gb=1073741824,de=102,fU="Unix.Unix_error",g="",fr=" || ",aH=100,dm="Kirc_OpenCL.ml",ga="#ifndef __FLOAT64_EXTENSION__ \n",fT="__shared__ int spoc_var",dw=103,bN=", ",fS="./",fF=1e3,fq="for (int ",f$="file_file",f_="spoc_var",ag=".",fE="else{\n",bR="+",dv="run",b3=65535,dl="#endif\n",aG=";\n",_="f",f9=785140586,f8="__shared__ double spoc_var",fD=-32,dd=111,fR=" > ",D=" ",f7="int spoc_var",ae=")",fQ="cuda_sources",b2=256,fC="nan",c7=116,f4="../",f5="kernel_name",f6=65520,f3="%.12g",fp=" && ",fP=32752,fB="/",fO="while (",c6="compile_and_run",b1=114,f2="* spoc_var",bM=" <= ",o="number",fA=" % ",uC=d.spoc_opencl_part_device_to_cpu_b!==undefined?d.spoc_opencl_part_device_to_cpu_b:function(){p("spoc_opencl_part_device_to_cpu_b not implemented")},uB=d.spoc_opencl_part_cpu_to_device_b!==undefined?d.spoc_opencl_part_cpu_to_device_b:function(){p("spoc_opencl_part_cpu_to_device_b not implemented")},uz=d.spoc_opencl_load_param_int64!==undefined?d.spoc_opencl_load_param_int64:function(){p("spoc_opencl_load_param_int64 not implemented")},ux=d.spoc_opencl_load_param_float64!==undefined?d.spoc_opencl_load_param_float64:function(){p("spoc_opencl_load_param_float64 not implemented")},uw=d.spoc_opencl_load_param_float!==undefined?d.spoc_opencl_load_param_float:function(){p("spoc_opencl_load_param_float not implemented")},ur=d.spoc_opencl_custom_part_device_to_cpu_b!==undefined?d.spoc_opencl_custom_part_device_to_cpu_b:function(){p("spoc_opencl_custom_part_device_to_cpu_b not implemented")},uq=d.spoc_opencl_custom_part_cpu_to_device_b!==undefined?d.spoc_opencl_custom_part_cpu_to_device_b:function(){p("spoc_opencl_custom_part_cpu_to_device_b not implemented")},up=d.spoc_opencl_custom_device_to_cpu!==undefined?d.spoc_opencl_custom_device_to_cpu:function(){p("spoc_opencl_custom_device_to_cpu not implemented")},uo=d.spoc_opencl_custom_cpu_to_device!==undefined?d.spoc_opencl_custom_cpu_to_device:function(){p("spoc_opencl_custom_cpu_to_device not implemented")},un=d.spoc_opencl_custom_alloc_vect!==undefined?d.spoc_opencl_custom_alloc_vect:function(){p("spoc_opencl_custom_alloc_vect not implemented")},uc=d.spoc_cuda_part_device_to_cpu_b!==undefined?d.spoc_cuda_part_device_to_cpu_b:function(){p("spoc_cuda_part_device_to_cpu_b not implemented")},ub=d.spoc_cuda_part_cpu_to_device_b!==undefined?d.spoc_cuda_part_cpu_to_device_b:function(){p("spoc_cuda_part_cpu_to_device_b not implemented")},ua=d.spoc_cuda_load_param_vec_b!==undefined?d.spoc_cuda_load_param_vec_b:function(){p("spoc_cuda_load_param_vec_b not implemented")},t$=d.spoc_cuda_load_param_int_b!==undefined?d.spoc_cuda_load_param_int_b:function(){p("spoc_cuda_load_param_int_b not implemented")},t_=d.spoc_cuda_load_param_int64_b!==undefined?d.spoc_cuda_load_param_int64_b:function(){p("spoc_cuda_load_param_int64_b not implemented")},t9=d.spoc_cuda_load_param_float_b!==undefined?d.spoc_cuda_load_param_float_b:function(){p("spoc_cuda_load_param_float_b not implemented")},t8=d.spoc_cuda_load_param_float64_b!==undefined?d.spoc_cuda_load_param_float64_b:function(){p("spoc_cuda_load_param_float64_b not implemented")},t7=d.spoc_cuda_launch_grid_b!==undefined?d.spoc_cuda_launch_grid_b:function(){p("spoc_cuda_launch_grid_b not implemented")},t6=d.spoc_cuda_flush_all!==undefined?d.spoc_cuda_flush_all:function(){p("spoc_cuda_flush_all not implemented")},t5=d.spoc_cuda_flush!==undefined?d.spoc_cuda_flush:function(){p("spoc_cuda_flush not implemented")},t4=d.spoc_cuda_device_to_cpu!==undefined?d.spoc_cuda_device_to_cpu:function(){p("spoc_cuda_device_to_cpu not implemented")},t2=d.spoc_cuda_custom_part_device_to_cpu_b!==undefined?d.spoc_cuda_custom_part_device_to_cpu_b:function(){p("spoc_cuda_custom_part_device_to_cpu_b not implemented")},t1=d.spoc_cuda_custom_part_cpu_to_device_b!==undefined?d.spoc_cuda_custom_part_cpu_to_device_b:function(){p("spoc_cuda_custom_part_cpu_to_device_b not implemented")},t0=d.spoc_cuda_custom_load_param_vec_b!==undefined?d.spoc_cuda_custom_load_param_vec_b:function(){p("spoc_cuda_custom_load_param_vec_b not implemented")},tZ=d.spoc_cuda_custom_device_to_cpu!==undefined?d.spoc_cuda_custom_device_to_cpu:function(){p("spoc_cuda_custom_device_to_cpu not implemented")},tY=d.spoc_cuda_custom_cpu_to_device!==undefined?d.spoc_cuda_custom_cpu_to_device:function(){p("spoc_cuda_custom_cpu_to_device not implemented")},gH=d.spoc_cuda_custom_alloc_vect!==undefined?d.spoc_cuda_custom_alloc_vect:function(){p("spoc_cuda_custom_alloc_vect not implemented")},tX=d.spoc_cuda_create_extra!==undefined?d.spoc_cuda_create_extra:function(){p("spoc_cuda_create_extra not implemented")},tW=d.spoc_cuda_cpu_to_device!==undefined?d.spoc_cuda_cpu_to_device:function(){p("spoc_cuda_cpu_to_device not implemented")},gG=d.spoc_cuda_alloc_vect!==undefined?d.spoc_cuda_alloc_vect:function(){p("spoc_cuda_alloc_vect not implemented")},tT=d.spoc_create_custom!==undefined?d.spoc_create_custom:function(){p("spoc_create_custom not implemented")},uF=1;function
gC(a,b){throw[0,a,b]}function
dH(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=d.console;b&&b.error&&b.error(a)}var
r=[0];function
bj(a,b){if(!a)return g;if(a&1)return bj(a-1,b)+b;var
c=bj(a>>1,b);return c+c}function
G(a){if(a!=null){this.bytes=this.fullBytes=a;this.last=this.len=a.length}}function
gF(){gC(r[4],new
G(b5))}G.prototype={string:null,bytes:null,fullBytes:null,array:null,len:null,last:0,toJsString:function(){var
a=this.getFullBytes();try{return this.string=decodeURIComponent(escape(a))}catch(f){dH('MlString.toJsString: wrong encoding for \"%s\" ',a);return a}},toBytes:function(){if(this.string!=null)try{var
a=unescape(encodeURIComponent(this.string))}catch(f){dH('MlString.toBytes: wrong encoding for \"%s\" ',this.string);var
a=this.string}else{var
a=g,c=this.array,d=c.length;for(var
b=0;b<d;b++)a+=String.fromCharCode(c[b])}this.bytes=this.fullBytes=a;this.last=this.len=a.length;return a},getBytes:function(){var
a=this.bytes;if(a==null)a=this.toBytes();return a},getFullBytes:function(){var
a=this.fullBytes;if(a!==null)return a;a=this.bytes;if(a==null)a=this.toBytes();if(this.last<this.len){this.bytes=a+=bj(this.len-this.last,"\0");this.last=this.len}this.fullBytes=a;return a},toArray:function(){var
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
b=this.bytes;if(b==null)b=this.toBytes();return a<this.last?b.charCodeAt(a):0},safeGet:function(a){if(this.len==null)this.toBytes();if(a<0||a>=this.len)gF();return this.get(a)},set:function(a,b){var
c=this.array;if(!c){if(this.last==a){this.bytes+=String.fromCharCode(b&s);this.last++;return 0}c=this.toArray()}else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;c[a]=b&s;return 0},safeSet:function(a,b){if(this.len==null)this.toBytes();if(a<0||a>=this.len)gF();this.set(a,b)},fill:function(a,b,c){if(a>=this.last&&this.last&&c==0)return;var
d=this.array;if(!d)d=this.toArray();else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;var
f=a+b;for(var
e=a;e<f;e++)d[e]=c},compare:function(a){if(this.string!=null&&a.string!=null){if(this.string<a.string)return-1;if(this.string>a.string)return 1;return 0}var
b=this.getFullBytes(),c=a.getFullBytes();if(b<c)return-1;if(b>c)return 1;return 0},equal:function(a){if(this.string!=null&&a.string!=null)return this.string==a.string;return this.getFullBytes()==a.getFullBytes()},lessThan:function(a){if(this.string!=null&&a.string!=null)return this.string<a.string;return this.getFullBytes()<a.getFullBytes()},lessEqual:function(a){if(this.string!=null&&a.string!=null)return this.string<=a.string;return this.getFullBytes()<=a.getFullBytes()}};function
an(a){this.string=a}an.prototype=new
G();function
sI(a,b,c,d,e){if(d<=b)for(var
f=1;f<=e;f++)c[d+f]=a[b+f];else
for(var
f=e;f>=1;f--)c[d+f]=a[b+f]}function
sJ(a){var
c=[0];while(a!==0){var
d=a[1];for(var
b=1;b<d.length;b++)c.push(d[b]);a=a[2]}return c}function
dG(a,b){gC(a,new
an(b))}function
ao(a){dG(r[4],a)}function
aK(){ao(b5)}function
sK(a,b){if(b<0||b>=a.length-1)aK();return a[b+1]}function
sL(a,b,c){if(b<0||b>=a.length-1)aK();a[b+1]=c;return 0}var
dz;function
sM(a,b,c){if(c.length!=2)ao("Bigarray.create: bad number of dimensions");if(b!=0)ao("Bigarray.create: unsupported layout");if(c[1]<0)ao("Bigarray.create: negative dimension");if(!dz){var
e=d;dz=[e.Float32Array,e.Float64Array,e.Int8Array,e.Uint8Array,e.Int16Array,e.Uint16Array,e.Int32Array,null,e.Int32Array,e.Int32Array,null,null,e.Uint8Array]}var
f=dz[a];if(!f)ao("Bigarray.create: unsupported kind");return new
f(c[1])}function
sN(a,b){if(b<0||b>=a.length)aK();return a[b]}function
sO(a,b,c){if(b<0||b>=a.length)aK();a[b]=c;return 0}function
dA(a,b,c,d,e){if(e===0)return;if(d===c.last&&c.bytes!=null){var
f=a.bytes;if(f==null)f=a.toBytes();if(b>0||a.last>e)f=f.slice(b,b+e);c.bytes+=f;c.last+=f.length;return}var
g=c.array;if(!g)g=c.toArray();else
c.bytes=c.string=null;a.blitToArray(b,g,d,e)}function
ah(c,b){if(c.fun)return ah(c.fun,b);var
a=c.length,d=a-b.length;if(d==0)return c.apply(null,b);else
if(d<0)return ah(c.apply(null,b.slice(0,a)),b.slice(a));else
return function(a){return ah(c,b.concat([a]))}}function
sP(a){if(isFinite(a)){if(Math.abs(a)>=2.22507385850720138e-308)return 0;if(a!=0)return 1;return 2}return isNaN(a)?4:3}function
s1(a,b){var
c=a[3]<<16,d=b[3]<<16;if(c>d)return 1;if(c<d)return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
gz(a,b){if(a<b)return-1;if(a==b)return 0;return 1}function
dB(a,b,c){var
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
g=a[0];if(g===aI){a=a[1];continue}else
if(b
instanceof
Array&&b[0]===(b[0]|0)){var
h=b[0];if(h===aI){b=b[1];continue}else
if(g!=h)return g<h?-1:1;else
switch(g){case
bS:{var
d=gz(a[2],b[2]);if(d!=0)return d;break}case
251:ao("equal: abstract value");case
s:{var
d=s1(a,b);if(d!=0)return d;break}default:if(a.length!=b.length)return a.length<b.length?-1:1;if(a.length>1)e.push(a,b,1)}}else
return 1}else
if(b
instanceof
G||b
instanceof
Array&&b[0]===(b[0]|0))return-1;else{if(a<b)return-1;if(a>b)return 1;if(c&&a!=b){if(a==a)return 1;if(b==b)return-1}}if(e.length==0)return 0;var
f=e.pop();b=e.pop();a=e.pop();if(f+1<a.length)e.push(a,b,f+1);a=a[f];b=b[f]}}function
gu(a,b){return dB(a,b,true)}function
gt(a){this.bytes=g;this.len=a}gt.prototype=new
G();function
gv(a){if(a<0)ao("String.create");return new
gt(a)}function
dF(a){throw[0,a]}function
gD(){dF(r[6])}function
sQ(a,b){if(b==0)gD();return a/b|0}function
sR(a,b){return+(dB(a,b,false)==0)}function
sS(a,b,c,d){a.fill(b,c,d)}function
dE(a){a=a.toString();var
e=a.length;if(e>31)ao("format_int: format too long");var
b={justify:bR,signstyle:ax,filler:D,alternate:false,base:0,signedconv:false,width:0,uppercase:false,sign:1,prec:-1,conv:_};for(var
d=0;d<e;d++){var
c=a.charAt(d);switch(c){case
ax:b.justify=ax;break;case
bR:case
D:b.signstyle=c;break;case
Z:b.filler=Z;break;case"#":b.alternate=true;break;case"1":case"2":case"3":case"4":case"5":case"6":case"7":case"8":case"9":b.width=0;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.width=b.width*10+c;d++}d--;break;case
ag:b.prec=0;d++;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.prec=b.prec*10+c;d++}d--;case"d":case"i":b.signedconv=true;case"u":b.base=10;break;case
dp:b.base=16;break;case"X":b.base=16;b.uppercase=true;break;case"o":b.base=8;break;case
bf:case
_:case
fx:b.signedconv=true;b.conv=c;break;case"E":case"F":case"G":b.signedconv=true;b.uppercase=true;b.conv=c.toLowerCase();break}}return b}function
dC(a,b){if(a.uppercase)b=b.toUpperCase();var
e=b.length;if(a.signedconv&&(a.sign<0||a.signstyle!=ax))e++;if(a.alternate){if(a.base==8)e+=1;if(a.base==16)e+=2}var
c=g;if(a.justify==bR&&a.filler==D)for(var
d=e;d<a.width;d++)c+=D;if(a.signedconv)if(a.sign<0)c+=ax;else
if(a.signstyle!=ax)c+=a.signstyle;if(a.alternate&&a.base==8)c+=Z;if(a.alternate&&a.base==16)c+="0x";if(a.justify==bR&&a.filler==Z)for(var
d=e;d<a.width;d++)c+=Z;c+=b;if(a.justify==ax)for(var
d=e;d<a.width;d++)c+=D;return new
an(c)}function
sT(a,b){var
c,f=dE(a),e=f.prec<0?6:f.prec;if(b<0){f.sign=-1;b=-b}if(isNaN(b)){c=fC;f.filler=D}else
if(!isFinite(b)){c="inf";f.filler=D}else
switch(f.conv){case
bf:var
c=b.toExponential(e),d=c.length;if(c.charAt(d-3)==bf)c=c.slice(0,d-1)+Z+c.slice(d-1);break;case
_:c=b.toFixed(e);break;case
fx:e=e?e:1;c=b.toExponential(e-1);var
i=c.indexOf(bf),h=+c.slice(i+1);if(h<-4||b.toFixed(0).length>e){var
d=i-1;while(c.charAt(d)==Z)d--;if(c.charAt(d)==ag)d--;c=c.slice(0,d+1)+c.slice(i);d=c.length;if(c.charAt(d-3)==bf)c=c.slice(0,d-1)+Z+c.slice(d-1);break}else{var
g=e;if(h<0){g-=h+1;c=b.toFixed(g)}else
while(c=b.toFixed(g),c.length>e+1)g--;if(g){var
d=c.length-1;while(c.charAt(d)==Z)d--;if(c.charAt(d)==ag)d--;c=c.slice(0,d+1)}}break}return dC(f,c)}function
sU(a,b){if(a.toString()=="%d")return new
an(g+b);var
c=dE(a);if(b<0)if(c.signedconv){c.sign=-1;b=-b}else
b>>>=0;var
d=b.toString(c.base);if(c.prec>=0){c.filler=D;var
e=c.prec-d.length;if(e>0)d=bj(e,Z)+d}return dC(c,d)}function
sV(){return 0}function
sW(){return 0}var
b_=[];function
sX(a,b,c){var
e=a[1],i=b_[c];if(i===null)for(var
h=b_.length;h<c;h++)b_[h]=0;else
if(e[i]===b)return e[i-1];var
d=3,g=e[1]*2+1,f;while(d<g){f=d+g>>1|1;if(b<e[f+1])g=f-2;else
d=f}b_[c]=d+1;return b==e[d+1]?e[d]:0}function
sY(a,b){return+(gu(a,b,false)>=0)}function
gw(a){if(!isFinite(a)){if(isNaN(a))return[s,1,0,f6];return a>0?[s,0,0,fP]:[s,0,0,f6]}var
f=a>=0?0:c9;if(f)a=-a;var
b=Math.floor(Math.LOG2E*Math.log(a))+1023;if(b<=0){b=0;a/=Math.pow(2,-gp)}else{a/=Math.pow(2,b-gd);if(a<16){a*=2;b-=1}if(b==0)a/=2}var
d=Math.pow(2,24),c=a|0;a=(a-c)*d;var
e=a|0;a=(a-e)*d;var
g=a|0;c=c&15|f|b<<4;return[s,g,e,c]}function
bi(a,b){return((a>>16)*b<<16)+(a&b3)*b|0}var
sZ=function(){var
p=b2;function
c(a,b){return a<<b|a>>>32-b}function
g(a,b){b=bi(b,3432918353);b=c(b,15);b=bi(b,461845907);a^=b;a=c(a,13);return(a*5|0)+3864292196|0}function
t(a){a^=a>>>16;a=bi(a,2246822507);a^=a>>>13;a=bi(a,3266489909);a^=a>>>16;return a}function
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
bS:f=g(f,e[2]);h--;break;case
aI:k[--l]=e[1];break;case
s:f=v(f,e);h--;break;default:var
r=e.length-1<<10|e[0];f=g(f,r);for(j=1,o=e.length;j<o;j++){if(m>=i)break;k[m++]=e[j]}break}else
if(e
instanceof
G){var
n=e.array;if(n)f=w(f,n);else{var
q=e.getFullBytes();f=x(f,q)}h--;break}else
if(e===(e|0)){f=g(f,e+e+1);h--}else
if(e===+e){f=u(f,gw(e));h--;break}}f=t(f);return f&bc}}();function
s_(a){return[a[3]>>8,a[3]&s,a[2]>>16,a[2]>>8&s,a[2]&s,a[1]>>16,a[1]>>8&s,a[1]&s]}function
s0(e,b,c){var
d=0;function
f(a){b--;if(e<0||b<0)return;if(a
instanceof
Array&&a[0]===(a[0]|0))switch(a[0]){case
bS:e--;d=d*dj+a[2]|0;break;case
aI:b++;f(a);break;case
s:e--;d=d*dj+a[1]+(a[2]<<24)|0;break;default:e--;d=d*19+a[0]|0;for(var
c=a.length-1;c>0;c--)f(a[c])}else
if(a
instanceof
G){e--;var
g=a.array,h=a.getLen();if(g)for(var
c=0;c<h;c++)d=d*19+g[c]|0;else{var
i=a.getFullBytes();for(var
c=0;c<h;c++)d=d*19+i.charCodeAt(c)|0}}else
if(a===(a|0)){e--;d=d*dj+a|0}else
if(a===+a){e--;var
j=s_(gw(a));for(var
c=7;c>=0;c--)d=d*19+j[c]|0}}f(c);return d&bc}function
s2(a){var
c=(a[3]&32767)>>4;if(c==2047)return(a[1]|a[2]|a[3]&15)==0?a[3]&c9?-Infinity:Infinity:NaN;var
d=Math.pow(2,-24),b=(a[1]*d+a[2])*d+(a[3]&15);if(c>0){b+=16;b*=Math.pow(2,c-gd)}else
b*=Math.pow(2,-gp);if(a[3]&c9)b=-b;return b}function
s5(a){return(a[3]|a[2]|a[1])==0}function
s8(a){return[s,a&ab,a>>24&ab,a>>31&b3]}function
s9(a,b){var
c=a[1]-b[1],d=a[2]-b[2]+(c>>24),e=a[3]-b[3]+(d>>24);return[s,c&ab,d&ab,e&b3]}function
gy(a,b){if(a[3]>b[3])return 1;if(a[3]<b[3])return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
gx(a){a[3]=a[3]<<1|a[2]>>23;a[2]=(a[2]<<1|a[1]>>23)&ab;a[1]=a[1]<<1&ab}function
s6(a){a[1]=(a[1]>>>1|a[2]<<23)&ab;a[2]=(a[2]>>>1|a[3]<<23)&ab;a[3]=a[3]>>>1}function
ta(a,b){var
e=0,d=a.slice(),c=b.slice(),f=[s,0,0,0];while(gy(d,c)>0){e++;gx(c)}while(e>=0){e--;gx(f);if(gy(d,c)>=0){f[1]++;d=s9(d,c)}s6(c)}return[0,f,d]}function
s$(a){return a[1]|a[2]<<24}function
s4(a){return a[3]<<16<0}function
s7(a){var
b=-a[1],c=-a[2]+(b>>24),d=-a[3]+(c>>24);return[s,b&ab,c&ab,d&b3]}function
s3(a,b){var
c=dE(a);if(c.signedconv&&s4(b)){c.sign=-1;b=s7(b)}var
d=g,i=s8(c.base),h="0123456789abcdef";do{var
f=ta(b,i);b=f[1];d=h.charAt(s$(f[2]))+d}while(!s5(b));if(c.prec>=0){c.filler=D;var
e=c.prec-d.length;if(e>0)d=bj(e,Z)+d}return dC(c,d)}function
tw(a){var
b=0,c=10,d=a.get(0)==45?(b++,-1):1;if(a.get(b)==48)switch(a.get(b+1)){case
ds:case
88:c=16;b+=2;break;case
dd:case
79:c=8;b+=2;break;case
98:case
66:c=2;b+=2;break}return[b,d,c]}function
gB(a){if(a>=48&&a<=57)return a-48;if(a>=65&&a<=90)return a-55;if(a>=97&&a<=122)return a-87;return-1}function
p(a){dG(r[3],a)}function
tb(a){var
g=tw(a),e=g[0],h=g[1],f=g[2],i=-1>>>0,d=a.get(e),c=gB(d);if(c<0||c>=f)p(bP);var
b=c;for(;;){e++;d=a.get(e);if(d==95)continue;c=gB(d);if(c<0||c>=f)break;b=f*b+c;if(b>i)p(bP)}if(e!=a.getLen())p(bP);b=h*b;if((b|0)!=b)p(bP);return b}function
tc(a){return+(a>31&&a<127)}var
b9={amp:/&/g,lt:/</g,quot:/\"/g,all:/[&<\"]/};function
td(a){if(!b9.all.test(a))return a;return a.replace(b9.amp,"&amp;").replace(b9.lt,"&lt;").replace(b9.quot,"&quot;")}function
te(a){var
c=Array.prototype.slice;return function(){var
b=arguments.length>0?c.call(arguments):[undefined];return ah(a,b)}}function
tf(a,b){var
d=[0];for(var
c=1;c<=a;c++)d[c]=b;return d}function
dy(a){var
b=a.length;this.array=a;this.len=this.last=b}dy.prototype=new
G();var
tg=function(){function
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
n=0;n<4;n++)o[g*4+n]=l[g]>>8*n&s;return o}return function(a,b,c){var
h=[];if(a.array){var
f=a.array;for(var
d=0;d<c;d+=4){var
e=d+b;h[d>>2]=f[e]|f[e+1]<<8|f[e+2]<<16|f[e+3]<<24}for(;d<c;d++)h[d>>2]|=f[d+b]<<8*(d&3)}else{var
g=a.getFullBytes();for(var
d=0;d<c;d+=4){var
e=d+b;h[d>>2]=g.charCodeAt(e)|g.charCodeAt(e+1)<<8|g.charCodeAt(e+2)<<16|g.charCodeAt(e+3)<<24}for(;d<c;d++)h[d>>2]|=g.charCodeAt(d+b)<<8*(d&3)}return new
dy(n(h,c))}}();function
th(a){return a.data.array.length}function
ap(a){dG(r[2],a)}function
dD(a){if(!a.opened)ap("Cannot flush a closed channel");if(a.buffer==g)return 0;if(a.output){switch(a.output.length){case
2:a.output(a,a.buffer);break;default:a.output(a.buffer)}}a.buffer=g}var
bh=new
Array();function
ti(a){dD(a);a.opened=false;delete
bh[a.fd];return 0}function
tj(a,b,c,d){var
e=a.data.array.length-a.data.offset;if(e<d)d=e;dA(new
dy(a.data.array),a.data.offset,b,c,d);a.data.offset+=d;return d}function
tx(){dF(r[5])}function
tk(a){if(a.data.offset>=a.data.array.length)tx();if(a.data.offset<0||a.data.offset>a.data.array.length)aK();var
b=a.data.array[a.data.offset];a.data.offset++;return b}function
tl(a){var
b=a.data.offset,c=a.data.array.length;if(b>=c)return 0;while(true){if(b>=c)return-(b-a.data.offset);if(b<0||b>a.data.array.length)aK();if(a.data.array[b]==10)return b-a.data.offset+1;b++}}function
tz(a,b){if(!r.files)r.files={};if(b
instanceof
G)var
c=b.getArray();else
if(b
instanceof
Array)var
c=b;else
var
c=new
G(b).getArray();r.files[a
instanceof
G?a.toString():a]=c}function
tG(a){return r.files&&r.files[a.toString()]?1:r.auto_register_file===undefined?0:r.auto_register_file(a)}function
bk(a,b,c){if(r.fds===undefined)r.fds=new
Array();c=c?c:{};var
d={};d.array=b;d.offset=c.append?d.array.length:0;d.flags=c;r.fds[a]=d;r.fd_last_idx=a;return a}function
tK(a,b,c){var
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
e=a.toString();if(d.rdonly&&d.wronly)ap(e+" : flags Open_rdonly and Open_wronly are not compatible");if(d.text&&d.binary)ap(e+" : flags Open_text and Open_binary are not compatible");if(tG(a)){if(d.create&&d.excl)ap(e+" : file already exists");var
f=r.fd_last_idx?r.fd_last_idx:0;if(d.truncate)r.files[e]=g;return bk(f+1,r.files[e],d)}else
if(d.create){var
f=r.fd_last_idx?r.fd_last_idx:0;tz(e,[]);return bk(f+1,r.files[e],d)}else
ap(e+": no such file or directory")}bk(0,[]);bk(1,[]);bk(2,[]);function
tm(a){var
b=r.fds[a];if(b.flags.wronly)ap(go+a+" is writeonly");return{data:b,fd:a,opened:true}}function
tR(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=d.console;b&&b.log&&b.log(a)}function
tC(a,b){var
e=new
G(b),d=e.getLen();for(var
c=0;c<d;c++)a.data.array[a.data.offset+c]=e.get(c);a.data.offset+=d;return 0}function
tn(a){var
b;switch(a){case
1:b=tR;break;case
2:b=dH;break;default:b=tC}var
d=r.fds[a];if(d.flags.rdonly)ap(go+a+" is readonly");var
c={data:d,fd:a,opened:true,buffer:g,output:b};bh[c.fd]=c;return c}function
to(){var
a=0;for(var
b
in
bh)if(bh[b].opened)a=[0,bh[b],a];return a}function
gA(a,b,c,d){if(!a.opened)ap("Cannot output to a closed channel");var
f;if(c==0&&b.getLen()==d)f=b;else{f=gv(d);dA(b,c,f,0,d)}var
e=f.toString(),g=e.lastIndexOf("\n");if(g<0)a.buffer+=e;else{a.buffer+=e.substr(0,g+1);dD(a);a.buffer+=e.substr(g+1)}}function
T(a){return new
G(a)}function
tp(a,b){var
c=T(String.fromCharCode(b));gA(a,c,0,1)}function
tq(a,b){if(b==0)gD();return a%b}function
ts(a,b){return+(dB(a,b,false)!=0)}function
tt(a,b){var
d=[a];for(var
c=1;c<=b;c++)d[c]=0;return d}function
tu(a,b){a[0]=b;return 0}function
tv(a){return a
instanceof
Array?a[0]:fF}function
tA(a,b){r[a+1]=b}var
tr={};function
tB(a,b){tr[a]=b;return 0}function
tD(a,b){return a.compare(b)}function
gE(a,b){var
c=a.fullBytes,d=b.fullBytes;if(c!=null&&d!=null)return c==d?1:0;return a.getFullBytes()==b.getFullBytes()?1:0}function
tE(a,b){return 1-gE(a,b)}function
tF(){return 32}function
tH(){var
a=new
an("a.out");return[0,a,[0,a]]}function
tI(){return[0,new
an(fw),32,0]}function
ty(){dF(r[7])}function
tJ(){ty()}function
tL(){var
a=new
Date()^4294967295*Math.random();return{valueOf:function(){return a},0:0,1:a,length:2}}function
tM(){console.log("caml_sys_system_command");return 0}function
tN(a){var
b=1;while(a&&a.joo_tramp){a=a.joo_tramp.apply(null,a.joo_args);b++}return a}function
tO(a,b){return{joo_tramp:a,joo_args:b}}function
tP(a,b){if(typeof
b==="function"){a.fun=b;return 0}if(b.fun){a.fun=b.fun;return 0}var
c=b.length;while(c--)a[c]=b[c];return 0}function
tQ(){return 0}var
dI=0;function
tS(){if(window.webcl==undefined){alert("Unfortunately your system does not support WebCL. "+"Make sure that you have both the OpenCL driver "+"and the WebCL browser extension installed.");dI=1}else{console.log("INIT OPENCL");dI=0}return 0}function
tU(){console.log(" spoc_cuInit");return 0}function
tV(){console.log(" spoc_cuda_compile");return 0}function
t3(){console.log(" spoc_cuda_debug_compile");return 0}function
ud(a,b,c){console.log(" spoc_debug_opencl_compile");console.log(a.bytes);var
e=c[9],f=e[0],d=f.createProgram(a.bytes),g=d.getInfo(WebCL.PROGRAM_DEVICES);d.build(g);var
h=d.createKernel(b.bytes);e[0]=f;c[9]=e;return h}function
ue(a){console.log("spoc_getCudaDevice");return 0}function
uf(){console.log(" spoc_getCudaDevicesCount");return 0}function
ug(a,b){console.log(" spoc_getOpenCLDevice");var
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
f=k[o],j=f.getDevices(),m=j.length;console.log("there "+g+D+m+D+a);if(g+m>=a)for(var
q
in
j){var
c=j[q];if(g==a){console.log("current ----------"+g);e[1]=T(c.getInfo(WebCL.DEVICE_NAME));console.log(e[1]);e[2]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_SIZE);e[3]=c.getInfo(WebCL.DEVICE_LOCAL_MEM_SIZE);e[4]=c.getInfo(WebCL.DEVICE_MAX_CLOCK_FREQUENCY);e[5]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE);e[6]=c.getInfo(WebCL.DEVICE_MAX_COMPUTE_UNITS);e[7]=c.getInfo(WebCL.DEVICE_ERROR_CORRECTION_SUPPORT);e[8]=b;var
i=new
Array(3);i[0]=webcl.createContext(c);i[1]=i[0].createCommandQueue();i[2]=i[0].createCommandQueue();e[9]=i;h[1]=T(f.getInfo(WebCL.PLATFORM_PROFILE));h[2]=T(f.getInfo(WebCL.PLATFORM_VERSION));h[3]=T(f.getInfo(WebCL.PLATFORM_NAME));h[4]=T(f.getInfo(WebCL.PLATFORM_VENDOR));h[5]=T(f.getInfo(WebCL.PLATFORM_EXTENSIONS));h[6]=m;var
l=c.getInfo(WebCL.DEVICE_TYPE),v=0;if(l&WebCL.DEVICE_TYPE_CPU)d[2]=0;if(l&WebCL.DEVICE_TYPE_GPU)d[2]=1;if(l&WebCL.DEVICE_TYPE_ACCELERATOR)d[2]=2;if(l&WebCL.DEVICE_TYPE_DEFAULT)d[2]=3;d[3]=T(c.getInfo(WebCL.DEVICE_PROFILE));d[4]=T(c.getInfo(WebCL.DEVICE_VERSION));d[5]=T(c.getInfo(WebCL.DEVICE_VENDOR));var
r=c.getInfo(WebCL.DEVICE_EXTENSIONS);d[6]=T(r);d[7]=c.getInfo(WebCL.DEVICE_VENDOR_ID);d[8]=c.getInfo(WebCL.DEVICE_MAX_WORK_ITEM_DIMENSIONS);d[9]=c.getInfo(WebCL.DEVICE_ADDRESS_BITS);d[10]=c.getInfo(WebCL.DEVICE_MAX_MEM_ALLOC_SIZE);d[11]=c.getInfo(WebCL.DEVICE_IMAGE_SUPPORT);d[12]=c.getInfo(WebCL.DEVICE_MAX_READ_IMAGE_ARGS);d[13]=c.getInfo(WebCL.DEVICE_MAX_WRITE_IMAGE_ARGS);d[14]=c.getInfo(WebCL.DEVICE_MAX_SAMPLERS);d[15]=c.getInfo(WebCL.DEVICE_MEM_BASE_ADDR_ALIGN);d[17]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHELINE_SIZE);d[18]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHE_SIZE);d[19]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_ARGS);d[20]=c.getInfo(WebCL.DEVICE_ENDIAN_LITTLE);d[21]=c.getInfo(WebCL.DEVICE_AVAILABLE);d[22]=c.getInfo(WebCL.DEVICE_COMPILER_AVAILABLE);d[23]=c.getInfo(WebCL.DEVICE_SINGLE_FP_CONFIG);d[24]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHE_TYPE);d[25]=c.getInfo(WebCL.DEVICE_QUEUE_PROPERTIES);d[26]=c.getInfo(WebCL.DEVICE_LOCAL_MEM_TYPE);d[28]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE);d[29]=c.getInfo(WebCL.DEVICE_EXECUTION_CAPABILITIES);d[31]=c.getInfo(WebCL.DEVICE_MAX_WORK_GROUP_SIZE);d[32]=c.getInfo(WebCL.DEVICE_IMAGE2D_MAX_HEIGHT);d[33]=c.getInfo(WebCL.DEVICE_IMAGE2D_MAX_WIDTH);d[34]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_DEPTH);d[35]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_HEIGHT);d[36]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_WIDTH);d[37]=c.getInfo(WebCL.DEVICE_MAX_PARAMETER_SIZE);d[38]=[0];var
n=c.getInfo(WebCL.DEVICE_MAX_WORK_ITEM_SIZES);d[38][1]=n[0];d[38][2]=n[1];d[38][3]=n[2];d[39]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);d[40]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);d[41]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_INT);d[42]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_LONG);d[43]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);d[45]=c.getInfo(WebCL.DEVICE_PROFILING_TIMER_RESOLUTION);d[46]=T(c.getInfo(WebCL.DRIVER_VERSION));g++;break}else
g++}else
g+=m}var
c=[0];d[1]=h;p[1]=d;c[1]=e;c[2]=p;return c}function
uh(){console.log(" spoc_getOpenCLDevicesCount");var
a=0,b=webcl.getPlatforms();for(var
d
in
b){var
e=b[d],c=e.getDevices();a+=c.length}return a}function
ui(){console.log(fu);return 0}function
uj(){console.log(fu);var
a=new
Array(3);a[0]=0;return a}function
dJ(a){if(a[1]instanceof
Float32Array||a[1].constructor.name=="Float32Array")return 4;else{console.log("unimplemented vector type");console.log(a[1].constructor.name);return 4}}function
uk(a,b,c){console.log("spoc_opencl_alloc_vect");var
f=a[2],i=a[4],h=i[b+1],j=a[5],k=dJ(f),d=c[9],e=d[0],d=c[9],e=d[0],g=e.createBuffer(WebCL.MEM_READ_WRITE,j*k);h[2]=g;d[0]=e;c[9]=d;return 0}function
ul(){console.log(" spoc_opencl_compile");return 0}function
um(a,b,c,d){console.log("spoc_opencl_cpu_to_device");var
f=a[2],k=a[4],j=k[b+1],l=a[5],m=dJ(f),e=c[9],h=e[0],g=e[d+1],i=j[2];g.enqueueWriteBuffer(i,false,0,l*m,f[1]);e[d+1]=g;e[0]=h;c[9]=e;return 0}function
us(a,b,c,d,e){console.log("spoc_opencl_device_to_cpu");var
g=a[2],l=a[4],k=l[b+1],n=a[5],o=dJ(g),f=c[9],i=f[0],h=f[e+1],j=k[2],m=g[1];h.enqueueReadBuffer(j,false,0,n*o,m);f[e+1]=h;f[0]=i;c[9]=f;return 0}function
ut(a,b){console.log("spoc_opencl_flush");var
c=a[9][b+1];c.flush();a[9][b+1]=c;return 0}function
uu(){console.log(" spoc_opencl_is_available");return!dI}function
uv(a,b,c,d,e){console.log("spoc_opencl_launch_grid");var
m=b[1],n=b[2],o=b[3],h=c[1],i=c[2],j=c[3],g=new
Array(3);g[0]=m*h;g[1]=n*i;g[2]=o*j;var
f=new
Array(3);f[0]=h;f[1]=i;f[2]=j;var
l=d[9],k=l[e+1];console.log(b);console.log(c);console.log(g);console.log(f);console.log("spoc_opencl_launch_grid----------");console.log(a);k.finish();k.enqueueNDRangeKernel(a,f.length,null,g,f);console.log("spoc_opencl_launch_grid+++++++++++++");return 0}function
uy(a,b,c,d){console.log("spoc_opencl_load_param_int");b.setArg(a[1],new
Uint32Array([c]));a[1]=a[1]+1;return 0}function
uA(a,b,c,d,e){console.log("spoc_opencl_load_param_vec");var
f=d[2];b.setArg(a[1],f);a[1]=a[1]+1;return 0}function
uD(){return new
Date().getTime()/fF}function
uE(){return 0}var
m=sK,j=sL,a7=dA,av=gu,C=gv,aw=sQ,cX=sT,bH=sU,a8=sW,aa=sX,e$=tb,c1=tc,fk=td,v=tf,e_=ti,cZ=dD,c3=tj,e8=tm,cY=tn,aF=tq,w=bi,b=T,c2=ts,fc=tt,aE=tA,c0=tB,fb=tD,bK=gE,x=tE,bI=tJ,e9=tK,fa=tL,fj=tM,Y=tN,F=tO,fi=tQ,fl=tS,fn=tU,fo=uf,fm=uh,ff=ui,fe=uj,fg=uk,fd=ut,bL=uE;function
k(a,b){return a.length==1?a(b):ah(a,[b])}function
i(a,b,c){return a.length==2?a(b,c):ah(a,[b,c])}function
q(a,b,c,d){return a.length==3?a(b,c,d):ah(a,[b,c,d])}function
fh(a,b,c,d,e,f,g){return a.length==6?a(b,c,d,e,f,g):ah(a,[b,c,d,e,f,g])}var
aL=[0,b("Failure")],bl=[0,b("Invalid_argument")],bm=[0,b("End_of_file")],t=[0,b("Not_found")],E=[0,b("Assert_failure")],cA=b(ag),cD=b(ag),cF=b(ag),eM=b(g),eL=[0,b(f$),b(f5),b(fQ),b(gi),b(gf)],e7=[0,1],e1=[0,b(gi),b(f5),b(f$),b(fQ),b(gf)],e2=[0,b(dq),b(c6),b(c_),b(da),b(dg),b(db),b(dn),b(dv),b(dc),b(df)],e3=[0,b(di),b(dt),b(dk)],cU=[0,b(dt),b(c_),b(da),b(dk),b(di),b(c6),b(dv),b(df),b(dq),b(db),b(dn),b(dg),b(dc)];aE(6,t);aE(5,[0,b("Division_by_zero")]);aE(4,bm);aE(3,bl);aE(2,aL);aE(1,[0,b("Sys_error")]);var
gQ=b("really_input"),gP=[0,0,[0,7,0]],gO=[0,1,[0,3,[0,4,[0,7,0]]]],gN=b(f3),gM=b(ag),gK=b("true"),gL=b("false"),gI=[s,0,0,fP],gR=b("Pervasives.do_at_exit"),gV=[0,b("array.ml"),163,4],gT=b("Array.blit"),gU=b("Array.Bottom"),gZ=b("List.iter2"),gX=b("tl"),gW=b("hd"),g3=b("\\b"),g4=b("\\t"),g5=b("\\n"),g6=b("\\r"),g2=b("\\\\"),g1=b("\\'"),g0=b("Char.chr"),g9=b("String.contains_from"),g8=b("String.blit"),g7=b("String.sub"),hg=b("Map.remove_min_elt"),hh=[0,0,0,0],hi=[0,b("map.ml"),270,10],hj=[0,0,0],hc=b(bU),hd=b(bU),he=b(bU),hf=b(bU),hk=b("CamlinternalLazy.Undefined"),hn=b("Buffer.add: cannot grow buffer"),hD=b(g),hE=b(g),hH=b(f3),hI=b(b0),hJ=b(b0),hF=b(bX),hG=b(bX),hC=b(fC),hA=b("neg_infinity"),hB=b("infinity"),hz=b(ag),hy=b("printf: bad positional specification (0)."),hx=b("%_"),hw=[0,b("printf.ml"),143,8],hu=b(bX),hv=b("Printf: premature end of format string '"),hq=b(bX),hr=b(" in format string '"),hs=b(", at char number "),ht=b("Printf: bad conversion %"),ho=b("Sformat.index_of_int: negative argument "),hL=b(dp),hM=[0,987910699,495797812,364182224,414272206,318284740,990407751,383018966,270373319,840823159,24560019,536292337,512266505,189156120,730249596,143776328,51606627,140166561,366354223,1003410265,700563762,981890670,913149062,526082594,1021425055,784300257,667753350,630144451,949649812,48546892,415514493,258888527,511570777,89983870,283659902,308386020,242688715,482270760,865188196,1027664170,207196989,193777847,619708188,671350186,149669678,257044018,87658204,558145612,183450813,28133145,901332182,710253903,510646120,652377910,409934019,801085050],sE=b("OCAMLRUNPARAM"),sC=b("CAMLRUNPARAM"),hN=b(g),h_=[0,b("camlinternalOO.ml"),287,50],h9=b(g),hP=b("CamlinternalOO.last_id"),iD=b(g),iA=b(fS),iz=b(".\\"),iy=b(f4),ix=b("..\\"),ip=b(fS),io=b(f4),ij=b(g),ii=b(g),ik=b(c$),il=b(fB),sA=b("TMPDIR"),ir=b("/tmp"),is=b("'\\''"),iv=b(c$),iw=b("\\"),sy=b("TEMP"),iB=b(ag),iG=b(c$),iH=b(fB),iK=b("Cygwin"),iL=b(fw),iM=b("Win32"),iN=[0,b("filename.ml"),189,9],iU=b("E2BIG"),iW=b("EACCES"),iX=b("EAGAIN"),iY=b("EBADF"),iZ=b("EBUSY"),i0=b("ECHILD"),i1=b("EDEADLK"),i2=b("EDOM"),i3=b("EEXIST"),i4=b("EFAULT"),i5=b("EFBIG"),i6=b("EINTR"),i7=b("EINVAL"),i8=b("EIO"),i9=b("EISDIR"),i_=b("EMFILE"),i$=b("EMLINK"),ja=b("ENAMETOOLONG"),jb=b("ENFILE"),jc=b("ENODEV"),jd=b("ENOENT"),je=b("ENOEXEC"),jf=b("ENOLCK"),jg=b("ENOMEM"),jh=b("ENOSPC"),ji=b("ENOSYS"),jj=b("ENOTDIR"),jk=b("ENOTEMPTY"),jl=b("ENOTTY"),jm=b("ENXIO"),jn=b("EPERM"),jo=b("EPIPE"),jp=b("ERANGE"),jq=b("EROFS"),jr=b("ESPIPE"),js=b("ESRCH"),jt=b("EXDEV"),ju=b("EWOULDBLOCK"),jv=b("EINPROGRESS"),jw=b("EALREADY"),jx=b("ENOTSOCK"),jy=b("EDESTADDRREQ"),jz=b("EMSGSIZE"),jA=b("EPROTOTYPE"),jB=b("ENOPROTOOPT"),jC=b("EPROTONOSUPPORT"),jD=b("ESOCKTNOSUPPORT"),jE=b("EOPNOTSUPP"),jF=b("EPFNOSUPPORT"),jG=b("EAFNOSUPPORT"),jH=b("EADDRINUSE"),jI=b("EADDRNOTAVAIL"),jJ=b("ENETDOWN"),jK=b("ENETUNREACH"),jL=b("ENETRESET"),jM=b("ECONNABORTED"),jN=b("ECONNRESET"),jO=b("ENOBUFS"),jP=b("EISCONN"),jQ=b("ENOTCONN"),jR=b("ESHUTDOWN"),jS=b("ETOOMANYREFS"),jT=b("ETIMEDOUT"),jU=b("ECONNREFUSED"),jV=b("EHOSTDOWN"),jW=b("EHOSTUNREACH"),jX=b("ELOOP"),jY=b("EOVERFLOW"),jZ=b("EUNKNOWNERR %d"),iV=b("Unix.Unix_error(Unix.%s, %S, %S)"),iQ=b(fU),iR=b(g),iS=b(g),iT=b(fU),j0=b("0.0.0.0"),j1=b("127.0.0.1"),sx=b("::"),sw=b("::1"),j$=[0,b("Vector.ml"),fY,25],ka=b("Cuda.No_Cuda_Device"),kb=b("Cuda.ERROR_DEINITIALIZED"),kc=b("Cuda.ERROR_NOT_INITIALIZED"),kd=b("Cuda.ERROR_INVALID_CONTEXT"),ke=b("Cuda.ERROR_INVALID_VALUE"),kf=b("Cuda.ERROR_OUT_OF_MEMORY"),kg=b("Cuda.ERROR_INVALID_DEVICE"),kh=b("Cuda.ERROR_NOT_FOUND"),ki=b("Cuda.ERROR_FILE_NOT_FOUND"),kj=b("Cuda.ERROR_UNKNOWN"),kk=b("Cuda.ERROR_LAUNCH_FAILED"),kl=b("Cuda.ERROR_LAUNCH_OUT_OF_RESOURCES"),km=b("Cuda.ERROR_LAUNCH_TIMEOUT"),kn=b("Cuda.ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"),ko=b("no_cuda_device"),kp=b("cuda_error_deinitialized"),kq=b("cuda_error_not_initialized"),kr=b("cuda_error_invalid_context"),ks=b("cuda_error_invalid_value"),kt=b("cuda_error_out_of_memory"),ku=b("cuda_error_invalid_device"),kv=b("cuda_error_not_found"),kw=b("cuda_error_file_not_found"),kx=b("cuda_error_launch_failed"),ky=b("cuda_error_launch_out_of_resources"),kz=b("cuda_error_launch_timeout"),kA=b("cuda_error_launch_incompatible_texturing"),kB=b("cuda_error_unknown"),kC=b("OpenCL.No_OpenCL_Device"),kD=b("OpenCL.OPENCL_ERROR_UNKNOWN"),kE=b("OpenCL.INVALID_CONTEXT"),kF=b("OpenCL.INVALID_DEVICE"),kG=b("OpenCL.INVALID_VALUE"),kH=b("OpenCL.INVALID_QUEUE_PROPERTIES"),kI=b("OpenCL.OUT_OF_RESOURCES"),kJ=b("OpenCL.MEM_OBJECT_ALLOCATION_FAILURE"),kK=b("OpenCL.OUT_OF_HOST_MEMORY"),kL=b("OpenCL.FILE_NOT_FOUND"),kM=b("OpenCL.INVALID_PROGRAM"),kN=b("OpenCL.INVALID_BINARY"),kO=b("OpenCL.INVALID_BUILD_OPTIONS"),kP=b("OpenCL.INVALID_OPERATION"),kQ=b("OpenCL.COMPILER_NOT_AVAILABLE"),kR=b("OpenCL.BUILD_PROGRAM_FAILURE"),kS=b("OpenCL.INVALID_KERNEL"),kT=b("OpenCL.INVALID_ARG_INDEX"),kU=b("OpenCL.INVALID_ARG_VALUE"),kV=b("OpenCL.INVALID_MEM_OBJECT"),kW=b("OpenCL.INVALID_SAMPLER"),kX=b("OpenCL.INVALID_ARG_SIZE"),kY=b("OpenCL.INVALID_COMMAND_QUEUE"),kZ=b("no_opencl_device"),k0=b("opencl_error_unknown"),k1=b("opencl_invalid_context"),k2=b("opencl_invalid_device"),k3=b("opencl_invalid_value"),k4=b("opencl_invalid_queue_properties"),k5=b("opencl_out_of_resources"),k6=b("opencl_mem_object_allocation_failure"),k7=b("opencl_out_of_host_memory"),k8=b("opencl_file_not_found"),k9=b("opencl_invalid_program"),k_=b("opencl_invalid_binary"),k$=b("opencl_invalid_build_options"),la=b("opencl_invalid_operation"),lb=b("opencl_compiler_not_available"),lc=b("opencl_build_program_failure"),ld=b("opencl_invalid_kernel"),le=b("opencl_invalid_arg_index"),lf=b("opencl_invalid_arg_value"),lg=b("opencl_invalid_mem_object"),lh=b("opencl_invalid_sampler"),li=b("opencl_invalid_arg_size"),lj=b("opencl_invalid_command_queue"),lk=b(b5),ll=b(b5),lC=b(fL),lB=b(fG),lA=b(fL),lz=b(fG),ly=[0,1],lx=b(g),lt=b(bO),lo=b("Cl LOAD ARG Type Not Implemented\n"),ln=b("CU LOAD ARG Type Not Implemented\n"),lm=[0,b(df),b(dc),b(dv),b(dn),b(db),b(di),b(dg),b(da),b(c_),b(dt),b(c6),b(dq),b(dk)],lp=b("Kernel.ERROR_BLOCK_SIZE"),lr=b("Kernel.ERROR_GRID_SIZE"),lu=b("Kernel.No_source_for_device"),lF=b("Empty"),lG=b("Unit"),lH=b("Kern"),lI=b("Params"),lJ=b("Plus"),lK=b("Plusf"),lL=b("Min"),lM=b("Minf"),lN=b("Mul"),lO=b("Mulf"),lP=b("Div"),lQ=b("Divf"),lR=b("Mod"),lS=b("Id "),lT=b("IdName "),lU=b("IntVar "),lV=b("FloatVar "),lW=b("UnitVar "),lX=b("CastDoubleVar "),lY=b("DoubleVar "),lZ=b("IntArr"),l0=b("Int32Arr"),l1=b("Int64Arr"),l2=b("Float32Arr"),l3=b("Float64Arr"),l4=b("VecVar "),l5=b("Concat"),l6=b("Seq"),l7=b("Return"),l8=b("Set"),l9=b("Decl"),l_=b("SetV"),l$=b("SetLocalVar"),ma=b("Intrinsics"),mb=b(D),mc=b("IntId "),md=b("Int "),mf=b("IntVecAcc"),mg=b("Local"),mh=b("Acc"),mi=b("Ife"),mj=b("If"),mk=b("Or"),ml=b("And"),mm=b("EqBool"),mn=b("LtBool"),mo=b("GtBool"),mp=b("LtEBool"),mq=b("GtEBool"),mr=b("DoLoop"),ms=b("While"),mt=b("App"),mu=b("GInt"),mv=b("GFloat"),me=b("Float "),lE=b("  "),lD=b("%s\n"),n9=b(f1),n_=[0,b(dh),166,14],my=b(g),mz=b(bO),mA=b("\n}\n#ifdef __cplusplus\n}\n#endif"),mB=b(" ) {\n"),mC=b(g),mD=b(bN),mF=b(g),mE=b('#ifdef __cplusplus\nextern "C" {\n#endif\n\n__global__ void spoc_dummy ( '),mG=b(ae),mH=b(b6),mI=b(af),mJ=b(ae),mK=b(b6),mL=b(af),mM=b(ae),mN=b(bV),mO=b(af),mP=b(ae),mQ=b(bV),mR=b(af),mS=b(ae),mT=b(bZ),mU=b(af),mV=b(ae),mW=b(bZ),mX=b(af),mY=b(ae),mZ=b(b8),m0=b(af),m1=b(ae),m2=b(b8),m3=b(af),m4=b(ae),m5=b(fA),m6=b(af),m7=b(f7),m8=b(fy),m9=[0,b(dh),65,17],m_=b(bW),m$=b(fM),na=b(L),nb=b(M),nc=b(fT),nd=b(L),ne=b(M),nf=b(ft),ng=b(L),nh=b(M),ni=b(fH),nj=b(L),nk=b(M),nl=b(f8),nm=b(f2),no=b("int"),np=b("float"),nq=b("double"),nn=[0,b(dh),60,12],ns=b(bN),nr=b(gn),nt=b(f0),nu=b(g),nv=b(g),ny=b(bQ),nz=b(ad),nA=b(aG),nC=b(bQ),nB=b(ad),nD=b(_),nE=b(L),nF=b(M),nG=b("}\n"),nH=b(aG),nI=b(aG),nJ=b("{"),nK=b(be),nL=b(fE),nM=b(be),nN=b(bb),nO=b(b7),nP=b(be),nQ=b(bb),nR=b(b7),nS=b(fr),nT=b(fp),nU=b(fJ),nV=b(ge),nW=b(fR),nX=b(bM),nY=b(gc),nZ=b(b4),n0=b(fv),n1=b(bT),n2=b(bM),n3=b(bT),n4=b(ad),n5=b(fq),n6=b(b4),n7=b(bb),n8=b(fO),ob=b(bY),oc=b(bY),od=b(D),oe=b(D),n$=b(fX),oa=b(gq),of=b(_),nw=b(bQ),nx=b(ad),og=b(L),oh=b(M),oj=b(bW),ok=b(_),ol=b(gk),om=b(L),on=b(M),oo=b(_),oi=b("cuda error parse_float"),mw=[0,b(g),b(g)],pK=b(f1),pL=[0,b(dm),162,14],or=b(g),os=b(bO),ot=b(b4),ou=b(" ) \n{\n"),ov=b(g),ow=b(bN),oy=b(g),ox=b("__kernel void spoc_dummy ( "),oz=b(b6),oA=b(b6),oB=b(bV),oC=b(bV),oD=b(bZ),oE=b(bZ),oF=b(b8),oG=b(b8),oH=b(fA),oI=b(f7),oJ=b(fy),oK=[0,b(dm),65,17],oL=b(bW),oM=b(fM),oN=b(L),oO=b(M),oP=b(fT),oQ=b(L),oR=b(M),oS=b(ft),oT=b(L),oU=b(M),oV=b(fH),oW=b(L),oX=b(M),oY=b(f8),oZ=b(f2),o1=b("__global int"),o2=b("__global float"),o3=b("__global double"),o0=[0,b(dm),60,12],o5=b(bN),o4=b(gn),o6=b(f0),o7=b(g),o8=b(g),o_=b(bQ),o$=b(ad),pa=b(aG),pb=b(ad),pc=b(_),pd=b(L),pe=b(M),pf=b(g),pg=b(bO),ph=b(aG),pi=b(g),pj=b(be),pk=b(fE),pl=b(be),pm=b(bb),pn=b(b7),po=b(b4),pp=b(aG),pq=b("{\n"),pr=b(")\n"),ps=b(b7),pt=b(fr),pu=b(fp),pv=b(fJ),pw=b(ge),px=b(fR),py=b(bM),pz=b(gc),pA=b(gg),pB=b(fv),pC=b(bT),pD=b(bM),pE=b(bT),pF=b(ad),pG=b(fq),pH=b(gg),pI=b(bb),pJ=b(fO),pO=b(bY),pP=b(bY),pQ=b(D),pR=b(D),pM=b(fX),pN=b(gq),pS=b(_),o9=b(ad),pT=b(L),pU=b(M),pW=b(bW),pX=b(_),pY=b(gk),pZ=b(L),p0=b(M),p1=b(_),pV=b("opencl error parse_float"),op=[0,b(g),b(g)],q0=[0,0],q1=[0,0],q2=[0,1],q3=[0,1],qU=b("kirc_kernel.cu"),qV=b("nvcc -m64 -arch=sm_10 -O3 -ptx kirc_kernel.cu -o kirc_kernel.ptx"),qW=b("kirc_kernel.ptx"),qX=b("rm kirc_kernel.cu kirc_kernel.ptx"),qR=[0,b(g),b(g)],qT=b(g),qS=[0,b("Kirc.ml"),407,81],qY=b(ad),qZ=b(f_),qO=[34,0],qJ=b(f_),p2=b("int spoc_xor (int a, int b ) { return (a^b);}\n"),p3=b("int spoc_powint (int a, int b ) { return ((int) pow (((float) a), ((float) b)));}\n"),p4=b("int logical_and (int a, int b ) { return (a & b);}\n"),p5=b("float spoc_fdiv ( float a, float b ) { return (a / b);}\n"),p6=b("float spoc_fmul ( float a, float b ) { return (a * b);}\n"),p7=b("float spoc_fminus ( float a, float b ) { return (a - b);}\n"),p8=b("float spoc_fadd ( float a, float b ) { return (a + b);}\n"),p9=b("float spoc_fdiv ( float a, float b );\n"),p_=b("float spoc_fmul ( float a, float b );\n"),p$=b("float spoc_fminus ( float a, float b );\n"),qa=b("float spoc_fadd ( float a, float b );\n"),qc=b(dl),qd=b("double spoc_ddiv ( double a, double b ) { return (a / b);}\n"),qe=b("double spoc_dmul ( double a, double b ) { return (a * b);}\n"),qf=b("double spoc_dminus ( double a, double b ) { return (a - b);}\n"),qg=b("double spoc_dadd ( double a, double b ) { return (a + b);}\n"),qh=b("double spoc_ddiv ( double a, double b );\n"),qi=b("double spoc_dmul ( double a, double b );\n"),qj=b("double spoc_dminus ( double a, double b );\n"),qk=b("double spoc_dadd ( double a, double b );\n"),ql=b(dl),qm=b("#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"),qn=b("#elif defined(cl_amd_fp64)  // AMD extension available?\n"),qo=b("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"),qp=b("#if defined(cl_khr_fp64)  // Khronos extension available?\n"),qq=b(gj),qr=b(ga),qt=b(dl),qu=b("__device__ double spoc_ddiv ( double a, double b ) { return (a / b);}\n"),qv=b("__device__ double spoc_dmul ( double a, double b ) { return (a * b);}\n"),qw=b("__device__ double spoc_dminus ( double a, double b ) { return (a - b);}\n"),qx=b("__device__ double spoc_dadd ( double a, double b ) { return (a + b);}\n"),qy=b(gj),qz=b(ga),qB=b("__device__ int spoc_xor (int a, int b ) { return (a^b);}\n"),qC=b("__device__ int spoc_powint (int a, int b ) { return ((int) pow (((double) a), ((double) b)));}\n"),qD=b("__device__ int logical_and (int a, int b ) { return (a & b);}\n"),qE=b("__device__ float spoc_fdiv ( float a, float b ) { return (a / b);}\n"),qF=b("__device__ float spoc_fmul ( float a, float b ) { return (a * b);}\n"),qG=b("__device__ float spoc_fminus ( float a, float b ) { return (a - b);}\n"),qH=b("__device__ float spoc_fadd ( float a, float b ) { return (a + b);}\n"),qP=[0,b(g),b(g)],rd=b("span"),rc=b("br"),rb=b(fs),ra=b("select"),q$=b("option"),sr=[0,b(fK),199,17],ss=b("This sample computes a bitonic sort over a vector of float"),st=b("Choose a computing device : "),su=b("Vector size :  2^"),sl=b("Will use device : %s  to sort %d floats\n%!"),sm=b("Sequential Array.sort"),sn=[0,1],so=b("Parallel Bitonic"),sq=b("error, %g <  %g"),sp=b("Check OK\n"),sk=b("time %s : %Fs\n%!"),ro=b("spoc_dummy"),rp=b("kirc_kernel"),rm=b("spoc_kernel_extension error"),re=[0,b(fK),12,15],rr=[7,[0,0,0]],rX=b(gr),rY=b(gr),r5=b(fz),r6=b(fz),r9=b("(get_group_id (0))"),r_=b("blockIdx.x"),sa=b("(get_local_size (0))"),sb=b("blockDim.x"),sc=b("(get_local_id (0))"),sd=b("threadIdx.x");function
N(a){throw[0,aL,a]}function
H(a){throw[0,bl,a]}var
gJ=s2(gI);function
h(a,b){var
c=a.getLen(),e=b.getLen(),d=C(c+e|0);a7(a,0,d,0,c);a7(b,0,d,c,e);return d}function
l(a){return b(g+a)}function
O(a){var
c=cX(gN,a),b=0,f=c.getLen();for(;;){if(f<=b)var
e=h(c,gM);else{var
d=c.safeGet(b),g=48<=d?58<=d?0:1:45===d?1:0;if(g){var
b=b+1|0;continue}var
e=c}return e}}function
b$(a,b){if(a){var
c=a[1];return[0,c,b$(a[2],b)]}return b}e8(0);var
dK=cY(1);cY(2);function
dL(a,b){return gA(a,b,0,b.getLen())}function
dM(a){return e8(e9(a,gP,0))}function
dN(a){var
b=to(0);for(;;){if(b){var
c=b[2],d=b[1];try{cZ(d)}catch(f){}var
b=c;continue}return 0}}c0(gR,dN);function
dO(a){return e_(a)}function
gS(a,b){return tp(a,b)}function
dP(a){return cZ(a)}function
dQ(a,b){var
d=b.length-1-1|0,e=0;if(!(d<0)){var
c=e;for(;;){k(a,b[c+1]);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return 0}function
aM(a,b){var
d=b.length-1;if(0===d)return[0];var
e=v(d,k(a,b[0+1])),f=d-1|0,g=1;if(!(f<1)){var
c=g;for(;;){e[c+1]=k(a,b[c+1]);var
h=c+1|0;if(f!==c){var
c=h;continue}break}}return e}function
bn(a,b){var
d=b.length-1-1|0,e=0;if(!(d<0)){var
c=e;for(;;){i(a,c,b[c+1]);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return 0}function
aN(a){var
b=a.length-1-1|0,c=0;for(;;){if(0<=b){var
d=[0,a[b+1],c],b=b-1|0,c=d;continue}return c}}function
dR(a,b,c){var
e=[0,b],f=c.length-1-1|0,g=0;if(!(f<0)){var
d=g;for(;;){e[1]=i(a,e[1],c[d+1]);var
h=d+1|0;if(f!==d){var
d=h;continue}break}}return e[1]}var
ca=[0,gU];function
dS(a){var
b=a,c=0;for(;;){if(b){var
d=[0,b[1],c],b=b[2],c=d;continue}return c}}function
cb(a,b){if(b){var
c=b[2],d=k(a,b[1]);return[0,d,cb(a,c)]}return 0}function
cd(a,b,c){if(b){var
d=b[1];return i(a,d,cd(a,b[2],c))}return c}function
dU(a,b,c){var
e=b,d=c;for(;;){if(e){if(d){var
f=d[2],g=e[2];i(a,e[1],d[1]);var
e=g,d=f;continue}}else
if(!d)return 0;return H(gZ)}}function
ce(a,b){var
c=b;for(;;){if(c){var
e=c[2],d=0===av(c[1],a)?1:0;if(d)return d;var
c=e;continue}return 0}}function
cf(a){if(0<=a)if(!(s<a))return a;return H(g0)}function
dV(a){var
b=65<=a?90<a?0:1:0;if(!b){var
c=192<=a?214<a?0:1:0;if(!c){var
d=216<=a?222<a?1:0:1;if(d)return a}}return a+32|0}function
ai(a,b){var
c=C(a);sS(c,0,a,b);return c}function
u(a,b,c){if(0<=b)if(0<=c)if(!((a.getLen()-c|0)<b)){var
d=C(c);a7(a,b,d,0,c);return d}return H(g7)}function
bp(a,b,c,d,e){if(0<=e)if(0<=b)if(!((a.getLen()-e|0)<b))if(0<=d)if(!((c.getLen()-e|0)<d))return a7(a,b,c,d,e);return H(g8)}function
dW(a){var
c=a.getLen();if(0===c)var
f=a;else{var
d=C(c),e=c-1|0,g=0;if(!(e<0)){var
b=g;for(;;){d.safeSet(b,dV(a.safeGet(b)));var
h=b+1|0;if(e!==b){var
b=h;continue}break}}var
f=d}return f}var
ch=tI(0)[1],az=tF(0),ci=(1<<(az-10|0))-1|0,aO=w(az/8|0,ci)-1|0,g$=tH(0)[2],ha=bS,hb=aI;function
cj(j){function
h(a){return a?a[5]:0}function
e(a,b,c,d){var
e=h(a),f=h(d),g=f<=e?e+1|0:f+1|0;return[0,a,b,c,d,g]}function
p(a,b){return[0,0,a,b,0,1]}function
f(a,b,c,d){var
i=a?a[5]:0,j=d?d[5]:0;if((j+2|0)<i){if(a){var
f=a[4],m=a[3],n=a[2],k=a[1],q=h(f);if(q<=h(k))return e(k,n,m,e(f,b,c,d));if(f){var
r=f[3],s=f[2],t=f[1],u=e(f[4],b,c,d);return e(e(k,n,m,t),s,r,u)}return H(hc)}return H(hd)}if((i+2|0)<j){if(d){var
l=d[4],o=d[3],p=d[2],g=d[1],v=h(g);if(v<=h(l))return e(e(a,b,c,g),p,o,l);if(g){var
w=g[3],x=g[2],y=g[1],z=e(g[4],p,o,l);return e(e(a,b,c,y),x,w,z)}return H(he)}return H(hf)}var
A=j<=i?i+1|0:j+1|0;return[0,a,b,c,d,A]}var
a=0;function
I(a){return a?0:1}function
r(a,b,c){if(c){var
d=c[4],h=c[3],e=c[2],g=c[1],l=c[5],k=i(j[1],a,e);return 0===k?[0,g,a,b,d,l]:0<=k?f(g,e,h,r(a,b,d)):f(r(a,b,g),e,h,d)}return[0,0,a,b,0,1]}function
J(a,b){var
c=b;for(;;){if(c){var
e=c[4],f=c[3],g=c[1],d=i(j[1],a,c[2]);if(0===d)return f;var
h=0<=d?e:g,c=h;continue}throw[0,t]}}function
K(a,b){var
c=b;for(;;){if(c){var
f=c[4],g=c[1],d=i(j[1],a,c[2]),e=0===d?1:0;if(e)return e;var
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
c=a[4],d=a[3],e=a[2];return f(s(b),e,d,c)}return a[4]}return H(hg)}function
u(a,b){if(b){var
c=b[4],k=b[3],e=b[2],d=b[1],l=i(j[1],a,e);if(0===l){if(d)if(c){var
h=n(c),m=h[2],o=h[1],g=f(d,o,m,s(c))}else
var
g=d;else
var
g=c;return g}return 0<=l?f(d,e,k,u(a,c)):f(u(a,d),e,k,c)}return 0}function
y(a,b){var
c=b;for(;;){if(c){var
d=c[4],e=c[3],f=c[2];y(a,c[1]);i(a,f,e);var
c=d;continue}return 0}}function
c(a,b){if(b){var
d=b[5],e=b[4],f=b[3],g=b[2],h=c(a,b[1]),i=k(a,f);return[0,h,g,i,c(a,e),d]}return 0}function
v(a,b){if(b){var
c=b[2],d=b[5],e=b[4],f=b[3],g=v(a,b[1]),h=i(a,c,f);return[0,g,c,h,v(a,e),d]}return 0}function
z(a,b,c){var
d=b,e=c;for(;;){if(d){var
f=d[4],g=d[3],h=d[2],i=q(a,h,g,z(a,d[1],e)),d=f,e=i;continue}return e}}function
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
d=c[4],e=c[3],g=c[2];return f(C(a,b,c[1]),g,e,d)}return p(a,b)}function
D(a,b,c){if(c){var
d=c[3],e=c[2],g=c[1];return f(g,e,d,D(a,b,c[4]))}return p(a,b)}function
g(a,b,c,d){if(a){if(d){var
h=d[5],i=a[5],j=d[4],k=d[3],l=d[2],m=d[1],n=a[4],o=a[3],p=a[2],q=a[1];return(h+2|0)<i?f(q,p,o,g(n,b,c,d)):(i+2|0)<h?f(g(a,b,c,m),l,k,j):e(a,b,c,d)}return D(b,c,a)}return C(b,c,d)}function
o(a,b){if(a){if(b){var
c=n(b),d=c[2],e=c[1];return g(a,e,d,s(b))}return a}return b}function
F(a,b,c,d){return c?g(a,b,c[1],d):o(a,d)}function
l(a,b){if(b){var
c=b[4],d=b[3],e=b[2],f=b[1],m=i(j[1],a,e);if(0===m)return[0,f,[0,d],c];if(0<=m){var
h=l(a,c),n=h[3],o=h[2];return[0,g(f,e,d,h[1]),o,n]}var
k=l(a,f),p=k[2],q=k[1];return[0,q,p,g(k[3],e,d,c)]}return hh}function
m(a,b,c){if(b){var
d=b[2],i=b[5],j=b[4],k=b[3],n=b[1];if(h(c)<=i){var
e=l(d,c),o=e[2],p=e[1],r=m(a,j,e[3]),s=q(a,d,[0,k],o);return F(m(a,n,p),d,s,r)}}else
if(!c)return 0;if(c){var
f=c[2],t=c[4],u=c[3],v=c[1],g=l(f,b),w=g[2],x=g[1],y=m(a,g[3],t),z=q(a,f,w,[0,u]);return F(m(a,x,v),f,z,y)}throw[0,E,hi]}function
w(a,b){if(b){var
c=b[3],d=b[2],h=b[4],e=w(a,b[1]),j=i(a,d,c),f=w(a,h);return j?g(e,d,c,f):o(e,f)}return 0}function
x(a,b){if(b){var
c=b[3],d=b[2],m=b[4],e=x(a,b[1]),f=e[2],h=e[1],n=i(a,d,c),j=x(a,m),k=j[2],l=j[1];if(n){var
p=o(f,k);return[0,g(h,d,c,l),p]}var
q=g(f,d,c,k);return[0,o(h,l),q]}return hj}function
d(a,b){var
c=a,d=b;for(;;){if(c){var
e=[0,c[2],c[3],c[4],d],c=c[1],d=e;continue}return d}}function
M(a,b,c){var
s=d(c,0),f=d(b,0),e=s;for(;;){if(f)if(e){var
l=e[4],m=e[3],n=e[2],o=f[4],p=f[3],q=f[2],h=i(j[1],f[1],e[1]);if(0===h){var
k=i(a,q,n);if(0===k){var
r=d(m,l),f=d(p,o),e=r;continue}var
g=k}else
var
g=h}else
var
g=1;else
var
g=e?-1:0;return g}}function
N(a,b,c){var
t=d(c,0),f=d(b,0),e=t;for(;;){if(f)if(e){var
m=e[4],n=e[3],o=e[2],p=f[4],q=f[3],r=f[2],h=0===i(j[1],f[1],e[1])?1:0;if(h){var
k=i(a,r,o);if(k){var
s=d(n,m),f=d(q,p),e=s;continue}var
l=k}else
var
l=h;var
g=l}else
var
g=0;else
var
g=e?0:1;return g}}function
b(a){if(a){var
c=a[1],d=b(a[4]);return(b(c)+1|0)+d|0}return 0}function
G(a,b){var
d=a,c=b;for(;;){if(c){var
e=c[3],f=c[2],g=c[1],d=[0,[0,f,e],G(d,c[4])],c=g;continue}return d}}return[0,a,I,K,r,p,u,m,M,N,y,z,A,B,w,x,b,function(a){return G(0,a)},n,L,n,l,J,c,v]}var
hl=[0,hk];function
hm(a){throw[0,hl]}function
aP(a){var
b=1<=a?a:1,c=aO<b?aO:b,d=C(c);return[0,d,0,c,d]}function
aQ(a){return u(a[1],0,a[2])}function
dZ(a,b){var
c=[0,a[3]];for(;;){if(c[1]<(a[2]+b|0)){c[1]=2*c[1]|0;continue}if(aO<c[1])if((a[2]+b|0)<=aO)c[1]=aO;else
N(hn);var
d=C(c[1]);bp(a[1],0,d,0,a[2]);a[1]=d;a[3]=c[1];return 0}}function
I(a,b){var
c=a[2];if(a[3]<=c)dZ(a,1);a[1].safeSet(c,b);a[2]=c+1|0;return 0}function
br(a,b){var
c=b.getLen(),d=a[2]+c|0;if(a[3]<d)dZ(a,c);bp(b,0,a[1],a[2],c);a[2]=d;return 0}function
ck(a){return 0<=a?a:N(h(ho,l(a)))}function
d0(a,b){return ck(a+b|0)}var
hp=1;function
d1(a){return d0(hp,a)}function
d2(a){return u(a,0,a.getLen())}function
d3(a,b,c){var
d=h(hr,h(a,hq)),e=h(hs,h(l(b),d));return H(h(ht,h(ai(1,c),e)))}function
aR(a,b,c){return d3(d2(a),b,c)}function
bs(a){return H(h(hv,h(d2(a),hu)))}function
aq(e,b,c,d){function
h(a){if((e.safeGet(a)+aJ|0)<0||9<(e.safeGet(a)+aJ|0))return a;var
b=a+1|0;for(;;){var
c=e.safeGet(b);if(48<=c){if(!(58<=c)){var
b=b+1|0;continue}var
d=0}else
if(36===c){var
f=b+1|0,d=1}else
var
d=0;if(!d)var
f=a;return f}}var
i=h(b+1|0),f=aP((c-i|0)+10|0);I(f,37);var
a=i,g=dS(d);for(;;){if(a<=c){var
j=e.safeGet(a);if(42===j){if(g){var
k=g[2];br(f,l(g[1]));var
a=h(a+1|0),g=k;continue}throw[0,E,hw]}I(f,j);var
a=a+1|0;continue}return aQ(f)}}function
d4(a,b,c,d,e){var
f=aq(b,c,d,e);if(78!==a)if(bd!==a)return f;f.safeSet(f.getLen()-1|0,dr);return f}function
d5(a){return function(c,b){var
m=c.getLen();function
n(a,b){var
o=40===a?41:c8;function
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
f=g===o?e+1|0:aR(c,b,g);break;case
2:break;default:var
f=k(n(g,e+1|0)+1|0)}}return f}var
d=d+1|0;continue}}return k(b)}return n(a,b)}}function
d6(j,b,c){var
m=j.getLen()-1|0;function
s(a){var
l=a;a:for(;;){if(l<m){if(37===j.safeGet(l)){var
e=0,h=l+1|0;for(;;){if(m<h)var
w=bs(j);else{var
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
h=q(b,e,h,ay);continue;default:var
h=h+1|0;continue}var
d=h;b:for(;;){if(m<d)var
f=bs(j);else{var
k=j.safeGet(d);if(fY<=k)var
g=0;else
switch(k){case
78:case
88:case
aH:case
ay:case
dd:case
dr:case
ds:var
f=q(b,e,d,ay),g=1;break;case
69:case
70:case
71:case
gh:case
de:case
dw:var
f=q(b,e,d,de),g=1;break;case
33:case
37:case
44:case
64:var
f=d+1|0,g=1;break;case
83:case
91:case
bg:var
f=q(b,e,d,bg),g=1;break;case
97:case
b1:case
c7:var
f=q(b,e,d,k),g=1;break;case
76:case
fN:case
bd:var
t=d+1|0;if(m<t){var
f=q(b,e,d,ay),g=1}else{var
p=j.safeGet(t)+gl|0;if(p<0||32<p)var
r=1;else
switch(p){case
0:case
12:case
17:case
23:case
29:case
32:var
f=i(c,q(b,e,d,k),ay),g=1,r=0;break;default:var
r=1}if(r){var
f=q(b,e,d,ay),g=1}}break;case
67:case
99:var
f=q(b,e,d,99),g=1;break;case
66:case
98:var
f=q(b,e,d,66),g=1;break;case
41:case
c8:var
f=q(b,e,d,k),g=1;break;case
40:var
f=s(q(b,e,d,k)),g=1;break;case
du:var
u=q(b,e,d,k),v=i(d5(k),j,u),o=u;for(;;){if(o<(v-2|0)){var
o=i(c,o,j.safeGet(o));continue}var
d=v-1|0;continue b}default:var
g=0}if(!g)var
f=aR(j,d,k)}var
w=f;break}}var
l=w;continue a}}var
l=l+1|0;continue}return l}}s(0);return 0}function
d7(a){var
d=[0,0,0,0];function
b(a,b,c){var
f=41!==c?1:0,g=f?c8!==c?1:0:f;if(g){var
e=97===c?2:1;if(b1===c)d[3]=d[3]+1|0;if(a)d[2]=d[2]+e|0;else
d[1]=d[1]+e|0}return b+1|0}d6(a,b,function(a,b){return a+1|0});return d[1]}function
d8(a,b,c){var
h=a.safeGet(c);if((h+aJ|0)<0||9<(h+aJ|0))return i(b,0,c);var
e=h+aJ|0,d=c+1|0;for(;;){var
f=a.safeGet(d);if(48<=f){if(!(58<=f)){var
e=(10*e|0)+(f+aJ|0)|0,d=d+1|0;continue}var
g=0}else
if(36===f)if(0===e){var
j=N(hy),g=1}else{var
j=i(b,[0,ck(e-1|0)],d+1|0),g=1}else
var
g=0;if(!g)var
j=i(b,0,c);return j}}function
P(a,b){return a?b:d1(b)}function
d9(a,b){return a?a[1]:b}function
d_(aJ,b,c,d,e,f,g){var
D=k(b,g);function
af(a){return i(d,D,a)}function
aK(a,b,n,aM){var
j=n.getLen();function
E(l,b){var
p=b;for(;;){if(j<=p)return k(a,D);var
d=n.safeGet(p);if(37===d){var
o=function(a,b){return m(aM,d9(a,b))},au=function(g,f,c,d){var
a=d;for(;;){var
aa=n.safeGet(a)+fD|0;if(!(aa<0||25<aa))switch(aa){case
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
10:return d8(n,function(a,b){var
d=[0,o(a,f),c];return au(g,P(a,f),d,b)},a+1|0);default:var
a=a+1|0;continue}var
q=n.safeGet(a);if(124<=q)var
j=0;else
switch(q){case
78:case
88:case
aH:case
ay:case
dd:case
dr:case
ds:var
a8=o(g,f),a9=bH(d4(q,n,p,a,c),a8),l=r(P(g,f),a9,a+1|0),j=1;break;case
69:case
71:case
gh:case
de:case
dw:var
a1=o(g,f),a2=cX(aq(n,p,a,c),a1),l=r(P(g,f),a2,a+1|0),j=1;break;case
76:case
fN:case
bd:var
ad=n.safeGet(a+1|0)+gl|0;if(ad<0||32<ad)var
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
ah=0,aj=0;break;case
2:var
a7=o(g,f),aB=bH(aq(n,p,U,c),a7),aj=1;break;default:var
a6=o(g,f),aB=bH(aq(n,p,U,c),a6),aj=1}if(aj){var
aA=aB,ah=1}}if(!ah){var
a5=o(g,f),aA=s3(aq(n,p,U,c),a5)}var
l=r(P(g,f),aA,U+1|0),j=1,ag=0;break;default:var
ag=1}if(ag){var
a3=o(g,f),a4=bH(d4(bd,n,p,a,c),a3),l=r(P(g,f),a4,a+1|0),j=1}break;case
37:case
64:var
l=r(f,ai(1,q),a+1|0),j=1;break;case
83:case
bg:var
y=o(g,f);if(bg===q)var
z=y;else{var
b=[0,0],an=y.getLen()-1|0,aN=0;if(!(an<0)){var
M=aN;for(;;){var
x=y.safeGet(M),be=14<=x?34===x?1:92===x?1:0:11<=x?13<=x?1:0:8<=x?1:0,aT=be?2:c1(x)?1:4;b[1]=b[1]+aT|0;var
aU=M+1|0;if(an!==M){var
M=aU;continue}break}}if(b[1]===y.getLen())var
aD=y;else{var
m=C(b[1]);b[1]=0;var
ao=y.getLen()-1|0,aO=0;if(!(ao<0)){var
L=aO;for(;;){var
w=y.safeGet(L),A=w-34|0;if(A<0||58<A)if(-20<=A)var
V=1;else{switch(A+34|0){case
8:m.safeSet(b[1],92);b[1]++;m.safeSet(b[1],98);var
K=1;break;case
9:m.safeSet(b[1],92);b[1]++;m.safeSet(b[1],c7);var
K=1;break;case
10:m.safeSet(b[1],92);b[1]++;m.safeSet(b[1],bd);var
K=1;break;case
13:m.safeSet(b[1],92);b[1]++;m.safeSet(b[1],b1);var
K=1;break;default:var
V=1,K=0}if(K)var
V=0}else
var
V=(A-1|0)<0||56<(A-1|0)?(m.safeSet(b[1],92),b[1]++,m.safeSet(b[1],w),0):1;if(V)if(c1(w))m.safeSet(b[1],w);else{m.safeSet(b[1],92);b[1]++;m.safeSet(b[1],48+(w/aH|0)|0);b[1]++;m.safeSet(b[1],48+((w/10|0)%10|0)|0);b[1]++;m.safeSet(b[1],48+(w%10|0)|0)}b[1]++;var
aS=L+1|0;if(ao!==L){var
L=aS;continue}break}}var
aD=m}var
z=h(hJ,h(aD,hI))}if(a===(p+1|0))var
aC=z;else{var
J=aq(n,p,a,c);try{var
W=0,t=1;for(;;){if(J.getLen()<=t)var
ap=[0,0,W];else{var
X=J.safeGet(t);if(49<=X)if(58<=X)var
ak=0;else{var
ap=[0,e$(u(J,t,(J.getLen()-t|0)-1|0)),W],ak=1}else{if(45===X){var
W=1,t=t+1|0;continue}var
ak=0}if(!ak){var
t=t+1|0;continue}}var
Z=ap;break}}catch(f){if(f[1]!==aL)throw f;var
Z=d3(J,0,bg)}var
N=Z[1],B=z.getLen(),aV=Z[2],O=0,aW=32;if(N===B)if(0===O){var
_=z,al=1}else
var
al=0;else
var
al=0;if(!al)if(N<=B)var
_=u(z,O,B);else{var
Y=ai(N,aW);if(aV)bp(z,O,Y,0,B);else
bp(z,O,Y,N-B|0,B);var
_=Y}var
aC=_}var
l=r(P(g,f),aC,a+1|0),j=1;break;case
67:case
99:var
s=o(g,f);if(99===q)var
ax=ai(1,s);else{if(39===s)var
v=g1;else
if(92===s)var
v=g2;else{if(14<=s)var
F=0;else
switch(s){case
8:var
v=g3,F=1;break;case
9:var
v=g4,F=1;break;case
10:var
v=g5,F=1;break;case
13:var
v=g6,F=1;break;default:var
F=0}if(!F)if(c1(s)){var
am=C(1);am.safeSet(0,s);var
v=am}else{var
G=C(4);G.safeSet(0,92);G.safeSet(1,48+(s/aH|0)|0);G.safeSet(2,48+((s/10|0)%10|0)|0);G.safeSet(3,48+(s%10|0)|0);var
v=G}}var
ax=h(hG,h(v,hF))}var
l=r(P(g,f),ax,a+1|0),j=1;break;case
66:case
98:var
aZ=a+1|0,a0=o(g,f)?gK:gL,l=r(P(g,f),a0,aZ),j=1;break;case
40:case
du:var
T=o(g,f),av=i(d5(q),n,a+1|0);if(du===q){var
Q=aP(T.getLen()),ar=function(a,b){I(Q,b);return a+1|0};d6(T,function(a,b,c){if(a)br(Q,hx);else
I(Q,37);return ar(b,c)},ar);var
aX=aQ(Q),l=r(P(g,f),aX,av),j=1}else{var
aw=P(g,f),bc=d0(d7(T),aw),l=aK(function(a){return E(bc,av)},aw,T,aM),j=1}break;case
33:k(e,D);var
l=E(f,a+1|0),j=1;break;case
41:var
l=r(f,hD,a+1|0),j=1;break;case
44:var
l=r(f,hE,a+1|0),j=1;break;case
70:var
ab=o(g,f);if(0===c)var
az=hH;else{var
$=aq(n,p,a,c);if(70===q)$.safeSet($.getLen()-1|0,dw);var
az=$}var
at=sP(ab);if(3===at)var
ac=ab<0?hA:hB;else
if(4<=at)var
ac=hC;else{var
S=cX(az,ab),R=0,aY=S.getLen();for(;;){if(aY<=R)var
as=h(S,hz);else{var
H=S.safeGet(R)-46|0,bf=H<0||23<H?55===H?1:0:(H-1|0)<0||21<(H-1|0)?1:0;if(!bf){var
R=R+1|0;continue}var
as=S}var
ac=as;break}}var
l=r(P(g,f),ac,a+1|0),j=1;break;case
91:var
l=aR(n,a,q),j=1;break;case
97:var
aE=o(g,f),aF=d1(d9(g,f)),aG=o(0,aF),a_=a+1|0,a$=P(g,aF);if(aJ)af(i(aE,0,aG));else
i(aE,D,aG);var
l=E(a$,a_),j=1;break;case
b1:var
l=aR(n,a,q),j=1;break;case
c7:var
aI=o(g,f),ba=a+1|0,bb=P(g,f);if(aJ)af(k(aI,0));else
k(aI,D);var
l=E(bb,ba),j=1;break;default:var
j=0}if(!j)var
l=aR(n,a,q);return l}},f=p+1|0,g=0;return d8(n,function(a,b){return au(a,l,g,b)},f)}i(c,D,d);var
p=p+1|0;continue}}function
r(a,b,c){af(b);return E(a,c)}return E(b,0)}var
p=ck(0);function
l(a,b){return aK(f,p,a,b)}var
n=d7(g);if(n<0||6<n){var
o=function(f,b){if(n<=f){var
h=v(n,0),i=function(a,b){return j(h,(n-a|0)-1|0,b)},c=0,a=b;for(;;){if(a){var
d=a[2],e=a[1];if(d){i(c,e);var
c=c+1|0,a=d;continue}i(c,e)}return l(g,h)}}return function(a){return o(f+1|0,[0,a,b])}},a=o(0,0)}else
switch(n){case
1:var
a=function(a){var
b=v(1,0);j(b,0,a);return l(g,b)};break;case
2:var
a=function(a,b){var
c=v(2,0);j(c,0,a);j(c,1,b);return l(g,c)};break;case
3:var
a=function(a,b,c){var
d=v(3,0);j(d,0,a);j(d,1,b);j(d,2,c);return l(g,d)};break;case
4:var
a=function(a,b,c,d){var
e=v(4,0);j(e,0,a);j(e,1,b);j(e,2,c);j(e,3,d);return l(g,e)};break;case
5:var
a=function(a,b,c,d,e){var
f=v(5,0);j(f,0,a);j(f,1,b);j(f,2,c);j(f,3,d);j(f,4,e);return l(g,f)};break;case
6:var
a=function(a,b,c,d,e,f){var
h=v(6,0);j(h,0,a);j(h,1,b);j(h,2,c);j(h,3,d);j(h,4,e);j(h,5,f);return l(g,h)};break;default:var
a=l(g,[0])}return a}function
d$(a){function
b(a){return 0}return d_(0,function(a){return dK},gS,dL,dP,b,a)}function
hK(a){return aP(2*a.getLen()|0)}function
ea(c){function
b(a){var
b=aQ(a);a[2]=0;return k(c,b)}function
d(a){return 0}var
e=1;return function(a){return d_(e,hK,I,br,d,b,a)}}function
cl(a){return k(ea(function(a){return a}),a)}var
eb=[0,0];function
ec(a){eb[1]=[0,a,eb[1]];return 0}function
ed(a,b){var
k=0===b.length-1?[0,0]:b,f=k.length-1,q=0,r=54;if(!(54<0)){var
d=q;for(;;){j(a[1],d,d);var
w=d+1|0;if(r!==d){var
d=w;continue}break}}var
g=[0,hL],n=0,s=55,t=sY(55,f)?s:f,o=54+t|0;if(!(o<n)){var
c=n;for(;;){var
p=c%55|0,u=g[1],i=h(u,l(m(k,aF(c,f))));g[1]=tg(i,0,i.getLen());var
e=g[1];j(a[1],p,(m(a[1],p)^(((e.safeGet(0)+(e.safeGet(1)<<8)|0)+(e.safeGet(2)<<16)|0)+(e.safeGet(3)<<24)|0))&bc);var
v=c+1|0;if(o!==c){var
c=v;continue}break}}a[2]=0;return 0}function
cm(a){a[2]=(a[2]+1|0)%55|0;var
b=m(a[1],a[2]),c=(m(a[1],(a[2]+24|0)%55|0)+(b^b>>>25&31)|0)&bc;j(a[1],a[2],c);return c}32===az;var
cn=[0,hM.slice(),0];try{var
sF=bI(sE),co=sF}catch(f){if(f[1]!==t)throw f;try{var
sD=bI(sC),ee=sD}catch(f){if(f[1]!==t)throw f;var
ee=hN}var
co=ee}var
dX=co.getLen(),hO=82,dY=0;if(0<=0)if(dX<dY)var
bJ=0;else
try{var
bq=dY;for(;;){if(dX<=bq)throw[0,t];if(co.safeGet(bq)!==hO){var
bq=bq+1|0;continue}var
g_=1,cg=g_,bJ=1;break}}catch(f){if(f[1]!==t)throw f;var
cg=0,bJ=1}else
var
bJ=0;if(!bJ)var
cg=H(g9);var
aj=[fV,function(a){var
b=[0,v(55,0),0];ed(b,fa(0));return b}];function
cp(a,b){var
i=a?a[1]:cg,c=16;for(;;){if(!(b<=c))if(!(ci<(c*2|0))){var
c=c*2|0;continue}if(i){var
f=tv(aj);if(aI===f)var
d=aj[1];else
if(fV===f){var
h=aj[0+1];aj[0+1]=hm;try{var
e=k(h,0);aj[0+1]=e;tu(aj,hb)}catch(f){aj[0+1]=function(a){throw f};throw f}var
d=e}else
var
d=aj;var
g=cm(d)}else
var
g=0;return[0,0,v(c,0),g,c]}}function
cq(a,b){return 3<=a.length-1?sZ(10,aH,a[3],b)&(a[2].length-1-1|0):aF(s0(10,aH,b),a[2].length-1)}function
ef(d,b,c){var
o=cq(d,b);j(d[2],o,[0,b,c,m(d[2],o)]);d[1]=d[1]+1|0;var
p=d[2].length-1<<1<d[1]?1:0;if(p){var
f=d[2],g=f.length-1,h=g*2|0,i=h<ci?1:0;if(i){var
e=v(h,0);d[2]=e;var
k=function(a){if(a){var
b=a[1],f=a[2];k(a[3]);var
c=cq(d,b);return j(e,c,[0,b,f,m(e,c)])}return 0},l=g-1|0,q=0;if(!(l<0)){var
a=q;for(;;){k(m(f,a));var
r=a+1|0;if(l!==a){var
a=r;continue}break}}var
n=0}else
var
n=i;return n}return p}function
bt(a,b){var
i=cq(a,b),d=m(a[2],i);if(d){var
e=d[3],j=d[2];if(0===av(b,d[1]))return j;if(e){var
f=e[3],k=e[2];if(0===av(b,e[1]))return k;if(f){var
l=f[3],n=f[2];if(0===av(b,f[1]))return n;var
c=l;for(;;){if(c){var
g=c[3],h=c[2];if(0===av(b,c[1]))return h;var
c=g;continue}throw[0,t]}}throw[0,t]}throw[0,t]}throw[0,t]}function
a(a,b){return c0(a,b[0+1])}var
cr=[0,0];c0(hP,cr);var
hQ=2;function
hR(a){var
b=[0,0],d=a.getLen()-1|0,e=0;if(!(d<0)){var
c=e;for(;;){b[1]=(223*b[1]|0)+a.safeGet(c)|0;var
g=c+1|0;if(d!==c){var
c=g;continue}break}}b[1]=b[1]&((1<<31)-1|0);var
f=bc<b[1]?b[1]-(1<<31)|0:b[1];return f}var
ac=cj([0,function(a,b){return fb(a,b)}]),ar=cj([0,function(a,b){return fb(a,b)}]),ak=cj([0,function(a,b){return gz(a,b)}]),eg=fc(0,0),hS=[0,0];function
eh(a){return 2<a?eh((a+1|0)/2|0)*2|0:a}function
ei(a){hS[1]++;var
c=a.length-1,d=v((c*2|0)+2|0,eg);j(d,0,c);j(d,1,(w(eh(c),az)/8|0)-1|0);var
e=c-1|0,f=0;if(!(e<0)){var
b=f;for(;;){j(d,(b*2|0)+3|0,m(a,b));var
g=b+1|0;if(e!==b){var
b=g;continue}break}}return[0,hQ,d,ar[1],ak[1],0,0,ac[1],0]}function
cs(a,b){var
c=a[2].length-1,g=c<b?1:0;if(g){var
d=v(b,eg),h=a[2],e=0,f=0,j=0<=c?0<=f?(h.length-1-c|0)<f?0:0<=e?(d.length-1-c|0)<e?0:(sI(h,f,d,e,c),1):0:0:0;if(!j)H(gT);a[2]=d;var
i=0}else
var
i=g;return i}var
ej=[0,0],hT=[0,0];function
ct(a){var
b=a[2].length-1;cs(a,b+1|0);return b}function
aS(a,b){try{var
d=i(ar[22],b,a[3])}catch(f){if(f[1]===t){var
c=ct(a);a[3]=q(ar[4],b,c,a[3]);a[4]=q(ak[4],c,1,a[4]);return c}throw f}return d}function
cv(a){return a===0?0:aN(a)}function
ep(a,b){try{var
d=i(ac[22],b,a[7])}catch(f){if(f[1]===t){var
c=a[1];a[1]=c+1|0;if(x(b,h9))a[7]=q(ac[4],b,c,a[7]);return c}throw f}return d}function
cx(a){return sR(a,0)?[0]:a}function
er(a,b){if(a)return a;var
c=fc(ha,b[1]);c[0+1]=b[2];var
d=cr[1];c[1+1]=d;cr[1]=d+1|0;return c}function
bu(a){var
b=ct(a);if(0===(b%2|0))var
d=0;else
if((2+aw(m(a[2],1)*16|0,az)|0)<b)var
d=0;else{var
c=ct(a),d=1}if(!d)var
c=b;j(a[2],c,0);return c}function
es(a,ap){var
g=[0,0],aq=ap.length-1;for(;;){if(g[1]<aq){var
l=m(ap,g[1]),e=function(a){g[1]++;return m(ap,g[1])},n=e(0);if(typeof
n===o)switch(n){case
1:var
q=e(0),f=function(q){return function(a){return a[q+1]}}(q);break;case
2:var
r=e(0),b=e(0),f=function(r,b){return function(a){return a[r+1][b+1]}}(r,b);break;case
3:var
s=e(0),f=function(s){return function(a){return k(a[1][s+1],a)}}(s);break;case
4:var
t=e(0),f=function(t){return function(a,b){a[t+1]=b;return 0}}(t);break;case
5:var
u=e(0),v=e(0),f=function(u,v){return function(a){return k(u,v)}}(u,v);break;case
6:var
w=e(0),x=e(0),f=function(w,x){return function(a){return k(w,a[x+1])}}(w,x);break;case
7:var
y=e(0),z=e(0),c=e(0),f=function(y,z,c){return function(a){return k(y,a[z+1][c+1])}}(y,z,c);break;case
8:var
A=e(0),B=e(0),f=function(A,B){return function(a){return k(A,k(a[1][B+1],a))}}(A,B);break;case
9:var
C=e(0),D=e(0),E=e(0),f=function(C,D,E){return function(a){return i(C,D,E)}}(C,D,E);break;case
10:var
F=e(0),G=e(0),H=e(0),f=function(F,G,H){return function(a){return i(F,G,a[H+1])}}(F,G,H);break;case
11:var
I=e(0),J=e(0),K=e(0),d=e(0),f=function(I,J,K,d){return function(a){return i(I,J,a[K+1][d+1])}}(I,J,K,d);break;case
12:var
L=e(0),M=e(0),N=e(0),f=function(L,M,N){return function(a){return i(L,M,k(a[1][N+1],a))}}(L,M,N);break;case
13:var
O=e(0),P=e(0),Q=e(0),f=function(O,P,Q){return function(a){return i(O,a[P+1],Q)}}(O,P,Q);break;case
14:var
R=e(0),S=e(0),T=e(0),U=e(0),f=function(R,S,T,U){return function(a){return i(R,a[S+1][T+1],U)}}(R,S,T,U);break;case
15:var
V=e(0),W=e(0),X=e(0),f=function(V,W,X){return function(a){return i(V,k(a[1][W+1],a),X)}}(V,W,X);break;case
16:var
Y=e(0),Z=e(0),f=function(Y,Z){return function(a){return i(a[1][Y+1],a,Z)}}(Y,Z);break;case
17:var
_=e(0),$=e(0),f=function(_,$){return function(a){return i(a[1][_+1],a,a[$+1])}}(_,$);break;case
18:var
ab=e(0),ac=e(0),ad=e(0),f=function(ab,ac,ad){return function(a){return i(a[1][ab+1],a,a[ac+1][ad+1])}}(ab,ac,ad);break;case
19:var
ae=e(0),af=e(0),f=function(ae,af){return function(a){var
b=k(a[1][af+1],a);return i(a[1][ae+1],a,b)}}(ae,af);break;case
20:var
ag=e(0),h=e(0);bu(a);var
f=function(ag,h){return function(a){return k(aa(h,ag,0),h)}}(ag,h);break;case
21:var
ah=e(0),ai=e(0);bu(a);var
f=function(ah,ai){return function(a){var
b=a[ai+1];return k(aa(b,ah,0),b)}}(ah,ai);break;case
22:var
aj=e(0),al=e(0),am=e(0);bu(a);var
f=function(aj,al,am){return function(a){var
b=a[al+1][am+1];return k(aa(b,aj,0),b)}}(aj,al,am);break;case
23:var
an=e(0),ao=e(0);bu(a);var
f=function(an,ao){return function(a){var
b=k(a[1][ao+1],a);return k(aa(b,an,0),b)}}(an,ao);break;default:var
p=e(0),f=function(p){return function(a){return p}}(p)}else
var
f=n;hT[1]++;if(i(ak[22],l,a[4])){cs(a,l+1|0);j(a[2],l,f)}else
a[6]=[0,[0,l,f],a[6]];g[1]++;continue}return 0}}function
cy(a,b,c){if(bK(c,ii))return b;var
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
cz(a,b,c){if(bK(c,ij))return b;var
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
cB(a,b){return 47===a.safeGet(b)?1:0}function
et(a){var
b=a.getLen()<1?1:0,c=b||(47!==a.safeGet(0)?1:0);return c}function
im(a){var
c=et(a);if(c){var
e=a.getLen()<2?1:0,d=e||x(u(a,0,2),ip);if(d){var
f=a.getLen()<3?1:0,b=f||x(u(a,0,3),io)}else
var
b=d}else
var
b=c;return b}function
iq(a,b){var
c=b.getLen()<=a.getLen()?1:0,d=c?bK(u(a,a.getLen()-b.getLen()|0,b.getLen()),b):c;return d}try{var
sB=bI(sA),cC=sB}catch(f){if(f[1]!==t)throw f;var
cC=ir}function
eu(a){var
d=a.getLen(),b=aP(d+20|0);I(b,39);var
e=d-1|0,f=0;if(!(e<0)){var
c=f;for(;;){if(39===a.safeGet(c))br(b,is);else
I(b,a.safeGet(c));var
g=c+1|0;if(e!==c){var
c=g;continue}break}}I(b,39);return aQ(b)}function
it(a){return cy(cB,cA,a)}function
iu(a){return cz(cB,cA,a)}function
aB(a,b){var
c=a.safeGet(b),d=47===c?1:0;if(d)var
e=d;else{var
f=92===c?1:0,e=f||(58===c?1:0)}return e}function
cE(a){var
e=a.getLen()<1?1:0,c=e||(47!==a.safeGet(0)?1:0);if(c){var
f=a.getLen()<1?1:0,d=f||(92!==a.safeGet(0)?1:0);if(d){var
g=a.getLen()<2?1:0,b=g||(58!==a.safeGet(1)?1:0)}else
var
b=d}else
var
b=c;return b}function
ev(a){var
c=cE(a);if(c){var
g=a.getLen()<2?1:0,d=g||x(u(a,0,2),iA);if(d){var
h=a.getLen()<2?1:0,e=h||x(u(a,0,2),iz);if(e){var
i=a.getLen()<3?1:0,f=i||x(u(a,0,3),iy);if(f){var
j=a.getLen()<3?1:0,b=j||x(u(a,0,3),ix)}else
var
b=f}else
var
b=e}else
var
b=d}else
var
b=c;return b}function
ew(a,b){var
c=b.getLen()<=a.getLen()?1:0;if(c){var
e=u(a,a.getLen()-b.getLen()|0,b.getLen()),f=dW(b),d=bK(dW(e),f)}else
var
d=c;return d}try{var
sz=bI(sy),ex=sz}catch(f){if(f[1]!==t)throw f;var
ex=iB}function
iC(h){var
i=h.getLen(),e=aP(i+20|0);I(e,34);function
g(a,b){var
c=b;for(;;){if(c===i)return I(e,34);var
f=h.safeGet(c);if(34===f)return a<50?d(1+a,0,c):F(d,[0,0,c]);if(92===f)return a<50?d(1+a,0,c):F(d,[0,0,c]);I(e,f);var
c=c+1|0;continue}}function
d(a,b,c){var
f=b,d=c;for(;;){if(d===i){I(e,34);return a<50?j(1+a,f):F(j,[0,f])}var
l=h.safeGet(d);if(34===l){k((2*f|0)+1|0);I(e,34);return a<50?g(1+a,d+1|0):F(g,[0,d+1|0])}if(92===l){var
f=f+1|0,d=d+1|0;continue}k(f);return a<50?g(1+a,d):F(g,[0,d])}}function
j(a,b){var
d=1;if(!(b<1)){var
c=d;for(;;){I(e,92);var
f=c+1|0;if(b!==c){var
c=f;continue}break}}return 0}function
a(b){return Y(g(0,b))}function
b(b,c){return Y(d(0,b,c))}function
k(b){return Y(j(0,b))}a(0);return aQ(e)}function
ey(a){var
c=2<=a.getLen()?1:0;if(c){var
b=a.safeGet(0),g=91<=b?(b+fI|0)<0||25<(b+fI|0)?0:1:65<=b?1:0,d=g?1:0,e=d?58===a.safeGet(1)?1:0:d}else
var
e=c;if(e){var
f=u(a,2,a.getLen()-2|0);return[0,u(a,0,2),f]}return[0,iD,a]}function
iE(a){var
b=ey(a),c=b[1];return h(c,cz(aB,cD,b[2]))}function
iF(a){return cy(aB,cD,ey(a)[2])}function
iI(a){return cy(aB,cF,a)}function
iJ(a){return cz(aB,cF,a)}if(x(ch,iK))if(x(ch,iL)){if(x(ch,iM))throw[0,E,iN];var
bv=[0,cD,iv,iw,aB,cE,ev,ew,ex,iC,iF,iE]}else
var
bv=[0,cA,ik,il,cB,et,im,iq,cC,eu,it,iu];else
var
bv=[0,cF,iG,iH,aB,cE,ev,ew,cC,eu,iI,iJ];var
ez=[0,iQ],iO=bv[11],iP=bv[3];a(iT,[0,ez,0,iS,iR]);ec(function(a){if(a[1]===ez){var
c=a[2],d=a[4],e=a[3];if(typeof
c===o)switch(c){case
1:var
b=iW;break;case
2:var
b=iX;break;case
3:var
b=iY;break;case
4:var
b=iZ;break;case
5:var
b=i0;break;case
6:var
b=i1;break;case
7:var
b=i2;break;case
8:var
b=i3;break;case
9:var
b=i4;break;case
10:var
b=i5;break;case
11:var
b=i6;break;case
12:var
b=i7;break;case
13:var
b=i8;break;case
14:var
b=i9;break;case
15:var
b=i_;break;case
16:var
b=i$;break;case
17:var
b=ja;break;case
18:var
b=jb;break;case
19:var
b=jc;break;case
20:var
b=jd;break;case
21:var
b=je;break;case
22:var
b=jf;break;case
23:var
b=jg;break;case
24:var
b=jh;break;case
25:var
b=ji;break;case
26:var
b=jj;break;case
27:var
b=jk;break;case
28:var
b=jl;break;case
29:var
b=jm;break;case
30:var
b=jn;break;case
31:var
b=jo;break;case
32:var
b=jp;break;case
33:var
b=jq;break;case
34:var
b=jr;break;case
35:var
b=js;break;case
36:var
b=jt;break;case
37:var
b=ju;break;case
38:var
b=jv;break;case
39:var
b=jw;break;case
40:var
b=jx;break;case
41:var
b=jy;break;case
42:var
b=jz;break;case
43:var
b=jA;break;case
44:var
b=jB;break;case
45:var
b=jC;break;case
46:var
b=jD;break;case
47:var
b=jE;break;case
48:var
b=jF;break;case
49:var
b=jG;break;case
50:var
b=jH;break;case
51:var
b=jI;break;case
52:var
b=jJ;break;case
53:var
b=jK;break;case
54:var
b=jL;break;case
55:var
b=jM;break;case
56:var
b=jN;break;case
57:var
b=jO;break;case
58:var
b=jP;break;case
59:var
b=jQ;break;case
60:var
b=jR;break;case
61:var
b=jS;break;case
62:var
b=jT;break;case
63:var
b=jU;break;case
64:var
b=jV;break;case
65:var
b=jW;break;case
66:var
b=jX;break;case
67:var
b=jY;break;default:var
b=iU}else{var
f=c[1],b=k(cl(jZ),f)}return[0,q(cl(iV),b,e,d)]}return 0});bL(j0);bL(j1);try{bL(sx)}catch(f){if(f[1]!==aL)throw f}try{bL(sw)}catch(f){if(f[1]!==aL)throw f}cp(0,7);function
eA(a){return uD(a)}ai(32,s);var
j2=0,j3=0,j8=C(b2),j9=0,j_=s;if(!(s<0)){var
a6=j9;for(;;){j8.safeSet(a6,dV(cf(a6)));var
sv=a6+1|0;if(j_!==a6){var
a6=sv;continue}break}}var
cG=ai(32,0);cG.safeSet(10>>>3,cf(cG.safeGet(10>>>3)|1<<(10&7)));var
j4=C(32),j5=0,j6=31;if(!(31<0)){var
aW=j5;for(;;){j4.safeSet(aW,cf(cG.safeGet(aW)^s));var
j7=aW+1|0;if(j6!==aW){var
aW=j7;continue}break}}var
aC=[0,0],aD=[0,0],eB=[0,0];function
J(a){return aC[1]}function
eC(a){return aD[1]}function
Q(a,b,c){return 0===a[2][0]?b?t5(a[1],a,b[1]):t6(a[1],a):b?fd(a[1],b[1]):fd(a[1],0)}var
eD=[0,j2],cH=[0,0];function
as(e,b,c){cH[1]++;switch(e[0]){case
7:case
8:throw[0,E,j$];case
6:var
g=e[1],m=cH[1],n=fe(0),o=v(eC(0)+1|0,n),p=ff(0),q=v(J(0)+1|0,p),f=[0,-1,[1,[0,tT(g,c),g]],q,o,c,0,e,0,0,m,0];break;default:var
h=e[1],i=cH[1],j=fe(0),k=v(eC(0)+1|0,j),l=ff(0),f=[0,-1,[0,sM(h,j3,[0,c])],v(J(0)+1|0,l),k,c,0,e,0,0,i,0]}if(b){var
d=b[1],a=function(a){{if(0===d[2][0])return 6===e[0]?gH(f,d[1][8],d[1]):gG(f,d[1][8],d[1]);{var
b=d[1],c=J(0);return fg(f,d[1][8]-c|0,b)}}};try{a(0)}catch(f){a8(0);a(0)}f[6]=[0,d]}return f}function
W(a){return a[5]}function
aX(a){return a[6]}function
bw(a){return a[8]}function
bx(a){return a[7]}function
$(a){return a[2]}function
by(a,b,c){a[1]=b;a[6]=c;return 0}function
cI(a,b,c){return dx<=b?m(a[3],c):m(a[4],c)}function
cJ(a,b){var
e=b[3].length-1-2|0,g=0;if(!(e<0)){var
d=g;for(;;){j(b[3],d,m(a[3],d));var
k=d+1|0;if(e!==d){var
d=k;continue}break}}var
f=b[4].length-1-2|0,h=0;if(!(f<0)){var
c=h;for(;;){j(b[4],c,m(a[4],c));var
i=c+1|0;if(f!==c){var
c=i;continue}break}}return 0}function
bz(a,b){b[8]=a[8];return 0}var
at=[0,kf];a(ko,[0,[0,ka]]);a(kp,[0,[0,kb]]);a(kq,[0,[0,kc]]);a(kr,[0,[0,kd]]);a(ks,[0,[0,ke]]);a(kt,[0,at]);a(ku,[0,[0,kg]]);a(kv,[0,[0,kh]]);a(kw,[0,[0,ki]]);a(kx,[0,[0,kk]]);a(ky,[0,[0,kl]]);a(kz,[0,[0,km]]);a(kA,[0,[0,kn]]);a(kB,[0,[0,kj]]);var
cK=[0,kJ];a(kZ,[0,[0,kC]]);a(k0,[0,[0,kD]]);a(k1,[0,[0,kE]]);a(k2,[0,[0,kF]]);a(k3,[0,[0,kG]]);a(k4,[0,[0,kH]]);a(k5,[0,[0,kI]]);a(k6,[0,cK]);a(k7,[0,[0,kK]]);a(k8,[0,[0,kL]]);a(k9,[0,[0,kM]]);a(k_,[0,[0,kN]]);a(k$,[0,[0,kO]]);a(la,[0,[0,kP]]);a(lb,[0,[0,kQ]]);a(lc,[0,[0,kR]]);a(ld,[0,[0,kS]]);a(le,[0,[0,kT]]);a(lf,[0,[0,kU]]);a(lg,[0,[0,kV]]);a(lh,[0,[0,kW]]);a(li,[0,[0,kX]]);a(lj,[0,[0,kY]]);var
bA=1,eE=0;function
aY(a,b,c){var
d=a[2];if(0===d[0])var
f=sO(d[1],b,c);else{var
e=d[1],f=q(e[2][4],e[1],b,c)}return f}function
aZ(a,b){var
c=a[2];if(0===c[0])var
e=sN(c[1],b);else{var
d=c[1],e=i(d[2][3],d[1],b)}return e}function
eF(a,b){Q(a,0,0);eK(b,0,0);return Q(a,0,0)}function
X(a,b,c){var
f=a,d=b;for(;;){if(eE)return aY(f,d,c);var
m=d<0?1:0,n=m||(W(f)<=d?1:0);if(n)throw[0,bl,lk];if(bA){var
i=aX(f);if(typeof
i!==o)eF(i[1],f)}var
j=bw(f);if(j){var
e=j[1];if(1===e[1]){var
k=e[4],g=e[3],l=e[2];return 0===k?aY(e[5],l+d|0,c):aY(e[5],(l+w(aw(d,g),k+g|0)|0)+aF(d,g)|0,c)}var
h=e[3],f=e[5],d=(e[2]+w(aw(d,h),e[4]+h|0)|0)+aF(d,h)|0;continue}return aY(f,d,c)}}function
z(a,b){var
e=a,c=b;for(;;){if(eE)return aZ(e,c);var
l=c<0?1:0,m=l||(W(e)<=c?1:0);if(m)throw[0,bl,ll];if(bA){var
h=aX(e);if(typeof
h!==o)eF(h[1],e)}var
i=bw(e);if(i){var
d=i[1];if(1===d[1]){var
j=d[4],f=d[3],k=d[2];return 0===j?aZ(d[5],k+c|0):aZ(d[5],(k+w(aw(c,f),j+f|0)|0)+aF(c,f)|0)}var
g=d[3],e=d[5],c=(d[2]+w(aw(c,g),d[4]+g|0)|0)+aF(c,g)|0;continue}return aZ(e,c)}}function
eG(a){if(a[8]){var
b=as(a[7],0,a[5]);b[1]=a[1];b[6]=a[6];cJ(a,b);var
c=b}else
var
c=a;return c}function
eH(d,b,c){{if(0===c[2][0]){var
a=function(a){return 0===$(d)[0]?tW(d,c[1][8],c[1],c[3],b):tY(d,c[1][8],c[1],b)};try{var
f=a(0)}catch(f){if(f[1]===at){try{Q(c,0,0);var
g=a(0)}catch(f){a8(0);return a(0)}return g}throw f}return f}var
e=function(a){{if(0===$(d)[0]){var
e=c[1],f=J(0);return um(d,c[1][8]-f|0,e,b)}var
g=c[1],h=J(0);return uo(d,c[1][8]-h|0,g,b)}};try{var
i=e(0)}catch(f){try{Q(c,0,0);var
h=e(0)}catch(f){a8(0);return e(0)}return h}return i}}function
eI(d,b,c){{if(0===c[2][0]){var
a=function(a){return 0===$(d)[0]?t4(d,c[1][8],c[1],c,b):tZ(d,c[1][8],c[1],b)};try{var
f=a(0)}catch(f){if(f[1]===at){try{Q(c,0,0);var
g=a(0)}catch(f){a8(0);return a(0)}return g}throw f}return f}var
e=function(a){{if(0===$(d)[0]){var
e=c[2],f=c[1],g=J(0);return us(d,c[1][8]-g|0,f,e,b)}var
h=c[2],i=c[1],j=J(0);return up(d,c[1][8]-j|0,i,h,b)}};try{var
i=e(0)}catch(f){try{Q(c,0,0);var
h=e(0)}catch(f){a8(0);return e(0)}return h}return i}}function
a0(a,b,c,d,e,f,g,h){{if(0===d[2][0])return 0===$(a)[0]?ub(a,b,d[1][8],d[1],d[3],c,e,f,g,h):t1(a,b,d[1][8],d[1],d[3],c,e,f,g,h);{if(0===$(a)[0]){var
i=d[3],j=d[1],k=J(0);return uB(a,b,d[1][8]-k|0,j,i,c,e,f,g,h)}var
l=d[3],m=d[1],n=J(0);return uq(a,b,d[1][8]-n|0,m,l,c,e,f,g,h)}}}function
a1(a,b,c,d,e,f,g,h){{if(0===d[2][0])return 0===$(a)[0]?uc(a,b,d[1][8],d[1],d[3],c,e,f,g,h):t2(a,b,d[1][8],d[1],d[3],c,e,f,g,h);{if(0===$(a)[0]){var
i=d[3],j=d[1],k=J(0);return uC(a,b,d[1][8]-k|0,j,i,c,e,f,g,h)}var
l=d[3],m=d[1],n=J(0);return ur(a,b,d[1][8]-n|0,m,l,c,e,f,g,h)}}}function
eJ(a,b,c){var
p=b;for(;;){var
d=p?p[1]:0,q=aX(a);if(typeof
q===o){by(a,c[1][8],[1,c]);try{cL(a,c)}catch(f){if(f[1]!==at)f[1]===cK;try{Q(c,[0,d],0);cL(a,c)}catch(f){if(f[1]!==at)if(f[1]!==cK)throw f;Q(c,0,0);sV(0);cL(a,c)}}var
A=bw(a);if(A){var
j=A[1];if(1===j[1]){var
k=j[5],r=j[4],f=j[3],l=j[2];if(0===f)a0(k,a,d,c,0,0,l,W(a));else
if(y<f){var
h=0,m=W(a);for(;;){if(f<m){a0(k,a,d,c,w(h,f+r|0),w(h,f),l,f);var
h=h+1|0,m=m-f|0;continue}if(0<m)a0(k,a,d,c,w(h,f+r|0),w(h,f),l,m);break}}else{var
e=0,i=0,g=W(a);for(;;){if(y<g){var
u=as(bx(a),0,y);bz(a,u);var
B=e+gm|0;if(!(B<e)){var
s=e;for(;;){X(u,s,z(a,e));var
I=s+1|0;if(B!==s){var
s=I;continue}break}}a0(k,u,d,c,w(i,y+r|0),i*y|0,l,y);var
e=e+y|0,i=i+1|0,g=g+fZ|0;continue}if(0<g){var
v=as(bx(a),0,g),C=(e+g|0)-1|0;if(!(C<e)){var
t=e;for(;;){X(v,t,z(a,e));var
J=t+1|0;if(C!==t){var
t=J;continue}break}}bz(a,v);a0(k,v,d,c,w(i,y+r|0),i*y|0,l,g)}break}}}else{var
x=eG(a),D=W(a)-1|0,K=0;if(!(D<0)){var
n=K;for(;;){aY(x,n,z(a,n));var
L=n+1|0;if(D!==n){var
n=L;continue}break}}eH(x,d,c);cJ(x,a)}}else
eH(a,d,c);return by(a,c[1][8],[0,c])}else{if(0===q[0]){var
E=q[1],F=c2(E,c);if(F){eK(a,[0,d],0);Q(E,0,0);var
p=[0,d];continue}return F}var
G=q[1],H=c2(G,c);if(H){Q(G,0,0);var
p=[0,d];continue}return H}}}function
cL(a,b){{if(0===b[2][0])return 0===$(a)[0]?gG(a,b[1][8],b[1]):gH(a,b[1][8],b[1]);{if(0===$(a)[0]){var
c=b[1],d=J(0);return fg(a,b[1][8]-d|0,c)}var
e=b[1],f=J(0);return un(a,b[1][8]-f|0,e)}}}function
eK(a,b,c){var
v=b;for(;;){var
f=v?v[1]:0,q=aX(a);if(typeof
q===o)return 0;else{if(0===q[0]){var
d=q[1];by(a,d[1][8],[1,d]);var
B=bw(a);if(B){var
k=B[1];if(1===k[1]){var
l=k[5],r=k[4],e=k[3],m=k[2];if(0===e)a1(l,a,f,d,0,0,m,W(a));else
if(y<e){var
i=0,n=W(a);for(;;){if(e<n){a1(l,a,f,d,w(i,e+r|0),w(i,e),m,e);var
i=i+1|0,n=n-e|0;continue}if(0<n)a1(l,a,f,d,w(i,e+r|0),w(i,e),m,n);break}}else{var
j=0,h=W(a),g=0;for(;;){if(y<h){var
x=as(bx(a),0,y);bz(a,x);var
C=g+gm|0;if(!(C<g)){var
s=g;for(;;){X(x,s,z(a,g));var
F=s+1|0;if(C!==s){var
s=F;continue}break}}a1(l,x,f,d,w(j,y+r|0),j*y|0,m,y);var
j=j+1|0,h=h+fZ|0;continue}if(0<h){var
A=as(bx(a),0,h),D=(g+h|0)-1|0;if(!(D<g)){var
t=g;for(;;){X(A,t,z(a,g));var
G=t+1|0;if(D!==t){var
t=G;continue}break}}bz(a,A);a1(l,A,f,d,w(j,y+r|0),j*y|0,m,h)}break}}}else{var
u=eG(a);cJ(u,a);eI(u,f,d);var
E=W(u)-1|0,H=0;if(!(E<0)){var
p=H;for(;;){X(a,p,aZ(u,p));var
I=p+1|0;if(E!==p){var
p=I;continue}break}}}}else
eI(a,f,d);return by(a,d[1][8],0)}Q(q[1],0,0);var
v=[0,f];continue}}}var
lq=[0,lp],ls=[0,lr];function
bB(a,b){var
q=m(g$,0),r=h(iP,h(a,b)),f=dM(h(iO(q),r));try{var
o=eM,g=eM;a:for(;;){if(1){var
k=function(a,b,c){var
e=b,d=c;for(;;){if(d){var
g=d[1],f=g.getLen(),h=d[2];a7(g,0,a,e-f|0,f);var
e=e-f|0,d=h;continue}return a}},d=0,e=0;for(;;){var
c=tl(f);if(0===c){if(!d)throw[0,bm];var
j=k(C(e),e,d)}else{if(!(0<c)){var
n=C(-c|0);c3(f,n,0,-c|0);var
d=[0,n,d],e=e-c|0;continue}var
i=C(c-1|0);c3(f,i,0,c-1|0);tk(f);if(d){var
l=(e+c|0)-1|0,j=k(C(l),l,[0,i,d])}else
var
j=i}var
g=h(g,h(j,lt)),o=g;continue a}}var
p=g;break}}catch(f){if(f[1]!==bm)throw f;var
p=o}dO(f);return p}var
eN=[0,lu],cM=[],lv=0,lw=0;tP(cM,[0,0,function(f){var
l=ep(f,lx),e=cx(lm),d=e.length-1,o=eL.length-1,a=v(d+o|0,0),p=d-1|0,u=0;if(!(p<0)){var
c=u;for(;;){j(a,c,aS(f,m(e,c)));var
y=c+1|0;if(p!==c){var
c=y;continue}break}}var
r=o-1|0,w=0;if(!(r<0)){var
b=w;for(;;){j(a,b+d|0,ep(f,m(eL,b)));var
x=b+1|0;if(r!==b){var
b=x;continue}break}}var
s=a[10],n=a[12],h=a[15],i=a[16],k=a[17],g=a[18],z=a[1],A=a[2],B=a[3],C=a[4],D=a[5],E=a[7],F=a[8],G=a[9],H=a[11],I=a[14];function
J(a,b,c,d,e,f){var
h=d?d[1]:d;q(a[1][n+1],a,[0,h],f);var
i=bt(a[g+1],f);return fh(a[1][s+1],a,b,[0,c[1],c[2]],e,f,i)}function
K(a,b,c,d,e){try{var
f=bt(a[g+1],e),h=f}catch(f){if(f[1]!==t)throw f;try{q(a[1][n+1],a,ly,e)}catch(f){throw f}var
h=bt(a[g+1],e)}return fh(a[1][s+1],a,b,[0,c[1],c[2]],d,e,h)}function
L(a,b,c){var
d=b?b[1]:b;try{bt(a[g+1],c);var
f=0}catch(f){if(f[1]===t){if(0===c[2][0]){var
e=a[i+1];if(!e)throw[0,eN,c];var
j=e[1],o=d?t3(j,a[h+1],c[1]):tV(j,a[h+1],c[1]),l=o}else{var
m=a[k+1];if(!m)throw[0,eN,c];var
n=m[1],p=d?ud(n,a[h+1],c[1]):ul(n,a[h+1],c[1]),l=p}return ef(a[g+1],c,l)}throw f}return f}function
M(a,b){try{var
f=[0,bB(a[l+1],lA),0],c=f}catch(f){var
c=0}a[i+1]=c;try{var
e=[0,bB(a[l+1],lz),0],d=e}catch(f){var
d=0}a[k+1]=d;return 0}function
N(a,b){a[k+1]=[0,b,0];return 0}function
O(a,b){return a[k+1]}function
P(a,b){a[i+1]=[0,b,0];return 0}function
Q(a,b){return a[i+1]}function
R(a,b){var
d=a[g+1];d[1]=0;var
e=d[2].length-1-1|0,f=0;if(!(e<0)){var
c=f;for(;;){j(d[2],c,0);var
h=c+1|0;if(e!==c){var
c=h;continue}break}}return 0}es(f,[0,G,function(a,b){return a[g+1]},C,R,F,Q,A,P,E,O,z,N,D,M,n,L,B,K,H,J]);return function(a,b,c,d){var
e=er(b,f);e[l+1]=c;e[I+1]=c;e[h+1]=d;try{var
o=[0,bB(c,lC),0],j=o}catch(f){var
j=0}e[i+1]=j;try{var
n=[0,bB(c,lB),0],m=n}catch(f){var
m=0}e[k+1]=m;e[g+1]=cp(0,8);return e}},lw,lv]);fi(0);fi(0);function
cN(a){function
e(a,b){var
d=a-1|0,e=0;if(!(d<0)){var
c=e;for(;;){d$(lE);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return k(d$(lD),b)}function
f(a,b){var
c=a,d=b;for(;;)if(typeof
d===o)return 0===d?e(c,lF):e(c,lG);else
switch(d[0]){case
0:e(c,lH);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
1:e(c,lI);var
c=c+1|0,d=d[1];continue;case
2:e(c,lJ);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
3:e(c,lK);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
4:e(c,lL);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
5:e(c,lM);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
6:e(c,lN);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
7:e(c,lO);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
8:e(c,lP);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
9:e(c,lQ);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
10:e(c,lR);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
11:return e(c,h(lS,d[1]));case
12:return e(c,h(lT,d[1]));case
13:return e(c,h(lU,l(d[1])));case
14:return e(c,h(lV,l(d[1])));case
15:return e(c,h(lW,l(d[1])));case
16:return e(c,h(lX,l(d[1])));case
17:return e(c,h(lY,l(d[1])));case
18:return e(c,lZ);case
19:return e(c,l0);case
20:return e(c,l1);case
21:return e(c,l2);case
22:return e(c,l3);case
23:return e(c,h(l4,l(d[2])));case
24:e(c,l5);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
25:e(c,l6);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
26:e(c,l7);var
c=c+1|0,d=d[1];continue;case
27:e(c,l8);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
28:e(c,l9);var
c=c+1|0,d=d[1];continue;case
29:e(c,l_);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
30:e(c,l$);f(c+1|0,d[1]);f(c+1|0,d[2]);var
c=c+1|0,d=d[3];continue;case
31:return e(c,ma);case
32:var
g=h(mb,l(d[2]));return e(c,h(mc,h(d[1],g)));case
33:return e(c,h(md,l(d[1])));case
36:e(c,mf);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
37:e(c,mg);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
38:e(c,mh);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
39:e(c,mi);f(c+1|0,d[1]);f(c+1|0,d[2]);var
c=c+1|0,d=d[3];continue;case
40:e(c,mj);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
41:e(c,mk);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
42:e(c,ml);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
43:e(c,mm);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
44:e(c,mn);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
45:e(c,mo);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
46:e(c,mp);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
47:e(c,mq);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
48:e(c,mr);f(c+1|0,d[1]);f(c+1|0,d[2]);f(c+1|0,d[3]);var
c=c+1|0,d=d[4];continue;case
49:e(c,ms);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
50:e(c,mt);f(c+1|0,d[1]);var
i=d[2],j=c+1|0;return dQ(function(a){return f(j,a)},i);case
51:return e(c,mu);case
52:return e(c,mv);default:return e(c,h(me,O(d[1])))}}return f(0,a)}function
K(a){return ai(a,32)}var
a2=[0,mw];function
a_(a,b,c){var
d=c;for(;;)if(typeof
d===o)return my;else
switch(d[0]){case
18:case
19:var
T=h(nb,h(e(b,d[2]),na));return h(nc,h(l(d[1]),T));case
27:case
38:var
ac=d[1],ad=h(nx,h(e(b,d[2]),nw));return h(e(b,ac),ad);case
0:var
g=d[2],B=e(b,d[1]);if(typeof
g===o)var
r=0;else
if(25===g[0]){var
t=e(b,g),r=1}else
var
r=0;if(!r){var
C=h(mz,K(b)),t=h(e(b,g),C)}return h(h(B,t),mA);case
1:var
D=h(e(b,d[1]),mB),G=x(a2[1][1],mC)?h(a2[1][1],mD):mF;return h(mE,h(G,D));case
2:var
H=h(mH,h(U(b,d[2]),mG));return h(mI,h(U(b,d[1]),H));case
3:var
I=h(mK,h(al(b,d[2]),mJ));return h(mL,h(al(b,d[1]),I));case
4:var
J=h(mN,h(U(b,d[2]),mM));return h(mO,h(U(b,d[1]),J));case
5:var
L=h(mQ,h(al(b,d[2]),mP));return h(mR,h(al(b,d[1]),L));case
6:var
M=h(mT,h(U(b,d[2]),mS));return h(mU,h(U(b,d[1]),M));case
7:var
P=h(mW,h(al(b,d[2]),mV));return h(mX,h(al(b,d[1]),P));case
8:var
Q=h(mZ,h(U(b,d[2]),mY));return h(m0,h(U(b,d[1]),Q));case
9:var
R=h(m2,h(al(b,d[2]),m1));return h(m3,h(al(b,d[1]),R));case
10:var
S=h(m5,h(U(b,d[2]),m4));return h(m6,h(U(b,d[1]),S));case
13:return h(m7,l(d[1]));case
14:return h(m8,l(d[1]));case
15:throw[0,E,m9];case
16:return h(m_,l(d[1]));case
17:return h(m$,l(d[1]));case
20:var
V=h(ne,h(e(b,d[2]),nd));return h(nf,h(l(d[1]),V));case
21:var
W=h(nh,h(e(b,d[2]),ng));return h(ni,h(l(d[1]),W));case
22:var
X=h(nk,h(e(b,d[2]),nj));return h(nl,h(l(d[1]),X));case
23:var
Y=h(nm,l(d[2])),u=d[1];if(typeof
u===o)var
f=0;else
switch(u[0]){case
33:var
n=no,f=1;break;case
34:var
n=np,f=1;break;case
35:var
n=nq,f=1;break;default:var
f=0}if(f)return h(n,Y);throw[0,E,nn];case
24:var
i=d[2],v=d[1];if(typeof
i===o){if(0===i){var
d=v;continue}}else
if(24===i[0]){var
Z=h(ns,e(b,i));return h(e(b,v),Z)}return N(nr);case
25:var
_=e(b,d[2]),$=h(nt,h(K(b),_));return h(e(b,d[1]),$);case
26:var
aa=e(b,d[1]),ab=x(a2[1][2],nu)?a2[1][2]:nv;return h(ab,aa);case
28:var
d=d[1];continue;case
29:var
ae=h(nz,h(e(b,d[2]),ny));return h(e(b,d[1]),ae);case
30:var
j=d[2],af=e(b,d[3]),ag=h(nA,h(K(b),af));if(typeof
j===o)var
s=0;else
if(31===j[0]){var
w=h(mx(j[1]),nC),s=1}else
var
s=0;if(!s)var
w=e(b,j);var
ah=h(nB,h(w,ag));return h(e(b,d[1]),ah);case
31:return a<50?a9(1+a,d[1]):F(a9,[0,d[1]]);case
33:return l(d[1]);case
34:return h(O(d[1]),nD);case
35:return O(d[1]);case
36:var
ai=h(nF,h(e(b,d[2]),nE));return h(e(b,d[1]),ai);case
37:var
aj=h(nH,h(K(b),nG)),ak=h(e(b,d[2]),aj),am=h(nI,h(K(b),ak)),an=h(nJ,h(e(b,d[1]),am));return h(K(b),an);case
39:var
ao=h(nK,K(b)),ap=h(e(b+2|0,d[3]),ao),aq=h(nL,h(K(b+2|0),ap)),ar=h(nM,h(K(b),aq)),as=h(e(b+2|0,d[2]),ar),at=h(nN,h(K(b+2|0),as));return h(nO,h(e(b,d[1]),at));case
40:var
au=h(nP,K(b)),av=h(e(b+2|0,d[2]),au),aw=h(nQ,h(K(b+2|0),av));return h(nR,h(e(b,d[1]),aw));case
41:var
ax=h(nS,e(b,d[2]));return h(e(b,d[1]),ax);case
42:var
ay=h(nT,e(b,d[2]));return h(e(b,d[1]),ay);case
43:var
az=h(nU,e(b,d[2]));return h(e(b,d[1]),az);case
44:var
aA=h(nV,e(b,d[2]));return h(e(b,d[1]),aA);case
45:var
aB=h(nW,e(b,d[2]));return h(e(b,d[1]),aB);case
46:var
aC=h(nX,e(b,d[2]));return h(e(b,d[1]),aC);case
47:var
aD=h(nY,e(b,d[2]));return h(e(b,d[1]),aD);case
48:var
p=e(b,d[1]),aE=e(b,d[2]),aF=e(b,d[3]),aG=h(e(b+2|0,d[4]),nZ);return h(n5,h(p,h(n4,h(aE,h(n3,h(p,h(n2,h(aF,h(n1,h(p,h(n0,h(K(b+2|0),aG))))))))))));case
49:var
aH=e(b,d[1]),aI=h(e(b+2|0,d[2]),n6);return h(n8,h(aH,h(n7,h(K(b+2|0),aI))));case
50:var
y=d[2],m=d[1],z=e(b,m),q=function(a){if(a){var
c=a[2],d=a[1];if(c){var
f=h(n9,q(c));return h(e(b,d),f)}return e(b,d)}throw[0,E,n_]};if(typeof
m!==o)if(31===m[0]){var
A=m[1];if(!x(A[1],ob))if(!x(A[2],oc))return h(z,h(oe,h(q(aN(y)),od)))}return h(z,h(oa,h(q(aN(y)),n$)));case
51:return l(k(d[1],0));case
52:return h(O(k(d[1],0)),of);default:return d[1]}}function
sG(a,b,c){if(typeof
c!==o)switch(c[0]){case
2:case
4:case
6:case
8:case
10:case
50:return a<50?a_(1+a,b,c):F(a_,[0,b,c]);case
32:return c[1];case
33:return l(c[1]);case
36:var
d=h(oh,h(U(b,c[2]),og));return h(e(b,c[1]),d);case
51:return l(k(c[1],0));default:}return a<50?c4(1+a,b,c):F(c4,[0,b,c])}function
c4(a,b,c){if(typeof
c!==o)switch(c[0]){case
3:case
5:case
7:case
9:case
29:case
50:return a<50?a_(1+a,b,c):F(a_,[0,b,c]);case
16:return h(oj,l(c[1]));case
31:return a<50?a9(1+a,c[1]):F(a9,[0,c[1]]);case
32:return c[1];case
34:return h(O(c[1]),ok);case
35:return h(ol,O(c[1]));case
36:var
d=h(on,h(U(b,c[2]),om));return h(e(b,c[1]),d);case
52:return h(O(k(c[1],0)),oo);default:}cN(c);return N(oi)}function
a9(a,b){return b[1]}function
e(b,c){return Y(a_(0,b,c))}function
U(b,c){return Y(sG(0,b,c))}function
al(b,c){return Y(c4(0,b,c))}function
mx(b){return Y(a9(0,b))}function
A(a){return ai(a,32)}var
a3=[0,op];function
ba(a,b,c){var
d=c;for(;;)if(typeof
d===o)return or;else
switch(d[0]){case
18:case
19:var
U=h(oO,h(f(b,d[2]),oN));return h(oP,h(l(d[1]),U));case
27:case
38:var
ac=d[1],ad=h(o9,R(b,d[2]));return h(f(b,ac),ad);case
0:var
g=d[2],C=f(b,d[1]);if(typeof
g===o)var
r=0;else
if(25===g[0]){var
t=f(b,g),r=1}else
var
r=0;if(!r){var
D=h(os,A(b)),t=h(f(b,g),D)}return h(h(C,t),ot);case
1:var
G=h(f(b,d[1]),ou),H=x(a3[1][1],ov)?h(a3[1][1],ow):oy;return h(ox,h(H,G));case
2:var
I=h(oz,R(b,d[2]));return h(R(b,d[1]),I);case
3:var
J=h(oA,am(b,d[2]));return h(am(b,d[1]),J);case
4:var
K=h(oB,R(b,d[2]));return h(R(b,d[1]),K);case
5:var
L=h(oC,am(b,d[2]));return h(am(b,d[1]),L);case
6:var
M=h(oD,R(b,d[2]));return h(R(b,d[1]),M);case
7:var
P=h(oE,am(b,d[2]));return h(am(b,d[1]),P);case
8:var
Q=h(oF,R(b,d[2]));return h(R(b,d[1]),Q);case
9:var
S=h(oG,am(b,d[2]));return h(am(b,d[1]),S);case
10:var
T=h(oH,R(b,d[2]));return h(R(b,d[1]),T);case
13:return h(oI,l(d[1]));case
14:return h(oJ,l(d[1]));case
15:throw[0,E,oK];case
16:return h(oL,l(d[1]));case
17:return h(oM,l(d[1]));case
20:var
V=h(oR,h(f(b,d[2]),oQ));return h(oS,h(l(d[1]),V));case
21:var
W=h(oU,h(f(b,d[2]),oT));return h(oV,h(l(d[1]),W));case
22:var
X=h(oX,h(f(b,d[2]),oW));return h(oY,h(l(d[1]),X));case
23:var
Y=h(oZ,l(d[2])),u=d[1];if(typeof
u===o)var
e=0;else
switch(u[0]){case
33:var
n=o1,e=1;break;case
34:var
n=o2,e=1;break;case
35:var
n=o3,e=1;break;default:var
e=0}if(e)return h(n,Y);throw[0,E,o0];case
24:var
i=d[2],v=d[1];if(typeof
i===o){if(0===i){var
d=v;continue}}else
if(24===i[0]){var
Z=h(o5,f(b,i));return h(f(b,v),Z)}return N(o4);case
25:var
_=f(b,d[2]),$=h(o6,h(A(b),_));return h(f(b,d[1]),$);case
26:var
aa=f(b,d[1]),ab=x(a3[1][2],o7)?a3[1][2]:o8;return h(ab,aa);case
28:var
d=d[1];continue;case
29:var
ae=h(o$,h(f(b,d[2]),o_));return h(f(b,d[1]),ae);case
30:var
j=d[2],af=f(b,d[3]),ag=h(pa,h(A(b),af));if(typeof
j===o)var
s=0;else
if(31===j[0]){var
w=oq(j[1]),s=1}else
var
s=0;if(!s)var
w=f(b,j);var
ah=h(pb,h(w,ag));return h(f(b,d[1]),ah);case
31:return a<50?a$(1+a,d[1]):F(a$,[0,d[1]]);case
33:return l(d[1]);case
34:return h(O(d[1]),pc);case
35:return O(d[1]);case
36:var
ai=h(pe,h(f(b,d[2]),pd));return h(f(b,d[1]),ai);case
37:var
aj=h(pg,h(A(b),pf)),ak=h(f(b,d[2]),aj),al=h(ph,h(A(b),ak)),an=h(pi,h(f(b,d[1]),al));return h(A(b),an);case
39:var
ao=h(pj,A(b)),ap=h(f(b+2|0,d[3]),ao),aq=h(pk,h(A(b+2|0),ap)),ar=h(pl,h(A(b),aq)),as=h(f(b+2|0,d[2]),ar),at=h(pm,h(A(b+2|0),as));return h(pn,h(f(b,d[1]),at));case
40:var
au=h(po,A(b)),av=h(pp,h(A(b),au)),aw=h(f(b+2|0,d[2]),av),ax=h(pq,h(A(b+2|0),aw)),ay=h(pr,h(A(b),ax));return h(ps,h(f(b,d[1]),ay));case
41:var
az=h(pt,f(b,d[2]));return h(f(b,d[1]),az);case
42:var
aA=h(pu,f(b,d[2]));return h(f(b,d[1]),aA);case
43:var
aB=h(pv,f(b,d[2]));return h(f(b,d[1]),aB);case
44:var
aC=h(pw,f(b,d[2]));return h(f(b,d[1]),aC);case
45:var
aD=h(px,f(b,d[2]));return h(f(b,d[1]),aD);case
46:var
aE=h(py,f(b,d[2]));return h(f(b,d[1]),aE);case
47:var
aF=h(pz,f(b,d[2]));return h(f(b,d[1]),aF);case
48:var
p=f(b,d[1]),aG=f(b,d[2]),aH=f(b,d[3]),aI=h(f(b+2|0,d[4]),pA);return h(pG,h(p,h(pF,h(aG,h(pE,h(p,h(pD,h(aH,h(pC,h(p,h(pB,h(A(b+2|0),aI))))))))))));case
49:var
aJ=f(b,d[1]),aK=h(f(b+2|0,d[2]),pH);return h(pJ,h(aJ,h(pI,h(A(b+2|0),aK))));case
50:var
y=d[2],m=d[1],z=f(b,m),q=function(a){if(a){var
c=a[2],d=a[1];if(c){var
e=h(pK,q(c));return h(f(b,d),e)}return f(b,d)}throw[0,E,pL]};if(typeof
m!==o)if(31===m[0]){var
B=m[1];if(!x(B[1],pO))if(!x(B[2],pP))return h(z,h(pR,h(q(aN(y)),pQ)))}return h(z,h(pN,h(q(aN(y)),pM)));case
51:return l(k(d[1],0));case
52:return h(O(k(d[1],0)),pS);default:return d[1]}}function
sH(a,b,c){if(typeof
c!==o)switch(c[0]){case
2:case
4:case
6:case
8:case
10:case
50:return a<50?ba(1+a,b,c):F(ba,[0,b,c]);case
32:return c[1];case
33:return l(c[1]);case
36:var
d=h(pU,h(R(b,c[2]),pT));return h(f(b,c[1]),d);case
51:return l(k(c[1],0));default:}return a<50?c5(1+a,b,c):F(c5,[0,b,c])}function
c5(a,b,c){if(typeof
c!==o)switch(c[0]){case
3:case
5:case
7:case
9:case
50:return a<50?ba(1+a,b,c):F(ba,[0,b,c]);case
16:return h(pW,l(c[1]));case
31:return a<50?a$(1+a,c[1]):F(a$,[0,c[1]]);case
32:return c[1];case
34:return h(O(c[1]),pX);case
35:return h(pY,O(c[1]));case
36:var
d=h(p0,h(R(b,c[2]),pZ));return h(f(b,c[1]),d);case
52:return h(O(k(c[1],0)),p1);default:}cN(c);return N(pV)}function
a$(a,b){return b[2]}function
f(b,c){return Y(ba(0,b,c))}function
R(b,c){return Y(sH(0,b,c))}function
am(b,c){return Y(c5(0,b,c))}function
oq(b){return Y(a$(0,b))}var
qb=h(qa,h(p$,h(p_,h(p9,h(p8,h(p7,h(p6,h(p5,h(p4,h(p3,p2)))))))))),qs=h(qr,h(qq,h(qp,h(qo,h(qn,h(qm,h(ql,h(qk,h(qj,h(qi,h(qh,h(qg,h(qf,h(qe,h(qd,qc))))))))))))))),qA=h(qz,h(qy,h(qx,h(qw,h(qv,h(qu,qt)))))),qI=h(qH,h(qG,h(qF,h(qE,h(qD,h(qC,qB))))));function
n(a){return[32,h(qJ,l(a)),a]}function
au(a,b){return[25,a,b]}function
eO(a,b){return[50,a,b]}function
eP(a,b,c){return[39,a,b,c]}function
eQ(a,b){return[40,a,b]}function
cO(a,b){return[24,a,b]}function
bC(a){return[13,a]}function
bD(a,b){return[29,a,b]}function
a4(a,b){return[31,[0,a,b]]}function
cP(a,b){return[37,a,b]}function
cQ(a,b){return[27,a,b]}function
cR(a){return[28,a]}function
eR(a,b){return[38,a,b]}function
V(a,b){return[36,a,b]}function
eS(a){var
e=[0,0];function
b(a){var
c=a;for(;;){if(typeof
c!==o)switch(c[0]){case
0:var
s=b(c[2]);return[0,b(c[1]),s];case
1:return[1,b(c[1])];case
2:var
t=b(c[2]);return[2,b(c[1]),t];case
3:var
h=c[2],i=c[1];if(typeof
i!==o)if(34===i[0])if(typeof
h!==o)if(34===h[0]){e[1]=1;return[34,i[1]+h[1]]}var
u=b(h);return[3,b(i),u];case
4:var
v=b(c[2]);return[4,b(c[1]),v];case
5:var
j=c[2],k=c[1];if(typeof
k!==o)if(34===k[0])if(typeof
j!==o)if(34===j[0]){e[1]=1;return[34,k[1]+j[1]]}var
w=b(j);return[5,b(k),w];case
6:var
x=b(c[2]);return[6,b(c[1]),x];case
7:var
l=c[2],m=c[1];if(typeof
m!==o)if(34===m[0])if(typeof
l!==o)if(34===l[0]){e[1]=1;return[34,m[1]+l[1]]}var
y=b(l);return[7,b(m),y];case
8:var
z=b(c[2]);return[8,b(c[1]),z];case
9:var
n=c[2],p=c[1];if(typeof
p!==o)if(34===p[0])if(typeof
n!==o)if(34===n[0]){e[1]=1;return[34,p[1]+n[1]]}var
A=b(n);return[9,b(p),A];case
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
d!==o)switch(d[0]){case
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
f!==o)switch(f[0]){case
25:var
P=b(f[2]),Q=[29,b(q),P];return[25,f[1],Q];case
39:e[1]=1;var
R=f[3],S=[29,b(q),R],T=b(f[2]),U=[29,b(q),T];return[39,b(f[1]),U,S];default:}var
O=b(f);return[29,b(q),O];case
30:var
V=b(c[3]),W=b(c[2]);return[30,b(c[1]),W,V];case
36:var
g=c[2],r=c[1];if(typeof
g!==o)if(25===g[0]){var
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
ao=aM(b,c[2]);return[50,b(c[1]),ao];default:}return c}}var
c=b(a);for(;;){if(e[1]){e[1]=0;var
c=b(c);continue}return c}}var
qQ=[0,qP];function
a5(a,b){var
d=b[2],c=b[1],s=a?a[1]:2,k=d[3],p=d[2];qQ[1]=qR;var
g=k[1];if(typeof
g===o)var
j=1===g?0:1;else
switch(g[0]){case
23:case
29:case
36:var
j=0;break;case
13:case
14:case
17:var
m=h(qZ,h(l(g[1]),qY)),j=2;break;default:var
j=1}switch(j){case
1:cN(k[1]);dP(dK);throw[0,E,qS];case
2:break;default:var
m=qT}var
n=[0,e(0,k[1]),m];a2[1]=n;a3[1]=n;function
q(a){var
q=d[4],r=dR(function(a,b){return 0===b?a:h(qA,a)},qI,q),s=h(r,e(0,eS(p))),j=cY(e9(qU,gO,438));dL(j,s);cZ(j);e_(j);fj(qV);var
m=dM(qW),f=th(m),n=C(f),o=0;if(0<=0)if(0<=f)if((n.getLen()-f|0)<o)var
g=0;else{var
k=o,b=f;for(;;){if(0<b){var
l=c3(m,n,k,b);if(0===l)throw[0,bm];var
k=k+l|0,b=b-l|0;continue}var
g=1;break}}else
var
g=0;else
var
g=0;if(!g)H(gQ);dO(m);i(aa(c,723535973,3),c,n);fj(qX);return 0}function
r(a){var
b=d[4],e=dR(function(a,b){return 0===b?a:h(qs,a)},qb,b);return i(aa(c,56985577,4),c,h(e,f(0,eS(p))))}switch(s){case
1:r(0);break;case
2:q(0);r(0);break;default:q(0)}i(aa(c,345714255,5),c,0);return[0,c,d]}var
cS=d,q4=1,q5=1,q6=1,q7=null,q8=undefined;function
eT(a,b){return a==q7?k(b,0):a}var
eU=Array,q9=true,q_=false;ec(function(a){return a
instanceof
eU?0:[0,new
an(a.toString())]});function
B(a,b){a.appendChild(b);return 0}function
eV(d){return te(function(a){if(a){var
e=k(d,a);if(!(e|0))a.preventDefault();return e}var
c=event,b=k(d,c);if(!(b|0))c.returnValue=b;return b})}var
S=cS.document;function
bE(a,b){return a?k(b,a[1]):0}function
bF(a,b){return a.createElement(b.toString())}function
eW(a,b){return bF(a,b)}var
eX=[0,f9];function
eY(a,b,c,d){for(;;){if(0===a)if(0===b)return bF(c,d);var
h=eX[1];if(f9===h){try{var
j=S.createElement('<input name="x">'),k=j.tagName.toLowerCase()===fs?1:0,m=k?j.name===dp?1:0:k,i=m}catch(f){var
i=0}var
l=i?fW:-1003883683;eX[1]=l;continue}if(fW<=h){var
e=new
eU();e.push("<",d.toString());bE(a,function(a){e.push(' type="',fk(a),b0);return 0});bE(b,function(a){e.push(' name="',fk(a),b0);return 0});e.push(">");return c.createElement(e.join(g))}var
f=bF(c,d);bE(a,function(a){return f.type=a});bE(b,function(a){return f.name=a});return f}}function
eZ(a,b,c){return eY(a,b,c,rb)}function
e0(a){return eW(a,rc)}cS.HTMLElement===q8;function
e4(a){return e0(S)}function
cT(a){function
c(a){throw[0,E,re]}var
b=eT(S.getElementById(gs),c);return k(ea(function(a){B(b,e4(0));B(b,S.createTextNode(a.toString()));return B(b,e4(0))}),a)}function
rf(a){var
k=[0,[1,a[1]],[6,a[2]],[6,a[3]]];return function(a,b,c,d){var
h=a[2],i=a[1],l=c[2];if(0===l[0]){var
g=l[1],e=[0,0],f=tX(k.length-1),m=g[7][1]<i[1]?1:0;if(m)var
n=m;else{var
s=g[7][2]<i[2]?1:0,n=s||(g[7][3]<i[3]?1:0)}if(n)throw[0,lq];var
o=g[8][1]<h[1]?1:0;if(o)var
p=o;else{var
r=g[8][2]<h[2]?1:0,p=r||(g[8][3]<h[3]?1:0)}if(p)throw[0,ls];bn(function(a,b){function
h(a){if(bA)try{eJ(a,0,c);Q(c,0,0)}catch(f){if(f[1]===at)throw[0,at];throw f}return 11===b[0]?t0(e,f,cI(b[1],dx,c[1][8]),a):ua(e,f,cI(a,dx,c[1][8]),a,c)}switch(b[0]){case
6:case
7:case
8:case
9:case
10:switch(b[0]){case
6:var
d=t$(e,f,b[1]);break;case
7:var
d=t_(e,f,b[1]);break;case
8:var
d=t9(e,f,b[1]);break;case
9:var
d=t8(e,f,b[1]);break;default:var
d=N(ln)}var
g=d;break;case
11:var
g=h(b[1]);break;default:var
g=h(b[1])}return g},k);var
q=t7(e,d,h,i,f,c[1],b)}else{var
j=[0,0];bn(function(a,b){switch(b[0]){case
6:case
7:case
8:case
9:case
10:switch(b[0]){case
6:var
e=uy(j,d,b[1],c[1]);break;case
7:var
e=uz(j,d,b[1],c[1]);break;case
8:var
e=uw(j,d,b[1],c[1]);break;case
9:var
e=ux(j,d,b[1],c[1]);break;default:var
e=N(lo)}var
g=e;break;default:var
f=b[1];if(bA){if(c2(aX(f),[0,c]))eJ(f,0,c);Q(c,0,0)}var
h=c[1],i=J(0),g=uA(j,d,a,cI(f,-701974253,c[1][8]-i|0),h)}return g},k);var
q=uv(d,h,i,c[1],b)}return q}}if(cU===0)var
c=ei([0]);else{var
aV=ei(aM(hR,cU));bn(function(a,b){var
c=(a*2|0)+2|0;aV[3]=q(ar[4],b,c,aV[3]);aV[4]=q(ak[4],c,1,aV[4]);return 0},cU);var
c=aV}var
cu=aM(function(a){return aS(c,a)},e3),eq=cM[2],rg=cu[1],rh=cu[2],ri=cu[3],h$=cM[4],ek=cv(e1),el=cv(e3),em=cv(e2),rj=1,cw=cb(function(a){return aS(c,a)},el),hU=cb(function(a){return aS(c,a)},em);c[5]=[0,[0,c[3],c[4],c[6],c[7],cw,ek],c[5]];var
hV=ac[1],hW=c[7];function
hX(a,b,c){return ce(a,ek)?q(ac[4],a,b,c):c}c[7]=q(ac[11],hX,hW,hV);var
aT=[0,ar[1]],aU=[0,ak[1]];dU(function(a,b){aT[1]=q(ar[4],a,b,aT[1]);var
e=aU[1];try{var
f=i(ak[22],b,c[4]),d=f}catch(f){if(f[1]!==t)throw f;var
d=1}aU[1]=q(ak[4],b,d,e);return 0},em,hU);dU(function(a,b){aT[1]=q(ar[4],a,b,aT[1]);aU[1]=q(ak[4],b,0,aU[1]);return 0},el,cw);c[3]=aT[1];c[4]=aU[1];var
hY=0,hZ=c[6];c[6]=cd(function(a,b){return ce(a[1],cw)?b:[0,a,b]},hZ,hY);var
ia=rj?i(eq,c,h$):k(eq,c),en=c[5],aA=en?en[1]:N(gW),eo=c[5],h0=aA[6],h1=aA[5],h2=aA[4],h3=aA[3],h4=aA[2],h5=aA[1],h6=eo?eo[2]:N(gX);c[5]=h6;var
cc=h2,bo=h0;for(;;){if(bo){var
dT=bo[1],gY=bo[2],h7=i(ac[22],dT,c[7]),cc=q(ac[4],dT,h7,cc),bo=gY;continue}c[7]=cc;c[3]=h5;c[4]=h4;var
h8=c[6];c[6]=cd(function(a,b){return ce(a[1],h1)?b:[0,a,b]},h8,h3);var
ib=0,ic=cx(e2),id=[0,aM(function(a){var
e=aS(c,a);try{var
b=c[6];for(;;){if(!b)throw[0,t];var
d=b[1],f=b[2],h=d[2];if(0!==av(d[1],e)){var
b=f;continue}var
g=h;break}}catch(f){if(f[1]!==t)throw f;var
g=m(c[2],e)}return g},ic),ib],ie=cx(e1),rk=sJ([0,[0,ia],[0,aM(function(a){try{var
b=i(ac[22],a,c[7])}catch(f){if(f[1]===t)throw[0,E,h_];throw f}return b},ie),id]])[1],rl=function(a,b){if(3===b.length-1){var
c=b[0+1];if(1===c[0]){var
d=b[1+1];if(6===d[0]){var
e=b[2+1];if(6===e[0])return[0,c[1],d[1],e[1]]}}}return N(rm)};es(c,[0,rh,0,rf,ri,function(a,b){return[0,[1,b[1]],[6,b[2]],[6,b[3]]]},rg,rl]);var
rn=function(a,b){var
e=er(b,c);q(rk,e,rp,ro);if(!b){var
f=c[8];if(0!==f){var
d=f;for(;;){if(d){var
g=d[2];k(d[1],e);var
d=g;continue}break}}}return e};ej[1]=(ej[1]+c[1]|0)-1|0;c[8]=dS(c[8]);cs(c,3+aw(m(c[2],1)*16|0,az)|0);var
ig=0,ih=function(a){var
b=a;return rn(ig,b)},rt=n(6),ru=n(4),rv=bD(V(n(0),ru),rt),rw=n(4),rx=V(n(0),rw),ry=n(5),rz=au(bD(V(n(0),ry),rx),rv),rA=n(5),rB=V(n(0),rA),rC=au(eR(n(6),rB),rz),rD=n(5),rE=V(n(0),rD),rF=n(4),rG=eQ([44,V(n(0),rF),rE],rC),rH=n(6),rI=n(4),rJ=bD(V(n(0),rI),rH),rK=n(4),rL=V(n(0),rK),rM=n(5),rN=au(bD(V(n(0),rM),rL),rJ),rO=n(5),rP=V(n(0),rO),rQ=au(eR(n(6),rP),rN),rR=n(5),rS=V(n(0),rR),rT=n(4),rU=eQ([45,V(n(0),rT),rS],rQ),rV=n(2),rW=[0,n(4),rV],rZ=eP([43,eO(a4(rY,rX),rW),[33,0]],rU,rG),r0=n(4),r1=eP([44,n(5),r0],1,rZ),r2=au(cQ(n(6),[34,0]),r1),r3=n(1),r4=[0,n(4),r3],r7=eO(a4(r6,r5),r4),r8=au(cQ(n(5),r7),r2),r$=a4(r_,r9),qM=[6,a4(sb,sa),r$],qL=[2,a4(sd,sc),qM],qN=[26,au(cQ(n(4),qL),r8)],se=cP(cR([14,6]),qN),sf=cP(cR(bC(5)),se),sg=cP(cR(bC(4)),sf),sh=cO(bC(2),0),rq=[0,0],rs=[0,1,rr],qK=[0,[1,cO([23,qO,0],cO(bC(1),sh))],sg],si=[0,function(a,b,c){var
d=q4+(q6*q5|0)|0,e=d^b;if(e<d)return 0;if(0===(d&c)){var
h=z(a,e),f=h<z(a,d)?1:0;if(f){var
i=z(a,e);X(a,e,z(a,d));return X(a,d,i)}return f}var
j=z(a,e),g=z(a,d)<j?1:0;if(g){var
k=z(a,e);X(a,e,z(a,d));return X(a,d,k)}return g},qK,rs,rq],cV=[0,ih(0),si],e5=[0,0],sj=[0,0],e6=function(a,b){var
c=eA(0),e=k(b,0),d=eA(0);i(cT(sk),a,d-c);e5[1]=e5[1]+(d-c);sj[1]++;return e},bG=function(a){return e0(S)},cW=function(a){var
c=a.toString(),b=bF(S,rd);B(b,S.createTextNode(c));return b};cS.onload=eV(function(a){var
J=e7?e7[1]:2;switch(J){case
1:fl(0);aD[1]=fm(0);break;case
2:fn(0);aC[1]=fo(0);fl(0);aD[1]=fm(0);break;default:fn(0);aC[1]=fo(0)}eB[1]=aC[1]+aD[1]|0;var
x=aC[1]-1|0,w=0,K=0;if(x<0)var
y=w;else{var
f=K,D=w;for(;;){var
F=b$(D,[0,ue(f),0]),L=f+1|0;if(x!==f){var
f=L,D=F;continue}var
y=F;break}}var
q=0,d=0,c=y;for(;;){if(q<aD[1]){if(uu(d)){var
C=d+1|0,A=b$(c,[0,ug(d,d+aC[1]|0),0])}else{var
C=d,A=c}var
q=q+1|0,d=C,c=A;continue}var
p=0,o=c;for(;;){if(o){var
p=p+1|0,o=o[2];continue}eB[1]=p;aD[1]=d;if(c){var
k=0,h=c,G=c[2],H=c[1];for(;;){if(h){var
k=k+1|0,h=h[2];continue}var
u=v(k,H),n=1,e=G;for(;;){if(e){var
I=e[2];u[n+1]=e[1];var
n=n+1|0,e=I;continue}var
g=u;break}break}}else
var
g=[0];var
M=cp(0,g.length-1);bn(function(a,b){return ef(M,b[1][1],a)},g);var
O=function(a){throw[0,E,sr]},b=eT(S.getElementById(gs),O);B(b,cW(ss));B(b,bG(0));B(b,bG(0));var
r=eY(0,0,S,ra);B(b,cW(st));dQ(function(a){var
b=eW(S,q$);B(b,S.createTextNode(a[1][1].toString()));return B(r,b)},g);B(b,r);B(b,bG(0));var
l=eZ([0,"text"],0,S);l.value="10";l.size=4;B(b,cW(su));B(b,l);B(b,bG(0));var
P=function(a){var
d=2,t=e$(new
an(l.value));for(;;){if(1===t){var
n=m(g,r.selectedIndex+0|0),B=n[1][1];i(cT(sl),B,d);var
e=as(eD,0,d),c=v(d,0),C=as(eD,0,d);ed(cn,fa(0));var
u=W(e)-1|0,D=0;if(!(u<0)){var
f=D;for(;;){var
A=cm(cn),q=(A/gb+cm(cn))/gb*s;X(e,f,q);X(C,f,q);j(c,f,q);var
M=f+1|0;if(u!==f){var
f=M;continue}break}}e6(sm,function(a){function
g(a,b){return av(a,b)}function
p(a,b){var
d=((b+b|0)+b|0)+1|0,e=[0,d];if((d+2|0)<a){if(g(m(c,d),m(c,d+1|0))<0)e[1]=d+1|0;if(g(m(c,e[1]),m(c,d+2|0))<0)e[1]=d+2|0;return e[1]}if((d+1|0)<a)if(!(0<=g(m(c,d),m(c,d+1|0))))return d+1|0;if(d<a)return d;throw[0,ca,b]}var
i=c.length-1,r=((i+1|0)/3|0)-1|0,v=0;if(!(r<0)){var
f=r;for(;;){var
o=m(c,f);try{var
h=f;for(;;){var
k=p(i,h);if(0<g(m(c,k),o)){j(c,h,m(c,k));var
h=k;continue}j(c,h,o);break}}catch(f){if(f[1]!==ca)throw f;j(c,f[2],o)}var
A=f-1|0;if(v!==f){var
f=A;continue}break}}var
s=i-1|0,w=2;if(!(s<2)){var
b=s;a:for(;;){var
n=m(c,b);j(c,b,m(c,0));var
y=0;try{var
l=y;for(;;){var
q=p(b,l);j(c,l,m(c,q));var
l=q;continue}}catch(f){if(f[1]!==ca)throw f;var
d=f[2];for(;;){var
e=(d-1|0)/3|0;if(d===e)throw[0,E,gV];if(0<=g(m(c,e),n))j(c,d,n);else{j(c,d,m(c,e));if(0<e){var
d=e;continue}j(c,0,n)}var
z=b-1|0;if(w!==b){var
b=z;continue a}break}}break}}var
t=1<i?1:0;if(t){var
x=m(c,1);j(c,1,m(c,0));var
u=j(c,0,x)}else
var
u=t;return u});var
w=n[2];if(0===w[0])var
o=b2;else{var
L=0===w[1][2]?1:b2,o=L}a5(sn,cV);var
h=[0,2],k=[0,0],F=[0,o,1,1],G=[0,aw((d+o|0)-1|0,o),1,1];e6(so,function(a){a:for(;;){if(h[1]<=d){k[1]=h[1]>>>1;for(;;){if(0<k[1]){var
c=cV[2],b=cV[1],j=0,l=[0,F,G],m=[0,e,k[1],h[1]],g=0,f=0?g[1]:g;if(0===n[2][0]){if(f)a5(q0,[0,b,c]);else
if(!i(aa(b,-723625231,7),b,0))a5(q1,[0,b,c])}else
if(f)a5(q2,[0,b,c]);else
if(!i(aa(b,649483637,8),b,0))a5(q3,[0,b,c]);(function(a,b,c,d,e,f){return a.length==5?a(b,c,d,e,f):ah(a,[b,c,d,e,f])}(aa(b,5695307,6),b,m,l,j,n));k[1]=k[1]>>>1;continue}h[1]=h[1]<<1;continue a}}return 0}});var
x=d-1|0,H=-gJ,I=0;if(!(x<0)){var
b=I,p=H;for(;;){if(z(e,b)<p){var
J=z(e,b);N(i(cl(sq),J,p));var
y=p}else
var
y=z(e,b);var
K=b+1|0;if(x!==b){var
b=K,p=y;continue}break}}cT(sp);return q9}var
d=d*2|0,t=t-1|0;continue}},t=eZ([0,"button"],0,S);t.value="Go";t.onclick=eV(P);B(b,t);return q_}}});dN(0);return}}(this));
