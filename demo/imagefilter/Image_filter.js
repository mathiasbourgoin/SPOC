// This program was compiled from OCaml by js_of_ocaml 1.99dev
(function(d){"use strict";var
c$="set_cuda_sources",dr=123,bP=";",fD=108,ge="section1",c_="reload_sources",bT="Map.bal",fQ=",",bZ='"',_=16777215,c9="get_cuda_sources",b7=" / ",fC="double spoc_var",dh="args_to_list",bY=" * ",ae="(",fq="float spoc_var",dg=65599,b6="if (",bX="return",fP=" ;\n",dq="exec",bh=115,bf=";}\n",fB=".ptx",p=512,dp=120,c8="..",fO=-512,L="]",dn=117,bS="; ",dm="compile",gd=" (",W="0",df="list_to_args",bR=248,fN=126,gc="fd ",c7="get_binaries",fA=" == ",de="Kirc_Cuda.ml",b5=" + ",fM=") ",dl="x",fz=-97,fp="g",bd=1073741823,gb="parse concat",av=105,dd="get_opencl_sources",ga=511,be=110,f$=-88,ac=" = ",dc="set_opencl_sources",M="[",bW="'",fo="Unix",bO="int_of_string",f_="(double) ",fL=982028505,bc="){\n",bg="e",f9="#define __FLOAT64_EXTENSION__ \n",au="-",aK=-48,bV="(double) spoc_var",fn="++){\n",fy="__shared__ float spoc_var",f7="Image_filter_js.ml",f8="opencl_sources",fx=".cl",dk="reset_binaries",bN="\n",f6=101,du=748841679,b4="index out of bounds",fm="spoc_init_opencl_device_vec",c6=125,bU=" - ",f5=";}",r=255,f4="binaries",b3="}",f3=" < ",fl="__shared__ long spoc_var",aJ=250,f2=" >= ",fk="input",fK=246,db=102,fJ="Unix.Unix_error",g="",fj=" || ",aI=100,dj="Kirc_OpenCL.ml",f1="#ifndef __FLOAT64_EXTENSION__ \n",fI="__shared__ int spoc_var",dt=103,bM=", ",fH="./",fw=1e3,fi="for (int ",f0="file_file",fZ="spoc_var",af=".",fv="else{\n",bQ="+",ds="run",b2=65535,di="#endif\n",aH=";\n",X="f",fY=785140586,fX="__shared__ double spoc_var",fu=-32,da=111,fG=" > ",B=" ",fW="int spoc_var",ad=")",fF="cuda_sources",b1=256,ft="nan",c5=116,fT="../",fU="kernel_name",fV=65520,fS="%.12g",fh=" && ",fs="/",fE="while (",c4="compile_and_run",b0=114,fR="* spoc_var",bL=" <= ",m="number",fr=" % ",t0=d.spoc_opencl_part_device_to_cpu_b!==undefined?d.spoc_opencl_part_device_to_cpu_b:function(){n("spoc_opencl_part_device_to_cpu_b not implemented")},tZ=d.spoc_opencl_part_cpu_to_device_b!==undefined?d.spoc_opencl_part_cpu_to_device_b:function(){n("spoc_opencl_part_cpu_to_device_b not implemented")},tX=d.spoc_opencl_load_param_int64!==undefined?d.spoc_opencl_load_param_int64:function(){n("spoc_opencl_load_param_int64 not implemented")},tV=d.spoc_opencl_load_param_float64!==undefined?d.spoc_opencl_load_param_float64:function(){n("spoc_opencl_load_param_float64 not implemented")},tU=d.spoc_opencl_load_param_float!==undefined?d.spoc_opencl_load_param_float:function(){n("spoc_opencl_load_param_float not implemented")},tP=d.spoc_opencl_custom_part_device_to_cpu_b!==undefined?d.spoc_opencl_custom_part_device_to_cpu_b:function(){n("spoc_opencl_custom_part_device_to_cpu_b not implemented")},tO=d.spoc_opencl_custom_part_cpu_to_device_b!==undefined?d.spoc_opencl_custom_part_cpu_to_device_b:function(){n("spoc_opencl_custom_part_cpu_to_device_b not implemented")},tN=d.spoc_opencl_custom_device_to_cpu!==undefined?d.spoc_opencl_custom_device_to_cpu:function(){n("spoc_opencl_custom_device_to_cpu not implemented")},tM=d.spoc_opencl_custom_cpu_to_device!==undefined?d.spoc_opencl_custom_cpu_to_device:function(){n("spoc_opencl_custom_cpu_to_device not implemented")},tL=d.spoc_opencl_custom_alloc_vect!==undefined?d.spoc_opencl_custom_alloc_vect:function(){n("spoc_opencl_custom_alloc_vect not implemented")},tA=d.spoc_cuda_part_device_to_cpu_b!==undefined?d.spoc_cuda_part_device_to_cpu_b:function(){n("spoc_cuda_part_device_to_cpu_b not implemented")},tz=d.spoc_cuda_part_cpu_to_device_b!==undefined?d.spoc_cuda_part_cpu_to_device_b:function(){n("spoc_cuda_part_cpu_to_device_b not implemented")},ty=d.spoc_cuda_load_param_vec_b!==undefined?d.spoc_cuda_load_param_vec_b:function(){n("spoc_cuda_load_param_vec_b not implemented")},tx=d.spoc_cuda_load_param_int_b!==undefined?d.spoc_cuda_load_param_int_b:function(){n("spoc_cuda_load_param_int_b not implemented")},tw=d.spoc_cuda_load_param_int64_b!==undefined?d.spoc_cuda_load_param_int64_b:function(){n("spoc_cuda_load_param_int64_b not implemented")},tv=d.spoc_cuda_load_param_float_b!==undefined?d.spoc_cuda_load_param_float_b:function(){n("spoc_cuda_load_param_float_b not implemented")},tu=d.spoc_cuda_load_param_float64_b!==undefined?d.spoc_cuda_load_param_float64_b:function(){n("spoc_cuda_load_param_float64_b not implemented")},tt=d.spoc_cuda_launch_grid_b!==undefined?d.spoc_cuda_launch_grid_b:function(){n("spoc_cuda_launch_grid_b not implemented")},ts=d.spoc_cuda_flush_all!==undefined?d.spoc_cuda_flush_all:function(){n("spoc_cuda_flush_all not implemented")},tr=d.spoc_cuda_flush!==undefined?d.spoc_cuda_flush:function(){n("spoc_cuda_flush not implemented")},tq=d.spoc_cuda_device_to_cpu!==undefined?d.spoc_cuda_device_to_cpu:function(){n("spoc_cuda_device_to_cpu not implemented")},to=d.spoc_cuda_custom_part_device_to_cpu_b!==undefined?d.spoc_cuda_custom_part_device_to_cpu_b:function(){n("spoc_cuda_custom_part_device_to_cpu_b not implemented")},tn=d.spoc_cuda_custom_part_cpu_to_device_b!==undefined?d.spoc_cuda_custom_part_cpu_to_device_b:function(){n("spoc_cuda_custom_part_cpu_to_device_b not implemented")},tm=d.spoc_cuda_custom_load_param_vec_b!==undefined?d.spoc_cuda_custom_load_param_vec_b:function(){n("spoc_cuda_custom_load_param_vec_b not implemented")},tl=d.spoc_cuda_custom_device_to_cpu!==undefined?d.spoc_cuda_custom_device_to_cpu:function(){n("spoc_cuda_custom_device_to_cpu not implemented")},tk=d.spoc_cuda_custom_cpu_to_device!==undefined?d.spoc_cuda_custom_cpu_to_device:function(){n("spoc_cuda_custom_cpu_to_device not implemented")},gt=d.spoc_cuda_custom_alloc_vect!==undefined?d.spoc_cuda_custom_alloc_vect:function(){n("spoc_cuda_custom_alloc_vect not implemented")},tj=d.spoc_cuda_create_extra!==undefined?d.spoc_cuda_create_extra:function(){n("spoc_cuda_create_extra not implemented")},ti=d.spoc_cuda_cpu_to_device!==undefined?d.spoc_cuda_cpu_to_device:function(){n("spoc_cuda_cpu_to_device not implemented")},gs=d.spoc_cuda_alloc_vect!==undefined?d.spoc_cuda_alloc_vect:function(){n("spoc_cuda_alloc_vect not implemented")},tf=d.spoc_create_custom!==undefined?d.spoc_create_custom:function(){n("spoc_create_custom not implemented")},t3=1;function
go(a,b){throw[0,a,b]}function
dE(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=d.console;b&&b.error&&b.error(a)}var
q=[0];function
bk(a,b){if(!a)return g;if(a&1)return bk(a-1,b)+b;var
c=bk(a>>1,b);return c+c}function
D(a){if(a!=null){this.bytes=this.fullBytes=a;this.last=this.len=a.length}}function
gr(){go(q[4],new
D(b4))}D.prototype={string:null,bytes:null,fullBytes:null,array:null,len:null,last:0,toJsString:function(){var
a=this.getFullBytes();try{return this.string=decodeURIComponent(escape(a))}catch(f){dE('MlString.toJsString: wrong encoding for \"%s\" ',a);return a}},toBytes:function(){if(this.string!=null)try{var
a=unescape(encodeURIComponent(this.string))}catch(f){dE('MlString.toBytes: wrong encoding for \"%s\" ',this.string);var
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
b=this.bytes;if(b==null)b=this.toBytes();return a<this.last?b.charCodeAt(a):0},safeGet:function(a){if(this.len==null)this.toBytes();if(a<0||a>=this.len)gr();return this.get(a)},set:function(a,b){var
c=this.array;if(!c){if(this.last==a){this.bytes+=String.fromCharCode(b&r);this.last++;return 0}c=this.toArray()}else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;c[a]=b&r;return 0},safeSet:function(a,b){if(this.len==null)this.toBytes();if(a<0||a>=this.len)gr();this.set(a,b)},fill:function(a,b,c){if(a>=this.last&&this.last&&c==0)return;var
d=this.array;if(!d)d=this.toArray();else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;var
f=a+b;for(var
e=a;e<f;e++)d[e]=c},compare:function(a){if(this.string!=null&&a.string!=null){if(this.string<a.string)return-1;if(this.string>a.string)return 1;return 0}var
b=this.getFullBytes(),c=a.getFullBytes();if(b<c)return-1;if(b>c)return 1;return 0},equal:function(a){if(this.string!=null&&a.string!=null)return this.string==a.string;return this.getFullBytes()==a.getFullBytes()},lessThan:function(a){if(this.string!=null&&a.string!=null)return this.string<a.string;return this.getFullBytes()<a.getFullBytes()},lessEqual:function(a){if(this.string!=null&&a.string!=null)return this.string<=a.string;return this.getFullBytes()<=a.getFullBytes()}};function
aw(a){this.string=a}aw.prototype=new
D();function
r7(a,b,c,d,e){if(d<=b)for(var
f=1;f<=e;f++)c[d+f]=a[b+f];else
for(var
f=e;f>=1;f--)c[d+f]=a[b+f]}function
r8(a){var
c=[0];while(a!==0){var
d=a[1];for(var
b=1;b<d.length;b++)c.push(d[b]);a=a[2]}return c}function
dD(a,b){go(a,new
aw(b))}function
an(a){dD(q[4],a)}function
aL(){an(b4)}function
r9(a,b){if(b<0||b>=a.length-1)aL();return a[b+1]}function
r_(a,b,c){if(b<0||b>=a.length-1)aL();a[b+1]=c;return 0}var
dw;function
r$(a,b,c){if(c.length!=2)an("Bigarray.create: bad number of dimensions");if(b!=0)an("Bigarray.create: unsupported layout");if(c[1]<0)an("Bigarray.create: negative dimension");if(!dw){var
e=d;dw=[e.Float32Array,e.Float64Array,e.Int8Array,e.Uint8Array,e.Int16Array,e.Uint16Array,e.Int32Array,null,e.Int32Array,e.Int32Array,null,null,e.Uint8Array]}var
f=dw[a];if(!f)an("Bigarray.create: unsupported kind");return new
f(c[1])}function
sa(a,b){if(b<0||b>=a.length)aL();return a[b]}function
sb(a,b,c){if(b<0||b>=a.length)aL();a[b]=c;return 0}function
dx(a,b,c,d,e){if(e===0)return;if(d===c.last&&c.bytes!=null){var
f=a.bytes;if(f==null)f=a.toBytes();if(b>0||a.last>e)f=f.slice(b,b+e);c.bytes+=f;c.last+=f.length;return}var
g=c.array;if(!g)g=c.toArray();else
c.bytes=c.string=null;a.blitToArray(b,g,d,e)}function
ag(c,b){if(c.fun)return ag(c.fun,b);var
a=c.length,d=a-b.length;if(d==0)return c.apply(null,b);else
if(d<0)return ag(c.apply(null,b.slice(0,a)),b.slice(a));else
return function(a){return ag(c,b.concat([a]))}}function
sc(a){if(isFinite(a)){if(Math.abs(a)>=2.22507385850720138e-308)return 0;if(a!=0)return 1;return 2}return isNaN(a)?4:3}function
so(a,b){var
c=a[3]<<16,d=b[3]<<16;if(c>d)return 1;if(c<d)return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
gl(a,b){if(a<b)return-1;if(a==b)return 0;return 1}function
dy(a,b,c){var
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
d=gl(a[2],b[2]);if(d!=0)return d;break}case
251:an("equal: abstract value");case
r:{var
d=so(a,b);if(d!=0)return d;break}default:if(a.length!=b.length)return a.length<b.length?-1:1;if(a.length>1)e.push(a,b,1)}}else
return 1}else
if(b
instanceof
D||b
instanceof
Array&&b[0]===(b[0]|0))return-1;else{if(a<b)return-1;if(a>b)return 1;if(c&&a!=b){if(a==a)return 1;if(b==b)return-1}}if(e.length==0)return 0;var
f=e.pop();b=e.pop();a=e.pop();if(f+1<a.length)e.push(a,b,f+1);a=a[f];b=b[f]}}function
gg(a,b){return dy(a,b,true)}function
gf(a){this.bytes=g;this.len=a}gf.prototype=new
D();function
gh(a){if(a<0)an("String.create");return new
gf(a)}function
dC(a){throw[0,a]}function
gp(){dC(q[6])}function
sd(a,b){if(b==0)gp();return a/b|0}function
se(a,b){return+(dy(a,b,false)==0)}function
sf(a,b,c,d){a.fill(b,c,d)}function
dB(a){a=a.toString();var
e=a.length;if(e>31)an("format_int: format too long");var
b={justify:bQ,signstyle:au,filler:B,alternate:false,base:0,signedconv:false,width:0,uppercase:false,sign:1,prec:-1,conv:X};for(var
d=0;d<e;d++){var
c=a.charAt(d);switch(c){case
au:b.justify=au;break;case
bQ:case
B:b.signstyle=c;break;case
W:b.filler=W;break;case"#":b.alternate=true;break;case"1":case"2":case"3":case"4":case"5":case"6":case"7":case"8":case"9":b.width=0;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.width=b.width*10+c;d++}d--;break;case
af:b.prec=0;d++;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.prec=b.prec*10+c;d++}d--;case"d":case"i":b.signedconv=true;case"u":b.base=10;break;case
dl:b.base=16;break;case"X":b.base=16;b.uppercase=true;break;case"o":b.base=8;break;case
bg:case
X:case
fp:b.signedconv=true;b.conv=c;break;case"E":case"F":case"G":b.signedconv=true;b.uppercase=true;b.conv=c.toLowerCase();break}}return b}function
dz(a,b){if(a.uppercase)b=b.toUpperCase();var
e=b.length;if(a.signedconv&&(a.sign<0||a.signstyle!=au))e++;if(a.alternate){if(a.base==8)e+=1;if(a.base==16)e+=2}var
c=g;if(a.justify==bQ&&a.filler==B)for(var
d=e;d<a.width;d++)c+=B;if(a.signedconv)if(a.sign<0)c+=au;else
if(a.signstyle!=au)c+=a.signstyle;if(a.alternate&&a.base==8)c+=W;if(a.alternate&&a.base==16)c+="0x";if(a.justify==bQ&&a.filler==W)for(var
d=e;d<a.width;d++)c+=W;c+=b;if(a.justify==au)for(var
d=e;d<a.width;d++)c+=B;return new
aw(c)}function
sg(a,b){var
c,f=dB(a),e=f.prec<0?6:f.prec;if(b<0){f.sign=-1;b=-b}if(isNaN(b)){c=ft;f.filler=B}else
if(!isFinite(b)){c="inf";f.filler=B}else
switch(f.conv){case
bg:var
c=b.toExponential(e),d=c.length;if(c.charAt(d-3)==bg)c=c.slice(0,d-1)+W+c.slice(d-1);break;case
X:c=b.toFixed(e);break;case
fp:e=e?e:1;c=b.toExponential(e-1);var
i=c.indexOf(bg),h=+c.slice(i+1);if(h<-4||b.toFixed(0).length>e){var
d=i-1;while(c.charAt(d)==W)d--;if(c.charAt(d)==af)d--;c=c.slice(0,d+1)+c.slice(i);d=c.length;if(c.charAt(d-3)==bg)c=c.slice(0,d-1)+W+c.slice(d-1);break}else{var
g=e;if(h<0){g-=h+1;c=b.toFixed(g)}else
while(c=b.toFixed(g),c.length>e+1)g--;if(g){var
d=c.length-1;while(c.charAt(d)==W)d--;if(c.charAt(d)==af)d--;c=c.slice(0,d+1)}}break}return dz(f,c)}function
sh(a,b){if(a.toString()=="%d")return new
aw(g+b);var
c=dB(a);if(b<0)if(c.signedconv){c.sign=-1;b=-b}else
b>>>=0;var
d=b.toString(c.base);if(c.prec>=0){c.filler=B;var
e=c.prec-d.length;if(e>0)d=bk(e,W)+d}return dz(c,d)}function
si(){return 0}function
sj(){return 0}var
b9=[];function
sk(a,b,c){var
e=a[1],i=b9[c];if(i===null)for(var
h=b9.length;h<c;h++)b9[h]=0;else
if(e[i]===b)return e[i-1];var
d=3,g=e[1]*2+1,f;while(d<g){f=d+g>>1|1;if(b<e[f+1])g=f-2;else
d=f}b9[c]=d+1;return b==e[d+1]?e[d]:0}function
sl(a,b){return+(gg(a,b,false)>=0)}function
gi(a){if(!isFinite(a)){if(isNaN(a))return[r,1,0,fV];return a>0?[r,0,0,32752]:[r,0,0,fV]}var
f=a>=0?0:32768;if(f)a=-a;var
b=Math.floor(Math.LOG2E*Math.log(a))+1023;if(b<=0){b=0;a/=Math.pow(2,-1026)}else{a/=Math.pow(2,b-1027);if(a<16){a*=2;b-=1}if(b==0)a/=2}var
d=Math.pow(2,24),c=a|0;a=(a-c)*d;var
e=a|0;a=(a-e)*d;var
g=a|0;c=c&15|f|b<<4;return[r,g,e,c]}function
bj(a,b){return((a>>16)*b<<16)+(a&b2)*b|0}var
sm=function(){var
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
if(e===+e){f=u(f,gi(e));h--;break}}f=t(f);return f&bd}}();function
sw(a){return[a[3]>>8,a[3]&r,a[2]>>16,a[2]>>8&r,a[2]&r,a[1]>>16,a[1]>>8&r,a[1]&r]}function
sn(e,b,c){var
d=0;function
f(a){b--;if(e<0||b<0)return;if(a
instanceof
Array&&a[0]===(a[0]|0))switch(a[0]){case
bR:e--;d=d*dg+a[2]|0;break;case
aJ:b++;f(a);break;case
r:e--;d=d*dg+a[1]+(a[2]<<24)|0;break;default:e--;d=d*19+a[0]|0;for(var
c=a.length-1;c>0;c--)f(a[c])}else
if(a
instanceof
D){e--;var
g=a.array,h=a.getLen();if(g)for(var
c=0;c<h;c++)d=d*19+g[c]|0;else{var
i=a.getFullBytes();for(var
c=0;c<h;c++)d=d*19+i.charCodeAt(c)|0}}else
if(a===(a|0)){e--;d=d*dg+a|0}else
if(a===+a){e--;var
j=sw(gi(a));for(var
c=7;c>=0;c--)d=d*19+j[c]|0}}f(c);return d&bd}function
sr(a){return(a[3]|a[2]|a[1])==0}function
su(a){return[r,a&_,a>>24&_,a>>31&b2]}function
sv(a,b){var
c=a[1]-b[1],d=a[2]-b[2]+(c>>24),e=a[3]-b[3]+(d>>24);return[r,c&_,d&_,e&b2]}function
gk(a,b){if(a[3]>b[3])return 1;if(a[3]<b[3])return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
gj(a){a[3]=a[3]<<1|a[2]>>23;a[2]=(a[2]<<1|a[1]>>23)&_;a[1]=a[1]<<1&_}function
ss(a){a[1]=(a[1]>>>1|a[2]<<23)&_;a[2]=(a[2]>>>1|a[3]<<23)&_;a[3]=a[3]>>>1}function
sy(a,b){var
e=0,d=a.slice(),c=b.slice(),f=[r,0,0,0];while(gk(d,c)>0){e++;gj(c)}while(e>=0){e--;gj(f);if(gk(d,c)>=0){f[1]++;d=sv(d,c)}ss(c)}return[0,f,d]}function
sx(a){return a[1]|a[2]<<24}function
sq(a){return a[3]<<16<0}function
st(a){var
b=-a[1],c=-a[2]+(b>>24),d=-a[3]+(c>>24);return[r,b&_,c&_,d&b2]}function
sp(a,b){var
c=dB(a);if(c.signedconv&&sq(b)){c.sign=-1;b=st(b)}var
d=g,i=su(c.base),h="0123456789abcdef";do{var
f=sy(b,i);b=f[1];d=h.charAt(sx(f[2]))+d}while(!sr(b));if(c.prec>=0){c.filler=B;var
e=c.prec-d.length;if(e>0)d=bk(e,W)+d}return dz(c,d)}function
sU(a){var
b=0,c=10,d=a.get(0)==45?(b++,-1):1;if(a.get(b)==48)switch(a.get(b+1)){case
dp:case
88:c=16;b+=2;break;case
da:case
79:c=8;b+=2;break;case
98:case
66:c=2;b+=2;break}return[b,d,c]}function
gn(a){if(a>=48&&a<=57)return a-48;if(a>=65&&a<=90)return a-55;if(a>=97&&a<=122)return a-87;return-1}function
n(a){dD(q[3],a)}function
sz(a){var
g=sU(a),e=g[0],h=g[1],f=g[2],i=-1>>>0,d=a.get(e),c=gn(d);if(c<0||c>=f)n(bO);var
b=c;for(;;){e++;d=a.get(e);if(d==95)continue;c=gn(d);if(c<0||c>=f)break;b=f*b+c;if(b>i)n(bO)}if(e!=a.getLen())n(bO);b=h*b;if((b|0)!=b)n(bO);return b}function
sA(a){return+(a>31&&a<127)}var
b8={amp:/&/g,lt:/</g,quot:/\"/g,all:/[&<\"]/};function
sB(a){if(!b8.all.test(a))return a;return a.replace(b8.amp,"&amp;").replace(b8.lt,"&lt;").replace(b8.quot,"&quot;")}function
sC(a){var
c=Array.prototype.slice;return function(){var
b=arguments.length>0?c.call(arguments):[undefined];return ag(a,b)}}function
sD(a,b){var
d=[0];for(var
c=1;c<=a;c++)d[c]=b;return d}function
dv(a){var
b=a.length;this.array=a;this.len=this.last=b}dv.prototype=new
D();var
sE=function(){function
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
dv(n(h,c))}}();function
sF(a){return a.data.array.length}function
ao(a){dD(q[2],a)}function
dA(a){if(!a.opened)ao("Cannot flush a closed channel");if(a.buffer==g)return 0;if(a.output){switch(a.output.length){case
2:a.output(a,a.buffer);break;default:a.output(a.buffer)}}a.buffer=g}var
bi=new
Array();function
sG(a){dA(a);a.opened=false;delete
bi[a.fd];return 0}function
sH(a,b,c,d){var
e=a.data.array.length-a.data.offset;if(e<d)d=e;dx(new
dv(a.data.array),a.data.offset,b,c,d);a.data.offset+=d;return d}function
sV(){dC(q[5])}function
sI(a){if(a.data.offset>=a.data.array.length)sV();if(a.data.offset<0||a.data.offset>a.data.array.length)aL();var
b=a.data.array[a.data.offset];a.data.offset++;return b}function
sJ(a){var
b=a.data.offset,c=a.data.array.length;if(b>=c)return 0;while(true){if(b>=c)return-(b-a.data.offset);if(b<0||b>a.data.array.length)aL();if(a.data.array[b]==10)return b-a.data.offset+1;b++}}function
sX(a,b){if(!q.files)q.files={};if(b
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
s4(a){return q.files&&q.files[a.toString()]?1:q.auto_register_file===undefined?0:q.auto_register_file(a)}function
bl(a,b,c){if(q.fds===undefined)q.fds=new
Array();c=c?c:{};var
d={};d.array=b;d.offset=c.append?d.array.length:0;d.flags=c;q.fds[a]=d;q.fd_last_idx=a;return a}function
s8(a,b,c){var
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
e=a.toString();if(d.rdonly&&d.wronly)ao(e+" : flags Open_rdonly and Open_wronly are not compatible");if(d.text&&d.binary)ao(e+" : flags Open_text and Open_binary are not compatible");if(s4(a)){if(d.create&&d.excl)ao(e+" : file already exists");var
f=q.fd_last_idx?q.fd_last_idx:0;if(d.truncate)q.files[e]=g;return bl(f+1,q.files[e],d)}else
if(d.create){var
f=q.fd_last_idx?q.fd_last_idx:0;sX(e,[]);return bl(f+1,q.files[e],d)}else
ao(e+": no such file or directory")}bl(0,[]);bl(1,[]);bl(2,[]);function
sK(a){var
b=q.fds[a];if(b.flags.wronly)ao(gc+a+" is writeonly");return{data:b,fd:a,opened:true}}function
td(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=d.console;b&&b.log&&b.log(a)}function
s0(a,b){var
e=new
D(b),d=e.getLen();for(var
c=0;c<d;c++)a.data.array[a.data.offset+c]=e.get(c);a.data.offset+=d;return 0}function
sL(a){var
b;switch(a){case
1:b=td;break;case
2:b=dE;break;default:b=s0}var
d=q.fds[a];if(d.flags.rdonly)ao(gc+a+" is readonly");var
c={data:d,fd:a,opened:true,buffer:g,output:b};bi[c.fd]=c;return c}function
sM(){var
a=0;for(var
b
in
bi)if(bi[b].opened)a=[0,bi[b],a];return a}function
gm(a,b,c,d){if(!a.opened)ao("Cannot output to a closed channel");var
f;if(c==0&&b.getLen()==d)f=b;else{f=gh(d);dx(b,c,f,0,d)}var
e=f.toString(),g=e.lastIndexOf("\n");if(g<0)a.buffer+=e;else{a.buffer+=e.substr(0,g+1);dA(a);a.buffer+=e.substr(g+1)}}function
R(a){return new
D(a)}function
sN(a,b){var
c=R(String.fromCharCode(b));gm(a,c,0,1)}function
sO(a,b){if(b==0)gp();return a%b}function
sQ(a,b){return+(dy(a,b,false)!=0)}function
sR(a,b){var
d=[a];for(var
c=1;c<=b;c++)d[c]=0;return d}function
sS(a,b){a[0]=b;return 0}function
sT(a){return a
instanceof
Array?a[0]:fw}function
sY(a,b){q[a+1]=b}var
sP={};function
sZ(a,b){sP[a]=b;return 0}function
s1(a,b){return a.compare(b)}function
gq(a,b){var
c=a.fullBytes,d=b.fullBytes;if(c!=null&&d!=null)return c==d?1:0;return a.getFullBytes()==b.getFullBytes()?1:0}function
s2(a,b){return 1-gq(a,b)}function
s3(){return 32}function
s5(){var
a=new
aw("a.out");return[0,a,[0,a]]}function
s6(){return[0,new
aw(fo),32,0]}function
sW(){dC(q[7])}function
s7(){sW()}function
s9(){var
a=new
Date()^4294967295*Math.random();return{valueOf:function(){return a},0:0,1:a,length:2}}function
s_(){console.log("caml_sys_system_command");return 0}function
s$(a){var
b=1;while(a&&a.joo_tramp){a=a.joo_tramp.apply(null,a.joo_args);b++}return a}function
ta(a,b){return{joo_tramp:a,joo_args:b}}function
tb(a,b){if(typeof
b==="function"){a.fun=b;return 0}if(b.fun){a.fun=b.fun;return 0}var
c=b.length;while(c--)a[c]=b[c];return 0}function
tc(){return 0}var
dF=0;function
te(){if(window.webcl==undefined){alert("Unfortunately your system does not support WebCL. "+"Make sure that you have both the OpenCL driver "+"and the WebCL browser extension installed.");dF=1}else{console.log("INIT OPENCL");dF=0}return 0}function
tg(){console.log(" spoc_cuInit");return 0}function
th(){console.log(" spoc_cuda_compile");return 0}function
tp(){console.log(" spoc_cuda_debug_compile");return 0}function
tB(a,b,c){console.log(" spoc_debug_opencl_compile");console.log(a.bytes);var
e=c[9],f=e[0],d=f.createProgram(a.bytes),g=d.getInfo(WebCL.PROGRAM_DEVICES);d.build(g);var
h=d.createKernel(b.bytes);e[0]=f;c[9]=e;return h}function
tC(a){console.log("spoc_getCudaDevice");return 0}function
tD(){console.log(" spoc_getCudaDevicesCount");return 0}function
tE(a,b){console.log(" spoc_getOpenCLDevice");var
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
tF(){console.log(" spoc_getOpenCLDevicesCount");var
a=0,b=webcl.getPlatforms();for(var
d
in
b){var
e=b[d],c=e.getDevices();a+=c.length}return a}function
tG(){console.log(fm);return 0}function
tH(){console.log(fm);var
a=new
Array(3);a[0]=0;return a}function
dG(a){if(a[1]instanceof
Float32Array||a[1].constructor.name=="Float32Array")return 4;if(a[1]instanceof
Int32Array||a[1].constructor.name=="Int32Array")return 4;{console.log("unimplemented vector type");console.log(a[1].constructor.name);return 4}}function
tI(a,b,c){console.log("spoc_opencl_alloc_vect");var
f=a[2],i=a[4],h=i[b+1],j=a[5],k=dG(f),d=c[9],e=d[0],d=c[9],e=d[0],g=e.createBuffer(WebCL.MEM_READ_WRITE,j*k);h[2]=g;d[0]=e;c[9]=d;return 0}function
tJ(){console.log(" spoc_opencl_compile");return 0}function
tK(a,b,c,d){console.log("spoc_opencl_cpu_to_device");var
f=a[2],k=a[4],j=k[b+1],l=a[5],m=dG(f),e=c[9],h=e[0],g=e[d+1],i=j[2];g.enqueueWriteBuffer(i,false,0,l*m,f[1]);e[d+1]=g;e[0]=h;c[9]=e;return 0}function
tQ(a,b,c,d,e){console.log("spoc_opencl_device_to_cpu");var
g=a[2],l=a[4],k=l[b+1],n=a[5],o=dG(g),f=c[9],i=f[0],h=f[e+1],j=k[2],m=g[1];h.enqueueReadBuffer(j,false,0,n*o,m);f[e+1]=h;f[0]=i;c[9]=f;return 0}function
tR(a,b){console.log("spoc_opencl_flush");var
c=a[9][b+1];c.flush();a[9][b+1]=c;return 0}function
tS(){console.log(" spoc_opencl_is_available");return!dF}function
tT(a,b,c,d,e){console.log("spoc_opencl_launch_grid");var
m=b[1],n=b[2],o=b[3],h=c[1],i=c[2],j=c[3],g=new
Array(3);g[0]=m*h;g[1]=n*i;g[2]=o*j;var
f=new
Array(3);f[0]=h;f[1]=i;f[2]=j;var
l=d[9],k=l[e+1];if(h==1&&i==1&&j==1)k.enqueueNDRangeKernel(a,f.length,null,g);else
k.enqueueNDRangeKernel(a,f.length,null,g,f);return 0}function
tW(a,b,c,d){console.log("spoc_opencl_load_param_int");b.setArg(a[1],new
Uint32Array([c]));a[1]=a[1]+1;return 0}function
tY(a,b,c,d,e){console.log("spoc_opencl_load_param_vec");var
f=d[2];b.setArg(a[1],f);a[1]=a[1]+1;return 0}function
t1(){return new
Date().getTime()/fw}function
t2(){return 0}var
s=r9,l=r_,a8=dx,aF=gg,A=gh,at=sd,cV=sg,bG=sh,a9=sj,Z=sk,cZ=sA,fc=sB,w=sD,e2=sG,cX=dA,c1=sH,e0=sK,cW=sL,aG=sO,x=bj,b=R,c0=sQ,e5=sR,aE=sY,cY=sZ,e4=s1,bJ=gq,y=s2,bH=s7,e1=s8,e3=s9,fb=s_,V=s$,C=ta,e$=tb,fa=tc,fd=te,ff=tg,fg=tD,fe=tF,e8=tG,e7=tH,e9=tI,e6=tR,bK=t2;function
j(a,b){return a.length==1?a(b):ag(a,[b])}function
i(a,b,c){return a.length==2?a(b,c):ag(a,[b,c])}function
o(a,b,c,d){return a.length==3?a(b,c,d):ag(a,[b,c,d])}function
e_(a,b,c,d,e,f,g){return a.length==6?a(b,c,d,e,f,g):ag(a,[b,c,d,e,f,g])}var
aM=[0,b("Failure")],bm=[0,b("Invalid_argument")],bn=[0,b("End_of_file")],t=[0,b("Not_found")],F=[0,b("Assert_failure")],cv=b(af),cy=b(af),cA=b(af),eK=b(g),eJ=[0,b(f0),b(fU),b(fF),b(f8),b(f4)],eZ=[0,1],eU=[0,b(f8),b(fU),b(f0),b(fF),b(f4)],eV=[0,b(dm),b(c4),b(c7),b(c9),b(dd),b(c_),b(dk),b(ds),b(c$),b(dc)],eW=[0,b(df),b(dq),b(dh)],cT=[0,b(dq),b(c7),b(c9),b(dh),b(df),b(c4),b(ds),b(dc),b(dm),b(c_),b(dk),b(dd),b(c$)];aE(6,t);aE(5,[0,b("Division_by_zero")]);aE(4,bn);aE(3,bm);aE(2,aM);aE(1,[0,b("Sys_error")]);var
gA=b("really_input"),gz=[0,0,[0,7,0]],gy=[0,1,[0,3,[0,4,[0,7,0]]]],gx=b(fS),gw=b(af),gu=b("true"),gv=b("false"),gB=b("Pervasives.do_at_exit"),gD=b("Array.blit"),gH=b("List.iter2"),gF=b("tl"),gE=b("hd"),gL=b("\\b"),gM=b("\\t"),gN=b("\\n"),gO=b("\\r"),gK=b("\\\\"),gJ=b("\\'"),gI=b("Char.chr"),gR=b("String.contains_from"),gQ=b("String.blit"),gP=b("String.sub"),g0=b("Map.remove_min_elt"),g1=[0,0,0,0],g2=[0,b("map.ml"),270,10],g3=[0,0,0],gW=b(bT),gX=b(bT),gY=b(bT),gZ=b(bT),g4=b("CamlinternalLazy.Undefined"),g7=b("Buffer.add: cannot grow buffer"),hl=b(g),hm=b(g),hp=b(fS),hq=b(bZ),hr=b(bZ),hn=b(bW),ho=b(bW),hk=b(ft),hi=b("neg_infinity"),hj=b("infinity"),hh=b(af),hg=b("printf: bad positional specification (0)."),hf=b("%_"),he=[0,b("printf.ml"),143,8],hc=b(bW),hd=b("Printf: premature end of format string '"),g_=b(bW),g$=b(" in format string '"),ha=b(", at char number "),hb=b("Printf: bad conversion %"),g8=b("Sformat.index_of_int: negative argument "),ht=b(dl),hu=[0,987910699,495797812,364182224,414272206,318284740,990407751,383018966,270373319,840823159,24560019,536292337,512266505,189156120,730249596,143776328,51606627,140166561,366354223,1003410265,700563762,981890670,913149062,526082594,1021425055,784300257,667753350,630144451,949649812,48546892,415514493,258888527,511570777,89983870,283659902,308386020,242688715,482270760,865188196,1027664170,207196989,193777847,619708188,671350186,149669678,257044018,87658204,558145612,183450813,28133145,901332182,710253903,510646120,652377910,409934019,801085050],r3=b("OCAMLRUNPARAM"),r1=b("CAMLRUNPARAM"),hw=b(g),hT=[0,b("camlinternalOO.ml"),287,50],hS=b(g),hy=b("CamlinternalOO.last_id"),il=b(g),ii=b(fH),ih=b(".\\"),ig=b(fT),ie=b("..\\"),h8=b(fH),h7=b(fT),h3=b(g),h2=b(g),h4=b(c8),h5=b(fs),rZ=b("TMPDIR"),h_=b("/tmp"),h$=b("'\\''"),ic=b(c8),id=b("\\"),rX=b("TEMP"),ij=b(af),ip=b(c8),iq=b(fs),it=b("Cygwin"),iu=b(fo),iv=b("Win32"),iw=[0,b("filename.ml"),189,9],iD=b("E2BIG"),iF=b("EACCES"),iG=b("EAGAIN"),iH=b("EBADF"),iI=b("EBUSY"),iJ=b("ECHILD"),iK=b("EDEADLK"),iL=b("EDOM"),iM=b("EEXIST"),iN=b("EFAULT"),iO=b("EFBIG"),iP=b("EINTR"),iQ=b("EINVAL"),iR=b("EIO"),iS=b("EISDIR"),iT=b("EMFILE"),iU=b("EMLINK"),iV=b("ENAMETOOLONG"),iW=b("ENFILE"),iX=b("ENODEV"),iY=b("ENOENT"),iZ=b("ENOEXEC"),i0=b("ENOLCK"),i1=b("ENOMEM"),i2=b("ENOSPC"),i3=b("ENOSYS"),i4=b("ENOTDIR"),i5=b("ENOTEMPTY"),i6=b("ENOTTY"),i7=b("ENXIO"),i8=b("EPERM"),i9=b("EPIPE"),i_=b("ERANGE"),i$=b("EROFS"),ja=b("ESPIPE"),jb=b("ESRCH"),jc=b("EXDEV"),jd=b("EWOULDBLOCK"),je=b("EINPROGRESS"),jf=b("EALREADY"),jg=b("ENOTSOCK"),jh=b("EDESTADDRREQ"),ji=b("EMSGSIZE"),jj=b("EPROTOTYPE"),jk=b("ENOPROTOOPT"),jl=b("EPROTONOSUPPORT"),jm=b("ESOCKTNOSUPPORT"),jn=b("EOPNOTSUPP"),jo=b("EPFNOSUPPORT"),jp=b("EAFNOSUPPORT"),jq=b("EADDRINUSE"),jr=b("EADDRNOTAVAIL"),js=b("ENETDOWN"),jt=b("ENETUNREACH"),ju=b("ENETRESET"),jv=b("ECONNABORTED"),jw=b("ECONNRESET"),jx=b("ENOBUFS"),jy=b("EISCONN"),jz=b("ENOTCONN"),jA=b("ESHUTDOWN"),jB=b("ETOOMANYREFS"),jC=b("ETIMEDOUT"),jD=b("ECONNREFUSED"),jE=b("EHOSTDOWN"),jF=b("EHOSTUNREACH"),jG=b("ELOOP"),jH=b("EOVERFLOW"),jI=b("EUNKNOWNERR %d"),iE=b("Unix.Unix_error(Unix.%s, %S, %S)"),iz=b(fJ),iA=b(g),iB=b(g),iC=b(fJ),jJ=b("0.0.0.0"),jK=b("127.0.0.1"),rW=b("::"),rV=b("::1"),jU=[0,b("Vector.ml"),fN,25],jV=b("Cuda.No_Cuda_Device"),jW=b("Cuda.ERROR_DEINITIALIZED"),jX=b("Cuda.ERROR_NOT_INITIALIZED"),jY=b("Cuda.ERROR_INVALID_CONTEXT"),jZ=b("Cuda.ERROR_INVALID_VALUE"),j0=b("Cuda.ERROR_OUT_OF_MEMORY"),j1=b("Cuda.ERROR_INVALID_DEVICE"),j2=b("Cuda.ERROR_NOT_FOUND"),j3=b("Cuda.ERROR_FILE_NOT_FOUND"),j4=b("Cuda.ERROR_UNKNOWN"),j5=b("Cuda.ERROR_LAUNCH_FAILED"),j6=b("Cuda.ERROR_LAUNCH_OUT_OF_RESOURCES"),j7=b("Cuda.ERROR_LAUNCH_TIMEOUT"),j8=b("Cuda.ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"),j9=b("no_cuda_device"),j_=b("cuda_error_deinitialized"),j$=b("cuda_error_not_initialized"),ka=b("cuda_error_invalid_context"),kb=b("cuda_error_invalid_value"),kc=b("cuda_error_out_of_memory"),kd=b("cuda_error_invalid_device"),ke=b("cuda_error_not_found"),kf=b("cuda_error_file_not_found"),kg=b("cuda_error_launch_failed"),kh=b("cuda_error_launch_out_of_resources"),ki=b("cuda_error_launch_timeout"),kj=b("cuda_error_launch_incompatible_texturing"),kk=b("cuda_error_unknown"),kl=b("OpenCL.No_OpenCL_Device"),km=b("OpenCL.OPENCL_ERROR_UNKNOWN"),kn=b("OpenCL.INVALID_CONTEXT"),ko=b("OpenCL.INVALID_DEVICE"),kp=b("OpenCL.INVALID_VALUE"),kq=b("OpenCL.INVALID_QUEUE_PROPERTIES"),kr=b("OpenCL.OUT_OF_RESOURCES"),ks=b("OpenCL.MEM_OBJECT_ALLOCATION_FAILURE"),kt=b("OpenCL.OUT_OF_HOST_MEMORY"),ku=b("OpenCL.FILE_NOT_FOUND"),kv=b("OpenCL.INVALID_PROGRAM"),kw=b("OpenCL.INVALID_BINARY"),kx=b("OpenCL.INVALID_BUILD_OPTIONS"),ky=b("OpenCL.INVALID_OPERATION"),kz=b("OpenCL.COMPILER_NOT_AVAILABLE"),kA=b("OpenCL.BUILD_PROGRAM_FAILURE"),kB=b("OpenCL.INVALID_KERNEL"),kC=b("OpenCL.INVALID_ARG_INDEX"),kD=b("OpenCL.INVALID_ARG_VALUE"),kE=b("OpenCL.INVALID_MEM_OBJECT"),kF=b("OpenCL.INVALID_SAMPLER"),kG=b("OpenCL.INVALID_ARG_SIZE"),kH=b("OpenCL.INVALID_COMMAND_QUEUE"),kI=b("no_opencl_device"),kJ=b("opencl_error_unknown"),kK=b("opencl_invalid_context"),kL=b("opencl_invalid_device"),kM=b("opencl_invalid_value"),kN=b("opencl_invalid_queue_properties"),kO=b("opencl_out_of_resources"),kP=b("opencl_mem_object_allocation_failure"),kQ=b("opencl_out_of_host_memory"),kR=b("opencl_file_not_found"),kS=b("opencl_invalid_program"),kT=b("opencl_invalid_binary"),kU=b("opencl_invalid_build_options"),kV=b("opencl_invalid_operation"),kW=b("opencl_compiler_not_available"),kX=b("opencl_build_program_failure"),kY=b("opencl_invalid_kernel"),kZ=b("opencl_invalid_arg_index"),k0=b("opencl_invalid_arg_value"),k1=b("opencl_invalid_mem_object"),k2=b("opencl_invalid_sampler"),k3=b("opencl_invalid_arg_size"),k4=b("opencl_invalid_command_queue"),k5=b(b4),k6=b(b4),ll=b(fB),lk=b(fx),lj=b(fB),li=b(fx),lh=[0,1],lg=b(g),lc=b(bN),k9=b("Cl LOAD ARG Type Not Implemented\n"),k8=b("CU LOAD ARG Type Not Implemented\n"),k7=[0,b(dc),b(c$),b(ds),b(dk),b(c_),b(df),b(dd),b(c9),b(c7),b(dq),b(c4),b(dm),b(dh)],k_=b("Kernel.ERROR_BLOCK_SIZE"),la=b("Kernel.ERROR_GRID_SIZE"),ld=b("Kernel.No_source_for_device"),lo=b("Empty"),lp=b("Unit"),lq=b("Kern"),lr=b("Params"),ls=b("Plus"),lt=b("Plusf"),lu=b("Min"),lv=b("Minf"),lw=b("Mul"),lx=b("Mulf"),ly=b("Div"),lz=b("Divf"),lA=b("Mod"),lB=b("Id "),lC=b("IdName "),lD=b("IntVar "),lE=b("FloatVar "),lF=b("UnitVar "),lG=b("CastDoubleVar "),lH=b("DoubleVar "),lI=b("IntArr"),lJ=b("Int32Arr"),lK=b("Int64Arr"),lL=b("Float32Arr"),lM=b("Float64Arr"),lN=b("VecVar "),lO=b("Concat"),lP=b("Seq"),lQ=b("Return"),lR=b("Set"),lS=b("Decl"),lT=b("SetV"),lU=b("SetLocalVar"),lV=b("Intrinsics"),lW=b(B),lX=b("IntId "),lY=b("Int "),l0=b("IntVecAcc"),l1=b("Local"),l2=b("Acc"),l3=b("Ife"),l4=b("If"),l5=b("Or"),l6=b("And"),l7=b("EqBool"),l8=b("LtBool"),l9=b("GtBool"),l_=b("LtEBool"),l$=b("GtEBool"),ma=b("DoLoop"),mb=b("While"),mc=b("App"),md=b("GInt"),me=b("GFloat"),lZ=b("Float "),ln=b("  "),lm=b("%s\n"),nS=b(fQ),nT=[0,b(de),166,14],mh=b(g),mi=b(bN),mj=b("\n}\n#ifdef __cplusplus\n}\n#endif"),mk=b(" ) {\n"),ml=b(g),mm=b(bM),mo=b(g),mn=b('#ifdef __cplusplus\nextern "C" {\n#endif\n\n__global__ void spoc_dummy ( '),mp=b(ad),mq=b(b5),mr=b(ae),ms=b(ad),mt=b(b5),mu=b(ae),mv=b(ad),mw=b(bU),mx=b(ae),my=b(ad),mz=b(bU),mA=b(ae),mB=b(ad),mC=b(bY),mD=b(ae),mE=b(ad),mF=b(bY),mG=b(ae),mH=b(ad),mI=b(b7),mJ=b(ae),mK=b(ad),mL=b(b7),mM=b(ae),mN=b(ad),mO=b(fr),mP=b(ae),mQ=b(fW),mR=b(fq),mS=[0,b(de),65,17],mT=b(bV),mU=b(fC),mV=b(L),mW=b(M),mX=b(fI),mY=b(L),mZ=b(M),m0=b(fl),m1=b(L),m2=b(M),m3=b(fy),m4=b(L),m5=b(M),m6=b(fX),m7=b(fR),m9=b("int"),m_=b("float"),m$=b("double"),m8=[0,b(de),60,12],nb=b(bM),na=b(gb),nc=b(fP),nd=b(g),ne=b(g),nh=b(bP),ni=b(ac),nj=b(aH),nl=b(bP),nk=b(ac),nm=b(X),nn=b(L),no=b(M),np=b("}\n"),nq=b(aH),nr=b(aH),ns=b("{"),nt=b(bf),nu=b(fv),nv=b(bf),nw=b(bc),nx=b(b6),ny=b(bf),nz=b(bc),nA=b(b6),nB=b(fj),nC=b(fh),nD=b(fA),nE=b(f3),nF=b(fG),nG=b(bL),nH=b(f2),nI=b(b3),nJ=b(fn),nK=b(bS),nL=b(bL),nM=b(bS),nN=b(ac),nO=b(fi),nP=b(b3),nQ=b(bc),nR=b(fE),nW=b(bX),nX=b(bX),nY=b(B),nZ=b(B),nU=b(fM),nV=b(gd),n0=b(X),nf=b(bP),ng=b(ac),n1=b(L),n2=b(M),n4=b(bV),n5=b(X),n6=b(f_),n7=b(L),n8=b(M),n9=b(X),n3=b("cuda error parse_float"),mf=[0,b(g),b(g)],pt=b(fQ),pu=[0,b(dj),162,14],oa=b(g),ob=b(bN),oc=b(b3),od=b(" ) \n{\n"),oe=b(g),of=b(bM),oh=b(g),og=b("__kernel void spoc_dummy ( "),oi=b(b5),oj=b(b5),ok=b(bU),ol=b(bU),om=b(bY),on=b(bY),oo=b(b7),op=b(b7),oq=b(fr),or=b(fW),os=b(fq),ot=[0,b(dj),65,17],ou=b(bV),ov=b(fC),ow=b(L),ox=b(M),oy=b(fI),oz=b(L),oA=b(M),oB=b(fl),oC=b(L),oD=b(M),oE=b(fy),oF=b(L),oG=b(M),oH=b(fX),oI=b(fR),oK=b("__global int"),oL=b("__global float"),oM=b("__global double"),oJ=[0,b(dj),60,12],oO=b(bM),oN=b(gb),oP=b(fP),oQ=b(g),oR=b(g),oT=b(bP),oU=b(ac),oV=b(aH),oW=b(ac),oX=b(X),oY=b(L),oZ=b(M),o0=b(g),o1=b(bN),o2=b(aH),o3=b(g),o4=b(bf),o5=b(fv),o6=b(bf),o7=b(bc),o8=b(b6),o9=b(b3),o_=b(aH),o$=b("{\n"),pa=b(")\n"),pb=b(b6),pc=b(fj),pd=b(fh),pe=b(fA),pf=b(f3),pg=b(fG),ph=b(bL),pi=b(f2),pj=b(f5),pk=b(fn),pl=b(bS),pm=b(bL),pn=b(bS),po=b(ac),pp=b(fi),pq=b(f5),pr=b(bc),ps=b(fE),px=b(bX),py=b(bX),pz=b(B),pA=b(B),pv=b(fM),pw=b(gd),pB=b(X),oS=b(ac),pC=b(L),pD=b(M),pF=b(bV),pG=b(X),pH=b(f_),pI=b(L),pJ=b(M),pK=b(X),pE=b("opencl error parse_float"),n_=[0,b(g),b(g)],qJ=[0,0],qK=[0,0],qL=[0,1],qM=[0,1],qD=b("kirc_kernel.cu"),qE=b("nvcc -m64 -arch=sm_10 -O3 -ptx kirc_kernel.cu -o kirc_kernel.ptx"),qF=b("kirc_kernel.ptx"),qG=b("rm kirc_kernel.cu kirc_kernel.ptx"),qA=[0,b(g),b(g)],qC=b(g),qB=[0,b("Kirc.ml"),407,81],qH=b(ac),qI=b(fZ),qx=[33,0],qs=b(fZ),pL=b("int spoc_xor (int a, int b ) { return (a^b);}\n"),pM=b("int spoc_powint (int a, int b ) { return ((int) pow (((float) a), ((float) b)));}\n"),pN=b("int logical_and (int a, int b ) { return (a & b);}\n"),pO=b("float spoc_fdiv ( float a, float b ) { return (a / b);}\n"),pP=b("float spoc_fmul ( float a, float b ) { return (a * b);}\n"),pQ=b("float spoc_fminus ( float a, float b ) { return (a - b);}\n"),pR=b("float spoc_fadd ( float a, float b ) { return (a + b);}\n"),pS=b("float spoc_fdiv ( float a, float b );\n"),pT=b("float spoc_fmul ( float a, float b );\n"),pU=b("float spoc_fminus ( float a, float b );\n"),pV=b("float spoc_fadd ( float a, float b );\n"),pX=b(di),pY=b("double spoc_ddiv ( double a, double b ) { return (a / b);}\n"),pZ=b("double spoc_dmul ( double a, double b ) { return (a * b);}\n"),p0=b("double spoc_dminus ( double a, double b ) { return (a - b);}\n"),p1=b("double spoc_dadd ( double a, double b ) { return (a + b);}\n"),p2=b("double spoc_ddiv ( double a, double b );\n"),p3=b("double spoc_dmul ( double a, double b );\n"),p4=b("double spoc_dminus ( double a, double b );\n"),p5=b("double spoc_dadd ( double a, double b );\n"),p6=b(di),p7=b("#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"),p8=b("#elif defined(cl_amd_fp64)  // AMD extension available?\n"),p9=b("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"),p_=b("#if defined(cl_khr_fp64)  // Khronos extension available?\n"),p$=b(f9),qa=b(f1),qc=b(di),qd=b("__device__ double spoc_ddiv ( double a, double b ) { return (a / b);}\n"),qe=b("__device__ double spoc_dmul ( double a, double b ) { return (a * b);}\n"),qf=b("__device__ double spoc_dminus ( double a, double b ) { return (a - b);}\n"),qg=b("__device__ double spoc_dadd ( double a, double b ) { return (a + b);}\n"),qh=b(f9),qi=b(f1),qk=b("__device__ int spoc_xor (int a, int b ) { return (a^b);}\n"),ql=b("__device__ int spoc_powint (int a, int b ) { return ((int) pow (((double) a), ((double) b)));}\n"),qm=b("__device__ int logical_and (int a, int b ) { return (a & b);}\n"),qn=b("__device__ float spoc_fdiv ( float a, float b ) { return (a / b);}\n"),qo=b("__device__ float spoc_fmul ( float a, float b ) { return (a * b);}\n"),qp=b("__device__ float spoc_fminus ( float a, float b ) { return (a - b);}\n"),qq=b("__device__ float spoc_fadd ( float a, float b ) { return (a + b);}\n"),qy=[0,b(g),b(g)],q1=b("canvas"),qY=b("span"),qX=b("img"),qW=b("br"),qV=b(fk),qU=b("select"),qT=b("option"),qZ=b("Dom_html.Canvas_not_available"),rT=[0,b(f7),135,17],rQ=b("Will use device : %s!"),rR=[0,1],rS=b(g),rP=b("Time %s : %Fs\n%!"),ra=b("spoc_dummy"),rb=b("kirc_kernel"),q_=b("spoc_kernel_extension error"),q2=[0,b(f7),12,15],rD=b("(get_group_id (0))"),rE=b("blockIdx.x"),rG=b("(get_local_size (0))"),rH=b("blockDim.x"),rJ=b("(get_local_id (0))"),rK=b("threadIdx.x");function
S(a){throw[0,aM,a]}function
E(a){throw[0,bm,a]}function
h(a,b){var
c=a.getLen(),e=b.getLen(),d=A(c+e|0);a8(a,0,d,0,c);a8(b,0,d,c,e);return d}function
k(a){return b(g+a)}function
N(a){var
c=cV(gx,a),b=0,f=c.getLen();for(;;){if(f<=b)var
e=h(c,gw);else{var
d=c.safeGet(b),g=48<=d?58<=d?0:1:45===d?1:0;if(g){var
b=b+1|0;continue}var
e=c}return e}}function
b_(a,b){if(a){var
c=a[1];return[0,c,b_(a[2],b)]}return b}e0(0);var
dH=cW(1);cW(2);function
dI(a,b){return gm(a,b,0,b.getLen())}function
dJ(a){return e0(e1(a,gz,0))}function
dK(a){var
b=sM(0);for(;;){if(b){var
c=b[2],d=b[1];try{cX(d)}catch(f){}var
b=c;continue}return 0}}cY(gB,dK);function
dL(a){return e2(a)}function
gC(a,b){return sN(a,b)}function
dM(a){return cX(a)}function
dN(a,b){var
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
dO(a,b,c){var
e=[0,b],f=c.length-1-1|0,g=0;if(!(f<0)){var
d=g;for(;;){e[1]=i(a,e[1],c[d+1]);var
h=d+1|0;if(f!==d){var
d=h;continue}break}}return e[1]}function
dP(a){var
b=a,c=0;for(;;){if(b){var
d=[0,b[1],c],b=b[2],c=d;continue}return c}}function
ca(a,b){if(b){var
c=b[2],d=j(a,b[1]);return[0,d,ca(a,c)]}return 0}function
cc(a,b,c){if(b){var
d=b[1];return i(a,d,cc(a,b[2],c))}return c}function
dR(a,b,c){var
e=b,d=c;for(;;){if(e){if(d){var
f=d[2],g=e[2];i(a,e[1],d[1]);var
e=g,d=f;continue}}else
if(!d)return 0;return E(gH)}}function
cd(a,b){var
c=b;for(;;){if(c){var
e=c[2],d=0===aF(c[1],a)?1:0;if(d)return d;var
c=e;continue}return 0}}function
ce(a){if(0<=a)if(!(r<a))return a;return E(gI)}function
dS(a){var
b=65<=a?90<a?0:1:0;if(!b){var
c=192<=a?214<a?0:1:0;if(!c){var
d=216<=a?222<a?1:0:1;if(d)return a}}return a+32|0}function
ah(a,b){var
c=A(a);sf(c,0,a,b);return c}function
u(a,b,c){if(0<=b)if(0<=c)if(!((a.getLen()-c|0)<b)){var
d=A(c);a8(a,b,d,0,c);return d}return E(gP)}function
bp(a,b,c,d,e){if(0<=e)if(0<=b)if(!((a.getLen()-e|0)<b))if(0<=d)if(!((c.getLen()-e|0)<d))return a8(a,b,c,d,e);return E(gQ)}function
dT(a){var
c=a.getLen();if(0===c)var
f=a;else{var
d=A(c),e=c-1|0,g=0;if(!(e<0)){var
b=g;for(;;){d.safeSet(b,dS(a.safeGet(b)));var
h=b+1|0;if(e!==b){var
b=h;continue}break}}var
f=d}return f}var
cg=s6(0)[1],ax=s3(0),ch=(1<<(ax-10|0))-1|0,aP=x(ax/8|0,ch)-1|0,gT=s5(0)[2],gU=bR,gV=aJ;function
ci(k){function
h(a){return a?a[5]:0}function
e(a,b,c,d){var
e=h(a),f=h(d),g=f<=e?e+1|0:f+1|0;return[0,a,b,c,d,g]}function
q(a,b){return[0,0,a,b,0,1]}function
f(a,b,c,d){var
i=a?a[5]:0,j=d?d[5]:0;if((j+2|0)<i){if(a){var
f=a[4],m=a[3],n=a[2],k=a[1],q=h(f);if(q<=h(k))return e(k,n,m,e(f,b,c,d));if(f){var
r=f[3],s=f[2],t=f[1],u=e(f[4],b,c,d);return e(e(k,n,m,t),s,r,u)}return E(gW)}return E(gX)}if((i+2|0)<j){if(d){var
l=d[4],o=d[3],p=d[2],g=d[1],v=h(g);if(v<=h(l))return e(e(a,b,c,g),p,o,l);if(g){var
w=g[3],x=g[2],y=g[1],z=e(g[4],p,o,l);return e(e(a,b,c,y),x,w,z)}return E(gY)}return E(gZ)}var
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
c=a[4],d=a[3],e=a[2];return f(s(b),e,d,c)}return a[4]}return E(g0)}function
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
j=l(a,f),p=j[2],q=j[1];return[0,q,p,g(j[3],e,d,c)]}return g1}function
m(a,b,c){if(b){var
d=b[2],i=b[5],j=b[4],k=b[3],n=b[1];if(h(c)<=i){var
e=l(d,c),p=e[2],q=e[1],r=m(a,j,e[3]),s=o(a,d,[0,k],p);return G(m(a,n,q),d,s,r)}}else
if(!c)return 0;if(c){var
f=c[2],t=c[4],u=c[3],v=c[1],g=l(f,b),w=g[2],x=g[1],y=m(a,g[3],t),z=o(a,f,w,[0,u]);return G(m(a,x,v),f,z,y)}throw[0,F,g2]}function
w(a,b){if(b){var
c=b[3],d=b[2],h=b[4],e=w(a,b[1]),j=i(a,d,c),f=w(a,h);return j?g(e,d,c,f):p(e,f)}return 0}function
x(a,b){if(b){var
c=b[3],d=b[2],m=b[4],e=x(a,b[1]),f=e[2],h=e[1],n=i(a,d,c),j=x(a,m),k=j[2],l=j[1];if(n){var
o=p(f,k);return[0,g(h,d,c,l),o]}var
q=g(f,d,c,k);return[0,p(h,l),q]}return g3}function
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
g5=[0,g4];function
g6(a){throw[0,g5]}function
aQ(a){var
b=1<=a?a:1,c=aP<b?aP:b,d=A(c);return[0,d,0,c,d]}function
aR(a){return u(a[1],0,a[2])}function
dW(a,b){var
c=[0,a[3]];for(;;){if(c[1]<(a[2]+b|0)){c[1]=2*c[1]|0;continue}if(aP<c[1])if((a[2]+b|0)<=aP)c[1]=aP;else
S(g7);var
d=A(c[1]);bp(a[1],0,d,0,a[2]);a[1]=d;a[3]=c[1];return 0}}function
G(a,b){var
c=a[2];if(a[3]<=c)dW(a,1);a[1].safeSet(c,b);a[2]=c+1|0;return 0}function
br(a,b){var
c=b.getLen(),d=a[2]+c|0;if(a[3]<d)dW(a,c);bp(b,0,a[1],a[2],c);a[2]=d;return 0}function
cj(a){return 0<=a?a:S(h(g8,k(a)))}function
dX(a,b){return cj(a+b|0)}var
g9=1;function
dY(a){return dX(g9,a)}function
dZ(a){return u(a,0,a.getLen())}function
d0(a,b,c){var
d=h(g$,h(a,g_)),e=h(ha,h(k(b),d));return E(h(hb,h(ah(1,c),e)))}function
aS(a,b,c){return d0(dZ(a),b,c)}function
bs(a){return E(h(hd,h(dZ(a),hc)))}function
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
a=i,g=dP(d);for(;;){if(a<=c){var
j=e.safeGet(a);if(42===j){if(g){var
l=g[2];br(f,k(g[1]));var
a=h(a+1|0),g=l;continue}throw[0,F,he]}G(f,j);var
a=a+1|0;continue}return aR(f)}}function
d1(a,b,c,d,e){var
f=ap(b,c,d,e);if(78!==a)if(be!==a)return f;f.safeSet(f.getLen()-1|0,dn);return f}function
d2(a){return function(c,b){var
m=c.getLen();function
n(a,b){var
o=40===a?41:c6;function
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
d3(j,b,c){var
m=j.getLen()-1|0;function
s(a){var
l=a;a:for(;;){if(l<m){if(37===j.safeGet(l)){var
e=0,h=l+1|0;for(;;){if(m<h)var
w=bs(j);else{var
n=j.safeGet(h);if(58<=n){if(95===n){var
e=1,h=h+1|0;continue}}else
if(32<=n)switch(n+fu|0){case
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
k=j.safeGet(d);if(fN<=k)var
g=0;else
switch(k){case
78:case
88:case
aI:case
av:case
da:case
dn:case
dp:var
f=o(b,e,d,av),g=1;break;case
69:case
70:case
71:case
f6:case
db:case
dt:var
f=o(b,e,d,db),g=1;break;case
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
c5:var
f=o(b,e,d,k),g=1;break;case
76:case
fD:case
be:var
t=d+1|0;if(m<t){var
f=o(b,e,d,av),g=1}else{var
q=j.safeGet(t)+f$|0;if(q<0||32<q)var
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
c6:var
f=o(b,e,d,k),g=1;break;case
40:var
f=s(o(b,e,d,k)),g=1;break;case
dr:var
u=o(b,e,d,k),v=i(d2(k),j,u),p=u;for(;;){if(p<(v-2|0)){var
p=i(c,p,j.safeGet(p));continue}var
d=v-1|0;continue b}default:var
g=0}if(!g)var
f=aS(j,d,k)}var
w=f;break}}var
l=w;continue a}}var
l=l+1|0;continue}return l}}s(0);return 0}function
d4(a){var
d=[0,0,0,0];function
b(a,b,c){var
f=41!==c?1:0,g=f?c6!==c?1:0:f;if(g){var
e=97===c?2:1;if(b0===c)d[3]=d[3]+1|0;if(a)d[2]=d[2]+e|0;else
d[1]=d[1]+e|0}return b+1|0}d3(a,b,function(a,b){return a+1|0});return d[1]}function
d5(a,b,c){var
h=a.safeGet(c);if((h+aK|0)<0||9<(h+aK|0))return i(b,0,c);var
e=h+aK|0,d=c+1|0;for(;;){var
f=a.safeGet(d);if(48<=f){if(!(58<=f)){var
e=(10*e|0)+(f+aK|0)|0,d=d+1|0;continue}var
g=0}else
if(36===f)if(0===e){var
j=S(hg),g=1}else{var
j=i(b,[0,cj(e-1|0)],d+1|0),g=1}else
var
g=0;if(!g)var
j=i(b,0,c);return j}}function
O(a,b){return a?b:dY(b)}function
d6(a,b){return a?a[1]:b}function
d7(aJ,b,c,d,e,f,g){var
D=j(b,g);function
af(a){return i(d,D,a)}function
aK(a,b,m,aL){var
k=m.getLen();function
E(l,b){var
p=b;for(;;){if(k<=p)return j(a,D);var
d=m.safeGet(p);if(37===d){var
o=function(a,b){return s(aL,d6(a,b))},au=function(g,f,c,d){var
a=d;for(;;){var
aa=m.safeGet(a)+fu|0;if(!(aa<0||25<aa))switch(aa){case
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
10:return d5(m,function(a,b){var
d=[0,o(a,f),c];return au(g,O(a,f),d,b)},a+1|0);default:var
a=a+1|0;continue}var
q=m.safeGet(a);if(124<=q)var
k=0;else
switch(q){case
78:case
88:case
aI:case
av:case
da:case
dn:case
dp:var
a8=o(g,f),a9=bG(d1(q,m,p,a,c),a8),l=r(O(g,f),a9,a+1|0),k=1;break;case
69:case
71:case
f6:case
db:case
dt:var
a1=o(g,f),a2=cV(ap(m,p,a,c),a1),l=r(O(g,f),a2,a+1|0),k=1;break;case
76:case
fD:case
be:var
ad=m.safeGet(a+1|0)+f$|0;if(ad<0||32<ad)var
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
a5=o(g,f),aA=sp(ap(m,p,U,c),a5)}var
l=r(O(g,f),aA,U+1|0),k=1,ag=0;break;default:var
ag=1}if(ag){var
a3=o(g,f),a4=bG(d1(be,m,p,a,c),a3),l=r(O(g,f),a4,a+1|0),k=1}break;case
37:case
64:var
l=r(f,ah(1,q),a+1|0),k=1;break;case
83:case
bh:var
y=o(g,f);if(bh===q)var
z=y;else{var
b=[0,0],an=y.getLen()-1|0,aN=0;if(!(an<0)){var
M=aN;for(;;){var
x=y.safeGet(M),bd=14<=x?34===x?1:92===x?1:0:11<=x?13<=x?1:0:8<=x?1:0,aT=bd?2:cZ(x)?1:4;b[1]=b[1]+aT|0;var
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
9:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],c5);var
K=1;break;case
10:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],be);var
K=1;break;case
13:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],b0);var
K=1;break;default:var
V=1,K=0}if(K)var
V=0}else
var
V=(B-1|0)<0||56<(B-1|0)?(n.safeSet(b[1],92),b[1]++,n.safeSet(b[1],w),0):1;if(V)if(cZ(w))n.safeSet(b[1],w);else{n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],48+(w/aI|0)|0);b[1]++;n.safeSet(b[1],48+((w/10|0)%10|0)|0);b[1]++;n.safeSet(b[1],48+(w%10|0)|0)}b[1]++;var
aP=L+1|0;if(ao!==L){var
L=aP;continue}break}}var
aD=n}var
z=h(hr,h(aD,hq))}if(a===(p+1|0))var
aC=z;else{var
J=ap(m,p,a,c);try{var
W=0,t=1;for(;;){if(J.getLen()<=t)var
aq=[0,0,W];else{var
X=J.safeGet(t);if(49<=X)if(58<=X)var
ak=0;else{var
aq=[0,sz(u(J,t,(J.getLen()-t|0)-1|0)),W],ak=1}else{if(45===X){var
W=1,t=t+1|0;continue}var
ak=0}if(!ak){var
t=t+1|0;continue}}var
Z=aq;break}}catch(f){if(f[1]!==aM)throw f;var
Z=d0(J,0,bh)}var
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
v=gJ;else
if(92===s)var
v=gK;else{if(14<=s)var
F=0;else
switch(s){case
8:var
v=gL,F=1;break;case
9:var
v=gM,F=1;break;case
10:var
v=gN,F=1;break;case
13:var
v=gO,F=1;break;default:var
F=0}if(!F)if(cZ(s)){var
am=A(1);am.safeSet(0,s);var
v=am}else{var
H=A(4);H.safeSet(0,92);H.safeSet(1,48+(s/aI|0)|0);H.safeSet(2,48+((s/10|0)%10|0)|0);H.safeSet(3,48+(s%10|0)|0);var
v=H}}var
ay=h(ho,h(v,hn))}var
l=r(O(g,f),ay,a+1|0),k=1;break;case
66:case
98:var
aZ=a+1|0,a0=o(g,f)?gu:gv,l=r(O(g,f),a0,aZ),k=1;break;case
40:case
dr:var
T=o(g,f),aw=i(d2(q),m,a+1|0);if(dr===q){var
Q=aQ(T.getLen()),ar=function(a,b){G(Q,b);return a+1|0};d3(T,function(a,b,c){if(a)br(Q,hf);else
G(Q,37);return ar(b,c)},ar);var
aX=aR(Q),l=r(O(g,f),aX,aw),k=1}else{var
ax=O(g,f),bc=dX(d4(T),ax),l=aK(function(a){return E(bc,aw)},ax,T,aL),k=1}break;case
33:j(e,D);var
l=E(f,a+1|0),k=1;break;case
41:var
l=r(f,hl,a+1|0),k=1;break;case
44:var
l=r(f,hm,a+1|0),k=1;break;case
70:var
ab=o(g,f);if(0===c)var
az=hp;else{var
$=ap(m,p,a,c);if(70===q)$.safeSet($.getLen()-1|0,dt);var
az=$}var
at=sc(ab);if(3===at)var
ac=ab<0?hi:hj;else
if(4<=at)var
ac=hk;else{var
S=cV(az,ab),R=0,aY=S.getLen();for(;;){if(aY<=R)var
as=h(S,hh);else{var
I=S.safeGet(R)-46|0,bf=I<0||23<I?55===I?1:0:(I-1|0)<0||21<(I-1|0)?1:0;if(!bf){var
R=R+1|0;continue}var
as=S}var
ac=as;break}}var
l=r(O(g,f),ac,a+1|0),k=1;break;case
91:var
l=aS(m,a,q),k=1;break;case
97:var
aE=o(g,f),aF=dY(d6(g,f)),aG=o(0,aF),a_=a+1|0,a$=O(g,aF);if(aJ)af(i(aE,0,aG));else
i(aE,D,aG);var
l=E(a$,a_),k=1;break;case
b0:var
l=aS(m,a,q),k=1;break;case
c5:var
aH=o(g,f),ba=a+1|0,bb=O(g,f);if(aJ)af(j(aH,0));else
j(aH,D);var
l=E(bb,ba),k=1;break;default:var
k=0}if(!k)var
l=aS(m,a,q);return l}},f=p+1|0,g=0;return d5(m,function(a,b){return au(a,l,g,b)},f)}i(c,D,d);var
p=p+1|0;continue}}function
r(a,b,c){af(b);return E(a,c)}return E(b,0)}var
o=cj(0);function
k(a,b){return aK(f,o,a,b)}var
m=d4(g);if(m<0||6<m){var
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
d8(a){function
b(a){return 0}return d7(0,function(a){return dH},gC,dI,dM,b,a)}function
hs(a){return aQ(2*a.getLen()|0)}function
d9(c){function
b(a){var
b=aR(a);a[2]=0;return j(c,b)}function
d(a){return 0}var
e=1;return function(a){return d7(e,hs,G,br,d,b,a)}}function
d_(a){return j(d9(function(a){return a}),a)}var
d$=[0,0];function
ea(a){d$[1]=[0,a,d$[1]];return 0}function
eb(a,b){var
j=0===b.length-1?[0,0]:b,f=j.length-1,p=0,q=54;if(!(54<0)){var
d=p;for(;;){l(a[1],d,d);var
w=d+1|0;if(q!==d){var
d=w;continue}break}}var
g=[0,ht],m=0,r=55,t=sl(55,f)?r:f,n=54+t|0;if(!(n<m)){var
c=m;for(;;){var
o=c%55|0,u=g[1],i=h(u,k(s(j,aG(c,f))));g[1]=sE(i,0,i.getLen());var
e=g[1];l(a[1],o,(s(a[1],o)^(((e.safeGet(0)+(e.safeGet(1)<<8)|0)+(e.safeGet(2)<<16)|0)+(e.safeGet(3)<<24)|0))&bd);var
v=c+1|0;if(n!==c){var
c=v;continue}break}}a[2]=0;return 0}32===ax;var
hv=[0,hu.slice(),0];try{var
r4=bH(r3),ck=r4}catch(f){if(f[1]!==t)throw f;try{var
r2=bH(r1),ec=r2}catch(f){if(f[1]!==t)throw f;var
ec=hw}var
ck=ec}var
dU=ck.getLen(),hx=82,dV=0;if(0<=0)if(dU<dV)var
bI=0;else
try{var
bq=dV;for(;;){if(dU<=bq)throw[0,t];if(ck.safeGet(bq)!==hx){var
bq=bq+1|0;continue}var
gS=1,cf=gS,bI=1;break}}catch(f){if(f[1]!==t)throw f;var
cf=0,bI=1}else
var
bI=0;if(!bI)var
cf=E(gR);var
ai=[fK,function(a){var
b=[0,w(55,0),0];eb(b,e3(0));return b}];function
ed(a,b){var
m=a?a[1]:cf,d=16;for(;;){if(!(b<=d))if(!(ch<(d*2|0))){var
d=d*2|0;continue}if(m){var
h=sT(ai);if(aJ===h)var
c=ai[1];else
if(fK===h){var
k=ai[0+1];ai[0+1]=g6;try{var
e=j(k,0);ai[0+1]=e;sS(ai,gV)}catch(f){ai[0+1]=function(a){throw f};throw f}var
c=e}else
var
c=ai;c[2]=(c[2]+1|0)%55|0;var
f=s(c[1],c[2]),g=(s(c[1],(c[2]+24|0)%55|0)+(f^f>>>25&31)|0)&bd;l(c[1],c[2],g);var
i=g}else
var
i=0;return[0,0,w(d,0),i,d]}}function
cl(a,b){return 3<=a.length-1?sm(10,aI,a[3],b)&(a[2].length-1-1|0):aG(sn(10,aI,b),a[2].length-1)}function
bt(a,b){var
i=cl(a,b),d=s(a[2],i);if(d){var
e=d[3],j=d[2];if(0===aF(b,d[1]))return j;if(e){var
f=e[3],k=e[2];if(0===aF(b,e[1]))return k;if(f){var
l=f[3],m=f[2];if(0===aF(b,f[1]))return m;var
c=l;for(;;){if(c){var
g=c[3],h=c[2];if(0===aF(b,c[1]))return h;var
c=g;continue}throw[0,t]}}throw[0,t]}throw[0,t]}throw[0,t]}function
a(a,b){return cY(a,b[0+1])}var
cm=[0,0];cY(hy,cm);var
hz=2;function
hA(a){var
b=[0,0],d=a.getLen()-1|0,e=0;if(!(d<0)){var
c=e;for(;;){b[1]=(223*b[1]|0)+a.safeGet(c)|0;var
g=c+1|0;if(d!==c){var
c=g;continue}break}}b[1]=b[1]&((1<<31)-1|0);var
f=bd<b[1]?b[1]-(1<<31)|0:b[1];return f}var
$=ci([0,function(a,b){return e4(a,b)}]),aq=ci([0,function(a,b){return e4(a,b)}]),aj=ci([0,function(a,b){return gl(a,b)}]),ee=e5(0,0),hB=[0,0];function
ef(a){return 2<a?ef((a+1|0)/2|0)*2|0:a}function
eg(a){hB[1]++;var
c=a.length-1,d=w((c*2|0)+2|0,ee);l(d,0,c);l(d,1,(x(ef(c),ax)/8|0)-1|0);var
e=c-1|0,f=0;if(!(e<0)){var
b=f;for(;;){l(d,(b*2|0)+3|0,s(a,b));var
g=b+1|0;if(e!==b){var
b=g;continue}break}}return[0,hz,d,aq[1],aj[1],0,0,$[1],0]}function
cn(a,b){var
c=a[2].length-1,g=c<b?1:0;if(g){var
d=w(b,ee),h=a[2],e=0,f=0,j=0<=c?0<=f?(h.length-1-c|0)<f?0:0<=e?(d.length-1-c|0)<e?0:(r7(h,f,d,e,c),1):0:0:0;if(!j)E(gD);a[2]=d;var
i=0}else
var
i=g;return i}var
eh=[0,0],hC=[0,0];function
co(a){var
b=a[2].length-1;cn(a,b+1|0);return b}function
aT(a,b){try{var
d=i(aq[22],b,a[3])}catch(f){if(f[1]===t){var
c=co(a);a[3]=o(aq[4],b,c,a[3]);a[4]=o(aj[4],c,1,a[4]);return c}throw f}return d}function
cq(a){return a===0?0:aO(a)}function
en(a,b){try{var
d=i($[22],b,a[7])}catch(f){if(f[1]===t){var
c=a[1];a[1]=c+1|0;if(y(b,hS))a[7]=o($[4],b,c,a[7]);return c}throw f}return d}function
cs(a){return se(a,0)?[0]:a}function
ep(a,b){if(a)return a;var
c=e5(gU,b[1]);c[0+1]=b[2];var
d=cm[1];c[1+1]=d;cm[1]=d+1|0;return c}function
bu(a){var
b=co(a);if(0===(b%2|0))var
d=0;else
if((2+at(s(a[2],1)*16|0,ax)|0)<b)var
d=0;else{var
c=co(a),d=1}if(!d)var
c=b;l(a[2],c,0);return c}function
eq(a,ap){var
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
f=n;hC[1]++;if(i(aj[22],k,a[4])){cn(a,k+1|0);l(a[2],k,f)}else
a[6]=[0,[0,k,f],a[6]];g[1]++;continue}return 0}}function
ct(a,b,c){if(bJ(c,h2))return b;var
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
cu(a,b,c){if(bJ(c,h3))return b;var
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
er(a){var
b=a.getLen()<1?1:0,c=b||(47!==a.safeGet(0)?1:0);return c}function
h6(a){var
c=er(a);if(c){var
e=a.getLen()<2?1:0,d=e||y(u(a,0,2),h8);if(d){var
f=a.getLen()<3?1:0,b=f||y(u(a,0,3),h7)}else
var
b=d}else
var
b=c;return b}function
h9(a,b){var
c=b.getLen()<=a.getLen()?1:0,d=c?bJ(u(a,a.getLen()-b.getLen()|0,b.getLen()),b):c;return d}try{var
r0=bH(rZ),cx=r0}catch(f){if(f[1]!==t)throw f;var
cx=h_}function
es(a){var
d=a.getLen(),b=aQ(d+20|0);G(b,39);var
e=d-1|0,f=0;if(!(e<0)){var
c=f;for(;;){if(39===a.safeGet(c))br(b,h$);else
G(b,a.safeGet(c));var
g=c+1|0;if(e!==c){var
c=g;continue}break}}G(b,39);return aR(b)}function
ia(a){return ct(cw,cv,a)}function
ib(a){return cu(cw,cv,a)}function
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
et(a){var
c=cz(a);if(c){var
g=a.getLen()<2?1:0,d=g||y(u(a,0,2),ii);if(d){var
h=a.getLen()<2?1:0,e=h||y(u(a,0,2),ih);if(e){var
i=a.getLen()<3?1:0,f=i||y(u(a,0,3),ig);if(f){var
j=a.getLen()<3?1:0,b=j||y(u(a,0,3),ie)}else
var
b=f}else
var
b=e}else
var
b=d}else
var
b=c;return b}function
eu(a,b){var
c=b.getLen()<=a.getLen()?1:0;if(c){var
e=u(a,a.getLen()-b.getLen()|0,b.getLen()),f=dT(b),d=bJ(dT(e),f)}else
var
d=c;return d}try{var
rY=bH(rX),ev=rY}catch(f){if(f[1]!==t)throw f;var
ev=ij}function
ik(h){var
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
ew(a){var
c=2<=a.getLen()?1:0;if(c){var
b=a.safeGet(0),g=91<=b?(b+fz|0)<0||25<(b+fz|0)?0:1:65<=b?1:0,d=g?1:0,e=d?58===a.safeGet(1)?1:0:d}else
var
e=c;if(e){var
f=u(a,2,a.getLen()-2|0);return[0,u(a,0,2),f]}return[0,il,a]}function
im(a){var
b=ew(a),c=b[1];return h(c,cu(az,cy,b[2]))}function
io(a){return ct(az,cy,ew(a)[2])}function
ir(a){return ct(az,cA,a)}function
is(a){return cu(az,cA,a)}if(y(cg,it))if(y(cg,iu)){if(y(cg,iv))throw[0,F,iw];var
bv=[0,cy,ic,id,az,cz,et,eu,ev,ik,io,im]}else
var
bv=[0,cv,h4,h5,cw,er,h6,h9,cx,es,ia,ib];else
var
bv=[0,cA,ip,iq,az,cz,et,eu,cx,es,ir,is];var
ex=[0,iz],ix=bv[11],iy=bv[3];a(iC,[0,ex,0,iB,iA]);ea(function(a){if(a[1]===ex){var
c=a[2],d=a[4],e=a[3];if(typeof
c===m)switch(c){case
1:var
b=iF;break;case
2:var
b=iG;break;case
3:var
b=iH;break;case
4:var
b=iI;break;case
5:var
b=iJ;break;case
6:var
b=iK;break;case
7:var
b=iL;break;case
8:var
b=iM;break;case
9:var
b=iN;break;case
10:var
b=iO;break;case
11:var
b=iP;break;case
12:var
b=iQ;break;case
13:var
b=iR;break;case
14:var
b=iS;break;case
15:var
b=iT;break;case
16:var
b=iU;break;case
17:var
b=iV;break;case
18:var
b=iW;break;case
19:var
b=iX;break;case
20:var
b=iY;break;case
21:var
b=iZ;break;case
22:var
b=i0;break;case
23:var
b=i1;break;case
24:var
b=i2;break;case
25:var
b=i3;break;case
26:var
b=i4;break;case
27:var
b=i5;break;case
28:var
b=i6;break;case
29:var
b=i7;break;case
30:var
b=i8;break;case
31:var
b=i9;break;case
32:var
b=i_;break;case
33:var
b=i$;break;case
34:var
b=ja;break;case
35:var
b=jb;break;case
36:var
b=jc;break;case
37:var
b=jd;break;case
38:var
b=je;break;case
39:var
b=jf;break;case
40:var
b=jg;break;case
41:var
b=jh;break;case
42:var
b=ji;break;case
43:var
b=jj;break;case
44:var
b=jk;break;case
45:var
b=jl;break;case
46:var
b=jm;break;case
47:var
b=jn;break;case
48:var
b=jo;break;case
49:var
b=jp;break;case
50:var
b=jq;break;case
51:var
b=jr;break;case
52:var
b=js;break;case
53:var
b=jt;break;case
54:var
b=ju;break;case
55:var
b=jv;break;case
56:var
b=jw;break;case
57:var
b=jx;break;case
58:var
b=jy;break;case
59:var
b=jz;break;case
60:var
b=jA;break;case
61:var
b=jB;break;case
62:var
b=jC;break;case
63:var
b=jD;break;case
64:var
b=jE;break;case
65:var
b=jF;break;case
66:var
b=jG;break;case
67:var
b=jH;break;default:var
b=iD}else{var
f=c[1],b=j(d_(jI),f)}return[0,o(d_(iE),b,e,d)]}return 0});bK(jJ);bK(jK);try{bK(rW)}catch(f){if(f[1]!==aM)throw f}try{bK(rV)}catch(f){if(f[1]!==aM)throw f}ed(0,7);function
ey(a){return t1(a)}ah(32,r);var
jL=6,jM=0,jR=A(b1),jS=0,jT=r;if(!(r<0)){var
a7=jS;for(;;){jR.safeSet(a7,dS(ce(a7)));var
rU=a7+1|0;if(jT!==a7){var
a7=rU;continue}break}}var
cB=ah(32,0);cB.safeSet(10>>>3,ce(cB.safeGet(10>>>3)|1<<(10&7)));var
jN=A(32),jO=0,jP=31;if(!(31<0)){var
aX=jO;for(;;){jN.safeSet(aX,ce(cB.safeGet(aX)^r));var
jQ=aX+1|0;if(jP!==aX){var
aX=jQ;continue}break}}var
aA=[0,0],aB=[0,0],ez=[0,0];function
H(a){return aA[1]}function
eA(a){return aB[1]}function
P(a,b,c){return 0===a[2][0]?b?tr(a[1],a,b[1]):ts(a[1],a):b?e6(a[1],b[1]):e6(a[1],0)}var
eB=[3,jL],cC=[0,0];function
aC(e,b,c){cC[1]++;switch(e[0]){case
7:case
8:throw[0,F,jU];case
6:var
g=e[1],m=cC[1],n=e7(0),o=w(eA(0)+1|0,n),p=e8(0),q=w(H(0)+1|0,p),f=[0,-1,[1,[0,tf(g,c),g]],q,o,c,0,e,0,0,m,0];break;default:var
h=e[1],i=cC[1],j=e7(0),k=w(eA(0)+1|0,j),l=e8(0),f=[0,-1,[0,r$(h,jM,[0,c])],w(H(0)+1|0,l),k,c,0,e,0,0,i,0]}if(b){var
d=b[1],a=function(a){{if(0===d[2][0])return 6===e[0]?gt(f,d[1][8],d[1]):gs(f,d[1][8],d[1]);{var
b=d[1],c=H(0);return e9(f,d[1][8]-c|0,b)}}};try{a(0)}catch(f){a9(0);a(0)}f[6]=[0,d]}return f}function
T(a){return a[5]}function
aY(a){return a[6]}function
bw(a){return a[8]}function
bx(a){return a[7]}function
Y(a){return a[2]}function
by(a,b,c){a[1]=b;a[6]=c;return 0}function
cD(a,b,c){return du<=b?s(a[3],c):s(a[4],c)}function
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
ar=[0,j0];a(j9,[0,[0,jV]]);a(j_,[0,[0,jW]]);a(j$,[0,[0,jX]]);a(ka,[0,[0,jY]]);a(kb,[0,[0,jZ]]);a(kc,[0,ar]);a(kd,[0,[0,j1]]);a(ke,[0,[0,j2]]);a(kf,[0,[0,j3]]);a(kg,[0,[0,j5]]);a(kh,[0,[0,j6]]);a(ki,[0,[0,j7]]);a(kj,[0,[0,j8]]);a(kk,[0,[0,j4]]);var
cF=[0,ks];a(kI,[0,[0,kl]]);a(kJ,[0,[0,km]]);a(kK,[0,[0,kn]]);a(kL,[0,[0,ko]]);a(kM,[0,[0,kp]]);a(kN,[0,[0,kq]]);a(kO,[0,[0,kr]]);a(kP,[0,cF]);a(kQ,[0,[0,kt]]);a(kR,[0,[0,ku]]);a(kS,[0,[0,kv]]);a(kT,[0,[0,kw]]);a(kU,[0,[0,kx]]);a(kV,[0,[0,ky]]);a(kW,[0,[0,kz]]);a(kX,[0,[0,kA]]);a(kY,[0,[0,kB]]);a(kZ,[0,[0,kC]]);a(k0,[0,[0,kD]]);a(k1,[0,[0,kE]]);a(k2,[0,[0,kF]]);a(k3,[0,[0,kG]]);a(k4,[0,[0,kH]]);var
bA=1,eC=0;function
aZ(a,b,c){var
d=a[2];if(0===d[0])var
f=sb(d[1],b,c);else{var
e=d[1],f=o(e[2][4],e[1],b,c)}return f}function
a0(a,b){var
c=a[2];if(0===c[0])var
e=sa(c[1],b);else{var
d=c[1],e=i(d[2][3],d[1],b)}return e}function
eD(a,b){P(a,0,0);eI(b,0,0);return P(a,0,0)}function
aa(a,b,c){var
f=a,d=b;for(;;){if(eC)return aZ(f,d,c);var
n=d<0?1:0,o=n||(T(f)<=d?1:0);if(o)throw[0,bm,k5];if(bA){var
i=aY(f);if(typeof
i!==m)eD(i[1],f)}var
j=bw(f);if(j){var
e=j[1];if(1===e[1]){var
k=e[4],g=e[3],l=e[2];return 0===k?aZ(e[5],l+d|0,c):aZ(e[5],(l+x(at(d,g),k+g|0)|0)+aG(d,g)|0,c)}var
h=e[3],f=e[5],d=(e[2]+x(at(d,h),e[4]+h|0)|0)+aG(d,h)|0;continue}return aZ(f,d,c)}}function
ab(a,b){var
e=a,c=b;for(;;){if(eC)return a0(e,c);var
l=c<0?1:0,n=l||(T(e)<=c?1:0);if(n)throw[0,bm,k6];if(bA){var
h=aY(e);if(typeof
h!==m)eD(h[1],e)}var
i=bw(e);if(i){var
d=i[1];if(1===d[1]){var
j=d[4],f=d[3],k=d[2];return 0===j?a0(d[5],k+c|0):a0(d[5],(k+x(at(c,f),j+f|0)|0)+aG(c,f)|0)}var
g=d[3],e=d[5],c=(d[2]+x(at(c,g),d[4]+g|0)|0)+aG(c,g)|0;continue}return a0(e,c)}}function
eE(a){if(a[8]){var
b=aC(a[7],0,a[5]);b[1]=a[1];b[6]=a[6];cE(a,b);var
c=b}else
var
c=a;return c}function
eF(d,b,c){{if(0===c[2][0]){var
a=function(a){return 0===Y(d)[0]?ti(d,c[1][8],c[1],c[3],b):tk(d,c[1][8],c[1],b)};try{var
f=a(0)}catch(f){if(f[1]===ar){try{P(c,0,0);var
g=a(0)}catch(f){a9(0);return a(0)}return g}throw f}return f}var
e=function(a){{if(0===Y(d)[0]){var
e=c[1],f=H(0);return tK(d,c[1][8]-f|0,e,b)}var
g=c[1],h=H(0);return tM(d,c[1][8]-h|0,g,b)}};try{var
i=e(0)}catch(f){try{P(c,0,0);var
h=e(0)}catch(f){a9(0);return e(0)}return h}return i}}function
eG(d,b,c){{if(0===c[2][0]){var
a=function(a){return 0===Y(d)[0]?tq(d,c[1][8],c[1],c,b):tl(d,c[1][8],c[1],b)};try{var
f=a(0)}catch(f){if(f[1]===ar){try{P(c,0,0);var
g=a(0)}catch(f){a9(0);return a(0)}return g}throw f}return f}var
e=function(a){{if(0===Y(d)[0]){var
e=c[2],f=c[1],g=H(0);return tQ(d,c[1][8]-g|0,f,e,b)}var
h=c[2],i=c[1],j=H(0);return tN(d,c[1][8]-j|0,i,h,b)}};try{var
i=e(0)}catch(f){try{P(c,0,0);var
h=e(0)}catch(f){a9(0);return e(0)}return h}return i}}function
a1(a,b,c,d,e,f,g,h){{if(0===d[2][0])return 0===Y(a)[0]?tz(a,b,d[1][8],d[1],d[3],c,e,f,g,h):tn(a,b,d[1][8],d[1],d[3],c,e,f,g,h);{if(0===Y(a)[0]){var
i=d[3],j=d[1],k=H(0);return tZ(a,b,d[1][8]-k|0,j,i,c,e,f,g,h)}var
l=d[3],m=d[1],n=H(0);return tO(a,b,d[1][8]-n|0,m,l,c,e,f,g,h)}}}function
a2(a,b,c,d,e,f,g,h){{if(0===d[2][0])return 0===Y(a)[0]?tA(a,b,d[1][8],d[1],d[3],c,e,f,g,h):to(a,b,d[1][8],d[1],d[3],c,e,f,g,h);{if(0===Y(a)[0]){var
i=d[3],j=d[1],k=H(0);return t0(a,b,d[1][8]-k|0,j,i,c,e,f,g,h)}var
l=d[3],m=d[1],n=H(0);return tP(a,b,d[1][8]-n|0,m,l,c,e,f,g,h)}}}function
eH(a,b,c){var
q=b;for(;;){var
d=q?q[1]:0,r=aY(a);if(typeof
r===m){by(a,c[1][8],[1,c]);try{cG(a,c)}catch(f){if(f[1]!==ar)f[1]===cF;try{P(c,[0,d],0);cG(a,c)}catch(f){if(f[1]!==ar)if(f[1]!==cF)throw f;P(c,0,0);si(0);cG(a,c)}}var
z=bw(a);if(z){var
j=z[1];if(1===j[1]){var
k=j[5],s=j[4],f=j[3],l=j[2];if(0===f)a1(k,a,d,c,0,0,l,T(a));else
if(p<f){var
h=0,n=T(a);for(;;){if(f<n){a1(k,a,d,c,x(h,f+s|0),x(h,f),l,f);var
h=h+1|0,n=n-f|0;continue}if(0<n)a1(k,a,d,c,x(h,f+s|0),x(h,f),l,n);break}}else{var
e=0,i=0,g=T(a);for(;;){if(p<g){var
v=aC(bx(a),0,p);bz(a,v);var
A=e+ga|0;if(!(A<e)){var
t=e;for(;;){aa(v,t,ab(a,e));var
H=t+1|0;if(A!==t){var
t=H;continue}break}}a1(k,v,d,c,x(i,p+s|0),i*p|0,l,p);var
e=e+p|0,i=i+1|0,g=g+fO|0;continue}if(0<g){var
w=aC(bx(a),0,g),B=(e+g|0)-1|0;if(!(B<e)){var
u=e;for(;;){aa(w,u,ab(a,e));var
I=u+1|0;if(B!==u){var
u=I;continue}break}}bz(a,w);a1(k,w,d,c,x(i,p+s|0),i*p|0,l,g)}break}}}else{var
y=eE(a),C=T(a)-1|0,J=0;if(!(C<0)){var
o=J;for(;;){aZ(y,o,ab(a,o));var
K=o+1|0;if(C!==o){var
o=K;continue}break}}eF(y,d,c);cE(y,a)}}else
eF(a,d,c);return by(a,c[1][8],[0,c])}else{if(0===r[0]){var
D=r[1],E=c0(D,c);if(E){eI(a,[0,d],0);P(D,0,0);var
q=[0,d];continue}return E}var
F=r[1],G=c0(F,c);if(G){P(F,0,0);var
q=[0,d];continue}return G}}}function
cG(a,b){{if(0===b[2][0])return 0===Y(a)[0]?gs(a,b[1][8],b[1]):gt(a,b[1][8],b[1]);{if(0===Y(a)[0]){var
c=b[1],d=H(0);return e9(a,b[1][8]-d|0,c)}var
e=b[1],f=H(0);return tL(a,b[1][8]-f|0,e)}}}function
eI(a,b,c){var
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
B=g+ga|0;if(!(B<g)){var
t=g;for(;;){aa(y,t,ab(a,g));var
E=t+1|0;if(B!==t){var
t=E;continue}break}}a2(l,y,f,d,x(j,p+s|0),j*p|0,n,p);var
j=j+1|0,h=h+fO|0;continue}if(0<h){var
z=aC(bx(a),0,h),C=(g+h|0)-1|0;if(!(C<g)){var
u=g;for(;;){aa(z,u,ab(a,g));var
F=u+1|0;if(C!==u){var
u=F;continue}break}}bz(a,z);a2(l,z,f,d,x(j,p+s|0),j*p|0,n,h)}break}}}else{var
v=eE(a);cE(v,a);eG(v,f,d);var
D=T(v)-1|0,G=0;if(!(D<0)){var
q=G;for(;;){aa(a,q,a0(v,q));var
H=q+1|0;if(D!==q){var
q=H;continue}break}}}}else
eG(a,f,d);return by(a,d[1][8],0)}P(r[1],0,0);var
w=[0,f];continue}}}var
k$=[0,k_],lb=[0,la];function
bB(a,b){var
p=s(gT,0),q=h(iy,h(a,b)),f=dJ(h(ix(p),q));try{var
n=eK,g=eK;a:for(;;){if(1){var
k=function(a,b,c){var
e=b,d=c;for(;;){if(d){var
g=d[1],f=g.getLen(),h=d[2];a8(g,0,a,e-f|0,f);var
e=e-f|0,d=h;continue}return a}},d=0,e=0;for(;;){var
c=sJ(f);if(0===c){if(!d)throw[0,bn];var
j=k(A(e),e,d)}else{if(!(0<c)){var
m=A(-c|0);c1(f,m,0,-c|0);var
d=[0,m,d],e=e-c|0;continue}var
i=A(c-1|0);c1(f,i,0,c-1|0);sI(f);if(d){var
l=(e+c|0)-1|0,j=k(A(l),l,[0,i,d])}else
var
j=i}var
g=h(g,h(j,lc)),n=g;continue a}}var
o=g;break}}catch(f){if(f[1]!==bn)throw f;var
o=n}dL(f);return o}var
eL=[0,ld],cH=[],le=0,lf=0;e$(cH,[0,0,function(f){var
k=en(f,lg),e=cs(k7),d=e.length-1,n=eJ.length-1,a=w(d+n|0,0),p=d-1|0,u=0;if(!(p<0)){var
c=u;for(;;){l(a,c,aT(f,s(e,c)));var
y=c+1|0;if(p!==c){var
c=y;continue}break}}var
q=n-1|0,v=0;if(!(q<0)){var
b=v;for(;;){l(a,b+d|0,en(f,s(eJ,b)));var
x=b+1|0;if(q!==b){var
b=x;continue}break}}var
r=a[10],m=a[12],h=a[15],i=a[16],j=a[17],g=a[18],z=a[1],A=a[2],B=a[3],C=a[4],D=a[5],E=a[7],F=a[8],G=a[9],H=a[11],I=a[14];function
J(a,b,c,d,e,f){var
h=d?d[1]:d;o(a[1][m+1],a,[0,h],f);var
i=bt(a[g+1],f);return e_(a[1][r+1],a,b,[0,c[1],c[2]],e,f,i)}function
K(a,b,c,d,e){try{var
f=bt(a[g+1],e),h=f}catch(f){if(f[1]!==t)throw f;try{o(a[1][m+1],a,lh,e)}catch(f){throw f}var
h=bt(a[g+1],e)}return e_(a[1][r+1],a,b,[0,c[1],c[2]],d,e,h)}function
L(a,b,c){var
y=b?b[1]:b;try{bt(a[g+1],c);var
f=0}catch(f){if(f[1]===t){if(0===c[2][0]){var
z=a[i+1];if(!z)throw[0,eL,c];var
A=z[1],H=y?tp(A,a[h+1],c[1]):th(A,a[h+1],c[1]),B=H}else{var
D=a[j+1];if(!D)throw[0,eL,c];var
E=D[1],I=y?tB(E,a[h+1],c[1]):tJ(E,a[h+1],c[1]),B=I}var
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
f=[0,bB(a[k+1],lj),0],c=f}catch(f){var
c=0}a[i+1]=c;try{var
e=[0,bB(a[k+1],li),0],d=e}catch(f){var
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
c=h;continue}break}}return 0}eq(f,[0,G,function(a,b){return a[g+1]},C,R,F,Q,A,P,E,O,z,N,D,M,m,L,B,K,H,J]);return function(a,b,c,d){var
e=ep(b,f);e[k+1]=c;e[I+1]=c;e[h+1]=d;try{var
o=[0,bB(c,ll),0],l=o}catch(f){var
l=0}e[i+1]=l;try{var
n=[0,bB(c,lk),0],m=n}catch(f){var
m=0}e[j+1]=m;e[g+1]=ed(0,8);return e}},lf,le]);fa(0);fa(0);function
cI(a){function
e(a,b){var
d=a-1|0,e=0;if(!(d<0)){var
c=e;for(;;){d8(ln);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return j(d8(lm),b)}function
f(a,b){var
c=a,d=b;for(;;)if(typeof
d===m)return 0===d?e(c,lo):e(c,lp);else
switch(d[0]){case
0:e(c,lq);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
1:e(c,lr);var
c=c+1|0,d=d[1];continue;case
2:e(c,ls);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
3:e(c,lt);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
4:e(c,lu);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
5:e(c,lv);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
6:e(c,lw);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
7:e(c,lx);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
8:e(c,ly);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
9:e(c,lz);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
10:e(c,lA);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
11:return e(c,h(lB,d[1]));case
12:return e(c,h(lC,d[1]));case
13:return e(c,h(lD,k(d[1])));case
14:return e(c,h(lE,k(d[1])));case
15:return e(c,h(lF,k(d[1])));case
16:return e(c,h(lG,k(d[1])));case
17:return e(c,h(lH,k(d[1])));case
18:return e(c,lI);case
19:return e(c,lJ);case
20:return e(c,lK);case
21:return e(c,lL);case
22:return e(c,lM);case
23:return e(c,h(lN,k(d[2])));case
24:e(c,lO);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
25:e(c,lP);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
26:e(c,lQ);var
c=c+1|0,d=d[1];continue;case
27:e(c,lR);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
28:e(c,lS);var
c=c+1|0,d=d[1];continue;case
29:e(c,lT);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
30:e(c,lU);f(c+1|0,d[1]);f(c+1|0,d[2]);var
c=c+1|0,d=d[3];continue;case
31:return e(c,lV);case
32:var
g=h(lW,k(d[2]));return e(c,h(lX,h(d[1],g)));case
33:return e(c,h(lY,k(d[1])));case
36:e(c,l0);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
37:e(c,l1);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
38:e(c,l2);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
39:e(c,l3);f(c+1|0,d[1]);f(c+1|0,d[2]);var
c=c+1|0,d=d[3];continue;case
40:e(c,l4);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
41:e(c,l5);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
42:e(c,l6);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
43:e(c,l7);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
44:e(c,l8);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
45:e(c,l9);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
46:e(c,l_);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
47:e(c,l$);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
48:e(c,ma);f(c+1|0,d[1]);f(c+1|0,d[2]);f(c+1|0,d[3]);var
c=c+1|0,d=d[4];continue;case
49:e(c,mb);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
50:e(c,mc);f(c+1|0,d[1]);var
i=d[2],j=c+1|0;return dN(function(a){return f(j,a)},i);case
51:return e(c,md);case
52:return e(c,me);default:return e(c,h(lZ,N(d[1])))}}return f(0,a)}function
I(a){return ah(a,32)}var
a3=[0,mf];function
a$(a,b,c){var
d=c;for(;;)if(typeof
d===m)return mh;else
switch(d[0]){case
18:case
19:var
T=h(mW,h(e(b,d[2]),mV));return h(mX,h(k(d[1]),T));case
27:case
38:var
ac=d[1],ad=h(ng,h(e(b,d[2]),nf));return h(e(b,ac),ad);case
0:var
g=d[2],B=e(b,d[1]);if(typeof
g===m)var
r=0;else
if(25===g[0]){var
t=e(b,g),r=1}else
var
r=0;if(!r){var
D=h(mi,I(b)),t=h(e(b,g),D)}return h(h(B,t),mj);case
1:var
E=h(e(b,d[1]),mk),G=y(a3[1][1],ml)?h(a3[1][1],mm):mo;return h(mn,h(G,E));case
2:var
H=h(mq,h(U(b,d[2]),mp));return h(mr,h(U(b,d[1]),H));case
3:var
J=h(mt,h(ak(b,d[2]),ms));return h(mu,h(ak(b,d[1]),J));case
4:var
K=h(mw,h(U(b,d[2]),mv));return h(mx,h(U(b,d[1]),K));case
5:var
L=h(mz,h(ak(b,d[2]),my));return h(mA,h(ak(b,d[1]),L));case
6:var
M=h(mC,h(U(b,d[2]),mB));return h(mD,h(U(b,d[1]),M));case
7:var
O=h(mF,h(ak(b,d[2]),mE));return h(mG,h(ak(b,d[1]),O));case
8:var
P=h(mI,h(U(b,d[2]),mH));return h(mJ,h(U(b,d[1]),P));case
9:var
Q=h(mL,h(ak(b,d[2]),mK));return h(mM,h(ak(b,d[1]),Q));case
10:var
R=h(mO,h(U(b,d[2]),mN));return h(mP,h(U(b,d[1]),R));case
13:return h(mQ,k(d[1]));case
14:return h(mR,k(d[1]));case
15:throw[0,F,mS];case
16:return h(mT,k(d[1]));case
17:return h(mU,k(d[1]));case
20:var
V=h(mZ,h(e(b,d[2]),mY));return h(m0,h(k(d[1]),V));case
21:var
W=h(m2,h(e(b,d[2]),m1));return h(m3,h(k(d[1]),W));case
22:var
X=h(m5,h(e(b,d[2]),m4));return h(m6,h(k(d[1]),X));case
23:var
Y=h(m7,k(d[2])),u=d[1];if(typeof
u===m)var
f=0;else
switch(u[0]){case
33:var
o=m9,f=1;break;case
34:var
o=m_,f=1;break;case
35:var
o=m$,f=1;break;default:var
f=0}if(f)return h(o,Y);throw[0,F,m8];case
24:var
i=d[2],v=d[1];if(typeof
i===m){if(0===i){var
d=v;continue}}else
if(24===i[0]){var
Z=h(nb,e(b,i));return h(e(b,v),Z)}return S(na);case
25:var
_=e(b,d[2]),$=h(nc,h(I(b),_));return h(e(b,d[1]),$);case
26:var
aa=e(b,d[1]),ab=y(a3[1][2],nd)?a3[1][2]:ne;return h(ab,aa);case
28:var
d=d[1];continue;case
29:var
ae=h(ni,h(e(b,d[2]),nh));return h(e(b,d[1]),ae);case
30:var
l=d[2],af=e(b,d[3]),ag=h(nj,h(I(b),af));if(typeof
l===m)var
s=0;else
if(31===l[0]){var
w=h(mg(l[1]),nl),s=1}else
var
s=0;if(!s)var
w=e(b,l);var
ah=h(nk,h(w,ag));return h(e(b,d[1]),ah);case
31:return a<50?a_(1+a,d[1]):C(a_,[0,d[1]]);case
33:return k(d[1]);case
34:return h(N(d[1]),nm);case
35:return N(d[1]);case
36:var
ai=h(no,h(e(b,d[2]),nn));return h(e(b,d[1]),ai);case
37:var
aj=h(nq,h(I(b),np)),al=h(e(b,d[2]),aj),am=h(nr,h(I(b),al)),an=h(ns,h(e(b,d[1]),am));return h(I(b),an);case
39:var
ao=h(nt,I(b)),ap=h(e(b+2|0,d[3]),ao),aq=h(nu,h(I(b+2|0),ap)),ar=h(nv,h(I(b),aq)),as=h(e(b+2|0,d[2]),ar),at=h(nw,h(I(b+2|0),as));return h(nx,h(e(b,d[1]),at));case
40:var
au=h(ny,I(b)),av=h(e(b+2|0,d[2]),au),aw=h(nz,h(I(b+2|0),av));return h(nA,h(e(b,d[1]),aw));case
41:var
ax=h(nB,e(b,d[2]));return h(e(b,d[1]),ax);case
42:var
ay=h(nC,e(b,d[2]));return h(e(b,d[1]),ay);case
43:var
az=h(nD,e(b,d[2]));return h(e(b,d[1]),az);case
44:var
aA=h(nE,e(b,d[2]));return h(e(b,d[1]),aA);case
45:var
aB=h(nF,e(b,d[2]));return h(e(b,d[1]),aB);case
46:var
aC=h(nG,e(b,d[2]));return h(e(b,d[1]),aC);case
47:var
aD=h(nH,e(b,d[2]));return h(e(b,d[1]),aD);case
48:var
p=e(b,d[1]),aE=e(b,d[2]),aF=e(b,d[3]),aG=h(e(b+2|0,d[4]),nI);return h(nO,h(p,h(nN,h(aE,h(nM,h(p,h(nL,h(aF,h(nK,h(p,h(nJ,h(I(b+2|0),aG))))))))))));case
49:var
aH=e(b,d[1]),aI=h(e(b+2|0,d[2]),nP);return h(nR,h(aH,h(nQ,h(I(b+2|0),aI))));case
50:var
x=d[2],n=d[1],z=e(b,n),q=function(a){if(a){var
c=a[2],d=a[1];if(c){var
f=h(nS,q(c));return h(e(b,d),f)}return e(b,d)}throw[0,F,nT]};if(typeof
n!==m)if(31===n[0]){var
A=n[1];if(!y(A[1],nW))if(!y(A[2],nX))return h(z,h(nZ,h(q(aO(x)),nY)))}return h(z,h(nV,h(q(aO(x)),nU)));case
51:return k(j(d[1],0));case
52:return h(N(j(d[1],0)),n0);default:return d[1]}}function
r5(a,b,c){if(typeof
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
d=h(n2,h(U(b,c[2]),n1));return h(e(b,c[1]),d);case
51:return k(j(c[1],0));default:}return a<50?c2(1+a,b,c):C(c2,[0,b,c])}function
c2(a,b,c){if(typeof
c!==m)switch(c[0]){case
3:case
5:case
7:case
9:case
29:case
50:return a<50?a$(1+a,b,c):C(a$,[0,b,c]);case
16:return h(n4,k(c[1]));case
31:return a<50?a_(1+a,c[1]):C(a_,[0,c[1]]);case
32:return c[1];case
34:return h(N(c[1]),n5);case
35:return h(n6,N(c[1]));case
36:var
d=h(n8,h(U(b,c[2]),n7));return h(e(b,c[1]),d);case
52:return h(N(j(c[1],0)),n9);default:}cI(c);return S(n3)}function
a_(a,b){return b[1]}function
e(b,c){return V(a$(0,b,c))}function
U(b,c){return V(r5(0,b,c))}function
ak(b,c){return V(c2(0,b,c))}function
mg(b){return V(a_(0,b))}function
z(a){return ah(a,32)}var
a4=[0,n_];function
bb(a,b,c){var
d=c;for(;;)if(typeof
d===m)return oa;else
switch(d[0]){case
18:case
19:var
U=h(ox,h(f(b,d[2]),ow));return h(oy,h(k(d[1]),U));case
27:case
38:var
ac=d[1],ad=h(oS,Q(b,d[2]));return h(f(b,ac),ad);case
0:var
g=d[2],D=f(b,d[1]);if(typeof
g===m)var
r=0;else
if(25===g[0]){var
t=f(b,g),r=1}else
var
r=0;if(!r){var
E=h(ob,z(b)),t=h(f(b,g),E)}return h(h(D,t),oc);case
1:var
G=h(f(b,d[1]),od),H=y(a4[1][1],oe)?h(a4[1][1],of):oh;return h(og,h(H,G));case
2:var
I=h(oi,Q(b,d[2]));return h(Q(b,d[1]),I);case
3:var
J=h(oj,al(b,d[2]));return h(al(b,d[1]),J);case
4:var
K=h(ok,Q(b,d[2]));return h(Q(b,d[1]),K);case
5:var
L=h(ol,al(b,d[2]));return h(al(b,d[1]),L);case
6:var
M=h(om,Q(b,d[2]));return h(Q(b,d[1]),M);case
7:var
O=h(on,al(b,d[2]));return h(al(b,d[1]),O);case
8:var
P=h(oo,Q(b,d[2]));return h(Q(b,d[1]),P);case
9:var
R=h(op,al(b,d[2]));return h(al(b,d[1]),R);case
10:var
T=h(oq,Q(b,d[2]));return h(Q(b,d[1]),T);case
13:return h(or,k(d[1]));case
14:return h(os,k(d[1]));case
15:throw[0,F,ot];case
16:return h(ou,k(d[1]));case
17:return h(ov,k(d[1]));case
20:var
V=h(oA,h(f(b,d[2]),oz));return h(oB,h(k(d[1]),V));case
21:var
W=h(oD,h(f(b,d[2]),oC));return h(oE,h(k(d[1]),W));case
22:var
X=h(oG,h(f(b,d[2]),oF));return h(oH,h(k(d[1]),X));case
23:var
Y=h(oI,k(d[2])),u=d[1];if(typeof
u===m)var
e=0;else
switch(u[0]){case
33:var
o=oK,e=1;break;case
34:var
o=oL,e=1;break;case
35:var
o=oM,e=1;break;default:var
e=0}if(e)return h(o,Y);throw[0,F,oJ];case
24:var
i=d[2],v=d[1];if(typeof
i===m){if(0===i){var
d=v;continue}}else
if(24===i[0]){var
Z=h(oO,f(b,i));return h(f(b,v),Z)}return S(oN);case
25:var
_=f(b,d[2]),$=h(oP,h(z(b),_));return h(f(b,d[1]),$);case
26:var
aa=f(b,d[1]),ab=y(a4[1][2],oQ)?a4[1][2]:oR;return h(ab,aa);case
28:var
d=d[1];continue;case
29:var
ae=h(oU,h(f(b,d[2]),oT));return h(f(b,d[1]),ae);case
30:var
l=d[2],af=f(b,d[3]),ag=h(oV,h(z(b),af));if(typeof
l===m)var
s=0;else
if(31===l[0]){var
w=n$(l[1]),s=1}else
var
s=0;if(!s)var
w=f(b,l);var
ah=h(oW,h(w,ag));return h(f(b,d[1]),ah);case
31:return a<50?ba(1+a,d[1]):C(ba,[0,d[1]]);case
33:return k(d[1]);case
34:return h(N(d[1]),oX);case
35:return N(d[1]);case
36:var
ai=h(oZ,h(f(b,d[2]),oY));return h(f(b,d[1]),ai);case
37:var
aj=h(o1,h(z(b),o0)),ak=h(f(b,d[2]),aj),am=h(o2,h(z(b),ak)),an=h(o3,h(f(b,d[1]),am));return h(z(b),an);case
39:var
ao=h(o4,z(b)),ap=h(f(b+2|0,d[3]),ao),aq=h(o5,h(z(b+2|0),ap)),ar=h(o6,h(z(b),aq)),as=h(f(b+2|0,d[2]),ar),at=h(o7,h(z(b+2|0),as));return h(o8,h(f(b,d[1]),at));case
40:var
au=h(o9,z(b)),av=h(o_,h(z(b),au)),aw=h(f(b+2|0,d[2]),av),ax=h(o$,h(z(b+2|0),aw)),ay=h(pa,h(z(b),ax));return h(pb,h(f(b,d[1]),ay));case
41:var
az=h(pc,f(b,d[2]));return h(f(b,d[1]),az);case
42:var
aA=h(pd,f(b,d[2]));return h(f(b,d[1]),aA);case
43:var
aB=h(pe,f(b,d[2]));return h(f(b,d[1]),aB);case
44:var
aC=h(pf,f(b,d[2]));return h(f(b,d[1]),aC);case
45:var
aD=h(pg,f(b,d[2]));return h(f(b,d[1]),aD);case
46:var
aE=h(ph,f(b,d[2]));return h(f(b,d[1]),aE);case
47:var
aF=h(pi,f(b,d[2]));return h(f(b,d[1]),aF);case
48:var
p=f(b,d[1]),aG=f(b,d[2]),aH=f(b,d[3]),aI=h(f(b+2|0,d[4]),pj);return h(pp,h(p,h(po,h(aG,h(pn,h(p,h(pm,h(aH,h(pl,h(p,h(pk,h(z(b+2|0),aI))))))))))));case
49:var
aJ=f(b,d[1]),aK=h(f(b+2|0,d[2]),pq);return h(ps,h(aJ,h(pr,h(z(b+2|0),aK))));case
50:var
x=d[2],n=d[1],A=f(b,n),q=function(a){if(a){var
c=a[2],d=a[1];if(c){var
e=h(pt,q(c));return h(f(b,d),e)}return f(b,d)}throw[0,F,pu]};if(typeof
n!==m)if(31===n[0]){var
B=n[1];if(!y(B[1],px))if(!y(B[2],py))return h(A,h(pA,h(q(aO(x)),pz)))}return h(A,h(pw,h(q(aO(x)),pv)));case
51:return k(j(d[1],0));case
52:return h(N(j(d[1],0)),pB);default:return d[1]}}function
r6(a,b,c){if(typeof
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
d=h(pD,h(Q(b,c[2]),pC));return h(f(b,c[1]),d);case
51:return k(j(c[1],0));default:}return a<50?c3(1+a,b,c):C(c3,[0,b,c])}function
c3(a,b,c){if(typeof
c!==m)switch(c[0]){case
3:case
5:case
7:case
9:case
50:return a<50?bb(1+a,b,c):C(bb,[0,b,c]);case
16:return h(pF,k(c[1]));case
31:return a<50?ba(1+a,c[1]):C(ba,[0,c[1]]);case
32:return c[1];case
34:return h(N(c[1]),pG);case
35:return h(pH,N(c[1]));case
36:var
d=h(pJ,h(Q(b,c[2]),pI));return h(f(b,c[1]),d);case
52:return h(N(j(c[1],0)),pK);default:}cI(c);return S(pE)}function
ba(a,b){return b[2]}function
f(b,c){return V(bb(0,b,c))}function
Q(b,c){return V(r6(0,b,c))}function
al(b,c){return V(c3(0,b,c))}function
n$(b){return V(ba(0,b))}var
pW=h(pV,h(pU,h(pT,h(pS,h(pR,h(pQ,h(pP,h(pO,h(pN,h(pM,pL)))))))))),qb=h(qa,h(p$,h(p_,h(p9,h(p8,h(p7,h(p6,h(p5,h(p4,h(p3,h(p2,h(p1,h(p0,h(pZ,h(pY,pX))))))))))))))),qj=h(qi,h(qh,h(qg,h(qf,h(qe,h(qd,qc)))))),qr=h(qq,h(qp,h(qo,h(qn,h(qm,h(ql,qk))))));function
v(a){return[32,h(qs,k(a)),a]}function
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
eM(a){var
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
qz=[0,qy];function
a6(a,b,c){var
g=c[2],d=c[1],t=a?a[1]:a,u=b?b[1]:2,n=g[3],p=g[2];qz[1]=qA;var
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
o=h(qI,h(k(j[1]),qH)),l=2;break;default:var
l=1}switch(l){case
1:cI(n[1]);dM(dH);throw[0,F,qB];case
2:break;default:var
o=qC}var
q=[0,e(0,n[1]),o];if(t){a3[1]=q;a4[1]=q}function
r(a){var
q=g[4],r=dO(function(a,b){return 0===b?a:h(qj,a)},qr,q),s=h(r,e(0,eM(p))),j=cW(e1(qD,gy,438));dI(j,s);cX(j);e2(j);fb(qE);var
m=dJ(qF),c=sF(m),n=A(c),o=0;if(0<=0)if(0<=c)if((n.getLen()-c|0)<o)var
f=0;else{var
k=o,b=c;for(;;){if(0<b){var
l=c1(m,n,k,b);if(0===l)throw[0,bn];var
k=k+l|0,b=b-l|0;continue}var
f=1;break}}else
var
f=0;else
var
f=0;if(!f)E(gA);dL(m);i(Z(d,723535973,3),d,n);fb(qG);return 0}function
s(a){var
b=g[4],c=dO(function(a,b){return 0===b?a:h(qb,a)},pW,b);return i(Z(d,56985577,4),d,h(c,f(0,eM(p))))}switch(u){case
1:s(0);break;case
2:r(0);s(0);break;default:r(0)}i(Z(d,345714255,5),d,0);return[0,d,g]}var
cQ=[];e$(cQ,[0,cQ,cQ]);var
cR=d,eN=null,qN=1,qO=1,qP=1,qQ=undefined;function
eO(a,b){return a==eN?j(b,0):a}var
eP=true,eQ=Array,qR=false;ea(function(a){return a
instanceof
eQ?0:[0,new
aw(a.toString())]});function
J(a,b){a.appendChild(b);return 0}function
cS(d){return sC(function(a){if(a){var
e=j(d,a);if(!(e|0))a.preventDefault();return e}var
c=event,b=j(d,c);if(!(b|0))c.returnValue=b;return b})}var
K=cR.document,qS="2d";function
bC(a,b){return a?j(b,a[1]):0}function
bD(a,b){return a.createElement(b.toString())}function
bE(a,b){return bD(a,b)}var
eR=[0,fY];function
eS(a,b,c,d){for(;;){if(0===a)if(0===b)return bD(c,d);var
h=eR[1];if(fY===h){try{var
j=K.createElement('<input name="x">'),k=j.tagName.toLowerCase()===fk?1:0,m=k?j.name===dl?1:0:k,i=m}catch(f){var
i=0}var
l=i?fL:-1003883683;eR[1]=l;continue}if(fL<=h){var
e=new
eQ();e.push("<",d.toString());bC(a,function(a){e.push(' type="',fc(a),bZ);return 0});bC(b,function(a){e.push(' name="',fc(a),bZ);return 0});e.push(">");return c.createElement(e.join(g))}var
f=bD(c,d);bC(a,function(a){return f.type=a});bC(b,function(a){return f.name=a});return f}}function
eT(a){return bE(a,qW)}var
q0=[0,qZ];cR.HTMLElement===qQ;function
eX(a){return eT(K)}function
eY(a){function
c(a){throw[0,F,q2]}var
b=eO(K.getElementById(ge),c);return j(d9(function(a){J(b,eX(0));J(b,K.createTextNode(a.toString()));return J(b,eX(0))}),a)}function
q3(a){var
k=[0,[4,a]];return function(a,b,c,d){var
h=a[2],i=a[1],l=c[2];if(0===l[0]){var
g=l[1],e=[0,0],f=tj(k.length-1),m=g[7][1]<i[1]?1:0;if(m)var
n=m;else{var
s=g[7][2]<i[2]?1:0,n=s||(g[7][3]<i[3]?1:0)}if(n)throw[0,k$];var
o=g[8][1]<h[1]?1:0;if(o)var
p=o;else{var
r=g[8][2]<h[2]?1:0,p=r||(g[8][3]<h[3]?1:0)}if(p)throw[0,lb];b$(function(a,b){function
h(a){if(bA)try{eH(a,0,c);P(c,0,0)}catch(f){if(f[1]===ar)throw[0,ar];throw f}return 11===b[0]?tm(e,f,cD(b[1],du,c[1][8]),a):ty(e,f,cD(a,du,c[1][8]),a,c)}switch(b[0]){case
6:case
7:case
8:case
9:case
10:switch(b[0]){case
6:var
d=tx(e,f,b[1]);break;case
7:var
d=tw(e,f,b[1]);break;case
8:var
d=tv(e,f,b[1]);break;case
9:var
d=tu(e,f,b[1]);break;default:var
d=S(k8)}var
g=d;break;case
11:var
g=h(b[1]);break;default:var
g=h(b[1])}return g},k);var
q=tt(e,d,h,i,f,c[1],b)}else{var
j=[0,0];b$(function(a,b){switch(b[0]){case
6:case
7:case
8:case
9:case
10:switch(b[0]){case
6:var
e=tW(j,d,b[1],c[1]);break;case
7:var
e=tX(j,d,b[1],c[1]);break;case
8:var
e=tU(j,d,b[1],c[1]);break;case
9:var
e=tV(j,d,b[1],c[1]);break;default:var
e=S(k9)}var
g=e;break;default:var
f=b[1];if(bA){if(c0(aY(f),[0,c]))eH(f,0,c);P(c,0,0)}var
h=c[1],i=H(0),g=tY(j,d,a,cD(f,-701974253,c[1][8]-i|0),h)}return g},k);var
q=tT(d,h,i,c[1],b)}return q}}if(cT===0)var
c=eg([0]);else{var
aW=eg(aN(hA,cT));b$(function(a,b){var
c=(a*2|0)+2|0;aW[3]=o(aq[4],b,c,aW[3]);aW[4]=o(aj[4],c,1,aW[4]);return 0},cT);var
c=aW}var
cp=aN(function(a){return aT(c,a)},eW),eo=cH[2],q4=cp[1],q5=cp[2],q6=cp[3],hU=cH[4],ei=cq(eU),ej=cq(eW),ek=cq(eV),q7=1,cr=ca(function(a){return aT(c,a)},ej),hD=ca(function(a){return aT(c,a)},ek);c[5]=[0,[0,c[3],c[4],c[6],c[7],cr,ei],c[5]];var
hE=$[1],hF=c[7];function
hG(a,b,c){return cd(a,ei)?o($[4],a,b,c):c}c[7]=o($[11],hG,hF,hE);var
aU=[0,aq[1]],aV=[0,aj[1]];dR(function(a,b){aU[1]=o(aq[4],a,b,aU[1]);var
e=aV[1];try{var
f=i(aj[22],b,c[4]),d=f}catch(f){if(f[1]!==t)throw f;var
d=1}aV[1]=o(aj[4],b,d,e);return 0},ek,hD);dR(function(a,b){aU[1]=o(aq[4],a,b,aU[1]);aV[1]=o(aj[4],b,0,aV[1]);return 0},ej,cr);c[3]=aU[1];c[4]=aV[1];var
hH=0,hI=c[6];c[6]=cc(function(a,b){return cd(a[1],cr)?b:[0,a,b]},hI,hH);var
hV=q7?i(eo,c,hU):j(eo,c),el=c[5],ay=el?el[1]:S(gE),em=c[5],hJ=ay[6],hK=ay[5],hL=ay[4],hM=ay[3],hN=ay[2],hO=ay[1],hP=em?em[2]:S(gF);c[5]=hP;var
cb=hL,bo=hJ;for(;;){if(bo){var
dQ=bo[1],gG=bo[2],hQ=i($[22],dQ,c[7]),cb=o($[4],dQ,hQ,cb),bo=gG;continue}c[7]=cb;c[3]=hO;c[4]=hN;var
hR=c[6];c[6]=cc(function(a,b){return cd(a[1],hK)?b:[0,a,b]},hR,hM);var
hW=0,hX=cs(eV),hY=[0,aN(function(a){var
e=aT(c,a);try{var
b=c[6];for(;;){if(!b)throw[0,t];var
d=b[1],f=b[2],h=d[2];if(0!==aF(d[1],e)){var
b=f;continue}var
g=h;break}}catch(f){if(f[1]!==t)throw f;var
g=s(c[2],e)}return g},hX),hW],hZ=cs(eU),q8=r8([0,[0,hV],[0,aN(function(a){try{var
b=i($[22],a,c[7])}catch(f){if(f[1]===t)throw[0,F,hT];throw f}return b},hZ),hY]])[1],q9=function(a,b){if(1===b.length-1){var
c=b[0+1];if(4===c[0])return c[1]}return S(q_)};eq(c,[0,q5,0,q3,q6,function(a,b){return[0,[4,b]]},q4,q9]);var
q$=function(a,b){var
e=ep(b,c);o(q8,e,rb,ra);if(!b){var
f=c[8];if(0!==f){var
d=f;for(;;){if(d){var
g=d[2];j(d[1],e);var
d=g;continue}break}}}return e};eh[1]=(eh[1]+c[1]|0)-1|0;c[8]=dP(c[8]);cn(c,3+at(s(c[2],1)*16|0,ax)|0);var
h0=0,h1=function(a){var
b=a;return q$(h0,b)},re=v(4),rf=am(2),rg=as(v(3),rf),rh=cL(aD(v(0),rg),re),ri=v(4),rj=am(1),rk=as(v(3),rj),rl=a5(cL(aD(v(0),rk),ri),rh),rm=v(4),rn=v(3),ro=a5(cL(aD(v(0),rn),rm),rl),rp=am(3),rq=am(2),rr=as(v(3),rq),rs=aD(v(0),rr),rt=am(1),ru=as(v(3),rt),rv=aD(v(0),ru),rw=v(3),qu=[8,as(as(aD(v(0),rw),rv),rs),rp],rx=a5(cO(v(4),qu),ro),ry=am(4),rz=cJ(v(2),ry),rA=a5(cO(v(3),rz),rx),rB=am(p),rC=cJ(am(p),rB),qv=[39,[45,v(2),rC],1,rA],rF=cM(rE,rD),rI=cJ(cM(rH,rG),rF),rL=as(cM(rK,rJ),rI),qw=[26,a5(cO(v(2),rL),qv)],rM=cN(cP(cK(4)),qw),rN=cN(cP(cK(3)),rM),rc=[0,0],rd=[0,[13,5],eB],qt=[0,[1,[24,[23,qx,0],0]],cN(cP(cK(2)),rN)],rO=[0,function(a){var
d=qN+(qP*qO|0)|0;if((p*p|0)<d)return 0;var
b=d*4|0,e=ab(a,b+2|0),f=ab(a,b+1|0),c=((ab(a,b)+f|0)+e|0)/3|0;aa(a,b,c);aa(a,b+1|0,c);return aa(a,b+2|0,c)},qt,rd,rc],cU=[0,h1(0),rO],bF=function(a){return eT(K)};cR.onload=cS(function(a){function
d(a){throw[0,F,rT]}var
c=eO(K.getElementById(ge),d);J(c,bF(0));var
r=eS(0,0,K,qU),b=bD(K,qY);J(b,K.createTextNode("Choose a computing device : "));J(c,b);J(c,r);J(c,bF(0));var
e=bE(K,q1);if(1-(e.getContext==eN?1:0)){e.width=p;e.height=p;var
u=bE(K,qX);u.src="lena.png";var
v=e.getContext(qS);u.onload=cS(function(a){v.drawImage(u,0,0);J(c,bF(0));J(c,e);var
M=eZ?eZ[1]:2;switch(M){case
1:fd(0);aB[1]=fe(0);break;case
2:ff(0);aA[1]=fg(0);fd(0);aB[1]=fe(0);break;default:ff(0);aA[1]=fg(0)}ez[1]=aA[1]+aB[1]|0;var
z=aA[1]-1|0,y=0,N=0;if(z<0)var
A=y;else{var
g=N,D=y;for(;;){var
E=b_(D,[0,tC(g),0]),O=g+1|0;if(z!==g){var
g=O,D=E;continue}var
A=E;break}}var
o=0,d=0,b=A;for(;;){if(o<aB[1]){if(tS(d)){var
C=d+1|0,B=b_(b,[0,tE(d,d+aA[1]|0),0])}else{var
C=d,B=b}var
o=o+1|0,d=C,b=B;continue}var
n=0,m=b;for(;;){if(m){var
n=n+1|0,m=m[2];continue}ez[1]=n;aB[1]=d;if(b){var
k=0,h=b,H=b[2],I=b[1];for(;;){if(h){var
k=k+1|0,h=h[2];continue}var
x=w(k,I),l=1,f=H;for(;;){if(f){var
L=f[2];x[l+1]=f[1];var
l=l+1|0,f=L;continue}var
q=x;break}break}}else
var
q=[0];var
F=v.getImageData(0,0,p,p),G=F.data;J(c,bF(0));dN(function(a){var
b=bE(K,qT);J(b,K.createTextNode(a[1][1].toString()));return J(r,b)},q);var
P=function(a){var
g=s(q,r.selectedIndex+0|0),w=g[1][1];j(eY(rQ),w);var
c=aC(eB,0,(p*p|0)*4|0);eb(hv,e3(0));var
l=T(c)-1|0,x=0;if(!(l<0)){var
e=x;for(;;){aa(c,e,G[e]);var
D=e+1|0;if(l!==e){var
e=D;continue}break}}var
m=g[2];if(0===m[0])var
h=b1;else{var
C=0===m[1][2]?1:b1,h=C}a6(0,rR,cU);var
t=ey(0),f=cU[2],b=cU[1],y=0,z=[0,[0,h,1,1],[0,at(((p*p|0)+h|0)-1|0,h),1,1]],n=0,k=0?n[1]:n;if(0===g[2][0]){if(k)a6(0,qJ,[0,b,f]);else
if(!i(Z(b,-723625231,7),b,0))a6(0,qK,[0,b,f])}else
if(k)a6(0,qL,[0,b,f]);else
if(!i(Z(b,649483637,8),b,0))a6(0,qM,[0,b,f]);(function(a,b,c,d,e,f){return a.length==5?a(b,c,d,e,f):ag(a,[b,c,d,e,f])}(Z(b,5695307,6),b,c,z,y,g));var
u=ey(0)-t;i(eY(rP),rS,u);var
o=T(c)-1|0,A=0;if(!(o<0)){var
d=A;for(;;){G[d]=ab(c,d);var
B=d+1|0;if(o!==d){var
d=B;continue}break}}v.putImageData(F,0,0);return eP},t=eS([0,"button"],0,K,qV);t.value="Go";t.onclick=cS(P);J(c,t);return eP}}});return qR}throw[0,q0]});dK(0);return}}(this));
