// This program was compiled from OCaml by js_of_ocaml 1.99dev
(function(ca){"use strict";var
dc="set_cuda_sources",du=123,bQ=";",fH=108,gl="section1",db="reload_sources",bU="Map.bal",fX=",",b0='"',Y=16777215,da="get_cuda_sources",fW=0.07,b8=" / ",fG="double spoc_var",dk="args_to_list",bZ=" * ",ac="(",ft="float spoc_var",dj=65599,b7="if (",bY="return",fV=" ;\n",dt="exec",bh=115,bf=";}\n",fF=".ptx",d=512,ds=120,c$="..",fU=-512,J="]",dr=117,bT="; ",dq="compile",gk=" (",U="0",di="list_to_args",bS=248,fT=126,gj="fd ",c_="get_binaries",fE=" == ",aJ="(float)",dh="Kirc_Cuda.ml",b6=" + ",fS=") ",dp="x",fD=-97,fs="g",bd=1073741823,gi="parse concat",at=105,dg="get_opencl_sources",gh=511,be=110,gg=-88,aa=" = ",df="set_opencl_sources",fC=0.21,K="[",bX="'",fr="Unix",bP="int_of_string",gf="(double) ",fR=982028505,bc="){\n",bg="e",ge="#define __FLOAT64_EXTENSION__ \n",as="-",aI=-48,bW="(double) spoc_var",fq="++){\n",fB="__shared__ float spoc_var",gc="Image_filter_js.ml",gd="opencl_sources",fA=".cl",dn="reset_binaries",bO="\n",gb=101,dx=748841679,b5="index out of bounds",fp="spoc_init_opencl_device_vec",c9=125,bV=" - ",ga=";}",p=255,f$="binaries",b4="}",f_=" < ",fo="__shared__ long spoc_var",aH=250,f9=" >= ",fn="input",fQ=246,de=102,fP="Unix.Unix_error",g="",fm=" || ",aG=100,dm="Kirc_OpenCL.ml",f8="#ifndef __FLOAT64_EXTENSION__ \n",fO="__shared__ int spoc_var",dw=103,bN=", ",fN="./",fz=1e3,fl="for (int ",f7="file_file",f6="spoc_var",ad=".",fy="else{\n",bR="+",fM="(int)",dv="run",b3=65535,dl="#endif\n",aF=";\n",V="f",f5=785140586,f4="__shared__ double spoc_var",fx=-32,dd=111,fL=" > ",z=" ",f3="int spoc_var",ab=")",fK="cuda_sources",b2=256,fw="nan",c8=116,f0="../",f1="kernel_name",f2=65520,fZ="%.12g",fk=" && ",fJ=0.71,fv="/",fI="while (",c7="compile_and_run",b1=114,fY="* spoc_var",bM=" <= ",m="number",fu=" % ",tT=1;function
gv(a,b){throw[0,a,b]}function
dH(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=ca.console;b&&b.error&&b.error(a)}var
o=[0];function
bk(a,b){if(!a)return g;if(a&1)return bk(a-1,b)+b;var
c=bk(a>>1,b);return c+c}function
B(a){if(a!=null){this.bytes=this.fullBytes=a;this.last=this.len=a.length}}function
gy(){gv(o[4],new
B(b5))}B.prototype={string:null,bytes:null,fullBytes:null,array:null,len:null,last:0,toJsString:function(){var
a=this.getFullBytes();try{return this.string=decodeURIComponent(escape(a))}catch(f){dH('MlString.toJsString: wrong encoding for \"%s\" ',a);return a}},toBytes:function(){if(this.string!=null)try{var
a=unescape(encodeURIComponent(this.string))}catch(f){dH('MlString.toBytes: wrong encoding for \"%s\" ',this.string);var
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
b=this.bytes;if(b==null)b=this.toBytes();return a<this.last?b.charCodeAt(a):0},safeGet:function(a){if(this.len==null)this.toBytes();if(a<0||a>=this.len)gy();return this.get(a)},set:function(a,b){var
c=this.array;if(!c){if(this.last==a){this.bytes+=String.fromCharCode(b&p);this.last++;return 0}c=this.toArray()}else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;c[a]=b&p;return 0},safeSet:function(a,b){if(this.len==null)this.toBytes();if(a<0||a>=this.len)gy();this.set(a,b)},fill:function(a,b,c){if(a>=this.last&&this.last&&c==0)return;var
d=this.array;if(!d)d=this.toArray();else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;var
f=a+b;for(var
e=a;e<f;e++)d[e]=c},compare:function(a){if(this.string!=null&&a.string!=null){if(this.string<a.string)return-1;if(this.string>a.string)return 1;return 0}var
b=this.getFullBytes(),c=a.getFullBytes();if(b<c)return-1;if(b>c)return 1;return 0},equal:function(a){if(this.string!=null&&a.string!=null)return this.string==a.string;return this.getFullBytes()==a.getFullBytes()},lessThan:function(a){if(this.string!=null&&a.string!=null)return this.string<a.string;return this.getFullBytes()<a.getFullBytes()},lessEqual:function(a){if(this.string!=null&&a.string!=null)return this.string<=a.string;return this.getFullBytes()<=a.getFullBytes()}};function
au(a){this.string=a}au.prototype=new
B();function
so(a,b,c,d,e){if(d<=b)for(var
f=1;f<=e;f++)c[d+f]=a[b+f];else
for(var
f=e;f>=1;f--)c[d+f]=a[b+f]}function
sp(a){var
c=[0];while(a!==0){var
d=a[1];for(var
b=1;b<d.length;b++)c.push(d[b]);a=a[2]}return c}function
dG(a,b){gv(a,new
au(b))}function
ak(a){dG(o[4],a)}function
aK(){ak(b5)}function
sq(a,b){if(b<0||b>=a.length-1)aK();return a[b+1]}function
sr(a,b,c){if(b<0||b>=a.length-1)aK();a[b+1]=c;return 0}var
dz;function
ss(a,b,c){if(c.length!=2)ak("Bigarray.create: bad number of dimensions");if(b!=0)ak("Bigarray.create: unsupported layout");if(c[1]<0)ak("Bigarray.create: negative dimension");if(!dz){var
d=ca;dz=[d.Float32Array,d.Float64Array,d.Int8Array,d.Uint8Array,d.Int16Array,d.Uint16Array,d.Int32Array,null,d.Int32Array,d.Int32Array,null,null,d.Uint8Array]}var
e=dz[a];if(!e)ak("Bigarray.create: unsupported kind");return new
e(c[1])}function
st(a,b){if(b<0||b>=a.length)aK();return a[b]}function
su(a,b,c){if(b<0||b>=a.length)aK();a[b]=c;return 0}function
dA(a,b,c,d,e){if(e===0)return;if(d===c.last&&c.bytes!=null){var
f=a.bytes;if(f==null)f=a.toBytes();if(b>0||a.last>e)f=f.slice(b,b+e);c.bytes+=f;c.last+=f.length;return}var
g=c.array;if(!g)g=c.toArray();else
c.bytes=c.string=null;a.blitToArray(b,g,d,e)}function
ae(c,b){if(c.fun)return ae(c.fun,b);var
a=c.length,d=a-b.length;if(d==0)return c.apply(null,b);else
if(d<0)return ae(c.apply(null,b.slice(0,a)),b.slice(a));else
return function(a){return ae(c,b.concat([a]))}}function
sv(a){if(isFinite(a)){if(Math.abs(a)>=2.22507385850720138e-308)return 0;if(a!=0)return 1;return 2}return isNaN(a)?4:3}function
sH(a,b){var
c=a[3]<<16,d=b[3]<<16;if(c>d)return 1;if(c<d)return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
gs(a,b){if(a<b)return-1;if(a==b)return 0;return 1}function
dB(a,b,c){var
e=[];for(;;){if(!(c&&a===b))if(a
instanceof
B)if(b
instanceof
B){if(a!==b){var
d=a.compare(b);if(d!=0)return d}}else
return 1;else
if(a
instanceof
Array&&a[0]===(a[0]|0)){var
g=a[0];if(g===aH){a=a[1];continue}else
if(b
instanceof
Array&&b[0]===(b[0]|0)){var
h=b[0];if(h===aH){b=b[1];continue}else
if(g!=h)return g<h?-1:1;else
switch(g){case
bS:{var
d=gs(a[2],b[2]);if(d!=0)return d;break}case
251:ak("equal: abstract value");case
p:{var
d=sH(a,b);if(d!=0)return d;break}default:if(a.length!=b.length)return a.length<b.length?-1:1;if(a.length>1)e.push(a,b,1)}}else
return 1}else
if(b
instanceof
B||b
instanceof
Array&&b[0]===(b[0]|0))return-1;else{if(a<b)return-1;if(a>b)return 1;if(c&&a!=b){if(a==a)return 1;if(b==b)return-1}}if(e.length==0)return 0;var
f=e.pop();b=e.pop();a=e.pop();if(f+1<a.length)e.push(a,b,f+1);a=a[f];b=b[f]}}function
gn(a,b){return dB(a,b,true)}function
gm(a){this.bytes=g;this.len=a}gm.prototype=new
B();function
go(a){if(a<0)ak("String.create");return new
gm(a)}function
dF(a){throw[0,a]}function
gw(){dF(o[6])}function
sw(a,b){if(b==0)gw();return a/b|0}function
sx(a,b){return+(dB(a,b,false)==0)}function
sy(a,b,c,d){a.fill(b,c,d)}function
dE(a){a=a.toString();var
e=a.length;if(e>31)ak("format_int: format too long");var
b={justify:bR,signstyle:as,filler:z,alternate:false,base:0,signedconv:false,width:0,uppercase:false,sign:1,prec:-1,conv:V};for(var
d=0;d<e;d++){var
c=a.charAt(d);switch(c){case
as:b.justify=as;break;case
bR:case
z:b.signstyle=c;break;case
U:b.filler=U;break;case"#":b.alternate=true;break;case"1":case"2":case"3":case"4":case"5":case"6":case"7":case"8":case"9":b.width=0;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.width=b.width*10+c;d++}d--;break;case
ad:b.prec=0;d++;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.prec=b.prec*10+c;d++}d--;case"d":case"i":b.signedconv=true;case"u":b.base=10;break;case
dp:b.base=16;break;case"X":b.base=16;b.uppercase=true;break;case"o":b.base=8;break;case
bg:case
V:case
fs:b.signedconv=true;b.conv=c;break;case"E":case"F":case"G":b.signedconv=true;b.uppercase=true;b.conv=c.toLowerCase();break}}return b}function
dC(a,b){if(a.uppercase)b=b.toUpperCase();var
e=b.length;if(a.signedconv&&(a.sign<0||a.signstyle!=as))e++;if(a.alternate){if(a.base==8)e+=1;if(a.base==16)e+=2}var
c=g;if(a.justify==bR&&a.filler==z)for(var
d=e;d<a.width;d++)c+=z;if(a.signedconv)if(a.sign<0)c+=as;else
if(a.signstyle!=as)c+=a.signstyle;if(a.alternate&&a.base==8)c+=U;if(a.alternate&&a.base==16)c+="0x";if(a.justify==bR&&a.filler==U)for(var
d=e;d<a.width;d++)c+=U;c+=b;if(a.justify==as)for(var
d=e;d<a.width;d++)c+=z;return new
au(c)}function
sz(a,b){var
c,f=dE(a),e=f.prec<0?6:f.prec;if(b<0){f.sign=-1;b=-b}if(isNaN(b)){c=fw;f.filler=z}else
if(!isFinite(b)){c="inf";f.filler=z}else
switch(f.conv){case
bg:var
c=b.toExponential(e),d=c.length;if(c.charAt(d-3)==bg)c=c.slice(0,d-1)+U+c.slice(d-1);break;case
V:c=b.toFixed(e);break;case
fs:e=e?e:1;c=b.toExponential(e-1);var
i=c.indexOf(bg),h=+c.slice(i+1);if(h<-4||b.toFixed(0).length>e){var
d=i-1;while(c.charAt(d)==U)d--;if(c.charAt(d)==ad)d--;c=c.slice(0,d+1)+c.slice(i);d=c.length;if(c.charAt(d-3)==bg)c=c.slice(0,d-1)+U+c.slice(d-1);break}else{var
g=e;if(h<0){g-=h+1;c=b.toFixed(g)}else
while(c=b.toFixed(g),c.length>e+1)g--;if(g){var
d=c.length-1;while(c.charAt(d)==U)d--;if(c.charAt(d)==ad)d--;c=c.slice(0,d+1)}}break}return dC(f,c)}function
sA(a,b){if(a.toString()=="%d")return new
au(g+b);var
c=dE(a);if(b<0)if(c.signedconv){c.sign=-1;b=-b}else
b>>>=0;var
d=b.toString(c.base);if(c.prec>=0){c.filler=z;var
e=c.prec-d.length;if(e>0)d=bk(e,U)+d}return dC(c,d)}function
sB(){return 0}function
sC(){return 0}var
b$=[];function
sD(a,b,c){var
e=a[1],i=b$[c];if(i===null)for(var
h=b$.length;h<c;h++)b$[h]=0;else
if(e[i]===b)return e[i-1];var
d=3,g=e[1]*2+1,f;while(d<g){f=d+g>>1|1;if(b<e[f+1])g=f-2;else
d=f}b$[c]=d+1;return b==e[d+1]?e[d]:0}function
sE(a,b){return+(gn(a,b,false)>=0)}function
gp(a){if(!isFinite(a)){if(isNaN(a))return[p,1,0,f2];return a>0?[p,0,0,32752]:[p,0,0,f2]}var
f=a>=0?0:32768;if(f)a=-a;var
b=Math.floor(Math.LOG2E*Math.log(a))+1023;if(b<=0){b=0;a/=Math.pow(2,-1026)}else{a/=Math.pow(2,b-1027);if(a<16){a*=2;b-=1}if(b==0)a/=2}var
d=Math.pow(2,24),c=a|0;a=(a-c)*d;var
e=a|0;a=(a-e)*d;var
g=a|0;c=c&15|f|b<<4;return[p,g,e,c]}function
bj(a,b){return((a>>16)*b<<16)+(a&b3)*b|0}var
sF=function(){var
q=b2;function
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
k,l,m,i,h,f,e,j,o;i=b;if(i<0||i>q)i=q;h=a;f=c;k=[d];l=0;m=1;while(l<m&&h>0){e=k[l++];if(e
instanceof
Array&&e[0]===(e[0]|0))switch(e[0]){case
bS:f=g(f,e[2]);h--;break;case
aH:k[--l]=e[1];break;case
p:f=v(f,e);h--;break;default:var
s=e.length-1<<10|e[0];f=g(f,s);for(j=1,o=e.length;j<o;j++){if(m>=i)break;k[m++]=e[j]}break}else
if(e
instanceof
B){var
n=e.array;if(n)f=w(f,n);else{var
r=e.getFullBytes();f=x(f,r)}h--;break}else
if(e===(e|0)){f=g(f,e+e+1);h--}else
if(e===+e){f=u(f,gp(e));h--;break}}f=t(f);return f&bd}}();function
sP(a){return[a[3]>>8,a[3]&p,a[2]>>16,a[2]>>8&p,a[2]&p,a[1]>>16,a[1]>>8&p,a[1]&p]}function
sG(e,b,c){var
d=0;function
f(a){b--;if(e<0||b<0)return;if(a
instanceof
Array&&a[0]===(a[0]|0))switch(a[0]){case
bS:e--;d=d*dj+a[2]|0;break;case
aH:b++;f(a);break;case
p:e--;d=d*dj+a[1]+(a[2]<<24)|0;break;default:e--;d=d*19+a[0]|0;for(var
c=a.length-1;c>0;c--)f(a[c])}else
if(a
instanceof
B){e--;var
g=a.array,h=a.getLen();if(g)for(var
c=0;c<h;c++)d=d*19+g[c]|0;else{var
i=a.getFullBytes();for(var
c=0;c<h;c++)d=d*19+i.charCodeAt(c)|0}}else
if(a===(a|0)){e--;d=d*dj+a|0}else
if(a===+a){e--;var
j=sP(gp(a));for(var
c=7;c>=0;c--)d=d*19+j[c]|0}}f(c);return d&bd}function
sK(a){return(a[3]|a[2]|a[1])==0}function
sN(a){return[p,a&Y,a>>24&Y,a>>31&b3]}function
sO(a,b){var
c=a[1]-b[1],d=a[2]-b[2]+(c>>24),e=a[3]-b[3]+(d>>24);return[p,c&Y,d&Y,e&b3]}function
gr(a,b){if(a[3]>b[3])return 1;if(a[3]<b[3])return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
gq(a){a[3]=a[3]<<1|a[2]>>23;a[2]=(a[2]<<1|a[1]>>23)&Y;a[1]=a[1]<<1&Y}function
sL(a){a[1]=(a[1]>>>1|a[2]<<23)&Y;a[2]=(a[2]>>>1|a[3]<<23)&Y;a[3]=a[3]>>>1}function
sR(a,b){var
e=0,d=a.slice(),c=b.slice(),f=[p,0,0,0];while(gr(d,c)>0){e++;gq(c)}while(e>=0){e--;gq(f);if(gr(d,c)>=0){f[1]++;d=sO(d,c)}sL(c)}return[0,f,d]}function
sQ(a){return a[1]|a[2]<<24}function
sJ(a){return a[3]<<16<0}function
sM(a){var
b=-a[1],c=-a[2]+(b>>24),d=-a[3]+(c>>24);return[p,b&Y,c&Y,d&b3]}function
sI(a,b){var
c=dE(a);if(c.signedconv&&sJ(b)){c.sign=-1;b=sM(b)}var
d=g,i=sN(c.base),h="0123456789abcdef";do{var
f=sR(b,i);b=f[1];d=h.charAt(sQ(f[2]))+d}while(!sK(b));if(c.prec>=0){c.filler=z;var
e=c.prec-d.length;if(e>0)d=bk(e,U)+d}return dC(c,d)}function
tb(a){var
b=0,c=10,d=a.get(0)==45?(b++,-1):1;if(a.get(b)==48)switch(a.get(b+1)){case
ds:case
88:c=16;b+=2;break;case
dd:case
79:c=8;b+=2;break;case
98:case
66:c=2;b+=2;break}return[b,d,c]}function
gu(a){if(a>=48&&a<=57)return a-48;if(a>=65&&a<=90)return a-55;if(a>=97&&a<=122)return a-87;return-1}function
b9(a){dG(o[3],a)}function
sS(a){var
g=tb(a),e=g[0],h=g[1],f=g[2],i=-1>>>0,d=a.get(e),c=gu(d);if(c<0||c>=f)b9(bP);var
b=c;for(;;){e++;d=a.get(e);if(d==95)continue;c=gu(d);if(c<0||c>=f)break;b=f*b+c;if(b>i)b9(bP)}if(e!=a.getLen())b9(bP);b=h*b;if((b|0)!=b)b9(bP);return b}function
sT(a){return+(a>31&&a<127)}var
b_={amp:/&/g,lt:/</g,quot:/\"/g,all:/[&<\"]/};function
sU(a){if(!b_.all.test(a))return a;return a.replace(b_.amp,"&amp;").replace(b_.lt,"&lt;").replace(b_.quot,"&quot;")}function
sV(a){var
c=Array.prototype.slice;return function(){var
b=arguments.length>0?c.call(arguments):[undefined];return ae(a,b)}}function
sW(a,b){var
d=[0];for(var
c=1;c<=a;c++)d[c]=b;return d}function
dy(a){var
b=a.length;this.array=a;this.len=this.last=b}dy.prototype=new
B();var
sX=function(){function
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
n=0;n<4;n++)o[g*4+n]=l[g]>>8*n&p;return o}return function(a,b,c){var
h=[];if(a.array){var
f=a.array;for(var
d=0;d<c;d+=4){var
e=d+b;h[d>>2]=f[e]|f[e+1]<<8|f[e+2]<<16|f[e+3]<<24}for(;d<c;d++)h[d>>2]|=f[d+b]<<8*(d&3)}else{var
g=a.getFullBytes();for(var
d=0;d<c;d+=4){var
e=d+b;h[d>>2]=g.charCodeAt(e)|g.charCodeAt(e+1)<<8|g.charCodeAt(e+2)<<16|g.charCodeAt(e+3)<<24}for(;d<c;d++)h[d>>2]|=g.charCodeAt(d+b)<<8*(d&3)}return new
dy(n(h,c))}}();function
sY(a){return a.data.array.length}function
al(a){dG(o[2],a)}function
dD(a){if(!a.opened)al("Cannot flush a closed channel");if(a.buffer==g)return 0;if(a.output){switch(a.output.length){case
2:a.output(a,a.buffer);break;default:a.output(a.buffer)}}a.buffer=g}var
bi=new
Array();function
sZ(a){dD(a);a.opened=false;delete
bi[a.fd];return 0}function
s0(a,b,c,d){var
e=a.data.array.length-a.data.offset;if(e<d)d=e;dA(new
dy(a.data.array),a.data.offset,b,c,d);a.data.offset+=d;return d}function
tc(){dF(o[5])}function
s1(a){if(a.data.offset>=a.data.array.length)tc();if(a.data.offset<0||a.data.offset>a.data.array.length)aK();var
b=a.data.array[a.data.offset];a.data.offset++;return b}function
s2(a){var
b=a.data.offset,c=a.data.array.length;if(b>=c)return 0;while(true){if(b>=c)return-(b-a.data.offset);if(b<0||b>a.data.array.length)aK();if(a.data.array[b]==10)return b-a.data.offset+1;b++}}function
te(a,b){if(!o.files)o.files={};if(b
instanceof
B)var
c=b.getArray();else
if(b
instanceof
Array)var
c=b;else
var
c=new
B(b).getArray();o.files[a
instanceof
B?a.toString():a]=c}function
tl(a){return o.files&&o.files[a.toString()]?1:0}function
bl(a,b,c){if(o.fds===undefined)o.fds=new
Array();c=c?c:{};var
d={};d.array=b;d.offset=c.append?d.array.length:0;d.flags=c;o.fds[a]=d;o.fd_last_idx=a;return a}function
tp(a,b,c){var
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
e=a.toString();if(d.rdonly&&d.wronly)al(e+" : flags Open_rdonly and Open_wronly are not compatible");if(d.text&&d.binary)al(e+" : flags Open_text and Open_binary are not compatible");if(tl(a)){if(d.create&&d.excl)al(e+" : file already exists");var
f=o.fd_last_idx?o.fd_last_idx:0;if(d.truncate)o.files[e]=g;return bl(f+1,o.files[e],d)}else
if(d.create){var
f=o.fd_last_idx?o.fd_last_idx:0;te(e,[]);return bl(f+1,o.files[e],d)}else
al(e+": no such file or directory")}bl(0,[]);bl(1,[]);bl(2,[]);function
s3(a){var
b=o.fds[a];if(b.flags.wronly)al(gj+a+" is writeonly");return{data:b,fd:a,opened:true}}function
tw(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=ca.console;b&&b.log&&b.log(a)}function
th(a,b){var
e=new
B(b),d=e.getLen();for(var
c=0;c<d;c++)a.data.array[a.data.offset+c]=e.get(c);a.data.offset+=d;return 0}function
s4(a){var
b;switch(a){case
1:b=tw;break;case
2:b=dH;break;default:b=th}var
d=o.fds[a];if(d.flags.rdonly)al(gj+a+" is readonly");var
c={data:d,fd:a,opened:true,buffer:g,output:b};bi[c.fd]=c;return c}function
s5(){var
a=0;for(var
b
in
bi)if(bi[b].opened)a=[0,bi[b],a];return a}function
gt(a,b,c,d){if(!a.opened)al("Cannot output to a closed channel");var
f;if(c==0&&b.getLen()==d)f=b;else{f=go(d);dA(b,c,f,0,d)}var
e=f.toString(),g=e.lastIndexOf("\n");if(g<0)a.buffer+=e;else{a.buffer+=e.substr(0,g+1);dD(a);a.buffer+=e.substr(g+1)}}function
P(a){return new
B(a)}function
s6(a,b){var
c=P(String.fromCharCode(b));gt(a,c,0,1)}function
s7(a,b){if(b==0)gw();return a%b}function
s9(a,b){return+(dB(a,b,false)!=0)}function
s_(a,b){var
d=[a];for(var
c=1;c<=b;c++)d[c]=0;return d}function
s$(a,b){a[0]=b;return 0}function
ta(a){return a
instanceof
Array?a[0]:fz}function
tf(a,b){o[a+1]=b}var
s8={};function
tg(a,b){s8[a]=b;return 0}function
ti(a,b){return a.compare(b)}function
gx(a,b){var
c=a.fullBytes,d=b.fullBytes;if(c!=null&&d!=null)return c==d?1:0;return a.getFullBytes()==b.getFullBytes()?1:0}function
tj(a,b){return 1-gx(a,b)}function
tk(){return 32}function
tm(){var
a=new
au("a.out");return[0,a,[0,a]]}function
tn(){return[0,new
au(fr),32,0]}function
td(){dF(o[7])}function
to(){td()}function
tq(){var
a=new
Date()^4294967295*Math.random();return{valueOf:function(){return a},0:0,1:a,length:2}}function
tr(){console.log("caml_sys_system_command");return 0}function
ts(a){var
b=1;while(a&&a.joo_tramp){a=a.joo_tramp.apply(null,a.joo_args);b++}return a}function
tt(a,b){return{joo_tramp:a,joo_args:b}}function
tu(a,b){if(typeof
b==="function"){a.fun=b;return 0}if(b.fun){a.fun=b.fun;return 0}var
c=b.length;while(c--)a[c]=b[c];return 0}function
tv(){return 0}var
dI=0;function
tx(){if(window.webcl==undefined){alert("Unfortunately your system does not support WebCL. "+"Make sure that you have both the OpenCL driver "+"and the WebCL browser extension installed.");dI=1}else{console.log("INIT OPENCL");dI=0}return 0}function
ty(){console.log(" spoc_cuInit");return 0}function
tz(){console.log(" spoc_cuda_compile");return 0}function
tA(){console.log(" spoc_cuda_debug_compile");return 0}function
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
f=k[o],j=f.getDevices(),m=j.length;console.log("there "+g+z+m+z+a);if(g+m>=a)for(var
q
in
j){var
c=j[q];if(g==a){console.log("current ----------"+g);e[1]=P(c.getInfo(WebCL.DEVICE_NAME));console.log(e[1]);e[2]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_SIZE);e[3]=c.getInfo(WebCL.DEVICE_LOCAL_MEM_SIZE);e[4]=c.getInfo(WebCL.DEVICE_MAX_CLOCK_FREQUENCY);e[5]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE);e[6]=c.getInfo(WebCL.DEVICE_MAX_COMPUTE_UNITS);e[7]=c.getInfo(WebCL.DEVICE_ERROR_CORRECTION_SUPPORT);e[8]=b;var
i=new
Array(3);i[0]=webcl.createContext(c);i[1]=i[0].createCommandQueue();i[2]=i[0].createCommandQueue();e[9]=i;h[1]=P(f.getInfo(WebCL.PLATFORM_PROFILE));h[2]=P(f.getInfo(WebCL.PLATFORM_VERSION));h[3]=P(f.getInfo(WebCL.PLATFORM_NAME));h[4]=P(f.getInfo(WebCL.PLATFORM_VENDOR));h[5]=P(f.getInfo(WebCL.PLATFORM_EXTENSIONS));h[6]=m;var
l=c.getInfo(WebCL.DEVICE_TYPE),v=0;if(l&WebCL.DEVICE_TYPE_CPU)d[2]=0;if(l&WebCL.DEVICE_TYPE_GPU)d[2]=1;if(l&WebCL.DEVICE_TYPE_ACCELERATOR)d[2]=2;if(l&WebCL.DEVICE_TYPE_DEFAULT)d[2]=3;d[3]=P(c.getInfo(WebCL.DEVICE_PROFILE));d[4]=P(c.getInfo(WebCL.DEVICE_VERSION));d[5]=P(c.getInfo(WebCL.DEVICE_VENDOR));var
r=c.getInfo(WebCL.DEVICE_EXTENSIONS);d[6]=P(r);d[7]=c.getInfo(WebCL.DEVICE_VENDOR_ID);d[8]=c.getInfo(WebCL.DEVICE_MAX_WORK_ITEM_DIMENSIONS);d[9]=c.getInfo(WebCL.DEVICE_ADDRESS_BITS);d[10]=c.getInfo(WebCL.DEVICE_MAX_MEM_ALLOC_SIZE);d[11]=c.getInfo(WebCL.DEVICE_IMAGE_SUPPORT);d[12]=c.getInfo(WebCL.DEVICE_MAX_READ_IMAGE_ARGS);d[13]=c.getInfo(WebCL.DEVICE_MAX_WRITE_IMAGE_ARGS);d[14]=c.getInfo(WebCL.DEVICE_MAX_SAMPLERS);d[15]=c.getInfo(WebCL.DEVICE_MEM_BASE_ADDR_ALIGN);d[17]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHELINE_SIZE);d[18]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHE_SIZE);d[19]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_ARGS);d[20]=c.getInfo(WebCL.DEVICE_ENDIAN_LITTLE);d[21]=c.getInfo(WebCL.DEVICE_AVAILABLE);d[22]=c.getInfo(WebCL.DEVICE_COMPILER_AVAILABLE);d[23]=c.getInfo(WebCL.DEVICE_SINGLE_FP_CONFIG);d[24]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHE_TYPE);d[25]=c.getInfo(WebCL.DEVICE_QUEUE_PROPERTIES);d[26]=c.getInfo(WebCL.DEVICE_LOCAL_MEM_TYPE);d[28]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE);d[29]=c.getInfo(WebCL.DEVICE_EXECUTION_CAPABILITIES);d[31]=c.getInfo(WebCL.DEVICE_MAX_WORK_GROUP_SIZE);d[32]=c.getInfo(WebCL.DEVICE_IMAGE2D_MAX_HEIGHT);d[33]=c.getInfo(WebCL.DEVICE_IMAGE2D_MAX_WIDTH);d[34]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_DEPTH);d[35]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_HEIGHT);d[36]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_WIDTH);d[37]=c.getInfo(WebCL.DEVICE_MAX_PARAMETER_SIZE);d[38]=[0];var
n=c.getInfo(WebCL.DEVICE_MAX_WORK_ITEM_SIZES);d[38][1]=n[0];d[38][2]=n[1];d[38][3]=n[2];d[39]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);d[40]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);d[41]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_INT);d[42]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_LONG);d[43]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);d[45]=c.getInfo(WebCL.DEVICE_PROFILING_TIMER_RESOLUTION);d[46]=P(c.getInfo(WebCL.DRIVER_VERSION));g++;break}else
g++}else
g+=m}var
c=[0];d[1]=h;p[1]=d;c[1]=e;c[2]=p;return c}function
tF(){console.log(" spoc_getOpenCLDevicesCount");var
a=0,b=webcl.getPlatforms();for(var
d
in
b){var
e=b[d],c=e.getDevices();a+=c.length}return a}function
tG(){console.log(fp);return 0}function
tH(){console.log(fp);var
a=new
Array(3);a[0]=0;return a}function
dJ(a){if(a[1]instanceof
Float32Array||a[1].constructor.name=="Float32Array")return 4;if(a[1]instanceof
Int32Array||a[1].constructor.name=="Int32Array")return 4;{console.log("unimplemented vector type");console.log(a[1].constructor.name);return 4}}function
tI(a,b,c){console.log("spoc_opencl_alloc_vect");var
f=a[2],i=a[4],h=i[b+1],j=a[5],k=dJ(f),d=c[9],e=d[0],d=c[9],e=d[0],g=e.createBuffer(WebCL.MEM_READ_WRITE,j*k);h[2]=g;d[0]=e;c[9]=d;return 0}function
tJ(){console.log(" spoc_opencl_compile");return 0}function
tK(a,b,c,d){console.log("spoc_opencl_cpu_to_device");var
f=a[2],k=a[4],j=k[b+1],l=a[5],m=dJ(f),e=c[9],h=e[0],g=e[d+1],i=j[2];g.enqueueWriteBuffer(i,false,0,l*m,f[1]);e[d+1]=g;e[0]=h;c[9]=e;return 0}function
tL(a,b,c,d,e){console.log("spoc_opencl_device_to_cpu");var
g=a[2],l=a[4],k=l[b+1],n=a[5],o=dJ(g),f=c[9],i=f[0],h=f[e+1],j=k[2],m=g[1];h.enqueueReadBuffer(j,false,0,n*o,m);f[e+1]=h;f[0]=i;c[9]=f;return 0}function
tM(a,b){console.log("spoc_opencl_flush");var
c=a[9][b+1];c.flush();a[9][b+1]=c;return 0}function
tN(){console.log(" spoc_opencl_is_available");return!dI}function
tO(a,b,c,d,e){console.log("spoc_opencl_launch_grid");var
m=b[1],n=b[2],o=b[3],h=c[1],i=c[2],j=c[3],g=new
Array(3);g[0]=m*h;g[1]=n*i;g[2]=o*j;var
f=new
Array(3);f[0]=h;f[1]=i;f[2]=j;var
l=d[9],k=l[e+1];if(h==1&&i==1&&j==1)k.enqueueNDRangeKernel(a,f.length,null,g);else
k.enqueueNDRangeKernel(a,f.length,null,g,f);return 0}function
tP(a,b,c,d){console.log("spoc_opencl_load_param_int");b.setArg(a[1],new
Uint32Array([c]));a[1]=a[1]+1;return 0}function
tQ(a,b,c,d,e){console.log("spoc_opencl_load_param_vec");var
f=d[2];b.setArg(a[1],f);a[1]=a[1]+1;return 0}function
tR(){return new
Date().getTime()/fz}function
tS(){return 0}var
q=sq,l=sr,a8=dA,aD=gn,y=go,ar=sw,cY=sz,bH=sA,a9=sC,X=sD,c2=sT,ff=sU,u=sW,e6=sZ,c0=dD,c4=s0,e4=s3,cZ=s4,aE=s7,v=bj,b=P,c3=s9,e9=s_,aC=tf,c1=tg,e8=ti,bK=gx,w=tj,bI=to,e5=tp,e7=tq,fe=tr,T=ts,A=tt,fd=tv,fg=tx,fi=ty,fj=tD,fh=tF,fa=tG,e$=tH,fb=tI,e_=tM,bL=tS;function
j(a,b){return a.length==1?a(b):ae(a,[b])}function
i(a,b,c){return a.length==2?a(b,c):ae(a,[b,c])}function
n(a,b,c,d){return a.length==3?a(b,c,d):ae(a,[b,c,d])}function
fc(a,b,c,d,e,f,g){return a.length==6?a(b,c,d,e,f,g):ae(a,[b,c,d,e,f,g])}var
aL=[0,b("Failure")],bm=[0,b("Invalid_argument")],bn=[0,b("End_of_file")],r=[0,b("Not_found")],D=[0,b("Assert_failure")],cy=b(ad),cB=b(ad),cD=b(ad),eN=b(g),eM=[0,b(f7),b(f1),b(fK),b(gd),b(f$)],e3=[0,1],eY=[0,b(gd),b(f1),b(f7),b(fK),b(f$)],eZ=[0,b(dq),b(c7),b(c_),b(da),b(dg),b(db),b(dn),b(dv),b(dc),b(df)],e0=[0,b(di),b(dt),b(dk)],cW=[0,b(dt),b(c_),b(da),b(dk),b(di),b(c7),b(dv),b(df),b(dq),b(db),b(dn),b(dg),b(dc)];aC(6,r);aC(5,[0,b("Division_by_zero")]);aC(4,bn);aC(3,bm);aC(2,aL);aC(1,[0,b("Sys_error")]);var
gF=b("really_input"),gE=[0,0,[0,7,0]],gD=[0,1,[0,3,[0,4,[0,7,0]]]],gC=b(fZ),gB=b(ad),gz=b("true"),gA=b("false"),gG=b("Pervasives.do_at_exit"),gI=b("Array.blit"),gM=b("List.iter2"),gK=b("tl"),gJ=b("hd"),gQ=b("\\b"),gR=b("\\t"),gS=b("\\n"),gT=b("\\r"),gP=b("\\\\"),gO=b("\\'"),gN=b("Char.chr"),gW=b("String.contains_from"),gV=b("String.blit"),gU=b("String.sub"),g5=b("Map.remove_min_elt"),g6=[0,0,0,0],g7=[0,b("map.ml"),270,10],g8=[0,0,0],g1=b(bU),g2=b(bU),g3=b(bU),g4=b(bU),g9=b("CamlinternalLazy.Undefined"),ha=b("Buffer.add: cannot grow buffer"),hq=b(g),hr=b(g),hu=b(fZ),hv=b(b0),hw=b(b0),hs=b(bX),ht=b(bX),hp=b(fw),hn=b("neg_infinity"),ho=b("infinity"),hm=b(ad),hl=b("printf: bad positional specification (0)."),hk=b("%_"),hj=[0,b("printf.ml"),143,8],hh=b(bX),hi=b("Printf: premature end of format string '"),hd=b(bX),he=b(" in format string '"),hf=b(", at char number "),hg=b("Printf: bad conversion %"),hb=b("Sformat.index_of_int: negative argument "),hy=b(dp),hz=[0,987910699,495797812,364182224,414272206,318284740,990407751,383018966,270373319,840823159,24560019,536292337,512266505,189156120,730249596,143776328,51606627,140166561,366354223,1003410265,700563762,981890670,913149062,526082594,1021425055,784300257,667753350,630144451,949649812,48546892,415514493,258888527,511570777,89983870,283659902,308386020,242688715,482270760,865188196,1027664170,207196989,193777847,619708188,671350186,149669678,257044018,87658204,558145612,183450813,28133145,901332182,710253903,510646120,652377910,409934019,801085050],sk=b("OCAMLRUNPARAM"),si=b("CAMLRUNPARAM"),hB=b(g),hY=[0,b("camlinternalOO.ml"),287,50],hX=b(g),hD=b("CamlinternalOO.last_id"),ir=b(g),io=b(fN),im=b(".\\"),il=b(f0),ik=b("..\\"),ib=b(fN),ia=b(f0),h8=b(g),h7=b(g),h9=b(c$),h_=b(fv),sg=b("TMPDIR"),id=b("/tmp"),ie=b("'\\''"),ii=b(c$),ij=b("\\"),se=b("TEMP"),ip=b(ad),iu=b(c$),iv=b(fv),iy=b("Cygwin"),iz=b(fr),iA=b("Win32"),iB=[0,b("filename.ml"),189,9],iI=b("E2BIG"),iK=b("EACCES"),iL=b("EAGAIN"),iM=b("EBADF"),iN=b("EBUSY"),iO=b("ECHILD"),iP=b("EDEADLK"),iQ=b("EDOM"),iR=b("EEXIST"),iS=b("EFAULT"),iT=b("EFBIG"),iU=b("EINTR"),iV=b("EINVAL"),iW=b("EIO"),iX=b("EISDIR"),iY=b("EMFILE"),iZ=b("EMLINK"),i0=b("ENAMETOOLONG"),i1=b("ENFILE"),i2=b("ENODEV"),i3=b("ENOENT"),i4=b("ENOEXEC"),i5=b("ENOLCK"),i6=b("ENOMEM"),i7=b("ENOSPC"),i8=b("ENOSYS"),i9=b("ENOTDIR"),i_=b("ENOTEMPTY"),i$=b("ENOTTY"),ja=b("ENXIO"),jb=b("EPERM"),jc=b("EPIPE"),jd=b("ERANGE"),je=b("EROFS"),jf=b("ESPIPE"),jg=b("ESRCH"),jh=b("EXDEV"),ji=b("EWOULDBLOCK"),jj=b("EINPROGRESS"),jk=b("EALREADY"),jl=b("ENOTSOCK"),jm=b("EDESTADDRREQ"),jn=b("EMSGSIZE"),jo=b("EPROTOTYPE"),jp=b("ENOPROTOOPT"),jq=b("EPROTONOSUPPORT"),jr=b("ESOCKTNOSUPPORT"),js=b("EOPNOTSUPP"),jt=b("EPFNOSUPPORT"),ju=b("EAFNOSUPPORT"),jv=b("EADDRINUSE"),jw=b("EADDRNOTAVAIL"),jx=b("ENETDOWN"),jy=b("ENETUNREACH"),jz=b("ENETRESET"),jA=b("ECONNABORTED"),jB=b("ECONNRESET"),jC=b("ENOBUFS"),jD=b("EISCONN"),jE=b("ENOTCONN"),jF=b("ESHUTDOWN"),jG=b("ETOOMANYREFS"),jH=b("ETIMEDOUT"),jI=b("ECONNREFUSED"),jJ=b("EHOSTDOWN"),jK=b("EHOSTUNREACH"),jL=b("ELOOP"),jM=b("EOVERFLOW"),jN=b("EUNKNOWNERR %d"),iJ=b("Unix.Unix_error(Unix.%s, %S, %S)"),iE=b(fP),iF=b(g),iG=b(g),iH=b(fP),jO=b("0.0.0.0"),jP=b("127.0.0.1"),sd=b("::"),sc=b("::1"),jZ=[0,b("Vector.ml"),fT,25],j0=b("Cuda.No_Cuda_Device"),j1=b("Cuda.ERROR_DEINITIALIZED"),j2=b("Cuda.ERROR_NOT_INITIALIZED"),j3=b("Cuda.ERROR_INVALID_CONTEXT"),j4=b("Cuda.ERROR_INVALID_VALUE"),j5=b("Cuda.ERROR_OUT_OF_MEMORY"),j6=b("Cuda.ERROR_INVALID_DEVICE"),j7=b("Cuda.ERROR_NOT_FOUND"),j8=b("Cuda.ERROR_FILE_NOT_FOUND"),j9=b("Cuda.ERROR_UNKNOWN"),j_=b("Cuda.ERROR_LAUNCH_FAILED"),j$=b("Cuda.ERROR_LAUNCH_OUT_OF_RESOURCES"),ka=b("Cuda.ERROR_LAUNCH_TIMEOUT"),kb=b("Cuda.ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"),kc=b("no_cuda_device"),kd=b("cuda_error_deinitialized"),ke=b("cuda_error_not_initialized"),kf=b("cuda_error_invalid_context"),kg=b("cuda_error_invalid_value"),kh=b("cuda_error_out_of_memory"),ki=b("cuda_error_invalid_device"),kj=b("cuda_error_not_found"),kk=b("cuda_error_file_not_found"),kl=b("cuda_error_launch_failed"),km=b("cuda_error_launch_out_of_resources"),kn=b("cuda_error_launch_timeout"),ko=b("cuda_error_launch_incompatible_texturing"),kp=b("cuda_error_unknown"),kq=b("OpenCL.No_OpenCL_Device"),kr=b("OpenCL.OPENCL_ERROR_UNKNOWN"),ks=b("OpenCL.INVALID_CONTEXT"),kt=b("OpenCL.INVALID_DEVICE"),ku=b("OpenCL.INVALID_VALUE"),kv=b("OpenCL.INVALID_QUEUE_PROPERTIES"),kw=b("OpenCL.OUT_OF_RESOURCES"),kx=b("OpenCL.MEM_OBJECT_ALLOCATION_FAILURE"),ky=b("OpenCL.OUT_OF_HOST_MEMORY"),kz=b("OpenCL.FILE_NOT_FOUND"),kA=b("OpenCL.INVALID_PROGRAM"),kB=b("OpenCL.INVALID_BINARY"),kC=b("OpenCL.INVALID_BUILD_OPTIONS"),kD=b("OpenCL.INVALID_OPERATION"),kE=b("OpenCL.COMPILER_NOT_AVAILABLE"),kF=b("OpenCL.BUILD_PROGRAM_FAILURE"),kG=b("OpenCL.INVALID_KERNEL"),kH=b("OpenCL.INVALID_ARG_INDEX"),kI=b("OpenCL.INVALID_ARG_VALUE"),kJ=b("OpenCL.INVALID_MEM_OBJECT"),kK=b("OpenCL.INVALID_SAMPLER"),kL=b("OpenCL.INVALID_ARG_SIZE"),kM=b("OpenCL.INVALID_COMMAND_QUEUE"),kN=b("no_opencl_device"),kO=b("opencl_error_unknown"),kP=b("opencl_invalid_context"),kQ=b("opencl_invalid_device"),kR=b("opencl_invalid_value"),kS=b("opencl_invalid_queue_properties"),kT=b("opencl_out_of_resources"),kU=b("opencl_mem_object_allocation_failure"),kV=b("opencl_out_of_host_memory"),kW=b("opencl_file_not_found"),kX=b("opencl_invalid_program"),kY=b("opencl_invalid_binary"),kZ=b("opencl_invalid_build_options"),k0=b("opencl_invalid_operation"),k1=b("opencl_compiler_not_available"),k2=b("opencl_build_program_failure"),k3=b("opencl_invalid_kernel"),k4=b("opencl_invalid_arg_index"),k5=b("opencl_invalid_arg_value"),k6=b("opencl_invalid_mem_object"),k7=b("opencl_invalid_sampler"),k8=b("opencl_invalid_arg_size"),k9=b("opencl_invalid_command_queue"),k_=b(b5),k$=b(b5),lq=b(fF),lp=b(fA),lo=b(fF),ln=b(fA),lm=[0,1],ll=b(g),lh=b(bO),lc=b("Cl LOAD ARG Type Not Implemented\n"),lb=b("CU LOAD ARG Type Not Implemented\n"),la=[0,b(df),b(dc),b(dv),b(dn),b(db),b(di),b(dg),b(da),b(c_),b(dt),b(c7),b(dq),b(dk)],ld=b("Kernel.ERROR_BLOCK_SIZE"),lf=b("Kernel.ERROR_GRID_SIZE"),li=b("Kernel.No_source_for_device"),lt=b("Empty"),lu=b("Unit"),lv=b("Kern"),lw=b("Params"),lx=b("Plus"),ly=b("Plusf"),lz=b("Min"),lA=b("Minf"),lB=b("Mul"),lC=b("Mulf"),lD=b("Div"),lE=b("Divf"),lF=b("Mod"),lG=b("Id "),lH=b("IdName "),lI=b("IntVar "),lJ=b("FloatVar "),lK=b("UnitVar "),lL=b("CastDoubleVar "),lM=b("DoubleVar "),lN=b("IntArr"),lO=b("Int32Arr"),lP=b("Int64Arr"),lQ=b("Float32Arr"),lR=b("Float64Arr"),lS=b("VecVar "),lT=b("Concat"),lU=b("Seq"),lV=b("Return"),lW=b("Set"),lX=b("Decl"),lY=b("SetV"),lZ=b("SetLocalVar"),l0=b("Intrinsics"),l1=b(z),l2=b("IntId "),l3=b("Int "),l5=b("IntVecAcc"),l6=b("Local"),l7=b("Acc"),l8=b("Ife"),l9=b("If"),l_=b("Or"),l$=b("And"),ma=b("EqBool"),mb=b("LtBool"),mc=b("GtBool"),md=b("LtEBool"),me=b("GtEBool"),mf=b("DoLoop"),mg=b("While"),mh=b("App"),mi=b("GInt"),mj=b("GFloat"),l4=b("Float "),ls=b("  "),lr=b("%s\n"),nX=b(fX),nY=[0,b(dh),166,14],mm=b(g),mn=b(bO),mo=b("\n}\n#ifdef __cplusplus\n}\n#endif"),mp=b(" ) {\n"),mq=b(g),mr=b(bN),mt=b(g),ms=b('#ifdef __cplusplus\nextern "C" {\n#endif\n\n__global__ void spoc_dummy ( '),mu=b(ab),mv=b(b6),mw=b(ac),mx=b(ab),my=b(b6),mz=b(ac),mA=b(ab),mB=b(bV),mC=b(ac),mD=b(ab),mE=b(bV),mF=b(ac),mG=b(ab),mH=b(bZ),mI=b(ac),mJ=b(ab),mK=b(bZ),mL=b(ac),mM=b(ab),mN=b(b8),mO=b(ac),mP=b(ab),mQ=b(b8),mR=b(ac),mS=b(ab),mT=b(fu),mU=b(ac),mV=b(f3),mW=b(ft),mX=[0,b(dh),65,17],mY=b(bW),mZ=b(fG),m0=b(J),m1=b(K),m2=b(fO),m3=b(J),m4=b(K),m5=b(fo),m6=b(J),m7=b(K),m8=b(fB),m9=b(J),m_=b(K),m$=b(f4),na=b(fY),nc=b("int"),nd=b("float"),ne=b("double"),nb=[0,b(dh),60,12],ng=b(bN),nf=b(gi),nh=b(fV),ni=b(g),nj=b(g),nm=b(bQ),nn=b(aa),no=b(aF),nq=b(bQ),np=b(aa),nr=b(V),ns=b(J),nt=b(K),nu=b("}\n"),nv=b(aF),nw=b(aF),nx=b("{"),ny=b(bf),nz=b(fy),nA=b(bf),nB=b(bc),nC=b(b7),nD=b(bf),nE=b(bc),nF=b(b7),nG=b(fm),nH=b(fk),nI=b(fE),nJ=b(f_),nK=b(fL),nL=b(bM),nM=b(f9),nN=b(b4),nO=b(fq),nP=b(bT),nQ=b(bM),nR=b(bT),nS=b(aa),nT=b(fl),nU=b(b4),nV=b(bc),nW=b(fI),n1=b(bY),n2=b(bY),n3=b(z),n4=b(z),nZ=b(fS),n0=b(gk),n5=b(V),nk=b(bQ),nl=b(aa),n6=b(J),n7=b(K),n9=b(bW),n_=b(V),n$=b(gf),oa=b(J),ob=b(K),oc=b(V),n8=b("cuda error parse_float"),mk=[0,b(g),b(g)],py=b(fX),pz=[0,b(dm),162,14],of=b(g),og=b(bO),oh=b(b4),oi=b(" ) \n{\n"),oj=b(g),ok=b(bN),om=b(g),ol=b("__kernel void spoc_dummy ( "),on=b(b6),oo=b(b6),op=b(bV),oq=b(bV),or=b(bZ),os=b(bZ),ot=b(b8),ou=b(b8),ov=b(fu),ow=b(f3),ox=b(ft),oy=[0,b(dm),65,17],oz=b(bW),oA=b(fG),oB=b(J),oC=b(K),oD=b(fO),oE=b(J),oF=b(K),oG=b(fo),oH=b(J),oI=b(K),oJ=b(fB),oK=b(J),oL=b(K),oM=b(f4),oN=b(fY),oP=b("__global int"),oQ=b("__global float"),oR=b("__global double"),oO=[0,b(dm),60,12],oT=b(bN),oS=b(gi),oU=b(fV),oV=b(g),oW=b(g),oY=b(bQ),oZ=b(aa),o0=b(aF),o1=b(aa),o2=b(V),o3=b(J),o4=b(K),o5=b(g),o6=b(bO),o7=b(aF),o8=b(g),o9=b(bf),o_=b(fy),o$=b(bf),pa=b(bc),pb=b(b7),pc=b(b4),pd=b(aF),pe=b("{\n"),pf=b(")\n"),pg=b(b7),ph=b(fm),pi=b(fk),pj=b(fE),pk=b(f_),pl=b(fL),pm=b(bM),pn=b(f9),po=b(ga),pp=b(fq),pq=b(bT),pr=b(bM),ps=b(bT),pt=b(aa),pu=b(fl),pv=b(ga),pw=b(bc),px=b(fI),pC=b(bY),pD=b(bY),pE=b(z),pF=b(z),pA=b(fS),pB=b(gk),pG=b(V),oX=b(aa),pH=b(J),pI=b(K),pK=b(bW),pL=b(V),pM=b(gf),pN=b(J),pO=b(K),pP=b(V),pJ=b("opencl error parse_float"),od=[0,b(g),b(g)],qN=[0,0],qO=[0,0],qP=[0,1],qQ=[0,1],qH=b("kirc_kernel.cu"),qI=b("nvcc -m64 -arch=sm_10 -O3 -ptx kirc_kernel.cu -o kirc_kernel.ptx"),qJ=b("kirc_kernel.ptx"),qK=b("rm kirc_kernel.cu kirc_kernel.ptx"),qE=[0,b(g),b(g)],qG=b(g),qF=[0,b("Kirc.ml"),407,81],qL=b(aa),qM=b(f6),qB=[33,0],qx=b(f6),pQ=b("int spoc_xor (int a, int b ) { return (a^b);}\n"),pR=b("int spoc_powint (int a, int b ) { return ((int) pow (((float) a), ((float) b)));}\n"),pS=b("int logical_and (int a, int b ) { return (a & b);}\n"),pT=b("float spoc_fdiv ( float a, float b ) { return (a / b);}\n"),pU=b("float spoc_fmul ( float a, float b ) { return (a * b);}\n"),pV=b("float spoc_fminus ( float a, float b ) { return (a - b);}\n"),pW=b("float spoc_fadd ( float a, float b ) { return (a + b);}\n"),pX=b("float spoc_fdiv ( float a, float b );\n"),pY=b("float spoc_fmul ( float a, float b );\n"),pZ=b("float spoc_fminus ( float a, float b );\n"),p0=b("float spoc_fadd ( float a, float b );\n"),p2=b(dl),p3=b("double spoc_ddiv ( double a, double b ) { return (a / b);}\n"),p4=b("double spoc_dmul ( double a, double b ) { return (a * b);}\n"),p5=b("double spoc_dminus ( double a, double b ) { return (a - b);}\n"),p6=b("double spoc_dadd ( double a, double b ) { return (a + b);}\n"),p7=b("double spoc_ddiv ( double a, double b );\n"),p8=b("double spoc_dmul ( double a, double b );\n"),p9=b("double spoc_dminus ( double a, double b );\n"),p_=b("double spoc_dadd ( double a, double b );\n"),p$=b(dl),qa=b("#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"),qb=b("#elif defined(cl_amd_fp64)  // AMD extension available?\n"),qc=b("#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"),qd=b("#if defined(cl_khr_fp64)  // Khronos extension available?\n"),qe=b(ge),qf=b(f8),qh=b(dl),qi=b("__device__ double spoc_ddiv ( double a, double b ) { return (a / b);}\n"),qj=b("__device__ double spoc_dmul ( double a, double b ) { return (a * b);}\n"),qk=b("__device__ double spoc_dminus ( double a, double b ) { return (a - b);}\n"),ql=b("__device__ double spoc_dadd ( double a, double b ) { return (a + b);}\n"),qm=b(ge),qn=b(f8),qp=b("__device__ int spoc_xor (int a, int b ) { return (a^b);}\n"),qq=b("__device__ int spoc_powint (int a, int b ) { return ((int) pow (((double) a), ((double) b)));}\n"),qr=b("__device__ int logical_and (int a, int b ) { return (a & b);}\n"),qs=b("__device__ float spoc_fdiv ( float a, float b ) { return (a / b);}\n"),qt=b("__device__ float spoc_fmul ( float a, float b ) { return (a * b);}\n"),qu=b("__device__ float spoc_fminus ( float a, float b ) { return (a - b);}\n"),qv=b("__device__ float spoc_fadd ( float a, float b ) { return (a + b);}\n"),qC=[0,b(g),b(g)],q5=b("canvas"),q2=b("span"),q1=b("img"),q0=b("br"),qZ=b(fn),qY=b("select"),qX=b("option"),q3=b("Dom_html.Canvas_not_available"),sa=[0,b(gc),131,17],r9=b("Will use device : %s!"),r_=[0,1],r$=b(g),r8=b("Time %s : %Fs\n%!"),re=b("spoc_dummy"),rf=b("kirc_kernel"),rc=b("spoc_kernel_extension error"),q6=[0,b(gc),12,15],rw=b(aJ),rx=b(aJ),rD=b(aJ),rE=b(aJ),rJ=b(aJ),rK=b(aJ),rN=b(fM),rO=b(fM),rW=b("(get_group_id (0))"),rX=b("blockIdx.x"),rZ=b("(get_local_size (0))"),r0=b("blockDim.x"),r2=b("(get_local_id (0))"),r3=b("threadIdx.x");function
Q(a){throw[0,aL,a]}function
C(a){throw[0,bm,a]}function
h(a,b){var
c=a.getLen(),e=b.getLen(),d=y(c+e|0);a8(a,0,d,0,c);a8(b,0,d,c,e);return d}function
k(a){return b(g+a)}function
L(a){var
c=cY(gC,a),b=0,f=c.getLen();for(;;){if(f<=b)var
e=h(c,gB);else{var
d=c.safeGet(b),g=48<=d?58<=d?0:1:45===d?1:0;if(g){var
b=b+1|0;continue}var
e=c}return e}}function
cb(a,b){if(a){var
c=a[1];return[0,c,cb(a[2],b)]}return b}e4(0);var
dK=cZ(1);cZ(2);function
dL(a,b){return gt(a,b,0,b.getLen())}function
dM(a){return e4(e5(a,gE,0))}function
dN(a){var
b=s5(0);for(;;){if(b){var
c=b[2],d=b[1];try{c0(d)}catch(f){}var
b=c;continue}return 0}}c1(gG,dN);function
dO(a){return e6(a)}function
gH(a,b){return s6(a,b)}function
dP(a){return c0(a)}function
dQ(a,b){var
d=b.length-1-1|0,e=0;if(!(d<0)){var
c=e;for(;;){j(a,b[c+1]);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return 0}function
aM(a,b){var
d=b.length-1;if(0===d)return[0];var
e=u(d,j(a,b[0+1])),f=d-1|0,g=1;if(!(f<1)){var
c=g;for(;;){e[c+1]=j(a,b[c+1]);var
h=c+1|0;if(f!==c){var
c=h;continue}break}}return e}function
cc(a,b){var
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
d=h;continue}break}}return e[1]}function
dS(a){var
b=a,c=0;for(;;){if(b){var
d=[0,b[1],c],b=b[2],c=d;continue}return c}}function
cd(a,b){if(b){var
c=b[2],d=j(a,b[1]);return[0,d,cd(a,c)]}return 0}function
cf(a,b,c){if(b){var
d=b[1];return i(a,d,cf(a,b[2],c))}return c}function
dU(a,b,c){var
e=b,d=c;for(;;){if(e){if(d){var
f=d[2],g=e[2];i(a,e[1],d[1]);var
e=g,d=f;continue}}else
if(!d)return 0;return C(gM)}}function
cg(a,b){var
c=b;for(;;){if(c){var
e=c[2],d=0===aD(c[1],a)?1:0;if(d)return d;var
c=e;continue}return 0}}function
ch(a){if(0<=a)if(!(p<a))return a;return C(gN)}function
dV(a){var
b=65<=a?90<a?0:1:0;if(!b){var
c=192<=a?214<a?0:1:0;if(!c){var
d=216<=a?222<a?1:0:1;if(d)return a}}return a+32|0}function
af(a,b){var
c=y(a);sy(c,0,a,b);return c}function
s(a,b,c){if(0<=b)if(0<=c)if(!((a.getLen()-c|0)<b)){var
d=y(c);a8(a,b,d,0,c);return d}return C(gU)}function
bp(a,b,c,d,e){if(0<=e)if(0<=b)if(!((a.getLen()-e|0)<b))if(0<=d)if(!((c.getLen()-e|0)<d))return a8(a,b,c,d,e);return C(gV)}function
dW(a){var
c=a.getLen();if(0===c)var
f=a;else{var
d=y(c),e=c-1|0,g=0;if(!(e<0)){var
b=g;for(;;){d.safeSet(b,dV(a.safeGet(b)));var
h=b+1|0;if(e!==b){var
b=h;continue}break}}var
f=d}return f}var
cj=tn(0)[1],av=tk(0),ck=(1<<(av-10|0))-1|0,aO=v(av/8|0,ck)-1|0,gY=tm(0)[2],gZ=bS,g0=aH;function
cl(k){function
h(a){return a?a[5]:0}function
e(a,b,c,d){var
e=h(a),f=h(d),g=f<=e?e+1|0:f+1|0;return[0,a,b,c,d,g]}function
q(a,b){return[0,0,a,b,0,1]}function
f(a,b,c,d){var
i=a?a[5]:0,j=d?d[5]:0;if((j+2|0)<i){if(a){var
f=a[4],m=a[3],n=a[2],k=a[1],q=h(f);if(q<=h(k))return e(k,n,m,e(f,b,c,d));if(f){var
r=f[3],s=f[2],t=f[1],u=e(f[4],b,c,d);return e(e(k,n,m,t),s,r,u)}return C(g1)}return C(g2)}if((i+2|0)<j){if(d){var
l=d[4],o=d[3],p=d[2],g=d[1],v=h(g);if(v<=h(l))return e(e(a,b,c,g),p,o,l);if(g){var
w=g[3],x=g[2],y=g[1],z=e(g[4],p,o,l);return e(e(a,b,c,y),x,w,z)}return C(g3)}return C(g4)}var
A=j<=i?i+1|0:j+1|0;return[0,a,b,c,d,A]}var
a=0;function
I(a){return a?0:1}function
s(a,b,c){if(c){var
d=c[4],h=c[3],e=c[2],g=c[1],l=c[5],j=i(k[1],a,e);return 0===j?[0,g,a,b,d,l]:0<=j?f(g,e,h,s(a,b,d)):f(s(a,b,g),e,h,d)}return[0,0,a,b,0,1]}function
J(a,b){var
c=b;for(;;){if(c){var
e=c[4],f=c[3],g=c[1],d=i(k[1],a,c[2]);if(0===d)return f;var
h=0<=d?e:g,c=h;continue}throw[0,r]}}function
K(a,b){var
c=b;for(;;){if(c){var
f=c[4],g=c[1],d=i(k[1],a,c[2]),e=0===d?1:0;if(e)return e;var
h=0<=d?f:g,c=h;continue}return 0}}function
o(a){var
b=a;for(;;){if(b){var
c=b[1];if(c){var
b=c;continue}return[0,b[2],b[3]]}throw[0,r]}}function
L(a){var
b=a;for(;;){if(b){var
c=b[4],d=b[3],e=b[2];if(c){var
b=c;continue}return[0,e,d]}throw[0,r]}}function
t(a){if(a){var
b=a[1];if(b){var
c=a[4],d=a[3],e=a[2];return f(t(b),e,d,c)}return a[4]}return C(g5)}function
u(a,b){if(b){var
c=b[4],j=b[3],e=b[2],d=b[1],l=i(k[1],a,e);if(0===l){if(d)if(c){var
h=o(c),m=h[2],n=h[1],g=f(d,n,m,t(c))}else
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
f=d[4],g=d[3],h=d[2],i=n(a,h,g,z(a,d[1],e)),d=f,e=i;continue}return e}}function
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
E(a,b,c){if(c){var
d=c[4],e=c[3],g=c[2];return f(E(a,b,c[1]),g,e,d)}return q(a,b)}function
F(a,b,c){if(c){var
d=c[3],e=c[2],g=c[1];return f(g,e,d,F(a,b,c[4]))}return q(a,b)}function
g(a,b,c,d){if(a){if(d){var
h=d[5],i=a[5],j=d[4],k=d[3],l=d[2],m=d[1],n=a[4],o=a[3],p=a[2],q=a[1];return(h+2|0)<i?f(q,p,o,g(n,b,c,d)):(i+2|0)<h?f(g(a,b,c,m),l,k,j):e(a,b,c,d)}return F(b,c,a)}return E(b,c,d)}function
p(a,b){if(a){if(b){var
c=o(b),d=c[2],e=c[1];return g(a,e,d,t(b))}return a}return b}function
G(a,b,c,d){return c?g(a,b,c[1],d):p(a,d)}function
l(a,b){if(b){var
c=b[4],d=b[3],e=b[2],f=b[1],m=i(k[1],a,e);if(0===m)return[0,f,[0,d],c];if(0<=m){var
h=l(a,c),n=h[3],o=h[2];return[0,g(f,e,d,h[1]),o,n]}var
j=l(a,f),p=j[2],q=j[1];return[0,q,p,g(j[3],e,d,c)]}return g6}function
m(a,b,c){if(b){var
d=b[2],i=b[5],j=b[4],k=b[3],o=b[1];if(h(c)<=i){var
e=l(d,c),p=e[2],q=e[1],r=m(a,j,e[3]),s=n(a,d,[0,k],p);return G(m(a,o,q),d,s,r)}}else
if(!c)return 0;if(c){var
f=c[2],t=c[4],u=c[3],v=c[1],g=l(f,b),w=g[2],x=g[1],y=m(a,g[3],t),z=n(a,f,w,[0,u]);return G(m(a,x,v),f,z,y)}throw[0,D,g7]}function
w(a,b){if(b){var
c=b[3],d=b[2],h=b[4],e=w(a,b[1]),j=i(a,d,c),f=w(a,h);return j?g(e,d,c,f):p(e,f)}return 0}function
x(a,b){if(b){var
c=b[3],d=b[2],m=b[4],e=x(a,b[1]),f=e[2],h=e[1],n=i(a,d,c),j=x(a,m),k=j[2],l=j[1];if(n){var
o=p(f,k);return[0,g(h,d,c,l),o]}var
q=g(f,d,c,k);return[0,p(h,l),q]}return g8}function
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
e=c[3],f=c[2],g=c[1],d=[0,[0,f,e],H(d,c[4])],c=g;continue}return d}}return[0,a,I,K,s,q,u,m,M,N,y,z,A,B,w,x,b,function(a){return H(0,a)},o,L,o,l,J,c,v]}var
g_=[0,g9];function
g$(a){throw[0,g_]}function
aP(a){var
b=1<=a?a:1,c=aO<b?aO:b,d=y(c);return[0,d,0,c,d]}function
aQ(a){return s(a[1],0,a[2])}function
dZ(a,b){var
c=[0,a[3]];for(;;){if(c[1]<(a[2]+b|0)){c[1]=2*c[1]|0;continue}if(aO<c[1])if((a[2]+b|0)<=aO)c[1]=aO;else
Q(ha);var
d=y(c[1]);bp(a[1],0,d,0,a[2]);a[1]=d;a[3]=c[1];return 0}}function
E(a,b){var
c=a[2];if(a[3]<=c)dZ(a,1);a[1].safeSet(c,b);a[2]=c+1|0;return 0}function
br(a,b){var
c=b.getLen(),d=a[2]+c|0;if(a[3]<d)dZ(a,c);bp(b,0,a[1],a[2],c);a[2]=d;return 0}function
cm(a){return 0<=a?a:Q(h(hb,k(a)))}function
d0(a,b){return cm(a+b|0)}var
hc=1;function
d1(a){return d0(hc,a)}function
d2(a){return s(a,0,a.getLen())}function
d3(a,b,c){var
d=h(he,h(a,hd)),e=h(hf,h(k(b),d));return C(h(hg,h(af(1,c),e)))}function
aR(a,b,c){return d3(d2(a),b,c)}function
bs(a){return C(h(hi,h(d2(a),hh)))}function
am(e,b,c,d){function
h(a){if((e.safeGet(a)+aI|0)<0||9<(e.safeGet(a)+aI|0))return a;var
b=a+1|0;for(;;){var
c=e.safeGet(b);if(48<=c){if(!(58<=c)){var
b=b+1|0;continue}var
d=0}else
if(36===c){var
f=b+1|0,d=1}else
var
d=0;if(!d)var
f=a;return f}}var
i=h(b+1|0),f=aP((c-i|0)+10|0);E(f,37);var
a=i,g=dS(d);for(;;){if(a<=c){var
j=e.safeGet(a);if(42===j){if(g){var
l=g[2];br(f,k(g[1]));var
a=h(a+1|0),g=l;continue}throw[0,D,hj]}E(f,j);var
a=a+1|0;continue}return aQ(f)}}function
d4(a,b,c,d,e){var
f=am(b,c,d,e);if(78!==a)if(be!==a)return f;f.safeSet(f.getLen()-1|0,dr);return f}function
d5(a){return function(c,b){var
m=c.getLen();function
n(a,b){var
o=40===a?41:c9;function
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
o=j.safeGet(h);if(58<=o){if(95===o){var
e=1,h=h+1|0;continue}}else
if(32<=o)switch(o+fx|0){case
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
h=n(b,e,h,at);continue;default:var
h=h+1|0;continue}var
d=h;b:for(;;){if(m<d)var
f=bs(j);else{var
k=j.safeGet(d);if(fT<=k)var
g=0;else
switch(k){case
78:case
88:case
aG:case
at:case
dd:case
dr:case
ds:var
f=n(b,e,d,at),g=1;break;case
69:case
70:case
71:case
gb:case
de:case
dw:var
f=n(b,e,d,de),g=1;break;case
33:case
37:case
44:case
64:var
f=d+1|0,g=1;break;case
83:case
91:case
bh:var
f=n(b,e,d,bh),g=1;break;case
97:case
b1:case
c8:var
f=n(b,e,d,k),g=1;break;case
76:case
fH:case
be:var
t=d+1|0;if(m<t){var
f=n(b,e,d,at),g=1}else{var
q=j.safeGet(t)+gg|0;if(q<0||32<q)var
r=1;else
switch(q){case
0:case
12:case
17:case
23:case
29:case
32:var
f=i(c,n(b,e,d,k),at),g=1,r=0;break;default:var
r=1}if(r){var
f=n(b,e,d,at),g=1}}break;case
67:case
99:var
f=n(b,e,d,99),g=1;break;case
66:case
98:var
f=n(b,e,d,66),g=1;break;case
41:case
c9:var
f=n(b,e,d,k),g=1;break;case
40:var
f=s(n(b,e,d,k)),g=1;break;case
du:var
u=n(b,e,d,k),v=i(d5(k),j,u),p=u;for(;;){if(p<(v-2|0)){var
p=i(c,p,j.safeGet(p));continue}var
d=v-1|0;continue b}default:var
g=0}if(!g)var
f=aR(j,d,k)}var
w=f;break}}var
l=w;continue a}}var
l=l+1|0;continue}return l}}s(0);return 0}function
d7(a){var
d=[0,0,0,0];function
b(a,b,c){var
f=41!==c?1:0,g=f?c9!==c?1:0:f;if(g){var
e=97===c?2:1;if(b1===c)d[3]=d[3]+1|0;if(a)d[2]=d[2]+e|0;else
d[1]=d[1]+e|0}return b+1|0}d6(a,b,function(a,b){return a+1|0});return d[1]}function
d8(a,b,c){var
h=a.safeGet(c);if((h+aI|0)<0||9<(h+aI|0))return i(b,0,c);var
e=h+aI|0,d=c+1|0;for(;;){var
f=a.safeGet(d);if(48<=f){if(!(58<=f)){var
e=(10*e|0)+(f+aI|0)|0,d=d+1|0;continue}var
g=0}else
if(36===f)if(0===e){var
j=Q(hl),g=1}else{var
j=i(b,[0,cm(e-1|0)],d+1|0),g=1}else
var
g=0;if(!g)var
j=i(b,0,c);return j}}function
M(a,b){return a?b:d1(b)}function
d9(a,b){return a?a[1]:b}function
d_(aJ,b,c,d,e,f,g){var
D=j(b,g);function
ag(a){return i(d,D,a)}function
aK(a,b,m,aM){var
k=m.getLen();function
F(l,b){var
p=b;for(;;){if(k<=p)return j(a,D);var
d=m.safeGet(p);if(37===d){var
o=function(a,b){return q(aM,d9(a,b))},av=function(g,f,c,d){var
a=d;for(;;){var
aa=m.safeGet(a)+fx|0;if(!(aa<0||25<aa))switch(aa){case
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
10:return d8(m,function(a,b){var
d=[0,o(a,f),c];return av(g,M(a,f),d,b)},a+1|0);default:var
a=a+1|0;continue}var
q=m.safeGet(a);if(124<=q)var
k=0;else
switch(q){case
78:case
88:case
aG:case
at:case
dd:case
dr:case
ds:var
a8=o(g,f),a9=bH(d4(q,m,p,a,c),a8),l=r(M(g,f),a9,a+1|0),k=1;break;case
69:case
71:case
gb:case
de:case
dw:var
a1=o(g,f),a2=cY(am(m,p,a,c),a1),l=r(M(g,f),a2,a+1|0),k=1;break;case
76:case
fH:case
be:var
ad=m.safeGet(a+1|0)+gg|0;if(ad<0||32<ad)var
ah=1;else
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
a7=o(g,f),aB=bH(am(m,p,U,c),a7),aj=1;break;default:var
a6=o(g,f),aB=bH(am(m,p,U,c),a6),aj=1}if(aj){var
aA=aB,ai=1}}if(!ai){var
a5=o(g,f),aA=sI(am(m,p,U,c),a5)}var
l=r(M(g,f),aA,U+1|0),k=1,ah=0;break;default:var
ah=1}if(ah){var
a3=o(g,f),a4=bH(d4(be,m,p,a,c),a3),l=r(M(g,f),a4,a+1|0),k=1}break;case
37:case
64:var
l=r(f,af(1,q),a+1|0),k=1;break;case
83:case
bh:var
z=o(g,f);if(bh===q)var
A=z;else{var
b=[0,0],ao=z.getLen()-1|0,aN=0;if(!(ao<0)){var
N=aN;for(;;){var
x=z.safeGet(N),bd=14<=x?34===x?1:92===x?1:0:11<=x?13<=x?1:0:8<=x?1:0,aT=bd?2:c2(x)?1:4;b[1]=b[1]+aT|0;var
aU=N+1|0;if(ao!==N){var
N=aU;continue}break}}if(b[1]===z.getLen())var
aD=z;else{var
n=y(b[1]);b[1]=0;var
ap=z.getLen()-1|0,aO=0;if(!(ap<0)){var
L=aO;for(;;){var
w=z.safeGet(L),B=w-34|0;if(B<0||58<B)if(-20<=B)var
V=1;else{switch(B+34|0){case
8:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],98);var
K=1;break;case
9:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],c8);var
K=1;break;case
10:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],be);var
K=1;break;case
13:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],b1);var
K=1;break;default:var
V=1,K=0}if(K)var
V=0}else
var
V=(B-1|0)<0||56<(B-1|0)?(n.safeSet(b[1],92),b[1]++,n.safeSet(b[1],w),0):1;if(V)if(c2(w))n.safeSet(b[1],w);else{n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],48+(w/aG|0)|0);b[1]++;n.safeSet(b[1],48+((w/10|0)%10|0)|0);b[1]++;n.safeSet(b[1],48+(w%10|0)|0)}b[1]++;var
aS=L+1|0;if(ap!==L){var
L=aS;continue}break}}var
aD=n}var
A=h(hw,h(aD,hv))}if(a===(p+1|0))var
aC=A;else{var
J=am(m,p,a,c);try{var
W=0,u=1;for(;;){if(J.getLen()<=u)var
aq=[0,0,W];else{var
X=J.safeGet(u);if(49<=X)if(58<=X)var
ak=0;else{var
aq=[0,sS(s(J,u,(J.getLen()-u|0)-1|0)),W],ak=1}else{if(45===X){var
W=1,u=u+1|0;continue}var
ak=0}if(!ak){var
u=u+1|0;continue}}var
Z=aq;break}}catch(f){if(f[1]!==aL)throw f;var
Z=d3(J,0,bh)}var
O=Z[1],C=A.getLen(),aV=Z[2],P=0,aW=32;if(O===C)if(0===P){var
_=A,al=1}else
var
al=0;else
var
al=0;if(!al)if(O<=C)var
_=s(A,P,C);else{var
Y=af(O,aW);if(aV)bp(A,P,Y,0,C);else
bp(A,P,Y,O-C|0,C);var
_=Y}var
aC=_}var
l=r(M(g,f),aC,a+1|0),k=1;break;case
67:case
99:var
t=o(g,f);if(99===q)var
ay=af(1,t);else{if(39===t)var
v=gO;else
if(92===t)var
v=gP;else{if(14<=t)var
G=0;else
switch(t){case
8:var
v=gQ,G=1;break;case
9:var
v=gR,G=1;break;case
10:var
v=gS,G=1;break;case
13:var
v=gT,G=1;break;default:var
G=0}if(!G)if(c2(t)){var
an=y(1);an.safeSet(0,t);var
v=an}else{var
H=y(4);H.safeSet(0,92);H.safeSet(1,48+(t/aG|0)|0);H.safeSet(2,48+((t/10|0)%10|0)|0);H.safeSet(3,48+(t%10|0)|0);var
v=H}}var
ay=h(ht,h(v,hs))}var
l=r(M(g,f),ay,a+1|0),k=1;break;case
66:case
98:var
aZ=a+1|0,a0=o(g,f)?gz:gA,l=r(M(g,f),a0,aZ),k=1;break;case
40:case
du:var
T=o(g,f),aw=i(d5(q),m,a+1|0);if(du===q){var
Q=aP(T.getLen()),ar=function(a,b){E(Q,b);return a+1|0};d6(T,function(a,b,c){if(a)br(Q,hk);else
E(Q,37);return ar(b,c)},ar);var
aX=aQ(Q),l=r(M(g,f),aX,aw),k=1}else{var
ax=M(g,f),bc=d0(d7(T),ax),l=aK(function(a){return F(bc,aw)},ax,T,aM),k=1}break;case
33:j(e,D);var
l=F(f,a+1|0),k=1;break;case
41:var
l=r(f,hq,a+1|0),k=1;break;case
44:var
l=r(f,hr,a+1|0),k=1;break;case
70:var
ab=o(g,f);if(0===c)var
az=hu;else{var
$=am(m,p,a,c);if(70===q)$.safeSet($.getLen()-1|0,dw);var
az=$}var
au=sv(ab);if(3===au)var
ac=ab<0?hn:ho;else
if(4<=au)var
ac=hp;else{var
S=cY(az,ab),R=0,aY=S.getLen();for(;;){if(aY<=R)var
as=h(S,hm);else{var
I=S.safeGet(R)-46|0,bf=I<0||23<I?55===I?1:0:(I-1|0)<0||21<(I-1|0)?1:0;if(!bf){var
R=R+1|0;continue}var
as=S}var
ac=as;break}}var
l=r(M(g,f),ac,a+1|0),k=1;break;case
91:var
l=aR(m,a,q),k=1;break;case
97:var
aE=o(g,f),aF=d1(d9(g,f)),aH=o(0,aF),a_=a+1|0,a$=M(g,aF);if(aJ)ag(i(aE,0,aH));else
i(aE,D,aH);var
l=F(a$,a_),k=1;break;case
b1:var
l=aR(m,a,q),k=1;break;case
c8:var
aI=o(g,f),ba=a+1|0,bb=M(g,f);if(aJ)ag(j(aI,0));else
j(aI,D);var
l=F(bb,ba),k=1;break;default:var
k=0}if(!k)var
l=aR(m,a,q);return l}},f=p+1|0,g=0;return d8(m,function(a,b){return av(a,l,g,b)},f)}i(c,D,d);var
p=p+1|0;continue}}function
r(a,b,c){ag(b);return F(a,c)}return F(b,0)}var
o=cm(0);function
k(a,b){return aK(f,o,a,b)}var
m=d7(g);if(m<0||6<m){var
n=function(f,b){if(m<=f){var
h=u(m,0),i=function(a,b){return l(h,(m-a|0)-1|0,b)},c=0,a=b;for(;;){if(a){var
d=a[2],e=a[1];if(d){i(c,e);var
c=c+1|0,a=d;continue}i(c,e)}return k(g,h)}}return function(a){return n(f+1|0,[0,a,b])}},a=n(0,0)}else
switch(m){case
1:var
a=function(a){var
b=u(1,0);l(b,0,a);return k(g,b)};break;case
2:var
a=function(a,b){var
c=u(2,0);l(c,0,a);l(c,1,b);return k(g,c)};break;case
3:var
a=function(a,b,c){var
d=u(3,0);l(d,0,a);l(d,1,b);l(d,2,c);return k(g,d)};break;case
4:var
a=function(a,b,c,d){var
e=u(4,0);l(e,0,a);l(e,1,b);l(e,2,c);l(e,3,d);return k(g,e)};break;case
5:var
a=function(a,b,c,d,e){var
f=u(5,0);l(f,0,a);l(f,1,b);l(f,2,c);l(f,3,d);l(f,4,e);return k(g,f)};break;case
6:var
a=function(a,b,c,d,e,f){var
h=u(6,0);l(h,0,a);l(h,1,b);l(h,2,c);l(h,3,d);l(h,4,e);l(h,5,f);return k(g,h)};break;default:var
a=k(g,[0])}return a}function
d$(a){function
b(a){return 0}return d_(0,function(a){return dK},gH,dL,dP,b,a)}function
hx(a){return aP(2*a.getLen()|0)}function
ea(c){function
b(a){var
b=aQ(a);a[2]=0;return j(c,b)}function
d(a){return 0}var
e=1;return function(a){return d_(e,hx,E,br,d,b,a)}}function
eb(a){return j(ea(function(a){return a}),a)}var
ec=[0,0];function
ed(a){ec[1]=[0,a,ec[1]];return 0}function
ee(a,b){var
j=0===b.length-1?[0,0]:b,f=j.length-1,p=0,r=54;if(!(54<0)){var
d=p;for(;;){l(a[1],d,d);var
w=d+1|0;if(r!==d){var
d=w;continue}break}}var
g=[0,hy],m=0,s=55,t=sE(55,f)?s:f,n=54+t|0;if(!(n<m)){var
c=m;for(;;){var
o=c%55|0,u=g[1],i=h(u,k(q(j,aE(c,f))));g[1]=sX(i,0,i.getLen());var
e=g[1];l(a[1],o,(q(a[1],o)^(((e.safeGet(0)+(e.safeGet(1)<<8)|0)+(e.safeGet(2)<<16)|0)+(e.safeGet(3)<<24)|0))&bd);var
v=c+1|0;if(n!==c){var
c=v;continue}break}}a[2]=0;return 0}32===av;var
hA=[0,hz.slice(),0];try{var
sl=bI(sk),cn=sl}catch(f){if(f[1]!==r)throw f;try{var
sj=bI(si),ef=sj}catch(f){if(f[1]!==r)throw f;var
ef=hB}var
cn=ef}var
dX=cn.getLen(),hC=82,dY=0;if(0<=0)if(dX<dY)var
bJ=0;else
try{var
bq=dY;for(;;){if(dX<=bq)throw[0,r];if(cn.safeGet(bq)!==hC){var
bq=bq+1|0;continue}var
gX=1,ci=gX,bJ=1;break}}catch(f){if(f[1]!==r)throw f;var
ci=0,bJ=1}else
var
bJ=0;if(!bJ)var
ci=C(gW);var
ag=[fQ,function(a){var
b=[0,u(55,0),0];ee(b,e7(0));return b}];function
eg(a,b){var
m=a?a[1]:ci,d=16;for(;;){if(!(b<=d))if(!(ck<(d*2|0))){var
d=d*2|0;continue}if(m){var
h=ta(ag);if(aH===h)var
c=ag[1];else
if(fQ===h){var
k=ag[0+1];ag[0+1]=g$;try{var
e=j(k,0);ag[0+1]=e;s$(ag,g0)}catch(f){ag[0+1]=function(a){throw f};throw f}var
c=e}else
var
c=ag;c[2]=(c[2]+1|0)%55|0;var
f=q(c[1],c[2]),g=(q(c[1],(c[2]+24|0)%55|0)+(f^f>>>25&31)|0)&bd;l(c[1],c[2],g);var
i=g}else
var
i=0;return[0,0,u(d,0),i,d]}}function
co(a,b){return 3<=a.length-1?sF(10,aG,a[3],b)&(a[2].length-1-1|0):aE(sG(10,aG,b),a[2].length-1)}function
bt(a,b){var
i=co(a,b),d=q(a[2],i);if(d){var
e=d[3],j=d[2];if(0===aD(b,d[1]))return j;if(e){var
f=e[3],k=e[2];if(0===aD(b,e[1]))return k;if(f){var
l=f[3],m=f[2];if(0===aD(b,f[1]))return m;var
c=l;for(;;){if(c){var
g=c[3],h=c[2];if(0===aD(b,c[1]))return h;var
c=g;continue}throw[0,r]}}throw[0,r]}throw[0,r]}throw[0,r]}function
a(a,b){return c1(a,b[0+1])}var
cp=[0,0];c1(hD,cp);var
hE=2;function
hF(a){var
b=[0,0],d=a.getLen()-1|0,e=0;if(!(d<0)){var
c=e;for(;;){b[1]=(223*b[1]|0)+a.safeGet(c)|0;var
g=c+1|0;if(d!==c){var
c=g;continue}break}}b[1]=b[1]&((1<<31)-1|0);var
f=bd<b[1]?b[1]-(1<<31)|0:b[1];return f}var
Z=cl([0,function(a,b){return e8(a,b)}]),an=cl([0,function(a,b){return e8(a,b)}]),ah=cl([0,function(a,b){return gs(a,b)}]),eh=e9(0,0),hG=[0,0];function
ei(a){return 2<a?ei((a+1|0)/2|0)*2|0:a}function
ej(a){hG[1]++;var
c=a.length-1,d=u((c*2|0)+2|0,eh);l(d,0,c);l(d,1,(v(ei(c),av)/8|0)-1|0);var
e=c-1|0,f=0;if(!(e<0)){var
b=f;for(;;){l(d,(b*2|0)+3|0,q(a,b));var
g=b+1|0;if(e!==b){var
b=g;continue}break}}return[0,hE,d,an[1],ah[1],0,0,Z[1],0]}function
cq(a,b){var
c=a[2].length-1,g=c<b?1:0;if(g){var
d=u(b,eh),h=a[2],e=0,f=0,j=0<=c?0<=f?(h.length-1-c|0)<f?0:0<=e?(d.length-1-c|0)<e?0:(so(h,f,d,e,c),1):0:0:0;if(!j)C(gI);a[2]=d;var
i=0}else
var
i=g;return i}var
ek=[0,0],hH=[0,0];function
cr(a){var
b=a[2].length-1;cq(a,b+1|0);return b}function
aS(a,b){try{var
d=i(an[22],b,a[3])}catch(f){if(f[1]===r){var
c=cr(a);a[3]=n(an[4],b,c,a[3]);a[4]=n(ah[4],c,1,a[4]);return c}throw f}return d}function
ct(a){return a===0?0:aN(a)}function
eq(a,b){try{var
d=i(Z[22],b,a[7])}catch(f){if(f[1]===r){var
c=a[1];a[1]=c+1|0;if(w(b,hX))a[7]=n(Z[4],b,c,a[7]);return c}throw f}return d}function
cv(a){return sx(a,0)?[0]:a}function
es(a,b){if(a)return a;var
c=e9(gZ,b[1]);c[0+1]=b[2];var
d=cp[1];c[1+1]=d;cp[1]=d+1|0;return c}function
bu(a){var
b=cr(a);if(0===(b%2|0))var
d=0;else
if((2+ar(q(a[2],1)*16|0,av)|0)<b)var
d=0;else{var
c=cr(a),d=1}if(!d)var
c=b;l(a[2],c,0);return c}function
et(a,ap){var
g=[0,0],aq=ap.length-1;for(;;){if(g[1]<aq){var
k=q(ap,g[1]),e=function(a){g[1]++;return q(ap,g[1])},n=e(0);if(typeof
n===m)switch(n){case
1:var
p=e(0),f=function(p){return function(a){return a[p+1]}}(p);break;case
2:var
r=e(0),b=e(0),f=function(r,b){return function(a){return a[r+1][b+1]}}(r,b);break;case
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
V=e(0),W=e(0),Y=e(0),f=function(V,W,Y){return function(a){return i(V,j(a[1][W+1],a),Y)}}(V,W,Y);break;case
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
ag=e(0),h=e(0);bu(a);var
f=function(ag,h){return function(a){return j(X(h,ag,0),h)}}(ag,h);break;case
21:var
ai=e(0),aj=e(0);bu(a);var
f=function(ai,aj){return function(a){var
b=a[aj+1];return j(X(b,ai,0),b)}}(ai,aj);break;case
22:var
ak=e(0),al=e(0),am=e(0);bu(a);var
f=function(ak,al,am){return function(a){var
b=a[al+1][am+1];return j(X(b,ak,0),b)}}(ak,al,am);break;case
23:var
an=e(0),ao=e(0);bu(a);var
f=function(an,ao){return function(a){var
b=j(a[1][ao+1],a);return j(X(b,an,0),b)}}(an,ao);break;default:var
o=e(0),f=function(o){return function(a){return o}}(o)}else
var
f=n;hH[1]++;if(i(ah[22],k,a[4])){cq(a,k+1|0);l(a[2],k,f)}else
a[6]=[0,[0,k,f],a[6]];g[1]++;continue}return 0}}function
cw(a,b,c){if(bK(c,h7))return b;var
d=c.getLen()-1|0;for(;;){if(0<=d){if(i(a,c,d)){var
d=d-1|0;continue}var
f=d+1|0,e=d;for(;;){if(0<=e){if(!i(a,c,e)){var
e=e-1|0;continue}var
g=s(c,e+1|0,(f-e|0)-1|0)}else
var
g=s(c,0,f);var
h=g;break}}else
var
h=s(c,0,1);return h}}function
cx(a,b,c){if(bK(c,h8))return b;var
d=c.getLen()-1|0;for(;;){if(0<=d){if(i(a,c,d)){var
d=d-1|0;continue}var
e=d;for(;;){if(0<=e){if(!i(a,c,e)){var
e=e-1|0;continue}var
f=e;for(;;){if(0<=f){if(i(a,c,f)){var
f=f-1|0;continue}var
h=s(c,0,f+1|0)}else
var
h=s(c,0,1);var
g=h;break}}else
var
g=b;var
j=g;break}}else
var
j=s(c,0,1);return j}}function
cz(a,b){return 47===a.safeGet(b)?1:0}function
eu(a){var
b=a.getLen()<1?1:0,c=b||(47!==a.safeGet(0)?1:0);return c}function
h$(a){var
c=eu(a);if(c){var
e=a.getLen()<2?1:0,d=e||w(s(a,0,2),ib);if(d){var
f=a.getLen()<3?1:0,b=f||w(s(a,0,3),ia)}else
var
b=d}else
var
b=c;return b}function
ic(a,b){var
c=b.getLen()<=a.getLen()?1:0,d=c?bK(s(a,a.getLen()-b.getLen()|0,b.getLen()),b):c;return d}try{var
sh=bI(sg),cA=sh}catch(f){if(f[1]!==r)throw f;var
cA=id}function
ev(a){var
d=a.getLen(),b=aP(d+20|0);E(b,39);var
e=d-1|0,f=0;if(!(e<0)){var
c=f;for(;;){if(39===a.safeGet(c))br(b,ie);else
E(b,a.safeGet(c));var
g=c+1|0;if(e!==c){var
c=g;continue}break}}E(b,39);return aQ(b)}function
ig(a){return cw(cz,cy,a)}function
ih(a){return cx(cz,cy,a)}function
ax(a,b){var
c=a.safeGet(b),d=47===c?1:0;if(d)var
e=d;else{var
f=92===c?1:0,e=f||(58===c?1:0)}return e}function
cC(a){var
e=a.getLen()<1?1:0,c=e||(47!==a.safeGet(0)?1:0);if(c){var
f=a.getLen()<1?1:0,d=f||(92!==a.safeGet(0)?1:0);if(d){var
g=a.getLen()<2?1:0,b=g||(58!==a.safeGet(1)?1:0)}else
var
b=d}else
var
b=c;return b}function
ew(a){var
c=cC(a);if(c){var
g=a.getLen()<2?1:0,d=g||w(s(a,0,2),io);if(d){var
h=a.getLen()<2?1:0,e=h||w(s(a,0,2),im);if(e){var
i=a.getLen()<3?1:0,f=i||w(s(a,0,3),il);if(f){var
j=a.getLen()<3?1:0,b=j||w(s(a,0,3),ik)}else
var
b=f}else
var
b=e}else
var
b=d}else
var
b=c;return b}function
ex(a,b){var
c=b.getLen()<=a.getLen()?1:0;if(c){var
e=s(a,a.getLen()-b.getLen()|0,b.getLen()),f=dW(b),d=bK(dW(e),f)}else
var
d=c;return d}try{var
sf=bI(se),ey=sf}catch(f){if(f[1]!==r)throw f;var
ey=ip}function
iq(h){var
i=h.getLen(),e=aP(i+20|0);E(e,34);function
g(a,b){var
c=b;for(;;){if(c===i)return E(e,34);var
f=h.safeGet(c);if(34===f)return a<50?d(1+a,0,c):A(d,[0,0,c]);if(92===f)return a<50?d(1+a,0,c):A(d,[0,0,c]);E(e,f);var
c=c+1|0;continue}}function
d(a,b,c){var
f=b,d=c;for(;;){if(d===i){E(e,34);return a<50?j(1+a,f):A(j,[0,f])}var
l=h.safeGet(d);if(34===l){k((2*f|0)+1|0);E(e,34);return a<50?g(1+a,d+1|0):A(g,[0,d+1|0])}if(92===l){var
f=f+1|0,d=d+1|0;continue}k(f);return a<50?g(1+a,d):A(g,[0,d])}}function
j(a,b){var
d=1;if(!(b<1)){var
c=d;for(;;){E(e,92);var
f=c+1|0;if(b!==c){var
c=f;continue}break}}return 0}function
a(b){return T(g(0,b))}function
b(b,c){return T(d(0,b,c))}function
k(b){return T(j(0,b))}a(0);return aQ(e)}function
ez(a){var
c=2<=a.getLen()?1:0;if(c){var
b=a.safeGet(0),g=91<=b?(b+fD|0)<0||25<(b+fD|0)?0:1:65<=b?1:0,d=g?1:0,e=d?58===a.safeGet(1)?1:0:d}else
var
e=c;if(e){var
f=s(a,2,a.getLen()-2|0);return[0,s(a,0,2),f]}return[0,ir,a]}function
is(a){var
b=ez(a),c=b[1];return h(c,cx(ax,cB,b[2]))}function
it(a){return cw(ax,cB,ez(a)[2])}function
iw(a){return cw(ax,cD,a)}function
ix(a){return cx(ax,cD,a)}if(w(cj,iy))if(w(cj,iz)){if(w(cj,iA))throw[0,D,iB];var
bv=[0,cB,ii,ij,ax,cC,ew,ex,ey,iq,it,is]}else
var
bv=[0,cy,h9,h_,cz,eu,h$,ic,cA,ev,ig,ih];else
var
bv=[0,cD,iu,iv,ax,cC,ew,ex,cA,ev,iw,ix];var
eA=[0,iE],iC=bv[11],iD=bv[3];a(iH,[0,eA,0,iG,iF]);ed(function(a){if(a[1]===eA){var
c=a[2],d=a[4],e=a[3];if(typeof
c===m)switch(c){case
1:var
b=iK;break;case
2:var
b=iL;break;case
3:var
b=iM;break;case
4:var
b=iN;break;case
5:var
b=iO;break;case
6:var
b=iP;break;case
7:var
b=iQ;break;case
8:var
b=iR;break;case
9:var
b=iS;break;case
10:var
b=iT;break;case
11:var
b=iU;break;case
12:var
b=iV;break;case
13:var
b=iW;break;case
14:var
b=iX;break;case
15:var
b=iY;break;case
16:var
b=iZ;break;case
17:var
b=i0;break;case
18:var
b=i1;break;case
19:var
b=i2;break;case
20:var
b=i3;break;case
21:var
b=i4;break;case
22:var
b=i5;break;case
23:var
b=i6;break;case
24:var
b=i7;break;case
25:var
b=i8;break;case
26:var
b=i9;break;case
27:var
b=i_;break;case
28:var
b=i$;break;case
29:var
b=ja;break;case
30:var
b=jb;break;case
31:var
b=jc;break;case
32:var
b=jd;break;case
33:var
b=je;break;case
34:var
b=jf;break;case
35:var
b=jg;break;case
36:var
b=jh;break;case
37:var
b=ji;break;case
38:var
b=jj;break;case
39:var
b=jk;break;case
40:var
b=jl;break;case
41:var
b=jm;break;case
42:var
b=jn;break;case
43:var
b=jo;break;case
44:var
b=jp;break;case
45:var
b=jq;break;case
46:var
b=jr;break;case
47:var
b=js;break;case
48:var
b=jt;break;case
49:var
b=ju;break;case
50:var
b=jv;break;case
51:var
b=jw;break;case
52:var
b=jx;break;case
53:var
b=jy;break;case
54:var
b=jz;break;case
55:var
b=jA;break;case
56:var
b=jB;break;case
57:var
b=jC;break;case
58:var
b=jD;break;case
59:var
b=jE;break;case
60:var
b=jF;break;case
61:var
b=jG;break;case
62:var
b=jH;break;case
63:var
b=jI;break;case
64:var
b=jJ;break;case
65:var
b=jK;break;case
66:var
b=jL;break;case
67:var
b=jM;break;default:var
b=iI}else{var
f=c[1],b=j(eb(jN),f)}return[0,n(eb(iJ),b,e,d)]}return 0});bL(jO);bL(jP);try{bL(sd)}catch(f){if(f[1]!==aL)throw f}try{bL(sc)}catch(f){if(f[1]!==aL)throw f}eg(0,7);function
eB(a){return tR(a)}af(32,p);var
jQ=6,jR=0,jW=y(b2),jX=0,jY=p;if(!(p<0)){var
a7=jX;for(;;){jW.safeSet(a7,dV(ch(a7)));var
sb=a7+1|0;if(jY!==a7){var
a7=sb;continue}break}}var
cE=af(32,0);cE.safeSet(10>>>3,ch(cE.safeGet(10>>>3)|1<<(10&7)));var
jS=y(32),jT=0,jU=31;if(!(31<0)){var
aW=jT;for(;;){jS.safeSet(aW,ch(cE.safeGet(aW)^p));var
jV=aW+1|0;if(jU!==aW){var
aW=jV;continue}break}}var
ay=[0,0],az=[0,0],eC=[0,0];function
F(a){return ay[1]}function
eD(a){return az[1]}function
N(a,b,c){return 0===a[2][0]?b?spoc_cuda_flush(a[1],a,b[1]):spoc_cuda_flush_all(a[1],a):b?e_(a[1],b[1]):e_(a[1],0)}var
eE=[3,jQ],cF=[0,0];function
aA(e,b,c){cF[1]++;switch(e[0]){case
7:case
8:throw[0,D,jZ];case
6:var
g=e[1],m=cF[1],n=e$(0),o=u(eD(0)+1|0,n),p=fa(0),q=u(F(0)+1|0,p),f=[0,-1,[1,[0,spoc_create_custom(g,c),g]],q,o,c,0,e,0,0,m,0];break;default:var
h=e[1],i=cF[1],j=e$(0),k=u(eD(0)+1|0,j),l=fa(0),f=[0,-1,[0,ss(h,jR,[0,c])],u(F(0)+1|0,l),k,c,0,e,0,0,i,0]}if(b){var
d=b[1],a=function(a){{if(0===d[2][0])return 6===e[0]?spoc_cuda_custom_alloc_vect(f,d[1][8],d[1]):spoc_cuda_alloc_vect(f,d[1][8],d[1]);{var
b=d[1],c=F(0);return fb(f,d[1][8]-c|0,b)}}};try{a(0)}catch(f){a9(0);a(0)}f[6]=[0,d]}return f}function
R(a){return a[5]}function
aX(a){return a[6]}function
bw(a){return a[8]}function
bx(a){return a[7]}function
W(a){return a[2]}function
by(a,b,c){a[1]=b;a[6]=c;return 0}function
cG(a,b,c){return dx<=b?q(a[3],c):q(a[4],c)}function
cH(a,b){var
e=b[3].length-1-2|0,g=0;if(!(e<0)){var
d=g;for(;;){l(b[3],d,q(a[3],d));var
j=d+1|0;if(e!==d){var
d=j;continue}break}}var
f=b[4].length-1-2|0,h=0;if(!(f<0)){var
c=h;for(;;){l(b[4],c,q(a[4],c));var
i=c+1|0;if(f!==c){var
c=i;continue}break}}return 0}function
bz(a,b){b[8]=a[8];return 0}var
ao=[0,j5];a(kc,[0,[0,j0]]);a(kd,[0,[0,j1]]);a(ke,[0,[0,j2]]);a(kf,[0,[0,j3]]);a(kg,[0,[0,j4]]);a(kh,[0,ao]);a(ki,[0,[0,j6]]);a(kj,[0,[0,j7]]);a(kk,[0,[0,j8]]);a(kl,[0,[0,j_]]);a(km,[0,[0,j$]]);a(kn,[0,[0,ka]]);a(ko,[0,[0,kb]]);a(kp,[0,[0,j9]]);var
cI=[0,kx];a(kN,[0,[0,kq]]);a(kO,[0,[0,kr]]);a(kP,[0,[0,ks]]);a(kQ,[0,[0,kt]]);a(kR,[0,[0,ku]]);a(kS,[0,[0,kv]]);a(kT,[0,[0,kw]]);a(kU,[0,cI]);a(kV,[0,[0,ky]]);a(kW,[0,[0,kz]]);a(kX,[0,[0,kA]]);a(kY,[0,[0,kB]]);a(kZ,[0,[0,kC]]);a(k0,[0,[0,kD]]);a(k1,[0,[0,kE]]);a(k2,[0,[0,kF]]);a(k3,[0,[0,kG]]);a(k4,[0,[0,kH]]);a(k5,[0,[0,kI]]);a(k6,[0,[0,kJ]]);a(k7,[0,[0,kK]]);a(k8,[0,[0,kL]]);a(k9,[0,[0,kM]]);var
bA=1,eF=0;function
aY(a,b,c){var
d=a[2];if(0===d[0])var
f=su(d[1],b,c);else{var
e=d[1],f=n(e[2][4],e[1],b,c)}return f}function
aZ(a,b){var
c=a[2];if(0===c[0])var
e=st(c[1],b);else{var
d=c[1],e=i(d[2][3],d[1],b)}return e}function
eG(a,b){N(a,0,0);eL(b,0,0);return N(a,0,0)}function
_(a,b,c){var
f=a,d=b;for(;;){if(eF)return aY(f,d,c);var
n=d<0?1:0,o=n||(R(f)<=d?1:0);if(o)throw[0,bm,k_];if(bA){var
i=aX(f);if(typeof
i!==m)eG(i[1],f)}var
j=bw(f);if(j){var
e=j[1];if(1===e[1]){var
k=e[4],g=e[3],l=e[2];return 0===k?aY(e[5],l+d|0,c):aY(e[5],(l+v(ar(d,g),k+g|0)|0)+aE(d,g)|0,c)}var
h=e[3],f=e[5],d=(e[2]+v(ar(d,h),e[4]+h|0)|0)+aE(d,h)|0;continue}return aY(f,d,c)}}function
$(a,b){var
e=a,c=b;for(;;){if(eF)return aZ(e,c);var
l=c<0?1:0,n=l||(R(e)<=c?1:0);if(n)throw[0,bm,k$];if(bA){var
h=aX(e);if(typeof
h!==m)eG(h[1],e)}var
i=bw(e);if(i){var
d=i[1];if(1===d[1]){var
j=d[4],f=d[3],k=d[2];return 0===j?aZ(d[5],k+c|0):aZ(d[5],(k+v(ar(c,f),j+f|0)|0)+aE(c,f)|0)}var
g=d[3],e=d[5],c=(d[2]+v(ar(c,g),d[4]+g|0)|0)+aE(c,g)|0;continue}return aZ(e,c)}}function
eH(a){if(a[8]){var
b=aA(a[7],0,a[5]);b[1]=a[1];b[6]=a[6];cH(a,b);var
c=b}else
var
c=a;return c}function
eI(d,b,c){{if(0===c[2][0]){var
a=function(a){return 0===W(d)[0]?spoc_cuda_cpu_to_device(d,c[1][8],c[1],c[3],b):spoc_cuda_custom_cpu_to_device(d,c[1][8],c[1],b)};try{var
f=a(0)}catch(f){if(f[1]===ao){try{N(c,0,0);var
g=a(0)}catch(f){a9(0);return a(0)}return g}throw f}return f}var
e=function(a){{if(0===W(d)[0]){var
e=c[1],f=F(0);return tK(d,c[1][8]-f|0,e,b)}var
g=c[1],h=F(0);return spoc_opencl_custom_cpu_to_device(d,c[1][8]-h|0,g,b)}};try{var
i=e(0)}catch(f){try{N(c,0,0);var
h=e(0)}catch(f){a9(0);return e(0)}return h}return i}}function
eJ(d,b,c){{if(0===c[2][0]){var
a=function(a){return 0===W(d)[0]?spoc_cuda_device_to_cpu(d,c[1][8],c[1],c,b):spoc_cuda_custom_device_to_cpu(d,c[1][8],c[1],b)};try{var
f=a(0)}catch(f){if(f[1]===ao){try{N(c,0,0);var
g=a(0)}catch(f){a9(0);return a(0)}return g}throw f}return f}var
e=function(a){{if(0===W(d)[0]){var
e=c[2],f=c[1],g=F(0);return tL(d,c[1][8]-g|0,f,e,b)}var
h=c[2],i=c[1],j=F(0);return spoc_opencl_custom_device_to_cpu(d,c[1][8]-j|0,i,h,b)}};try{var
i=e(0)}catch(f){try{N(c,0,0);var
h=e(0)}catch(f){a9(0);return e(0)}return h}return i}}function
a0(a,b,c,d,e,f,g,h){{if(0===d[2][0])return 0===W(a)[0]?spoc_cuda_part_cpu_to_device_b(a,b,d[1][8],d[1],d[3],c,e,f,g,h):spoc_cuda_custom_part_cpu_to_device_b(a,b,d[1][8],d[1],d[3],c,e,f,g,h);{if(0===W(a)[0]){var
i=d[3],j=d[1],k=F(0);return spoc_opencl_part_cpu_to_device_b(a,b,d[1][8]-k|0,j,i,c,e,f,g,h)}var
l=d[3],m=d[1],n=F(0);return spoc_opencl_custom_part_cpu_to_device_b(a,b,d[1][8]-n|0,m,l,c,e,f,g,h)}}}function
a1(a,b,c,d,e,f,g,h){{if(0===d[2][0])return 0===W(a)[0]?spoc_cuda_part_device_to_cpu_b(a,b,d[1][8],d[1],d[3],c,e,f,g,h):spoc_cuda_custom_part_device_to_cpu_b(a,b,d[1][8],d[1],d[3],c,e,f,g,h);{if(0===W(a)[0]){var
i=d[3],j=d[1],k=F(0);return spoc_opencl_part_device_to_cpu_b(a,b,d[1][8]-k|0,j,i,c,e,f,g,h)}var
l=d[3],m=d[1],n=F(0);return spoc_opencl_custom_part_device_to_cpu_b(a,b,d[1][8]-n|0,m,l,c,e,f,g,h)}}}function
eK(a,b,c){var
q=b;for(;;){var
e=q?q[1]:0,r=aX(a);if(typeof
r===m){by(a,c[1][8],[1,c]);try{cJ(a,c)}catch(f){if(f[1]!==ao)f[1]===cI;try{N(c,[0,e],0);cJ(a,c)}catch(f){if(f[1]!==ao)if(f[1]!==cI)throw f;N(c,0,0);sB(0);cJ(a,c)}}var
z=bw(a);if(z){var
k=z[1];if(1===k[1]){var
l=k[5],s=k[4],g=k[3],n=k[2];if(0===g)a0(l,a,e,c,0,0,n,R(a));else
if(d<g){var
i=0,o=R(a);for(;;){if(g<o){a0(l,a,e,c,v(i,g+s|0),v(i,g),n,g);var
i=i+1|0,o=o-g|0;continue}if(0<o)a0(l,a,e,c,v(i,g+s|0),v(i,g),n,o);break}}else{var
f=0,j=0,h=R(a);for(;;){if(d<h){var
w=aA(bx(a),0,d);bz(a,w);var
A=f+gh|0;if(!(A<f)){var
t=f;for(;;){_(w,t,$(a,f));var
H=t+1|0;if(A!==t){var
t=H;continue}break}}a0(l,w,e,c,v(j,d+s|0),j*d|0,n,d);var
f=f+d|0,j=j+1|0,h=h+fU|0;continue}if(0<h){var
x=aA(bx(a),0,h),B=(f+h|0)-1|0;if(!(B<f)){var
u=f;for(;;){_(x,u,$(a,f));var
I=u+1|0;if(B!==u){var
u=I;continue}break}}bz(a,x);a0(l,x,e,c,v(j,d+s|0),j*d|0,n,h)}break}}}else{var
y=eH(a),C=R(a)-1|0,J=0;if(!(C<0)){var
p=J;for(;;){aY(y,p,$(a,p));var
K=p+1|0;if(C!==p){var
p=K;continue}break}}eI(y,e,c);cH(y,a)}}else
eI(a,e,c);return by(a,c[1][8],[0,c])}else{if(0===r[0]){var
D=r[1],E=c3(D,c);if(E){eL(a,[0,e],0);N(D,0,0);var
q=[0,e];continue}return E}var
F=r[1],G=c3(F,c);if(G){N(F,0,0);var
q=[0,e];continue}return G}}}function
cJ(a,b){{if(0===b[2][0])return 0===W(a)[0]?spoc_cuda_alloc_vect(a,b[1][8],b[1]):spoc_cuda_custom_alloc_vect(a,b[1][8],b[1]);{if(0===W(a)[0]){var
c=b[1],d=F(0);return fb(a,b[1][8]-d|0,c)}var
e=b[1],f=F(0);return spoc_opencl_custom_alloc_vect(a,b[1][8]-f|0,e)}}}function
eL(a,b,c){var
x=b;for(;;){var
g=x?x[1]:0,r=aX(a);if(typeof
r===m)return 0;else{if(0===r[0]){var
e=r[1];by(a,e[1][8],[1,e]);var
A=bw(a);if(A){var
l=A[1];if(1===l[1]){var
n=l[5],s=l[4],f=l[3],o=l[2];if(0===f)a1(n,a,g,e,0,0,o,R(a));else
if(d<f){var
j=0,p=R(a);for(;;){if(f<p){a1(n,a,g,e,v(j,f+s|0),v(j,f),o,f);var
j=j+1|0,p=p-f|0;continue}if(0<p)a1(n,a,g,e,v(j,f+s|0),v(j,f),o,p);break}}else{var
k=0,i=R(a),h=0;for(;;){if(d<i){var
y=aA(bx(a),0,d);bz(a,y);var
B=h+gh|0;if(!(B<h)){var
t=h;for(;;){_(y,t,$(a,h));var
E=t+1|0;if(B!==t){var
t=E;continue}break}}a1(n,y,g,e,v(k,d+s|0),k*d|0,o,d);var
k=k+1|0,i=i+fU|0;continue}if(0<i){var
z=aA(bx(a),0,i),C=(h+i|0)-1|0;if(!(C<h)){var
u=h;for(;;){_(z,u,$(a,h));var
F=u+1|0;if(C!==u){var
u=F;continue}break}}bz(a,z);a1(n,z,g,e,v(k,d+s|0),k*d|0,o,i)}break}}}else{var
w=eH(a);cH(w,a);eJ(w,g,e);var
D=R(w)-1|0,G=0;if(!(D<0)){var
q=G;for(;;){_(a,q,aZ(w,q));var
H=q+1|0;if(D!==q){var
q=H;continue}break}}}}else
eJ(a,g,e);return by(a,e[1][8],0)}N(r[1],0,0);var
x=[0,g];continue}}}var
le=[0,ld],lg=[0,lf];function
bB(a,b){var
p=q(gY,0),r=h(iD,h(a,b)),f=dM(h(iC(p),r));try{var
n=eN,g=eN;a:for(;;){if(1){var
k=function(a,b,c){var
e=b,d=c;for(;;){if(d){var
g=d[1],f=g.getLen(),h=d[2];a8(g,0,a,e-f|0,f);var
e=e-f|0,d=h;continue}return a}},d=0,e=0;for(;;){var
c=s2(f);if(0===c){if(!d)throw[0,bn];var
j=k(y(e),e,d)}else{if(!(0<c)){var
m=y(-c|0);c4(f,m,0,-c|0);var
d=[0,m,d],e=e-c|0;continue}var
i=y(c-1|0);c4(f,i,0,c-1|0);s1(f);if(d){var
l=(e+c|0)-1|0,j=k(y(l),l,[0,i,d])}else
var
j=i}var
g=h(g,h(j,lh)),n=g;continue a}}var
o=g;break}}catch(f){if(f[1]!==bn)throw f;var
o=n}dO(f);return o}var
eO=[0,li],cK=[],lj=0,lk=0;tu(cK,[0,0,function(f){var
k=eq(f,ll),e=cv(la),d=e.length-1,o=eM.length-1,a=u(d+o|0,0),p=d-1|0,v=0;if(!(p<0)){var
c=v;for(;;){l(a,c,aS(f,q(e,c)));var
y=c+1|0;if(p!==c){var
c=y;continue}break}}var
s=o-1|0,w=0;if(!(s<0)){var
b=w;for(;;){l(a,b+d|0,eq(f,q(eM,b)));var
x=b+1|0;if(s!==b){var
b=x;continue}break}}var
t=a[10],m=a[12],h=a[15],i=a[16],j=a[17],g=a[18],z=a[1],A=a[2],B=a[3],C=a[4],D=a[5],E=a[7],F=a[8],G=a[9],H=a[11],I=a[14];function
J(a,b,c,d,e,f){var
h=d?d[1]:d;n(a[1][m+1],a,[0,h],f);var
i=bt(a[g+1],f);return fc(a[1][t+1],a,b,[0,c[1],c[2]],e,f,i)}function
K(a,b,c,d,e){try{var
f=bt(a[g+1],e),h=f}catch(f){if(f[1]!==r)throw f;try{n(a[1][m+1],a,lm,e)}catch(f){throw f}var
h=bt(a[g+1],e)}return fc(a[1][t+1],a,b,[0,c[1],c[2]],d,e,h)}function
L(a,b,c){var
y=b?b[1]:b;try{bt(a[g+1],c);var
f=0}catch(f){if(f[1]===r){if(0===c[2][0]){var
z=a[i+1];if(!z)throw[0,eO,c];var
A=z[1],H=y?tA(A,a[h+1],c[1]):tz(A,a[h+1],c[1]),B=H}else{var
D=a[j+1];if(!D)throw[0,eO,c];var
E=D[1],I=y?tB(E,a[h+1],c[1]):tJ(E,a[h+1],c[1]),B=I}var
d=a[g+1],w=co(d,c);l(d[2],w,[0,c,B,q(d[2],w)]);d[1]=d[1]+1|0;var
x=d[2].length-1<<1<d[1]?1:0;if(x){var
m=d[2],n=m.length-1,o=n*2|0,p=o<ck?1:0;if(p){var
k=u(o,0);d[2]=k;var
s=function(a){if(a){var
b=a[1],e=a[2];s(a[3]);var
c=co(d,b);return l(k,c,[0,b,e,q(k,c)])}return 0},t=n-1|0,F=0;if(!(t<0)){var
e=F;for(;;){s(q(m,e));var
G=e+1|0;if(t!==e){var
e=G;continue}break}}var
v=0}else
var
v=p;var
C=v}else
var
C=x;return C}throw f}return f}function
M(a,b){try{var
f=[0,bB(a[k+1],lo),0],c=f}catch(f){var
c=0}a[i+1]=c;try{var
e=[0,bB(a[k+1],ln),0],d=e}catch(f){var
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
c=h;continue}break}}return 0}et(f,[0,G,function(a,b){return a[g+1]},C,R,F,Q,A,P,E,O,z,N,D,M,m,L,B,K,H,J]);return function(a,b,c,d){var
e=es(b,f);e[k+1]=c;e[I+1]=c;e[h+1]=d;try{var
o=[0,bB(c,lq),0],l=o}catch(f){var
l=0}e[i+1]=l;try{var
n=[0,bB(c,lp),0],m=n}catch(f){var
m=0}e[j+1]=m;e[g+1]=eg(0,8);return e}},lk,lj]);fd(0);fd(0);function
cL(a){function
e(a,b){var
d=a-1|0,e=0;if(!(d<0)){var
c=e;for(;;){d$(ls);var
f=c+1|0;if(d!==c){var
c=f;continue}break}}return j(d$(lr),b)}function
f(a,b){var
c=a,d=b;for(;;)if(typeof
d===m)return 0===d?e(c,lt):e(c,lu);else
switch(d[0]){case
0:e(c,lv);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
1:e(c,lw);var
c=c+1|0,d=d[1];continue;case
2:e(c,lx);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
3:e(c,ly);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
4:e(c,lz);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
5:e(c,lA);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
6:e(c,lB);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
7:e(c,lC);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
8:e(c,lD);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
9:e(c,lE);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
10:e(c,lF);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
11:return e(c,h(lG,d[1]));case
12:return e(c,h(lH,d[1]));case
13:return e(c,h(lI,k(d[1])));case
14:return e(c,h(lJ,k(d[1])));case
15:return e(c,h(lK,k(d[1])));case
16:return e(c,h(lL,k(d[1])));case
17:return e(c,h(lM,k(d[1])));case
18:return e(c,lN);case
19:return e(c,lO);case
20:return e(c,lP);case
21:return e(c,lQ);case
22:return e(c,lR);case
23:return e(c,h(lS,k(d[2])));case
24:e(c,lT);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
25:e(c,lU);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
26:e(c,lV);var
c=c+1|0,d=d[1];continue;case
27:e(c,lW);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
28:e(c,lX);var
c=c+1|0,d=d[1];continue;case
29:e(c,lY);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
30:e(c,lZ);f(c+1|0,d[1]);f(c+1|0,d[2]);var
c=c+1|0,d=d[3];continue;case
31:return e(c,l0);case
32:var
g=h(l1,k(d[2]));return e(c,h(l2,h(d[1],g)));case
33:return e(c,h(l3,k(d[1])));case
36:e(c,l5);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
37:e(c,l6);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
38:e(c,l7);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
39:e(c,l8);f(c+1|0,d[1]);f(c+1|0,d[2]);var
c=c+1|0,d=d[3];continue;case
40:e(c,l9);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
41:e(c,l_);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
42:e(c,l$);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
43:e(c,ma);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
44:e(c,mb);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
45:e(c,mc);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
46:e(c,md);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
47:e(c,me);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
48:e(c,mf);f(c+1|0,d[1]);f(c+1|0,d[2]);f(c+1|0,d[3]);var
c=c+1|0,d=d[4];continue;case
49:e(c,mg);f(c+1|0,d[1]);var
c=c+1|0,d=d[2];continue;case
50:e(c,mh);f(c+1|0,d[1]);var
i=d[2],j=c+1|0;return dQ(function(a){return f(j,a)},i);case
51:return e(c,mi);case
52:return e(c,mj);default:return e(c,h(l4,L(d[1])))}}return f(0,a)}function
G(a){return af(a,32)}var
a2=[0,mk];function
a$(a,b,c){var
d=c;for(;;)if(typeof
d===m)return mm;else
switch(d[0]){case
18:case
19:var
U=h(m1,h(e(b,d[2]),m0));return h(m2,h(k(d[1]),U));case
27:case
38:var
ac=d[1],ad=h(nl,h(e(b,d[2]),nk));return h(e(b,ac),ad);case
0:var
g=d[2],C=e(b,d[1]);if(typeof
g===m)var
r=0;else
if(25===g[0]){var
t=e(b,g),r=1}else
var
r=0;if(!r){var
E=h(mn,G(b)),t=h(e(b,g),E)}return h(h(C,t),mo);case
1:var
F=h(e(b,d[1]),mp),H=w(a2[1][1],mq)?h(a2[1][1],mr):mt;return h(ms,h(H,F));case
2:var
I=h(mv,h(S(b,d[2]),mu));return h(mw,h(S(b,d[1]),I));case
3:var
J=h(my,h(ai(b,d[2]),mx));return h(mz,h(ai(b,d[1]),J));case
4:var
K=h(mB,h(S(b,d[2]),mA));return h(mC,h(S(b,d[1]),K));case
5:var
M=h(mE,h(ai(b,d[2]),mD));return h(mF,h(ai(b,d[1]),M));case
6:var
N=h(mH,h(S(b,d[2]),mG));return h(mI,h(S(b,d[1]),N));case
7:var
O=h(mK,h(ai(b,d[2]),mJ));return h(mL,h(ai(b,d[1]),O));case
8:var
P=h(mN,h(S(b,d[2]),mM));return h(mO,h(S(b,d[1]),P));case
9:var
R=h(mQ,h(ai(b,d[2]),mP));return h(mR,h(ai(b,d[1]),R));case
10:var
T=h(mT,h(S(b,d[2]),mS));return h(mU,h(S(b,d[1]),T));case
13:return h(mV,k(d[1]));case
14:return h(mW,k(d[1]));case
15:throw[0,D,mX];case
16:return h(mY,k(d[1]));case
17:return h(mZ,k(d[1]));case
20:var
V=h(m4,h(e(b,d[2]),m3));return h(m5,h(k(d[1]),V));case
21:var
W=h(m7,h(e(b,d[2]),m6));return h(m8,h(k(d[1]),W));case
22:var
X=h(m_,h(e(b,d[2]),m9));return h(m$,h(k(d[1]),X));case
23:var
Y=h(na,k(d[2])),u=d[1];if(typeof
u===m)var
f=0;else
switch(u[0]){case
33:var
o=nc,f=1;break;case
34:var
o=nd,f=1;break;case
35:var
o=ne,f=1;break;default:var
f=0}if(f)return h(o,Y);throw[0,D,nb];case
24:var
i=d[2],v=d[1];if(typeof
i===m){if(0===i){var
d=v;continue}}else
if(24===i[0]){var
Z=h(ng,e(b,i));return h(e(b,v),Z)}return Q(nf);case
25:var
_=e(b,d[2]),$=h(nh,h(G(b),_));return h(e(b,d[1]),$);case
26:var
aa=e(b,d[1]),ab=w(a2[1][2],ni)?a2[1][2]:nj;return h(ab,aa);case
28:var
d=d[1];continue;case
29:var
ae=h(nn,h(e(b,d[2]),nm));return h(e(b,d[1]),ae);case
30:var
l=d[2],af=e(b,d[3]),ag=h(no,h(G(b),af));if(typeof
l===m)var
s=0;else
if(31===l[0]){var
x=h(ml(l[1]),nq),s=1}else
var
s=0;if(!s)var
x=e(b,l);var
ah=h(np,h(x,ag));return h(e(b,d[1]),ah);case
31:return a<50?a_(1+a,d[1]):A(a_,[0,d[1]]);case
33:return k(d[1]);case
34:return h(L(d[1]),nr);case
35:return L(d[1]);case
36:var
aj=h(nt,h(e(b,d[2]),ns));return h(e(b,d[1]),aj);case
37:var
ak=h(nv,h(G(b),nu)),al=h(e(b,d[2]),ak),am=h(nw,h(G(b),al)),an=h(nx,h(e(b,d[1]),am));return h(G(b),an);case
39:var
ao=h(ny,G(b)),ap=h(e(b+2|0,d[3]),ao),aq=h(nz,h(G(b+2|0),ap)),ar=h(nA,h(G(b),aq)),as=h(e(b+2|0,d[2]),ar),at=h(nB,h(G(b+2|0),as));return h(nC,h(e(b,d[1]),at));case
40:var
au=h(nD,G(b)),av=h(e(b+2|0,d[2]),au),aw=h(nE,h(G(b+2|0),av));return h(nF,h(e(b,d[1]),aw));case
41:var
ax=h(nG,e(b,d[2]));return h(e(b,d[1]),ax);case
42:var
ay=h(nH,e(b,d[2]));return h(e(b,d[1]),ay);case
43:var
az=h(nI,e(b,d[2]));return h(e(b,d[1]),az);case
44:var
aA=h(nJ,e(b,d[2]));return h(e(b,d[1]),aA);case
45:var
aB=h(nK,e(b,d[2]));return h(e(b,d[1]),aB);case
46:var
aC=h(nL,e(b,d[2]));return h(e(b,d[1]),aC);case
47:var
aD=h(nM,e(b,d[2]));return h(e(b,d[1]),aD);case
48:var
p=e(b,d[1]),aE=e(b,d[2]),aF=e(b,d[3]),aG=h(e(b+2|0,d[4]),nN);return h(nT,h(p,h(nS,h(aE,h(nR,h(p,h(nQ,h(aF,h(nP,h(p,h(nO,h(G(b+2|0),aG))))))))))));case
49:var
aH=e(b,d[1]),aI=h(e(b+2|0,d[2]),nU);return h(nW,h(aH,h(nV,h(G(b+2|0),aI))));case
50:var
y=d[2],n=d[1],z=e(b,n),q=function(a){if(a){var
c=a[2],d=a[1];if(c){var
f=h(nX,q(c));return h(e(b,d),f)}return e(b,d)}throw[0,D,nY]};if(typeof
n!==m)if(31===n[0]){var
B=n[1];if(!w(B[1],n1))if(!w(B[2],n2))return h(z,h(n4,h(q(aN(y)),n3)))}return h(z,h(n0,h(q(aN(y)),nZ)));case
51:return k(j(d[1],0));case
52:return h(L(j(d[1],0)),n5);default:return d[1]}}function
sm(a,b,c){if(typeof
c!==m)switch(c[0]){case
2:case
4:case
6:case
8:case
10:case
50:return a<50?a$(1+a,b,c):A(a$,[0,b,c]);case
32:return c[1];case
33:return k(c[1]);case
36:var
d=h(n7,h(S(b,c[2]),n6));return h(e(b,c[1]),d);case
51:return k(j(c[1],0));default:}return a<50?c5(1+a,b,c):A(c5,[0,b,c])}function
c5(a,b,c){if(typeof
c!==m)switch(c[0]){case
3:case
5:case
7:case
9:case
29:case
50:return a<50?a$(1+a,b,c):A(a$,[0,b,c]);case
16:return h(n9,k(c[1]));case
31:return a<50?a_(1+a,c[1]):A(a_,[0,c[1]]);case
32:return c[1];case
34:return h(L(c[1]),n_);case
35:return h(n$,L(c[1]));case
36:var
d=h(ob,h(S(b,c[2]),oa));return h(e(b,c[1]),d);case
52:return h(L(j(c[1],0)),oc);default:}cL(c);return Q(n8)}function
a_(a,b){return b[1]}function
e(b,c){return T(a$(0,b,c))}function
S(b,c){return T(sm(0,b,c))}function
ai(b,c){return T(c5(0,b,c))}function
ml(b){return T(a_(0,b))}function
x(a){return af(a,32)}var
a3=[0,od];function
bb(a,b,c){var
d=c;for(;;)if(typeof
d===m)return of;else
switch(d[0]){case
18:case
19:var
U=h(oC,h(f(b,d[2]),oB));return h(oD,h(k(d[1]),U));case
27:case
38:var
ac=d[1],ad=h(oX,O(b,d[2]));return h(f(b,ac),ad);case
0:var
g=d[2],E=f(b,d[1]);if(typeof
g===m)var
r=0;else
if(25===g[0]){var
t=f(b,g),r=1}else
var
r=0;if(!r){var
F=h(og,x(b)),t=h(f(b,g),F)}return h(h(E,t),oh);case
1:var
G=h(f(b,d[1]),oi),H=w(a3[1][1],oj)?h(a3[1][1],ok):om;return h(ol,h(H,G));case
2:var
I=h(on,O(b,d[2]));return h(O(b,d[1]),I);case
3:var
J=h(oo,aj(b,d[2]));return h(aj(b,d[1]),J);case
4:var
K=h(op,O(b,d[2]));return h(O(b,d[1]),K);case
5:var
M=h(oq,aj(b,d[2]));return h(aj(b,d[1]),M);case
6:var
N=h(or,O(b,d[2]));return h(O(b,d[1]),N);case
7:var
P=h(os,aj(b,d[2]));return h(aj(b,d[1]),P);case
8:var
R=h(ot,O(b,d[2]));return h(O(b,d[1]),R);case
9:var
S=h(ou,aj(b,d[2]));return h(aj(b,d[1]),S);case
10:var
T=h(ov,O(b,d[2]));return h(O(b,d[1]),T);case
13:return h(ow,k(d[1]));case
14:return h(ox,k(d[1]));case
15:throw[0,D,oy];case
16:return h(oz,k(d[1]));case
17:return h(oA,k(d[1]));case
20:var
V=h(oF,h(f(b,d[2]),oE));return h(oG,h(k(d[1]),V));case
21:var
W=h(oI,h(f(b,d[2]),oH));return h(oJ,h(k(d[1]),W));case
22:var
X=h(oL,h(f(b,d[2]),oK));return h(oM,h(k(d[1]),X));case
23:var
Y=h(oN,k(d[2])),u=d[1];if(typeof
u===m)var
e=0;else
switch(u[0]){case
33:var
o=oP,e=1;break;case
34:var
o=oQ,e=1;break;case
35:var
o=oR,e=1;break;default:var
e=0}if(e)return h(o,Y);throw[0,D,oO];case
24:var
i=d[2],v=d[1];if(typeof
i===m){if(0===i){var
d=v;continue}}else
if(24===i[0]){var
Z=h(oT,f(b,i));return h(f(b,v),Z)}return Q(oS);case
25:var
_=f(b,d[2]),$=h(oU,h(x(b),_));return h(f(b,d[1]),$);case
26:var
aa=f(b,d[1]),ab=w(a3[1][2],oV)?a3[1][2]:oW;return h(ab,aa);case
28:var
d=d[1];continue;case
29:var
ae=h(oZ,h(f(b,d[2]),oY));return h(f(b,d[1]),ae);case
30:var
l=d[2],af=f(b,d[3]),ag=h(o0,h(x(b),af));if(typeof
l===m)var
s=0;else
if(31===l[0]){var
y=oe(l[1]),s=1}else
var
s=0;if(!s)var
y=f(b,l);var
ah=h(o1,h(y,ag));return h(f(b,d[1]),ah);case
31:return a<50?ba(1+a,d[1]):A(ba,[0,d[1]]);case
33:return k(d[1]);case
34:return h(L(d[1]),o2);case
35:return L(d[1]);case
36:var
ai=h(o4,h(f(b,d[2]),o3));return h(f(b,d[1]),ai);case
37:var
ak=h(o6,h(x(b),o5)),al=h(f(b,d[2]),ak),am=h(o7,h(x(b),al)),an=h(o8,h(f(b,d[1]),am));return h(x(b),an);case
39:var
ao=h(o9,x(b)),ap=h(f(b+2|0,d[3]),ao),aq=h(o_,h(x(b+2|0),ap)),ar=h(o$,h(x(b),aq)),as=h(f(b+2|0,d[2]),ar),at=h(pa,h(x(b+2|0),as));return h(pb,h(f(b,d[1]),at));case
40:var
au=h(pc,x(b)),av=h(pd,h(x(b),au)),aw=h(f(b+2|0,d[2]),av),ax=h(pe,h(x(b+2|0),aw)),ay=h(pf,h(x(b),ax));return h(pg,h(f(b,d[1]),ay));case
41:var
az=h(ph,f(b,d[2]));return h(f(b,d[1]),az);case
42:var
aA=h(pi,f(b,d[2]));return h(f(b,d[1]),aA);case
43:var
aB=h(pj,f(b,d[2]));return h(f(b,d[1]),aB);case
44:var
aC=h(pk,f(b,d[2]));return h(f(b,d[1]),aC);case
45:var
aD=h(pl,f(b,d[2]));return h(f(b,d[1]),aD);case
46:var
aE=h(pm,f(b,d[2]));return h(f(b,d[1]),aE);case
47:var
aF=h(pn,f(b,d[2]));return h(f(b,d[1]),aF);case
48:var
p=f(b,d[1]),aG=f(b,d[2]),aH=f(b,d[3]),aI=h(f(b+2|0,d[4]),po);return h(pu,h(p,h(pt,h(aG,h(ps,h(p,h(pr,h(aH,h(pq,h(p,h(pp,h(x(b+2|0),aI))))))))))));case
49:var
aJ=f(b,d[1]),aK=h(f(b+2|0,d[2]),pv);return h(px,h(aJ,h(pw,h(x(b+2|0),aK))));case
50:var
z=d[2],n=d[1],B=f(b,n),q=function(a){if(a){var
c=a[2],d=a[1];if(c){var
e=h(py,q(c));return h(f(b,d),e)}return f(b,d)}throw[0,D,pz]};if(typeof
n!==m)if(31===n[0]){var
C=n[1];if(!w(C[1],pC))if(!w(C[2],pD))return h(B,h(pF,h(q(aN(z)),pE)))}return h(B,h(pB,h(q(aN(z)),pA)));case
51:return k(j(d[1],0));case
52:return h(L(j(d[1],0)),pG);default:return d[1]}}function
sn(a,b,c){if(typeof
c!==m)switch(c[0]){case
2:case
4:case
6:case
8:case
10:case
50:return a<50?bb(1+a,b,c):A(bb,[0,b,c]);case
32:return c[1];case
33:return k(c[1]);case
36:var
d=h(pI,h(O(b,c[2]),pH));return h(f(b,c[1]),d);case
51:return k(j(c[1],0));default:}return a<50?c6(1+a,b,c):A(c6,[0,b,c])}function
c6(a,b,c){if(typeof
c!==m)switch(c[0]){case
3:case
5:case
7:case
9:case
50:return a<50?bb(1+a,b,c):A(bb,[0,b,c]);case
16:return h(pK,k(c[1]));case
31:return a<50?ba(1+a,c[1]):A(ba,[0,c[1]]);case
32:return c[1];case
34:return h(L(c[1]),pL);case
35:return h(pM,L(c[1]));case
36:var
d=h(pO,h(O(b,c[2]),pN));return h(f(b,c[1]),d);case
52:return h(L(j(c[1],0)),pP);default:}cL(c);return Q(pJ)}function
ba(a,b){return b[2]}function
f(b,c){return T(bb(0,b,c))}function
O(b,c){return T(sn(0,b,c))}function
aj(b,c){return T(c6(0,b,c))}function
oe(b){return T(ba(0,b))}var
p1=h(p0,h(pZ,h(pY,h(pX,h(pW,h(pV,h(pU,h(pT,h(pS,h(pR,pQ)))))))))),qg=h(qf,h(qe,h(qd,h(qc,h(qb,h(qa,h(p$,h(p_,h(p9,h(p8,h(p7,h(p6,h(p5,h(p4,h(p3,p2))))))))))))))),qo=h(qn,h(qm,h(ql,h(qk,h(qj,h(qi,qh)))))),qw=h(qv,h(qu,h(qt,h(qs,h(qr,h(qq,qp))))));function
t(a){return[32,h(qx,k(a)),a]}function
a4(a,b){return[25,a,b]}function
bC(a,b){return[50,a,b]}function
ap(a){return[33,a]}function
cM(a){return[34,a]}function
a5(a,b){return[2,a,b]}function
eP(a,b){return[3,a,b]}function
cN(a,b){return[6,a,b]}function
cO(a,b){return[7,a,b]}function
cP(a){return[13,a]}function
cQ(a,b){return[29,a,b]}function
aq(a,b){return[31,[0,a,b]]}function
cR(a,b){return[37,a,b]}function
cS(a,b){return[27,a,b]}function
cT(a){return[28,a]}function
aB(a,b){return[36,a,b]}function
eQ(a){var
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
ao=aM(b,c[2]);return[50,b(c[1]),ao];default:}return c}}var
c=b(a);for(;;){if(e[1]){e[1]=0;var
c=b(c);continue}return c}}var
qD=[0,qC];function
a6(a,b,c){var
g=c[2],d=c[1],t=a?a[1]:a,u=b?b[1]:2,n=g[3],p=g[2];qD[1]=qE;var
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
o=h(qM,h(k(j[1]),qL)),l=2;break;default:var
l=1}switch(l){case
1:cL(n[1]);dP(dK);throw[0,D,qF];case
2:break;default:var
o=qG}var
q=[0,e(0,n[1]),o];if(t){a2[1]=q;a3[1]=q}function
r(a){var
q=g[4],r=dR(function(a,b){return 0===b?a:h(qo,a)},qw,q),s=h(r,e(0,eQ(p))),j=cZ(e5(qH,gD,438));dL(j,s);c0(j);e6(j);fe(qI);var
m=dM(qJ),c=sY(m),n=y(c),o=0;if(0<=0)if(0<=c)if((n.getLen()-c|0)<o)var
f=0;else{var
k=o,b=c;for(;;){if(0<b){var
l=c4(m,n,k,b);if(0===l)throw[0,bn];var
k=k+l|0,b=b-l|0;continue}var
f=1;break}}else
var
f=0;else
var
f=0;if(!f)C(gF);dO(m);i(X(d,723535973,3),d,n);fe(qK);return 0}function
s(a){var
b=g[4],c=dR(function(a,b){return 0===b?a:h(qg,a)},p1,b);return i(X(d,56985577,4),d,h(c,f(0,eQ(p))))}switch(u){case
1:s(0);break;case
2:r(0);s(0);break;default:r(0)}i(X(d,345714255,5),d,0);return[0,d,g]}var
cU=ca,eR=null,qR=1,qS=1,qT=1,qU=undefined;function
eS(a,b){return a==eR?j(b,0):a}var
eT=true,eU=Array,qV=false;ed(function(a){return a
instanceof
eU?0:[0,new
au(a.toString())]});function
H(a,b){a.appendChild(b);return 0}function
cV(d){return sV(function(a){if(a){var
e=j(d,a);if(!(e|0))a.preventDefault();return e}var
c=event,b=j(d,c);if(!(b|0))c.returnValue=b;return b})}var
I=cU.document,qW="2d";function
bD(a,b){return a?j(b,a[1]):0}function
bE(a,b){return a.createElement(b.toString())}function
bF(a,b){return bE(a,b)}var
eV=[0,f5];function
eW(a,b,c,d){for(;;){if(0===a)if(0===b)return bE(c,d);var
h=eV[1];if(f5===h){try{var
j=I.createElement('<input name="x">'),k=j.tagName.toLowerCase()===fn?1:0,m=k?j.name===dp?1:0:k,i=m}catch(f){var
i=0}var
l=i?fR:-1003883683;eV[1]=l;continue}if(fR<=h){var
e=new
eU();e.push("<",d.toString());bD(a,function(a){e.push(' type="',ff(a),b0);return 0});bD(b,function(a){e.push(' name="',ff(a),b0);return 0});e.push(">");return c.createElement(e.join(g))}var
f=bE(c,d);bD(a,function(a){return f.type=a});bD(b,function(a){return f.name=a});return f}}function
eX(a){return bF(a,q0)}var
q4=[0,q3];cU.HTMLElement===qU;function
e1(a){return eX(I)}function
e2(a){function
c(a){throw[0,D,q6]}var
b=eS(I.getElementById(gl),c);return j(ea(function(a){H(b,e1(0));H(b,I.createTextNode(a.toString()));return H(b,e1(0))}),a)}function
q7(a){var
k=[0,[4,a]];return function(a,b,c,d){var
h=a[2],i=a[1],l=c[2];if(0===l[0]){var
g=l[1],e=[0,0],f=spoc_cuda_create_extra(k.length-1),m=g[7][1]<i[1]?1:0;if(m)var
n=m;else{var
s=g[7][2]<i[2]?1:0,n=s||(g[7][3]<i[3]?1:0)}if(n)throw[0,le];var
o=g[8][1]<h[1]?1:0;if(o)var
p=o;else{var
r=g[8][2]<h[2]?1:0,p=r||(g[8][3]<h[3]?1:0)}if(p)throw[0,lg];cc(function(a,b){function
h(a){if(bA)try{eK(a,0,c);N(c,0,0)}catch(f){if(f[1]===ao)throw[0,ao];throw f}return 11===b[0]?spoc_cuda_custom_load_param_vec_b(e,f,cG(b[1],dx,c[1][8]),a):spoc_cuda_load_param_vec_b(e,f,cG(a,dx,c[1][8]),a,c)}switch(b[0]){case
6:case
7:case
8:case
9:case
10:switch(b[0]){case
6:var
d=spoc_cuda_load_param_int_b(e,f,b[1]);break;case
7:var
d=spoc_cuda_load_param_int64_b(e,f,b[1]);break;case
8:var
d=spoc_cuda_load_param_float_b(e,f,b[1]);break;case
9:var
d=spoc_cuda_load_param_float64_b(e,f,b[1]);break;default:var
d=Q(lb)}var
g=d;break;case
11:var
g=h(b[1]);break;default:var
g=h(b[1])}return g},k);var
q=spoc_cuda_launch_grid_b(e,d,h,i,f,c[1],b)}else{var
j=[0,0];cc(function(a,b){switch(b[0]){case
6:case
7:case
8:case
9:case
10:switch(b[0]){case
6:var
e=tP(j,d,b[1],c[1]);break;case
7:var
e=spoc_opencl_load_param_int64(j,d,b[1],c[1]);break;case
8:var
e=spoc_opencl_load_param_float(j,d,b[1],c[1]);break;case
9:var
e=spoc_opencl_load_param_float64(j,d,b[1],c[1]);break;default:var
e=Q(lc)}var
g=e;break;default:var
f=b[1];if(bA){if(c3(aX(f),[0,c]))eK(f,0,c);N(c,0,0)}var
h=c[1],i=F(0),g=tQ(j,d,a,cG(f,-701974253,c[1][8]-i|0),h)}return g},k);var
q=tO(d,h,i,c[1],b)}return q}}if(cW===0)var
c=ej([0]);else{var
aV=ej(aM(hF,cW));cc(function(a,b){var
c=(a*2|0)+2|0;aV[3]=n(an[4],b,c,aV[3]);aV[4]=n(ah[4],c,1,aV[4]);return 0},cW);var
c=aV}var
cs=aM(function(a){return aS(c,a)},e0),er=cK[2],q8=cs[1],q9=cs[2],q_=cs[3],hZ=cK[4],el=ct(eY),em=ct(e0),en=ct(eZ),q$=1,cu=cd(function(a){return aS(c,a)},em),hI=cd(function(a){return aS(c,a)},en);c[5]=[0,[0,c[3],c[4],c[6],c[7],cu,el],c[5]];var
hJ=Z[1],hK=c[7];function
hL(a,b,c){return cg(a,el)?n(Z[4],a,b,c):c}c[7]=n(Z[11],hL,hK,hJ);var
aT=[0,an[1]],aU=[0,ah[1]];dU(function(a,b){aT[1]=n(an[4],a,b,aT[1]);var
e=aU[1];try{var
f=i(ah[22],b,c[4]),d=f}catch(f){if(f[1]!==r)throw f;var
d=1}aU[1]=n(ah[4],b,d,e);return 0},en,hI);dU(function(a,b){aT[1]=n(an[4],a,b,aT[1]);aU[1]=n(ah[4],b,0,aU[1]);return 0},em,cu);c[3]=aT[1];c[4]=aU[1];var
hM=0,hN=c[6];c[6]=cf(function(a,b){return cg(a[1],cu)?b:[0,a,b]},hN,hM);var
h0=q$?i(er,c,hZ):j(er,c),eo=c[5],aw=eo?eo[1]:Q(gJ),ep=c[5],hO=aw[6],hP=aw[5],hQ=aw[4],hR=aw[3],hS=aw[2],hT=aw[1],hU=ep?ep[2]:Q(gK);c[5]=hU;var
ce=hQ,bo=hO;for(;;){if(bo){var
dT=bo[1],gL=bo[2],hV=i(Z[22],dT,c[7]),ce=n(Z[4],dT,hV,ce),bo=gL;continue}c[7]=ce;c[3]=hT;c[4]=hS;var
hW=c[6];c[6]=cf(function(a,b){return cg(a[1],hP)?b:[0,a,b]},hW,hR);var
h1=0,h2=cv(eZ),h3=[0,aM(function(a){var
e=aS(c,a);try{var
b=c[6];for(;;){if(!b)throw[0,r];var
d=b[1],f=b[2],h=d[2];if(0!==aD(d[1],e)){var
b=f;continue}var
g=h;break}}catch(f){if(f[1]!==r)throw f;var
g=q(c[2],e)}return g},h2),h1],h4=cv(eY),ra=sp([0,[0,h0],[0,aM(function(a){try{var
b=i(Z[22],a,c[7])}catch(f){if(f[1]===r)throw[0,D,hY];throw f}return b},h4),h3]])[1],rb=function(a,b){if(1===b.length-1){var
c=b[0+1];if(4===c[0])return c[1]}return Q(rc)};et(c,[0,q9,0,q7,q_,function(a,b){return[0,[4,b]]},q8,rb]);var
rd=function(a,b){var
e=es(b,c);n(ra,e,rf,re);if(!b){var
f=c[8];if(0!==f){var
d=f;for(;;){if(d){var
g=d[2];j(d[1],e);var
d=g;continue}break}}}return e};ek[1]=(ek[1]+c[1]|0)-1|0;c[8]=dS(c[8]);cq(c,3+ar(q(c[2],1)*16|0,av)|0);var
h5=0,h6=function(a){var
b=a;return rd(h5,b)},ri=t(4),rj=ap(2),rk=a5(t(3),rj),rl=cQ(aB(t(0),rk),ri),rm=t(4),rn=ap(1),ro=a5(t(3),rn),rp=a4(cQ(aB(t(0),ro),rm),rl),rq=t(4),rr=t(3),rs=a4(cQ(aB(t(0),rr),rq),rp),rt=ap(2),ru=a5(t(3),rt),rv=[0,aB(t(0),ru)],ry=bC(aq(rx,rw),rv),rz=cO(cM(fW),ry),rA=ap(1),rB=a5(t(3),rA),rC=[0,aB(t(0),rB)],rF=bC(aq(rE,rD),rC),rG=cO(cM(fJ),rF),rH=t(3),rI=[0,aB(t(0),rH)],rL=bC(aq(rK,rJ),rI),rM=[0,eP(eP(cO(cM(fC),rL),rG),rz)],rP=bC(aq(rO,rN),rM),rQ=a4(cS(t(4),rP),rs),rR=ap(4),rS=cN(t(2),rR),rT=a4(cS(t(3),rS),rQ),rU=ap(d),rV=cN(ap(d),rU),qz=[40,[46,t(2),rV],rT],rY=aq(rX,rW),r1=cN(aq(r0,rZ),rY),r4=a5(aq(r3,r2),r1),qA=[26,a4(cS(t(2),r4),qz)],r5=cR(cT(cP(4)),qA),r6=cR(cT(cP(3)),r5),rg=[0,0],rh=[0,[13,5],eE],qy=[0,[1,[24,[23,qB,0],0]],cR(cT(cP(2)),r6)],r7=[0,function(a){var
e=qR+(qT*qS|0)|0,f=e<=(d*d|0)?1:0;if(f){var
b=e*4|0,g=fW*$(a,b+2|0),h=fJ*$(a,b+1|0),c=fC*$(a,b)+h+g|0;_(a,b,c);_(a,b+1|0,c);return _(a,b+2|0,c)}return f},qy,rh,rg],cX=[0,h6(0),r7],bG=function(a){return eX(I)};cU.onload=cV(function(a){function
e(a){throw[0,D,sa]}var
c=eS(I.getElementById(gl),e);H(c,bG(0));var
s=eW(0,0,I,qY),b=bE(I,q2);H(b,I.createTextNode("Choose a computing device : "));H(c,b);H(c,s);H(c,bG(0));var
f=bF(I,q5);if(1-(f.getContext==eR?1:0)){f.width=d;f.height=d;var
v=bF(I,q1);v.src="lena.png";var
w=f.getContext(qW);v.onload=cV(function(a){w.drawImage(v,0,0);H(c,bG(0));H(c,f);var
M=e3?e3[1]:2;switch(M){case
1:fg(0);az[1]=fh(0);break;case
2:fi(0);ay[1]=fj(0);fg(0);az[1]=fh(0);break;default:fi(0);ay[1]=fj(0)}eC[1]=ay[1]+az[1]|0;var
z=ay[1]-1|0,y=0,N=0;if(z<0)var
A=y;else{var
h=N,D=y;for(;;){var
E=cb(D,[0,tC(h),0]),O=h+1|0;if(z!==h){var
h=O,D=E;continue}var
A=E;break}}var
p=0,e=0,b=A;for(;;){if(p<az[1]){if(tN(e)){var
C=e+1|0,B=cb(b,[0,tE(e,e+ay[1]|0),0])}else{var
C=e,B=b}var
p=p+1|0,e=C,b=B;continue}var
o=0,n=b;for(;;){if(n){var
o=o+1|0,n=n[2];continue}eC[1]=o;az[1]=e;if(b){var
l=0,k=b,J=b[2],K=b[1];for(;;){if(k){var
l=l+1|0,k=k[2];continue}var
x=u(l,K),m=1,g=J;for(;;){if(g){var
L=g[2];x[m+1]=g[1];var
m=m+1|0,g=L;continue}var
r=x;break}break}}else
var
r=[0];var
F=w.getImageData(0,0,d,d),G=F.data;H(c,bG(0));dQ(function(a){var
b=bF(I,qX);H(b,I.createTextNode(a[1][1].toString()));return H(s,b)},r);var
P=function(a){var
h=q(r,s.selectedIndex+0|0),v=h[1][1];j(e2(r9),v);var
c=aA(eE,0,(d*d|0)*4|0);ee(hA,e7(0));var
m=R(c)-1|0,x=0;if(!(m<0)){var
f=x;for(;;){_(c,f,G[f]);var
D=f+1|0;if(m!==f){var
f=D;continue}break}}var
n=h[2];if(0===n[0])var
k=b2;else{var
C=0===n[1][2]?1:b2,k=C}a6(0,r_,cX);var
t=eB(0),g=cX[2],b=cX[1],y=0,z=[0,[0,k,1,1],[0,ar(((d*d|0)+k|0)-1|0,k),1,1]],o=0,l=0?o[1]:o;if(0===h[2][0]){if(l)a6(0,qN,[0,b,g]);else
if(!i(X(b,-723625231,7),b,0))a6(0,qO,[0,b,g])}else
if(l)a6(0,qP,[0,b,g]);else
if(!i(X(b,649483637,8),b,0))a6(0,qQ,[0,b,g]);(function(a,b,c,d,e,f){return a.length==5?a(b,c,d,e,f):ae(a,[b,c,d,e,f])}(X(b,5695307,6),b,c,z,y,h));var
u=eB(0)-t;i(e2(r8),r$,u);var
p=R(c)-1|0,A=0;if(!(p<0)){var
e=A;for(;;){G[e]=$(c,e);var
B=e+1|0;if(p!==e){var
e=B;continue}break}}w.putImageData(F,0,0);return eT},t=eW([0,"button"],0,I,qZ);t.value="Go";t.onclick=cV(P);H(c,t);return eT}}});return qV}throw[0,q4]});dN(0);return}}(this));
