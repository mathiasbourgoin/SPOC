// This program was compiled from OCaml by js_of_ocaml 1.99dev
(function(by){"use strict";var
be=125,bp=123,h=255,cz=108,cE="x",A=".",ax="+",aC=65535,cx=-97,bl="FP DENORM\n",az="Map.bal",cF='"',y=16777215,cq="g",aw=1073741823,N=250,bm="f",bc="FP ROUND TO ZERO\n",F=105,bk=65599,Y=110,cL=-88,bb="FP ROUND TO NEAREST\n",cD=246,aA="'",_=115,cp="Unix",av="int_of_string",cu=-32,bj=102,cy=".ptx",bg=111,cC="Unix.Unix_error",bo=120,v=" ",bf="..",Z="e",bd="FP FMA\n",bi="FP INF NAN\n",bn=117,cI=256,E="-",ct="nan",k="",O=-48,ba=116,cG="../",cH=65520,bh="FP ROUND TO INF\n",M=100,w="0",ay=248,cs="/",cw=".cl",aB=114,bq=103,cM="fd ",cK=101,cA="FP NONE\n",cJ="index out of bounds",cB="./",cr="number",cv=1e3,kg=1;function
cW(a,b){throw[0,a,b]}function
bz(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=by.console;b&&b.error&&b.error(a)}var
d=[0];function
ab(a,b){if(!a)return k;if(a&1)return ab(a-1,b)+b;var
c=ab(a>>1,b);return c+c}function
o(a){if(a!=null){this.bytes=this.fullBytes=a;this.last=this.len=a.length}}function
cZ(){cW(d[4],new
o(cJ))}o.prototype={string:null,bytes:null,fullBytes:null,array:null,len:null,last:0,toJsString:function(){var
a=this.getFullBytes();try{return this.string=decodeURIComponent(escape(a))}catch(f){bz('MlString.toJsString: wrong encoding for \"%s\" ',a);return a}},toBytes:function(){if(this.string!=null)try{var
a=unescape(encodeURIComponent(this.string))}catch(f){bz('MlString.toBytes: wrong encoding for \"%s\" ',this.string);var
a=this.string}else{var
a=k,c=this.array,d=c.length;for(var
b=0;b<d;b++)a+=String.fromCharCode(c[b])}this.bytes=this.fullBytes=a;this.last=this.len=a.length;return a},getBytes:function(){var
a=this.bytes;if(a==null)a=this.toBytes();return a},getFullBytes:function(){var
a=this.fullBytes;if(a!==null)return a;a=this.bytes;if(a==null)a=this.toBytes();if(this.last<this.len){this.bytes=a+=ab(this.len-this.last,"\0");this.last=this.len}this.fullBytes=a;return a},toArray:function(){var
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
b=this.bytes;if(b==null)b=this.toBytes();return a<this.last?b.charCodeAt(a):0},safeGet:function(a){if(this.len==null)this.toBytes();if(a<0||a>=this.len)cZ();return this.get(a)},set:function(a,b){var
c=this.array;if(!c){if(this.last==a){this.bytes+=String.fromCharCode(b&h);this.last++;return 0}c=this.toArray()}else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;c[a]=b&h;return 0},safeSet:function(a,b){if(this.len==null)this.toBytes();if(a<0||a>=this.len)cZ();this.set(a,b)},fill:function(a,b,c){if(a>=this.last&&this.last&&c==0)return;var
d=this.array;if(!d)d=this.toArray();else
if(this.bytes!=null)this.bytes=this.fullBytes=this.string=null;var
f=a+b;for(var
e=a;e<f;e++)d[e]=c},compare:function(a){if(this.string!=null&&a.string!=null){if(this.string<a.string)return-1;if(this.string>a.string)return 1;return 0}var
b=this.getFullBytes(),c=a.getFullBytes();if(b<c)return-1;if(b>c)return 1;return 0},equal:function(a){if(this.string!=null&&a.string!=null)return this.string==a.string;return this.getFullBytes()==a.getFullBytes()},lessThan:function(a){if(this.string!=null&&a.string!=null)return this.string<a.string;return this.getFullBytes()<a.getFullBytes()},lessEqual:function(a){if(this.string!=null&&a.string!=null)return this.string<=a.string;return this.getFullBytes()<=a.getFullBytes()}};function
G(a){this.string=a}G.prototype=new
o();function
i6(a,b,c,d,e){if(d<=b)for(var
f=1;f<=e;f++)c[d+f]=a[b+f];else
for(var
f=e;f>=1;f--)c[d+f]=a[b+f]}function
bx(a,b){cW(a,new
G(b))}function
aF(a){bx(d[4],a)}function
aD(){aF(cJ)}function
i7(a,b){if(b<0||b>=a.length-1)aD();return a[b+1]}function
i8(a,b,c){if(b<0||b>=a.length-1)aD();a[b+1]=c;return 0}function
bs(a,b,c,d,e){if(e===0)return;if(d===c.last&&c.bytes!=null){var
f=a.bytes;if(f==null)f=a.toBytes();if(b>0||a.last>e)f=f.slice(b,b+e);c.bytes+=f;c.last+=f.length;return}var
g=c.array;if(!g)g=c.toArray();else
c.bytes=c.string=null;a.blitToArray(b,g,d,e)}function
B(c,b){if(c.fun)return B(c.fun,b);var
a=c.length,d=a-b.length;if(d==0)return c.apply(null,b);else
if(d<0)return B(c.apply(null,b.slice(0,a)),b.slice(a));else
return function(a){return B(c,b.concat([a]))}}function
i9(a){if(isFinite(a)){if(Math.abs(a)>=2.22507385850720138e-308)return 0;if(a!=0)return 1;return 2}return isNaN(a)?4:3}function
jh(a,b){var
c=a[3]<<16,d=b[3]<<16;if(c>d)return 1;if(c<d)return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
cU(a,b){if(a<b)return-1;if(a==b)return 0;return 1}function
cP(a,b,c){var
e=[];for(;;){if(!(c&&a===b))if(a
instanceof
o)if(b
instanceof
o){if(a!==b){var
d=a.compare(b);if(d!=0)return d}}else
return 1;else
if(a
instanceof
Array&&a[0]===(a[0]|0)){var
g=a[0];if(g===N){a=a[1];continue}else
if(b
instanceof
Array&&b[0]===(b[0]|0)){var
i=b[0];if(i===N){b=b[1];continue}else
if(g!=i)return g<i?-1:1;else
switch(g){case
ay:{var
d=cU(a[2],b[2]);if(d!=0)return d;break}case
251:aF("equal: abstract value");case
h:{var
d=jh(a,b);if(d!=0)return d;break}default:if(a.length!=b.length)return a.length<b.length?-1:1;if(a.length>1)e.push(a,b,1)}}else
return 1}else
if(b
instanceof
o||b
instanceof
Array&&b[0]===(b[0]|0))return-1;else{if(a<b)return-1;if(a>b)return 1;if(c&&a!=b){if(a==a)return 1;if(b==b)return-1}}if(e.length==0)return 0;var
f=e.pop();b=e.pop();a=e.pop();if(f+1<a.length)e.push(a,b,f+1);a=a[f];b=b[f]}}function
cO(a,b){return cP(a,b,true)}function
cN(a){this.bytes=k;this.len=a}cN.prototype=new
o();function
cQ(a){if(a<0)aF("String.create");return new
cN(a)}function
bw(a){throw[0,a]}function
cX(){bw(d[6])}function
i_(a,b){if(b==0)cX();return a/b|0}function
i$(a,b){return+(cP(a,b,false)==0)}function
ja(a,b,c,d){a.fill(b,c,d)}function
bv(a){a=a.toString();var
e=a.length;if(e>31)aF("format_int: format too long");var
b={justify:ax,signstyle:E,filler:v,alternate:false,base:0,signedconv:false,width:0,uppercase:false,sign:1,prec:-1,conv:bm};for(var
d=0;d<e;d++){var
c=a.charAt(d);switch(c){case
E:b.justify=E;break;case
ax:case
v:b.signstyle=c;break;case
w:b.filler=w;break;case"#":b.alternate=true;break;case"1":case"2":case"3":case"4":case"5":case"6":case"7":case"8":case"9":b.width=0;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.width=b.width*10+c;d++}d--;break;case
A:b.prec=0;d++;while(c=a.charCodeAt(d)-48,c>=0&&c<=9){b.prec=b.prec*10+c;d++}d--;case"d":case"i":b.signedconv=true;case"u":b.base=10;break;case
cE:b.base=16;break;case"X":b.base=16;b.uppercase=true;break;case"o":b.base=8;break;case
Z:case
bm:case
cq:b.signedconv=true;b.conv=c;break;case"E":case"F":case"G":b.signedconv=true;b.uppercase=true;b.conv=c.toLowerCase();break}}return b}function
bt(a,b){if(a.uppercase)b=b.toUpperCase();var
e=b.length;if(a.signedconv&&(a.sign<0||a.signstyle!=E))e++;if(a.alternate){if(a.base==8)e+=1;if(a.base==16)e+=2}var
c=k;if(a.justify==ax&&a.filler==v)for(var
d=e;d<a.width;d++)c+=v;if(a.signedconv)if(a.sign<0)c+=E;else
if(a.signstyle!=E)c+=a.signstyle;if(a.alternate&&a.base==8)c+=w;if(a.alternate&&a.base==16)c+="0x";if(a.justify==ax&&a.filler==w)for(var
d=e;d<a.width;d++)c+=w;c+=b;if(a.justify==E)for(var
d=e;d<a.width;d++)c+=v;return new
G(c)}function
jb(a,b){var
c,f=bv(a),e=f.prec<0?6:f.prec;if(b<0){f.sign=-1;b=-b}if(isNaN(b)){c=ct;f.filler=v}else
if(!isFinite(b)){c="inf";f.filler=v}else
switch(f.conv){case
Z:var
c=b.toExponential(e),d=c.length;if(c.charAt(d-3)==Z)c=c.slice(0,d-1)+w+c.slice(d-1);break;case
bm:c=b.toFixed(e);break;case
cq:e=e?e:1;c=b.toExponential(e-1);var
i=c.indexOf(Z),h=+c.slice(i+1);if(h<-4||b.toFixed(0).length>e){var
d=i-1;while(c.charAt(d)==w)d--;if(c.charAt(d)==A)d--;c=c.slice(0,d+1)+c.slice(i);d=c.length;if(c.charAt(d-3)==Z)c=c.slice(0,d-1)+w+c.slice(d-1);break}else{var
g=e;if(h<0){g-=h+1;c=b.toFixed(g)}else
while(c=b.toFixed(g),c.length>e+1)g--;if(g){var
d=c.length-1;while(c.charAt(d)==w)d--;if(c.charAt(d)==A)d--;c=c.slice(0,d+1)}}break}return bt(f,c)}function
jc(a,b){if(a.toString()=="%d")return new
G(k+b);var
c=bv(a);if(b<0)if(c.signedconv){c.sign=-1;b=-b}else
b>>>=0;var
d=b.toString(c.base);if(c.prec>=0){c.filler=v;var
e=c.prec-d.length;if(e>0)d=ab(e,w)+d}return bt(c,d)}var
aG=[];function
jd(a,b,c){var
e=a[1],i=aG[c];if(i===null)for(var
h=aG.length;h<c;h++)aG[h]=0;else
if(e[i]===b)return e[i-1];var
d=3,g=e[1]*2+1,f;while(d<g){f=d+g>>1|1;if(b<e[f+1])g=f-2;else
d=f}aG[c]=d+1;return b==e[d+1]?e[d]:0}function
je(a,b){return+(cO(a,b,false)>=0)}function
cR(a){if(!isFinite(a)){if(isNaN(a))return[h,1,0,cH];return a>0?[h,0,0,32752]:[h,0,0,cH]}var
f=a>=0?0:32768;if(f)a=-a;var
b=Math.floor(Math.LOG2E*Math.log(a))+1023;if(b<=0){b=0;a/=Math.pow(2,-1026)}else{a/=Math.pow(2,b-1027);if(a<16){a*=2;b-=1}if(b==0)a/=2}var
d=Math.pow(2,24),c=a|0;a=(a-c)*d;var
e=a|0;a=(a-e)*d;var
g=a|0;c=c&15|f|b<<4;return[h,g,e,c]}function
aa(a,b){return((a>>16)*b<<16)+(a&aC)*b|0}var
jf=function(){var
r=cI;function
c(a,b){return a<<b|a>>>32-b}function
g(a,b){b=aa(b,3432918353);b=c(b,15);b=aa(b,461845907);a^=b;a=c(a,13);return(a*5|0)+3864292196|0}function
u(a){a^=a>>>16;a=aa(a,2246822507);a^=a>>>13;a=aa(a,3266489909);a^=a>>>16;return a}function
v(a,b){var
d=b[1]|b[2]<<24,c=b[2]>>>8|b[3]<<16;a=g(a,d);a=g(a,c);return a}function
w(a,b){var
d=b[1]|b[2]<<24,c=b[2]>>>8|b[3]<<16;a=g(a,c^d);return a}function
y(a,b){var
e=b.length,c,d;for(c=0;c+4<=e;c+=4){d=b.charCodeAt(c)|b.charCodeAt(c+1)<<8|b.charCodeAt(c+2)<<16|b.charCodeAt(c+3)<<24;a=g(a,d)}d=0;switch(e&3){case
3:d=b.charCodeAt(c+2)<<16;case
2:d|=b.charCodeAt(c+1)<<8;case
1:d|=b.charCodeAt(c);a=g(a,d);default:}a^=e;return a}function
x(a,b){var
e=b.length,c,d;for(c=0;c+4<=e;c+=4){d=b[c]|b[c+1]<<8|b[c+2]<<16|b[c+3]<<24;a=g(a,d)}d=0;switch(e&3){case
3:d=b[c+2]<<16;case
2:d|=b[c+1]<<8;case
1:d|=b[c];a=g(a,d);default:}a^=e;return a}return function(a,b,c,d){var
l,m,n,j,i,f,e,k,q;j=b;if(j<0||j>r)j=r;i=a;f=c;l=[d];m=0;n=1;while(m<n&&i>0){e=l[m++];if(e
instanceof
Array&&e[0]===(e[0]|0))switch(e[0]){case
ay:f=g(f,e[2]);i--;break;case
N:l[--m]=e[1];break;case
h:f=w(f,e);i--;break;default:var
t=e.length-1<<10|e[0];f=g(f,t);for(k=1,q=e.length;k<q;k++){if(n>=j)break;l[n++]=e[k]}break}else
if(e
instanceof
o){var
p=e.array;if(p)f=x(f,p);else{var
s=e.getFullBytes();f=y(f,s)}i--;break}else
if(e===(e|0)){f=g(f,e+e+1);i--}else
if(e===+e){f=v(f,cR(e));i--;break}}f=u(f);return f&aw}}();function
jp(a){return[a[3]>>8,a[3]&h,a[2]>>16,a[2]>>8&h,a[2]&h,a[1]>>16,a[1]>>8&h,a[1]&h]}function
jg(e,b,c){var
d=0;function
f(a){b--;if(e<0||b<0)return;if(a
instanceof
Array&&a[0]===(a[0]|0))switch(a[0]){case
ay:e--;d=d*bk+a[2]|0;break;case
N:b++;f(a);break;case
h:e--;d=d*bk+a[1]+(a[2]<<24)|0;break;default:e--;d=d*19+a[0]|0;for(var
c=a.length-1;c>0;c--)f(a[c])}else
if(a
instanceof
o){e--;var
g=a.array,i=a.getLen();if(g)for(var
c=0;c<i;c++)d=d*19+g[c]|0;else{var
j=a.getFullBytes();for(var
c=0;c<i;c++)d=d*19+j.charCodeAt(c)|0}}else
if(a===(a|0)){e--;d=d*bk+a|0}else
if(a===+a){e--;var
k=jp(cR(a));for(var
c=7;c>=0;c--)d=d*19+k[c]|0}}f(c);return d&aw}function
jk(a){return(a[3]|a[2]|a[1])==0}function
jn(a){return[h,a&y,a>>24&y,a>>31&aC]}function
jo(a,b){var
c=a[1]-b[1],d=a[2]-b[2]+(c>>24),e=a[3]-b[3]+(d>>24);return[h,c&y,d&y,e&aC]}function
cT(a,b){if(a[3]>b[3])return 1;if(a[3]<b[3])return-1;if(a[2]>b[2])return 1;if(a[2]<b[2])return-1;if(a[1]>b[1])return 1;if(a[1]<b[1])return-1;return 0}function
cS(a){a[3]=a[3]<<1|a[2]>>23;a[2]=(a[2]<<1|a[1]>>23)&y;a[1]=a[1]<<1&y}function
jl(a){a[1]=(a[1]>>>1|a[2]<<23)&y;a[2]=(a[2]>>>1|a[3]<<23)&y;a[3]=a[3]>>>1}function
jr(a,b){var
e=0,d=a.slice(),c=b.slice(),f=[h,0,0,0];while(cT(d,c)>0){e++;cS(c)}while(e>=0){e--;cS(f);if(cT(d,c)>=0){f[1]++;d=jo(d,c)}jl(c)}return[0,f,d]}function
jq(a){return a[1]|a[2]<<24}function
jj(a){return a[3]<<16<0}function
jm(a){var
b=-a[1],c=-a[2]+(b>>24),d=-a[3]+(c>>24);return[h,b&y,c&y,d&aC]}function
ji(a,b){var
c=bv(a);if(c.signedconv&&jj(b)){c.sign=-1;b=jm(b)}var
d=k,h=jn(c.base),g="0123456789abcdef";do{var
f=jr(b,h);b=f[1];d=g.charAt(jq(f[2]))+d}while(!jk(b));if(c.prec>=0){c.filler=v;var
e=c.prec-d.length;if(e>0)d=ab(e,w)+d}return bt(c,d)}function
jL(a){var
b=0,c=10,d=a.get(0)==45?(b++,-1):1;if(a.get(b)==48)switch(a.get(b+1)){case
bo:case
88:c=16;b+=2;break;case
bg:case
79:c=8;b+=2;break;case
98:case
66:c=2;b+=2;break}return[b,d,c]}function
cV(a){if(a>=48&&a<=57)return a-48;if(a>=65&&a<=90)return a-55;if(a>=97&&a<=122)return a-87;return-1}function
aE(a){bx(d[3],a)}function
js(a){var
g=jL(a),e=g[0],h=g[1],f=g[2],i=-1>>>0,d=a.get(e),c=cV(d);if(c<0||c>=f)aE(av);var
b=c;for(;;){e++;d=a.get(e);if(d==95)continue;c=cV(d);if(c<0||c>=f)break;b=f*b+c;if(b>i)aE(av)}if(e!=a.getLen())aE(av);b=h*b;if((b|0)!=b)aE(av);return b}function
jt(a){return+(a>31&&a<127)}function
ju(a){var
c=Array.prototype.slice;return function(){var
b=arguments.length>0?c.call(arguments):[undefined];return B(a,b)}}function
jv(a,b){var
d=[0];for(var
c=1;c<=a;c++)d[c]=b;return d}function
br(a){var
b=a.length;this.array=a;this.len=this.last=b}br.prototype=new
o();var
jw=function(){function
n(a,b){return a+b|0}function
m(a,b,c,d,e,f){b=n(n(b,a),n(d,f));return n(b<<e|b>>>32-e,c)}function
i(a,b,c,d,e,f,g){return m(b&c|~b&d,a,b,e,f,g)}function
j(a,b,c,d,e,f,g){return m(b&d|c&~d,a,b,e,f,g)}function
k(a,b,c,d,e,f,g){return m(b^c^d,a,b,e,f,g)}function
l(a,b,c,d,e,f,g){return m(c^(b|~d),a,b,e,f,g)}function
o(a,b){var
g=b;a[g>>2]|=128<<8*(g&3);for(g=(g&~3)+8;(g&63)<60;g+=4)a[(g>>2)-1]=0;a[(g>>2)-1]=b<<3;a[g>>2]=b>>29&536870911;var
m=[1732584193,4023233417,2562383102,271733878];for(g=0;g<a.length;g+=16){var
c=m[0],d=m[1],e=m[2],f=m[3];c=i(c,d,e,f,a[g+0],7,3614090360);f=i(f,c,d,e,a[g+1],12,3905402710);e=i(e,f,c,d,a[g+2],17,606105819);d=i(d,e,f,c,a[g+3],22,3250441966);c=i(c,d,e,f,a[g+4],7,4118548399);f=i(f,c,d,e,a[g+5],12,1200080426);e=i(e,f,c,d,a[g+6],17,2821735955);d=i(d,e,f,c,a[g+7],22,4249261313);c=i(c,d,e,f,a[g+8],7,1770035416);f=i(f,c,d,e,a[g+9],12,2336552879);e=i(e,f,c,d,a[g+10],17,4294925233);d=i(d,e,f,c,a[g+11],22,2304563134);c=i(c,d,e,f,a[g+12],7,1804603682);f=i(f,c,d,e,a[g+13],12,4254626195);e=i(e,f,c,d,a[g+14],17,2792965006);d=i(d,e,f,c,a[g+15],22,1236535329);c=j(c,d,e,f,a[g+1],5,4129170786);f=j(f,c,d,e,a[g+6],9,3225465664);e=j(e,f,c,d,a[g+11],14,643717713);d=j(d,e,f,c,a[g+0],20,3921069994);c=j(c,d,e,f,a[g+5],5,3593408605);f=j(f,c,d,e,a[g+10],9,38016083);e=j(e,f,c,d,a[g+15],14,3634488961);d=j(d,e,f,c,a[g+4],20,3889429448);c=j(c,d,e,f,a[g+9],5,568446438);f=j(f,c,d,e,a[g+14],9,3275163606);e=j(e,f,c,d,a[g+3],14,4107603335);d=j(d,e,f,c,a[g+8],20,1163531501);c=j(c,d,e,f,a[g+13],5,2850285829);f=j(f,c,d,e,a[g+2],9,4243563512);e=j(e,f,c,d,a[g+7],14,1735328473);d=j(d,e,f,c,a[g+12],20,2368359562);c=k(c,d,e,f,a[g+5],4,4294588738);f=k(f,c,d,e,a[g+8],11,2272392833);e=k(e,f,c,d,a[g+11],16,1839030562);d=k(d,e,f,c,a[g+14],23,4259657740);c=k(c,d,e,f,a[g+1],4,2763975236);f=k(f,c,d,e,a[g+4],11,1272893353);e=k(e,f,c,d,a[g+7],16,4139469664);d=k(d,e,f,c,a[g+10],23,3200236656);c=k(c,d,e,f,a[g+13],4,681279174);f=k(f,c,d,e,a[g+0],11,3936430074);e=k(e,f,c,d,a[g+3],16,3572445317);d=k(d,e,f,c,a[g+6],23,76029189);c=k(c,d,e,f,a[g+9],4,3654602809);f=k(f,c,d,e,a[g+12],11,3873151461);e=k(e,f,c,d,a[g+15],16,530742520);d=k(d,e,f,c,a[g+2],23,3299628645);c=l(c,d,e,f,a[g+0],6,4096336452);f=l(f,c,d,e,a[g+7],10,1126891415);e=l(e,f,c,d,a[g+14],15,2878612391);d=l(d,e,f,c,a[g+5],21,4237533241);c=l(c,d,e,f,a[g+12],6,1700485571);f=l(f,c,d,e,a[g+3],10,2399980690);e=l(e,f,c,d,a[g+10],15,4293915773);d=l(d,e,f,c,a[g+1],21,2240044497);c=l(c,d,e,f,a[g+8],6,1873313359);f=l(f,c,d,e,a[g+15],10,4264355552);e=l(e,f,c,d,a[g+6],15,2734768916);d=l(d,e,f,c,a[g+13],21,1309151649);c=l(c,d,e,f,a[g+4],6,4149444226);f=l(f,c,d,e,a[g+11],10,3174756917);e=l(e,f,c,d,a[g+2],15,718787259);d=l(d,e,f,c,a[g+9],21,3951481745);m[0]=n(c,m[0]);m[1]=n(d,m[1]);m[2]=n(e,m[2]);m[3]=n(f,m[3])}var
p=[];for(var
g=0;g<4;g++)for(var
o=0;o<4;o++)p[g*4+o]=m[g]>>8*o&h;return p}return function(a,b,c){var
h=[];if(a.array){var
f=a.array;for(var
d=0;d<c;d+=4){var
e=d+b;h[d>>2]=f[e]|f[e+1]<<8|f[e+2]<<16|f[e+3]<<24}for(;d<c;d++)h[d>>2]|=f[d+b]<<8*(d&3)}else{var
g=a.getFullBytes();for(var
d=0;d<c;d+=4){var
e=d+b;h[d>>2]=g.charCodeAt(e)|g.charCodeAt(e+1)<<8|g.charCodeAt(e+2)<<16|g.charCodeAt(e+3)<<24}for(;d<c;d++)h[d>>2]|=g.charCodeAt(d+b)<<8*(d&3)}return new
br(o(h,c))}}();function
C(a){bx(d[2],a)}function
bu(a){if(!a.opened)C("Cannot flush a closed channel");if(a.buffer==k)return 0;if(a.output){switch(a.output.length){case
2:a.output(a,a.buffer);break;default:a.output(a.buffer)}}a.buffer=k}var
$=new
Array();function
jx(a){bu(a);a.opened=false;delete
$[a.fd];return 0}function
jy(a,b,c,d){var
e=a.data.array.length-a.data.offset;if(e<d)d=e;bs(new
br(a.data.array),a.data.offset,b,c,d);a.data.offset+=d;return d}function
jM(){bw(d[5])}function
jz(a){if(a.data.offset>=a.data.array.length)jM();if(a.data.offset<0||a.data.offset>a.data.array.length)aD();var
b=a.data.array[a.data.offset];a.data.offset++;return b}function
jA(a){var
b=a.data.offset,c=a.data.array.length;if(b>=c)return 0;while(true){if(b>=c)return-(b-a.data.offset);if(b<0||b>a.data.array.length)aD();if(a.data.array[b]==10)return b-a.data.offset+1;b++}}function
jO(a,b){if(!d.files)d.files={};if(b
instanceof
o)var
c=b.getArray();else
if(b
instanceof
Array)var
c=b;else
var
c=new
o(b).getArray();d.files[a
instanceof
o?a.toString():a]=c}function
jV(a){return d.files&&d.files[a.toString()]?1:0}function
ac(a,b,c){if(d.fds===undefined)d.fds=new
Array();c=c?c:{};var
e={};e.array=b;e.offset=c.append?e.array.length:0;e.flags=c;d.fds[a]=e;d.fd_last_idx=a;return a}function
jZ(a,b,c){var
e={};while(b){switch(b[1]){case
0:e.rdonly=1;break;case
1:e.wronly=1;break;case
2:e.append=1;break;case
3:e.create=1;break;case
4:e.truncate=1;break;case
5:e.excl=1;break;case
6:e.binary=1;break;case
7:e.text=1;break;case
8:e.nonblock=1;break}b=b[2]}var
f=a.toString();if(e.rdonly&&e.wronly)C(f+" : flags Open_rdonly and Open_wronly are not compatible");if(e.text&&e.binary)C(f+" : flags Open_text and Open_binary are not compatible");if(jV(a)){if(e.create&&e.excl)C(f+" : file already exists");var
g=d.fd_last_idx?d.fd_last_idx:0;if(e.truncate)d.files[f]=k;return ac(g+1,d.files[f],e)}else
if(e.create){var
g=d.fd_last_idx?d.fd_last_idx:0;jO(f,[]);return ac(g+1,d.files[f],e)}else
C(f+": no such file or directory")}ac(0,[]);ac(1,[]);ac(2,[]);function
jB(a){var
b=d.fds[a];if(b.flags.wronly)C(cM+a+" is writeonly");return{data:b,fd:a,opened:true}}function
j5(a){if(a.charCodeAt(a.length-1)==10)a=a.substr(0,a.length-1);var
b=by.console;b&&b.log&&b.log(a)}function
jR(a,b){var
e=new
o(b),d=e.getLen();for(var
c=0;c<d;c++)a.data.array[a.data.offset+c]=e.get(c);a.data.offset+=d;return 0}function
jC(a){var
b;switch(a){case
1:b=j5;break;case
2:b=bz;break;default:b=jR}var
e=d.fds[a];if(e.flags.rdonly)C(cM+a+" is readonly");var
c={data:e,fd:a,opened:true,buffer:k,output:b};$[c.fd]=c;return c}function
jD(){var
a=0;for(var
b
in
$)if($[b].opened)a=[0,$[b],a];return a}function
t(a){return new
o(a)}function
jE(a,b,c,d){if(!a.opened)C("Cannot output to a closed channel");var
f;if(c==0&&b.getLen()==d)f=b;else{f=cQ(d);bs(b,c,f,0,d)}var
e=f.toString(),g=e.lastIndexOf("\n");if(g<0)a.buffer+=e;else{a.buffer+=e.substr(0,g+1);bu(a);a.buffer+=e.substr(g+1)}}function
jF(a,b){var
c=t(String.fromCharCode(b));jE(a,c,0,1)}function
jG(a,b){if(b==0)cX();return a%b}function
jI(a,b){var
d=[a];for(var
c=1;c<=b;c++)d[c]=0;return d}function
jJ(a,b){a[0]=b;return 0}function
jK(a){return a
instanceof
Array?a[0]:cv}function
jP(a,b){d[a+1]=b}var
jH={};function
jQ(a,b){jH[a]=b;return 0}function
jS(a,b){return a.compare(b)}function
cY(a,b){var
c=a.fullBytes,d=b.fullBytes;if(c!=null&&d!=null)return c==d?1:0;return a.getFullBytes()==b.getFullBytes()?1:0}function
jT(a,b){return 1-cY(a,b)}function
jU(){return 32}function
jW(){var
a=new
G("a.out");return[0,a,[0,a]]}function
jX(){return[0,new
G(cp),32,0]}function
jN(){bw(d[7])}function
jY(){jN()}function
j0(){var
a=new
Date()^4294967295*Math.random();return{valueOf:function(){return a},0:0,1:a,length:2}}function
j1(a){var
b=1;while(a&&a.joo_tramp){a=a.joo_tramp.apply(null,a.joo_args);b++}return a}function
j2(a,b){return{joo_tramp:a,joo_args:b}}function
j3(a,b){if(typeof
b==="function"){a.fun=b;return 0}if(b.fun){a.fun=b.fun;return 0}var
c=b.length;while(c--)a[c]=b[c];return 0}function
j4(){return 0}var
bA=0;function
j6(){if(window.webcl==undefined){alert("Unfortunately your system does not support WebCL. "+"Make sure that you have both the OpenCL driver "+"and the WebCL browser extension installed.");bA=1}else{alert("CONGRATULATIONS! Your system supports WebCL");console.log("INIT OPENCL");bA=0}return 0}function
j7(){console.log(" spoc_cuInit");return 0}function
j8(){console.log(" spoc_cuda_compile");return 0}function
j9(){console.log(" spoc_cuda_debug_compile");return 0}function
j_(){console.log(" spoc_debug_opencl_compile");return 0}function
j$(a){console.log("spoc_getCudaDevice");return 0}function
ka(){console.log(" spoc_getCudaDevicesCount");return 0}function
kb(a,b){console.log(" spoc_getOpenCLDevice");var
u=[["DEVICE_ADDRESS_BITS",WebCL.DEVICE_ADDRESS_BITS],["DEVICE_AVAILABLE",WebCL.DEVICE_AVAILABLE],["DEVICE_COMPILER_AVAILABLE",WebCL.DEVICE_COMPILER_AVAILABLE],["DEVICE_ENDIAN_LITTLE",WebCL.DEVICE_ENDIAN_LITTLE],["DEVICE_ERROR_CORRECTION_SUPPORT",WebCL.DEVICE_ERROR_CORRECTION_SUPPORT],["DEVICE_EXECUTION_CAPABILITIES",WebCL.DEVICE_EXECUTION_CAPABILITIES],["DEVICE_EXTENSIONS",WebCL.DEVICE_EXTENSIONS],["DEVICE_GLOBAL_MEM_CACHE_SIZE",WebCL.DEVICE_GLOBAL_MEM_CACHE_SIZE],["DEVICE_GLOBAL_MEM_CACHE_TYPE",WebCL.DEVICE_GLOBAL_MEM_CACHE_TYPE],["DEVICE_GLOBAL_MEM_CACHELINE_SIZE",WebCL.DEVICE_GLOBAL_MEM_CACHELINE_SIZE],["DEVICE_GLOBAL_MEM_SIZE",WebCL.DEVICE_GLOBAL_MEM_SIZE],["DEVICE_HALF_FP_CONFIG",WebCL.DEVICE_HALF_FP_CONFIG],["DEVICE_IMAGE_SUPPORT",WebCL.DEVICE_IMAGE_SUPPORT],["DEVICE_IMAGE2D_MAX_HEIGHT",WebCL.DEVICE_IMAGE2D_MAX_HEIGHT],["DEVICE_IMAGE2D_MAX_WIDTH",WebCL.DEVICE_IMAGE2D_MAX_WIDTH],["DEVICE_IMAGE3D_MAX_DEPTH",WebCL.DEVICE_IMAGE3D_MAX_DEPTH],["DEVICE_IMAGE3D_MAX_HEIGHT",WebCL.DEVICE_IMAGE3D_MAX_HEIGHT],["DEVICE_IMAGE3D_MAX_WIDTH",WebCL.DEVICE_IMAGE3D_MAX_WIDTH],["DEVICE_LOCAL_MEM_SIZE",WebCL.DEVICE_LOCAL_MEM_SIZE],["DEVICE_LOCAL_MEM_TYPE",WebCL.DEVICE_LOCAL_MEM_TYPE],["DEVICE_MAX_CLOCK_FREQUENCY",WebCL.DEVICE_MAX_CLOCK_FREQUENCY],["DEVICE_MAX_COMPUTE_UNITS",WebCL.DEVICE_MAX_COMPUTE_UNITS],["DEVICE_MAX_CONSTANT_ARGS",WebCL.DEVICE_MAX_CONSTANT_ARGS],["DEVICE_MAX_CONSTANT_BUFFER_SIZE",WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE],["DEVICE_MAX_MEM_ALLOC_SIZE",WebCL.DEVICE_MAX_MEM_ALLOC_SIZE],["DEVICE_MAX_PARAMETER_SIZE",WebCL.DEVICE_MAX_PARAMETER_SIZE],["DEVICE_MAX_READ_IMAGE_ARGS",WebCL.DEVICE_MAX_READ_IMAGE_ARGS],["DEVICE_MAX_SAMPLERS",WebCL.DEVICE_MAX_SAMPLERS],["DEVICE_MAX_WORK_GROUP_SIZE",WebCL.DEVICE_MAX_WORK_GROUP_SIZE],["DEVICE_MAX_WORK_ITEM_DIMENSIONS",WebCL.DEVICE_MAX_WORK_ITEM_DIMENSIONS],["DEVICE_MAX_WORK_ITEM_SIZES",WebCL.DEVICE_MAX_WORK_ITEM_SIZES],["DEVICE_MAX_WRITE_IMAGE_ARGS",WebCL.DEVICE_MAX_WRITE_IMAGE_ARGS],["DEVICE_MEM_BASE_ADDR_ALIGN",WebCL.DEVICE_MEM_BASE_ADDR_ALIGN],["DEVICE_NAME",WebCL.DEVICE_NAME],["DEVICE_PLATFORM",WebCL.DEVICE_PLATFORM],["DEVICE_PREFERRED_VECTOR_WIDTH_CHAR",WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_CHAR],["DEVICE_PREFERRED_VECTOR_WIDTH_SHORT",WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_SHORT],["DEVICE_PREFERRED_VECTOR_WIDTH_INT",WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_INT],["DEVICE_PREFERRED_VECTOR_WIDTH_LONG",WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_LONG],["DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT",WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT],["DEVICE_PROFILE",WebCL.DEVICE_PROFILE],["DEVICE_PROFILING_TIMER_RESOLUTION",WebCL.DEVICE_PROFILING_TIMER_RESOLUTION],["DEVICE_QUEUE_PROPERTIES",WebCL.DEVICE_QUEUE_PROPERTIES],["DEVICE_SINGLE_FP_CONFIG",WebCL.DEVICE_SINGLE_FP_CONFIG],["DEVICE_TYPE",WebCL.DEVICE_TYPE],["DEVICE_VENDOR",WebCL.DEVICE_VENDOR],["DEVICE_VENDOR_ID",WebCL.DEVICE_VENDOR_ID],["DEVICE_VERSION",WebCL.DEVICE_VERSION],["DRIVER_VERSION",WebCL.DRIVER_VERSION]],r=0,e=[0],n=[1],d=new
Array(48);d[0]=0;var
g=[0],j=webcl.getPlatforms();for(var
s
in
j){var
f=j[s],i=f.getDevices();r+=i.length}var
h=0;j=webcl.getPlatforms();for(var
m
in
j){console.log("here "+m);var
f=j[m],i=f.getDevices(),l=i.length;console.log("there "+h+v+l+v+a);if(h+l>=a)for(var
p
in
i){var
c=i[p];if(h==a){e[1]=t(c.getInfo(WebCL.DEVICE_NAME));e[2]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_SIZE);e[3]=c.getInfo(WebCL.DEVICE_LOCAL_MEM_SIZE);e[4]=c.getInfo(WebCL.DEVICE_MAX_CLOCK_FREQUENCY);e[5]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE);e[6]=c.getInfo(WebCL.DEVICE_MAX_COMPUTE_UNITS);e[7]=c.getInfo(WebCL.DEVICE_ERROR_CORRECTION_SUPPORT);e[8]=b;var
o=0;e[9]=o;g[1]=t(f.getInfo(WebCL.PLATFORM_PROFILE));g[2]=t(f.getInfo(WebCL.PLATFORM_VERSION));g[3]=t(f.getInfo(WebCL.PLATFORM_NAME));g[4]=t(f.getInfo(WebCL.PLATFORM_VENDOR));g[5]=t(f.getInfo(WebCL.PLATFORM_EXTENSIONS));g[6]=l;var
k=c.getInfo(WebCL.DEVICE_TYPE),w=0;if(k&WebCL.DEVICE_TYPE_CPU)d[2]=0;if(k&WebCL.DEVICE_TYPE_GPU)d[2]=1;if(k&WebCL.DEVICE_TYPE_ACCELERATOR)d[2]=2;if(k&WebCL.DEVICE_TYPE_DEFAULT)d[2]=3;d[3]=t(c.getInfo(WebCL.DEVICE_PROFILE));d[4]=t(c.getInfo(WebCL.DEVICE_VERSION));d[5]=t(c.getInfo(WebCL.DEVICE_VENDOR));var
q=c.getInfo(WebCL.DEVICE_EXTENSIONS);d[6]=t(q);d[7]=c.getInfo(WebCL.DEVICE_VENDOR_ID);d[8]=c.getInfo(WebCL.DEVICE_MAX_WORK_ITEM_DIMENSIONS);d[9]=c.getInfo(WebCL.DEVICE_ADDRESS_BITS);d[10]=c.getInfo(WebCL.DEVICE_MAX_MEM_ALLOC_SIZE);d[11]=c.getInfo(WebCL.DEVICE_IMAGE_SUPPORT);d[12]=c.getInfo(WebCL.DEVICE_MAX_READ_IMAGE_ARGS);d[13]=c.getInfo(WebCL.DEVICE_MAX_WRITE_IMAGE_ARGS);d[14]=c.getInfo(WebCL.DEVICE_MAX_SAMPLERS);d[15]=c.getInfo(WebCL.DEVICE_MEM_BASE_ADDR_ALIGN);d[17]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHELINE_SIZE);d[18]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHE_SIZE);d[19]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_ARGS);d[20]=c.getInfo(WebCL.DEVICE_ENDIAN_LITTLE);d[21]=c.getInfo(WebCL.DEVICE_AVAILABLE);d[22]=c.getInfo(WebCL.DEVICE_COMPILER_AVAILABLE);d[23]=c.getInfo(WebCL.DEVICE_SINGLE_FP_CONFIG);d[24]=c.getInfo(WebCL.DEVICE_GLOBAL_MEM_CACHE_TYPE);d[25]=c.getInfo(WebCL.DEVICE_QUEUE_PROPERTIES);d[26]=c.getInfo(WebCL.DEVICE_LOCAL_MEM_TYPE);d[28]=c.getInfo(WebCL.DEVICE_MAX_CONSTANT_BUFFER_SIZE);d[29]=c.getInfo(WebCL.DEVICE_EXECUTION_CAPABILITIES);d[31]=c.getInfo(WebCL.DEVICE_MAX_WORK_GROUP_SIZE);d[32]=c.getInfo(WebCL.DEVICE_IMAGE2D_MAX_HEIGHT);d[33]=c.getInfo(WebCL.DEVICE_IMAGE2D_MAX_WIDTH);d[34]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_DEPTH);d[35]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_HEIGHT);d[36]=c.getInfo(WebCL.DEVICE_IMAGE3D_MAX_WIDTH);d[37]=c.getInfo(WebCL.DEVICE_MAX_PARAMETER_SIZE);d[38]=c.getInfo(WebCL.DEVICE_MAX_WORK_ITEM_SIZES);d[39]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);d[40]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);d[41]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_INT);d[42]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_LONG);d[43]=c.getInfo(WebCL.DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);d[45]=c.getInfo(WebCL.DEVICE_PROFILING_TIMER_RESOLUTION);d[46]=t(c.getInfo(WebCL.DRIVER_VERSION));break}else
h++}else
h+=l}var
c=[0];d[1]=g;n[1]=d;c[1]=e;c[2]=n;return c}function
kc(){console.log(" spoc_getOpenCLDevicesCount");var
a=0,b=webcl.getPlatforms();for(var
d
in
b){var
e=b[d],c=e.getDevices();a+=c.length}return a}function
kd(){console.log(" spoc_opencl_compile");return 0}function
ke(){console.log(" spoc_opencl_is_available");return!bA}function
kf(){return 0}var
n=i7,g=i8,W=bs,ar=cO,p=cQ,cf=jb,ao=jc,au=jd,a9=jt,r=jv,a7=bu,ci=jy,cd=jB,ce=jC,cg=jG,b=t,ch=jI,L=jP,a8=jQ,a_=jS,as=cY,x=jT,ap=jY,a$=j1,X=j2,ck=j4,cl=j6,cn=j7,co=ka,cm=kc,at=kf;function
e(a,b){return a.length==1?a(b):B(a,[b])}function
f(a,b,c){return a.length==2?a(b,c):B(a,[b,c])}function
i(a,b,c,d){return a.length==3?a(b,c,d):B(a,[b,c,d])}function
cj(a,b,c,d,e,f,g){return a.length==6?a(b,c,d,e,f,g):B(a,[b,c,d,e,f,g])}var
P=[0,b("Failure")],bB=[0,b("Invalid_argument")],aJ=[0,b("End_of_file")],m=[0,b("Not_found")],aK=[0,b("Assert_failure")],aX=b(A),a0=b(A),a2=b(A),ca=b(k),b_=[0,b("file_file"),b("kernel_name"),b("cuda_sources"),b("opencl_sources"),b("binaries")],b$=[0,b("set_opencl_sources"),b("set_cuda_sources"),b("run"),b("reset_binaries"),b("reload_sources"),b("list_to_args"),b("get_opencl_sources"),b("get_cuda_sources"),b("get_binaries"),b("exec"),b("compile_and_run"),b("compile"),b("args_to_list")];L(6,m);L(5,[0,b("Division_by_zero")]);L(4,aJ);L(3,bB);L(2,P);L(1,[0,b("Sys_error")]);var
c2=[0,0,[0,7,0]],c0=b("true"),c1=b("false"),c3=b("Pervasives.do_at_exit"),c4=b("Array.blit"),c8=b("\\b"),c9=b("\\t"),c_=b("\\n"),c$=b("\\r"),c7=b("\\\\"),c6=b("\\'"),c5=b("Char.chr"),dc=b("String.contains_from"),db=b("String.blit"),da=b("String.sub"),dl=b("Map.remove_min_elt"),dm=[0,0,0,0],dn=[0,b("map.ml"),270,10],dp=[0,0,0],dh=b(az),di=b(az),dj=b(az),dk=b(az),dq=b("CamlinternalLazy.Undefined"),dt=b("Buffer.add: cannot grow buffer"),dJ=b(k),dK=b(k),dN=b("%.12g"),dO=b(cF),dP=b(cF),dL=b(aA),dM=b(aA),dI=b(ct),dG=b("neg_infinity"),dH=b("infinity"),dF=b(A),dE=b("printf: bad positional specification (0)."),dD=b("%_"),dC=[0,b("printf.ml"),143,8],dA=b(aA),dB=b("Printf: premature end of format string '"),dw=b(aA),dx=b(" in format string '"),dy=b(", at char number "),dz=b("Printf: bad conversion %"),du=b("Sformat.index_of_int: negative argument "),dQ=b(cE),i4=b("OCAMLRUNPARAM"),i2=b("CAMLRUNPARAM"),dR=b(k),dW=b(k),dT=b("CamlinternalOO.last_id"),ef=b(k),ec=b(cB),eb=b(".\\"),ea=b(cG),d$=b("..\\"),d3=b(cB),d2=b(cG),dY=b(k),dX=b(k),dZ=b(bf),d0=b(cs),i0=b("TMPDIR"),d5=b("/tmp"),d6=b("'\\''"),d9=b(bf),d_=b("\\"),iY=b("TEMP"),ed=b(A),ei=b(bf),ej=b(cs),em=b("Cygwin"),en=b(cp),eo=b("Win32"),ep=[0,b("filename.ml"),189,9],ew=b("E2BIG"),ey=b("EACCES"),ez=b("EAGAIN"),eA=b("EBADF"),eB=b("EBUSY"),eC=b("ECHILD"),eD=b("EDEADLK"),eE=b("EDOM"),eF=b("EEXIST"),eG=b("EFAULT"),eH=b("EFBIG"),eI=b("EINTR"),eJ=b("EINVAL"),eK=b("EIO"),eL=b("EISDIR"),eM=b("EMFILE"),eN=b("EMLINK"),eO=b("ENAMETOOLONG"),eP=b("ENFILE"),eQ=b("ENODEV"),eR=b("ENOENT"),eS=b("ENOEXEC"),eT=b("ENOLCK"),eU=b("ENOMEM"),eV=b("ENOSPC"),eW=b("ENOSYS"),eX=b("ENOTDIR"),eY=b("ENOTEMPTY"),eZ=b("ENOTTY"),e0=b("ENXIO"),e1=b("EPERM"),e2=b("EPIPE"),e3=b("ERANGE"),e4=b("EROFS"),e5=b("ESPIPE"),e6=b("ESRCH"),e7=b("EXDEV"),e8=b("EWOULDBLOCK"),e9=b("EINPROGRESS"),e_=b("EALREADY"),e$=b("ENOTSOCK"),fa=b("EDESTADDRREQ"),fb=b("EMSGSIZE"),fc=b("EPROTOTYPE"),fd=b("ENOPROTOOPT"),fe=b("EPROTONOSUPPORT"),ff=b("ESOCKTNOSUPPORT"),fg=b("EOPNOTSUPP"),fh=b("EPFNOSUPPORT"),fi=b("EAFNOSUPPORT"),fj=b("EADDRINUSE"),fk=b("EADDRNOTAVAIL"),fl=b("ENETDOWN"),fm=b("ENETUNREACH"),fn=b("ENETRESET"),fo=b("ECONNABORTED"),fp=b("ECONNRESET"),fq=b("ENOBUFS"),fr=b("EISCONN"),fs=b("ENOTCONN"),ft=b("ESHUTDOWN"),fu=b("ETOOMANYREFS"),fv=b("ETIMEDOUT"),fw=b("ECONNREFUSED"),fx=b("EHOSTDOWN"),fy=b("EHOSTUNREACH"),fz=b("ELOOP"),fA=b("EOVERFLOW"),fB=b("EUNKNOWNERR %d"),ex=b("Unix.Unix_error(Unix.%s, %S, %S)"),es=b(cC),et=b(k),eu=b(k),ev=b(cC),fC=b("0.0.0.0"),fD=b("127.0.0.1"),iX=b("::"),iW=b("::1"),fL=b("Cuda.No_Cuda_Device"),fM=b("Cuda.ERROR_DEINITIALIZED"),fN=b("Cuda.ERROR_NOT_INITIALIZED"),fO=b("Cuda.ERROR_INVALID_CONTEXT"),fP=b("Cuda.ERROR_INVALID_VALUE"),fQ=b("Cuda.ERROR_OUT_OF_MEMORY"),fR=b("Cuda.ERROR_INVALID_DEVICE"),fS=b("Cuda.ERROR_NOT_FOUND"),fT=b("Cuda.ERROR_FILE_NOT_FOUND"),fU=b("Cuda.ERROR_UNKNOWN"),fV=b("Cuda.ERROR_LAUNCH_FAILED"),fW=b("Cuda.ERROR_LAUNCH_OUT_OF_RESOURCES"),fX=b("Cuda.ERROR_LAUNCH_TIMEOUT"),fY=b("Cuda.ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"),fZ=b("no_cuda_device"),f0=b("cuda_error_deinitialized"),f1=b("cuda_error_not_initialized"),f2=b("cuda_error_invalid_context"),f3=b("cuda_error_invalid_value"),f4=b("cuda_error_out_of_memory"),f5=b("cuda_error_invalid_device"),f6=b("cuda_error_not_found"),f7=b("cuda_error_file_not_found"),f8=b("cuda_error_launch_failed"),f9=b("cuda_error_launch_out_of_resources"),f_=b("cuda_error_launch_timeout"),f$=b("cuda_error_launch_incompatible_texturing"),ga=b("cuda_error_unknown"),gb=b("OpenCL.No_OpenCL_Device"),gc=b("OpenCL.OPENCL_ERROR_UNKNOWN"),gd=b("OpenCL.INVALID_CONTEXT"),ge=b("OpenCL.INVALID_DEVICE"),gf=b("OpenCL.INVALID_VALUE"),gg=b("OpenCL.INVALID_QUEUE_PROPERTIES"),gh=b("OpenCL.OUT_OF_RESOURCES"),gi=b("OpenCL.MEM_OBJECT_ALLOCATION_FAILURE"),gj=b("OpenCL.OUT_OF_HOST_MEMORY"),gk=b("OpenCL.FILE_NOT_FOUND"),gl=b("OpenCL.INVALID_PROGRAM"),gm=b("OpenCL.INVALID_BINARY"),gn=b("OpenCL.INVALID_BUILD_OPTIONS"),go=b("OpenCL.INVALID_OPERATION"),gp=b("OpenCL.COMPILER_NOT_AVAILABLE"),gq=b("OpenCL.BUILD_PROGRAM_FAILURE"),gr=b("OpenCL.INVALID_KERNEL"),gs=b("OpenCL.INVALID_ARG_INDEX"),gt=b("OpenCL.INVALID_ARG_VALUE"),gu=b("OpenCL.INVALID_MEM_OBJECT"),gv=b("OpenCL.INVALID_SAMPLER"),gw=b("OpenCL.INVALID_ARG_SIZE"),gx=b("OpenCL.INVALID_COMMAND_QUEUE"),gy=b("no_opencl_device"),gz=b("opencl_error_unknown"),gA=b("opencl_invalid_context"),gB=b("opencl_invalid_device"),gC=b("opencl_invalid_value"),gD=b("opencl_invalid_queue_properties"),gE=b("opencl_out_of_resources"),gF=b("opencl_mem_object_allocation_failure"),gG=b("opencl_out_of_host_memory"),gH=b("opencl_file_not_found"),gI=b("opencl_invalid_program"),gJ=b("opencl_invalid_binary"),gK=b("opencl_invalid_build_options"),gL=b("opencl_invalid_operation"),gM=b("opencl_compiler_not_available"),gN=b("opencl_build_program_failure"),gO=b("opencl_invalid_kernel"),gP=b("opencl_invalid_arg_index"),gQ=b("opencl_invalid_arg_value"),gR=b("opencl_invalid_mem_object"),gS=b("opencl_invalid_sampler"),gT=b("opencl_invalid_arg_size"),gU=b("opencl_invalid_command_queue"),g5=b(cy),g4=b(cw),g3=b(cy),g2=b(cw),g1=[0,1],g0=b(k),gV=b("\n"),gW=b("Kernel.No_source_for_device"),he=b("  Device  : %d\n"),hf=b("    Name : %s\n"),hg=b("    Total Global Memory : %d\n"),hh=b("    Local Memory Size : %d\n"),hi=b("    Clock Rate : %d\n"),hj=b("    Total Constant Memory : %d\n"),hk=b("    Multi Processor Count : %d\n"),hl=b("    ECC Enabled : %b\n"),hm=b("    Powered by Cuda\n"),iU=b("    Powered by OpenCL\n"),hn=b("    Driver Version %d\n"),ho=b("    Cuda %d.%d compatible\n"),hp=b("    Regs Per Block : %d\n"),hq=b("    Warp Size : %d\n"),hr=b("    Memory Pitch : %d\n"),hs=b("    Max Threads Per Block : %d\n"),ht=b("    Max Threads Dim : %dx%dx%d\n"),hu=b("    Max Grid Size : %dx%dx%d\n"),hv=b("    Texture Alignment : %d\n"),hw=b("    Device Overlap : %b\n"),hx=b("    Kernel Exec Timeout Enabled : %b\n"),hy=b("    Integrated : %b\n"),hz=b("    Can Map Host Memory : %b\n"),hA=b("    Compute Mode : %d\n"),hB=b("    Concurrent Kernels : %b\n"),hC=b("    PCI Bus ID : %d\n"),hD=b("    PCI Device ID : %d\n"),hE=b("    OpenCL compatible (via Platform : %s)\n"),hF=b("    Platform Profile : %s\n"),hG=b("    Platform Version : %s\n"),hH=b("    Platform Vendor : %s\n"),hI=b("    Platform Extensions : %s\n"),hJ=b("    Platform Number of Devices : %d\n"),hK=b("    Type : "),hL=b("CPU\n"),iR=b("GPU\n"),iS=b("ACCELERATOR\n"),iT=b("DEFAULT\n"),hM=b("    Profile : %s\n"),hN=b("    Version : %s\n"),hO=b("    Vendor : %s\n"),hP=b("    Driver : %s\n"),hQ=b("    Extensions : %s\n"),hR=b("    Vendor ID : %d\n"),hS=b("    Max Work Item Dimensions : %d\n"),hT=b("    Max Work Group Size : %d\n"),hU=b("    Max Work Item Size : %dx%dx%d\n"),hV=b("    Address Bits : %d\n"),hW=b("    Max Memory Alloc Size : %d\n"),hX=b("    Image Support : %b\n"),hY=b("    Max Read Image Args : %d\n"),hZ=b("    Max Write Image Args : %d\n"),h0=b("    Max Samplers : %d\n"),h1=b("    Memory Base Addr Align : %d\n"),h2=b("    Min Data Type Align Size : %d\n"),h3=b("    Global Mem Cacheline Size : %d\n"),h4=b("    Global Mem Cache Size : %d\n"),h5=b("    Max Constant Args : %d\n"),h6=b("    Endian Little : %b\n"),h7=b("    Available : %b\n"),h8=b("    Compiler Available : %b\n"),h9=b("    CL Device Single FP Config : "),h_=b(bl),iL=b(bi),iM=b(bb),iN=b(bc),iO=b(bh),iP=b(bd),iQ=b("error single_fp_config"),h$=b("    CL Device Double FP Config : "),ia=b(bl),iF=b(bi),iG=b(bb),iH=b(bc),iI=b(bh),iJ=b(bd),iK=b(cA),ib=b("    CL Device Half FP Config : "),ic=b(bl),iz=b(bi),iA=b(bb),iB=b(bc),iC=b(bh),iD=b(bd),iE=b(cA),id=b("    CL Device Global Mem Cache Type : "),ie=b("READ WRITE CACHE\n"),ix=b("READ ONLY CACHE\n"),iy=b("NONE\n"),ig=b("    CL Device Queue Properties : "),iw=b("PROFILING ENABLE\n"),ih=b("OUT OF ORDER EXEC MODE ENABLE\n"),ii=b("    CL Local Mem Type : "),iv=b("Global\n"),ij=b("Local\n"),ik=b("    Image2D DIM : %dx%d\n"),il=b("    Image3D DIM : %dx%dx%d\n"),im=b("    Preferred Vector Width Char : %d\n"),io=b("    Preferred Vector Width Short : %d\n"),ip=b("    Preferred Vector Width Int : %d\n"),iq=b("    Preferred Vector Width Long : %d\n"),ir=b("    Preferred Vector Width Float : %d\n"),is=b("    Preferred Vector Width Double : %d\n"),it=b("    Profiling Timer Resolution : %d\n"),iu=b("    !!Warning!! could be Device %d\n"),g$=b("DeviceQuery\nThis application prints informations about every\ndevice compatible with Spoc found on your computer.\n"),ha=b("Found %d devices: \n"),hb=b("  ** %d Cuda devices \n"),hc=b("  ** %d OpenCL devices \n"),hd=b("Devices Info:\n"),g_=b("<BR>");function
ad(a){throw[0,P,a]}function
u(a){throw[0,bB,a]}function
j(a,b){var
c=a.getLen(),e=b.getLen(),d=p(c+e|0);W(a,0,d,0,c);W(b,0,d,c,e);return d}function
ae(a){return b(k+a)}function
aH(a,b){if(a){var
c=a[1];return[0,c,aH(a[2],b)]}return b}cd(0);var
aI=ce(1);ce(2);function
bC(a){var
b=jD(0);for(;;){if(b){var
c=b[2],d=b[1];try{a7(d)}catch(f){}var
b=c;continue}return 0}}a8(c3,bC);function
aL(a){if(0<=a)if(!(h<a))return a;return u(c5)}function
bD(a){var
b=65<=a?90<a?0:1:0;if(!b){var
c=192<=a?214<a?0:1:0;if(!c){var
d=216<=a?222<a?1:0:1;if(d)return a}}return a+32|0}function
H(a,b){var
c=p(a);ja(c,0,a,b);return c}function
l(a,b,c){if(0<=b)if(0<=c)if(!((a.getLen()-c|0)<b)){var
d=p(c);W(a,b,d,0,c);return d}return u(da)}function
af(a,b,c,d,e){if(0<=e)if(0<=b)if(!((a.getLen()-e|0)<b))if(0<=d)if(!((c.getLen()-e|0)<d))return W(a,b,c,d,e);return u(db)}function
bE(a){var
c=a.getLen();if(0===c)var
f=a;else{var
d=p(c),e=c-1|0,g=0;if(!(e<0)){var
b=g;for(;;){d.safeSet(b,bD(a.safeGet(b)));var
h=b+1|0;if(e!==b){var
b=h;continue}break}}var
f=d}return f}var
aN=jX(0)[1],ah=jU(0),aO=(1<<(ah-10|0))-1|0,Q=aa(ah/8|0,aO)-1|0,de=jW(0)[2],df=ay,dg=N;function
aP(l){function
k(a){return a?a[5]:0}function
g(a,b,c,d){var
e=k(a),f=k(d),g=f<=e?e+1|0:f+1|0;return[0,a,b,c,d,g]}function
r(a,b){return[0,0,a,b,0,1]}function
h(a,b,c,d){var
h=a?a[5]:0,i=d?d[5]:0;if((i+2|0)<h){if(a){var
e=a[4],m=a[3],n=a[2],j=a[1],q=k(e);if(q<=k(j))return g(j,n,m,g(e,b,c,d));if(e){var
r=e[3],s=e[2],t=e[1],v=g(e[4],b,c,d);return g(g(j,n,m,t),s,r,v)}return u(dh)}return u(di)}if((h+2|0)<i){if(d){var
l=d[4],o=d[3],p=d[2],f=d[1],w=k(f);if(w<=k(l))return g(g(a,b,c,f),p,o,l);if(f){var
x=f[3],y=f[2],z=f[1],A=g(f[4],p,o,l);return g(g(a,b,c,z),y,x,A)}return u(dj)}return u(dk)}var
B=i<=h?h+1|0:i+1|0;return[0,a,b,c,d,B]}var
a=0;function
H(a){return a?0:1}function
s(a,b,c){if(c){var
d=c[4],i=c[3],e=c[2],g=c[1],k=c[5],j=f(l[1],a,e);return 0===j?[0,g,a,b,d,k]:0<=j?h(g,e,i,s(a,b,d)):h(s(a,b,g),e,i,d)}return[0,0,a,b,0,1]}function
I(a,b){var
c=b;for(;;){if(c){var
e=c[4],g=c[3],h=c[1],d=f(l[1],a,c[2]);if(0===d)return g;var
i=0<=d?e:h,c=i;continue}throw[0,m]}}function
J(a,b){var
c=b;for(;;){if(c){var
g=c[4],h=c[1],d=f(l[1],a,c[2]),e=0===d?1:0;if(e)return e;var
i=0<=d?g:h,c=i;continue}return 0}}function
p(a){var
b=a;for(;;){if(b){var
c=b[1];if(c){var
b=c;continue}return[0,b[2],b[3]]}throw[0,m]}}function
K(a){var
b=a;for(;;){if(b){var
c=b[4],d=b[3],e=b[2];if(c){var
b=c;continue}return[0,e,d]}throw[0,m]}}function
t(a){if(a){var
b=a[1];if(b){var
c=a[4],d=a[3],e=a[2];return h(t(b),e,d,c)}return a[4]}return u(dl)}function
v(a,b){if(b){var
c=b[4],j=b[3],e=b[2],d=b[1],k=f(l[1],a,e);if(0===k){if(d)if(c){var
i=p(c),m=i[2],n=i[1],g=h(d,n,m,t(c))}else
var
g=d;else
var
g=c;return g}return 0<=k?h(d,e,j,v(a,c)):h(v(a,d),e,j,c)}return 0}function
z(a,b){var
c=b;for(;;){if(c){var
d=c[4],e=c[3],g=c[2];z(a,c[1]);f(a,g,e);var
c=d;continue}return 0}}function
c(a,b){if(b){var
d=b[5],f=b[4],g=b[3],h=b[2],i=c(a,b[1]),j=e(a,g);return[0,i,h,j,c(a,f),d]}return 0}function
w(a,b){if(b){var
c=b[2],d=b[5],e=b[4],g=b[3],h=w(a,b[1]),i=f(a,c,g);return[0,h,c,i,w(a,e),d]}return 0}function
A(a,b,c){var
d=b,e=c;for(;;){if(d){var
f=d[4],g=d[3],h=d[2],j=i(a,h,g,A(a,d[1],e)),d=f,e=j;continue}return e}}function
B(a,b){var
c=b;for(;;){if(c){var
h=c[4],i=c[1],d=f(a,c[2],c[3]);if(d){var
e=B(a,i);if(e){var
c=h;continue}var
g=e}else
var
g=d;return g}return 1}}function
C(a,b){var
c=b;for(;;){if(c){var
h=c[4],i=c[1],d=f(a,c[2],c[3]);if(d)var
e=d;else{var
g=C(a,i);if(!g){var
c=h;continue}var
e=g}return e}return 0}}function
D(a,b,c){if(c){var
d=c[4],e=c[3],f=c[2];return h(D(a,b,c[1]),f,e,d)}return r(a,b)}function
E(a,b,c){if(c){var
d=c[3],e=c[2],f=c[1];return h(f,e,d,E(a,b,c[4]))}return r(a,b)}function
j(a,b,c,d){if(a){if(d){var
e=d[5],f=a[5],i=d[4],k=d[3],l=d[2],m=d[1],n=a[4],o=a[3],p=a[2],q=a[1];return(e+2|0)<f?h(q,p,o,j(n,b,c,d)):(f+2|0)<e?h(j(a,b,c,m),l,k,i):g(a,b,c,d)}return E(b,c,a)}return D(b,c,d)}function
q(a,b){if(a){if(b){var
c=p(b),d=c[2],e=c[1];return j(a,e,d,t(b))}return a}return b}function
F(a,b,c,d){return c?j(a,b,c[1],d):q(a,d)}function
n(a,b){if(b){var
c=b[4],d=b[3],e=b[2],g=b[1],k=f(l[1],a,e);if(0===k)return[0,g,[0,d],c];if(0<=k){var
h=n(a,c),m=h[3],o=h[2];return[0,j(g,e,d,h[1]),o,m]}var
i=n(a,g),p=i[2],q=i[1];return[0,q,p,j(i[3],e,d,c)]}return dm}function
o(a,b,c){if(b){var
d=b[2],h=b[5],j=b[4],l=b[3],m=b[1];if(k(c)<=h){var
e=n(d,c),p=e[2],q=e[1],r=o(a,j,e[3]),s=i(a,d,[0,l],p);return F(o(a,m,q),d,s,r)}}else
if(!c)return 0;if(c){var
f=c[2],t=c[4],u=c[3],v=c[1],g=n(f,b),w=g[2],x=g[1],y=o(a,g[3],t),z=i(a,f,w,[0,u]);return F(o(a,x,v),f,z,y)}throw[0,aK,dn]}function
x(a,b){if(b){var
c=b[3],d=b[2],h=b[4],e=x(a,b[1]),i=f(a,d,c),g=x(a,h);return i?j(e,d,c,g):q(e,g)}return 0}function
y(a,b){if(b){var
c=b[3],d=b[2],m=b[4],e=y(a,b[1]),g=e[2],h=e[1],n=f(a,d,c),i=y(a,m),k=i[2],l=i[1];if(n){var
o=q(g,k);return[0,j(h,d,c,l),o]}var
p=j(g,d,c,k);return[0,q(h,l),p]}return dp}function
d(a,b){var
c=a,d=b;for(;;){if(c){var
e=[0,c[2],c[3],c[4],d],c=c[1],d=e;continue}return d}}function
L(a,b,c){var
s=d(c,0),g=d(b,0),e=s;for(;;){if(g)if(e){var
k=e[4],m=e[3],n=e[2],o=g[4],p=g[3],q=g[2],i=f(l[1],g[1],e[1]);if(0===i){var
j=f(a,q,n);if(0===j){var
r=d(m,k),g=d(p,o),e=r;continue}var
h=j}else
var
h=i}else
var
h=1;else
var
h=e?-1:0;return h}}function
M(a,b,c){var
t=d(c,0),g=d(b,0),e=t;for(;;){if(g)if(e){var
m=e[4],n=e[3],o=e[2],p=g[4],q=g[3],r=g[2],i=0===f(l[1],g[1],e[1])?1:0;if(i){var
j=f(a,r,o);if(j){var
s=d(n,m),g=d(q,p),e=s;continue}var
k=j}else
var
k=i;var
h=k}else
var
h=0;else
var
h=e?0:1;return h}}function
b(a){if(a){var
c=a[1],d=b(a[4]);return(b(c)+1|0)+d|0}return 0}function
G(a,b){var
d=a,c=b;for(;;){if(c){var
e=c[3],f=c[2],g=c[1],d=[0,[0,f,e],G(d,c[4])],c=g;continue}return d}}return[0,a,H,J,s,r,v,o,L,M,z,A,B,C,x,y,b,function(a){return G(0,a)},p,K,p,n,I,c,w]}var
dr=[0,dq];function
ds(a){throw[0,dr]}function
R(a){var
b=1<=a?a:1,c=Q<b?Q:b,d=p(c);return[0,d,0,c,d]}function
S(a){return l(a[1],0,a[2])}function
bH(a,b){var
c=[0,a[3]];for(;;){if(c[1]<(a[2]+b|0)){c[1]=2*c[1]|0;continue}if(Q<c[1])if((a[2]+b|0)<=Q)c[1]=Q;else
ad(dt);var
d=p(c[1]);af(a[1],0,d,0,a[2]);a[1]=d;a[3]=c[1];return 0}}function
q(a,b){var
c=a[2];if(a[3]<=c)bH(a,1);a[1].safeSet(c,b);a[2]=c+1|0;return 0}function
ai(a,b){var
c=b.getLen(),d=a[2]+c|0;if(a[3]<d)bH(a,c);af(b,0,a[1],a[2],c);a[2]=d;return 0}function
aQ(a){return 0<=a?a:ad(j(du,ae(a)))}function
bI(a,b){return aQ(a+b|0)}var
dv=1;function
bJ(a){return bI(dv,a)}function
bK(a){return l(a,0,a.getLen())}function
bL(a,b,c){var
d=j(dx,j(a,dw)),e=j(dy,j(ae(b),d));return u(j(dz,j(H(1,c),e)))}function
T(a,b,c){return bL(bK(a),b,c)}function
aj(a){return u(j(dB,j(bK(a),dA)))}function
D(f,b,c,d){function
j(a){if((f.safeGet(a)+O|0)<0||9<(f.safeGet(a)+O|0))return a;var
b=a+1|0;for(;;){var
c=f.safeGet(b);if(48<=c){if(!(58<=c)){var
b=b+1|0;continue}var
d=0}else
if(36===c){var
e=b+1|0,d=1}else
var
d=0;if(!d)var
e=a;return e}}var
k=j(b+1|0),g=R((c-k|0)+10|0);q(g,37);var
e=d,i=0;for(;;){if(e){var
m=[0,e[1],i],e=e[2],i=m;continue}var
a=k,h=i;for(;;){if(a<=c){var
l=f.safeGet(a);if(42===l){if(h){var
n=h[2];ai(g,ae(h[1]));var
a=j(a+1|0),h=n;continue}throw[0,aK,dC]}q(g,l);var
a=a+1|0;continue}return S(g)}}}function
bM(a,b,c,d,e){var
f=D(b,c,d,e);if(78!==a)if(Y!==a)return f;f.safeSet(f.getLen()-1|0,bn);return f}function
bN(a){return function(c,b){var
m=c.getLen();function
n(a,b){var
o=40===a?41:be;function
k(a){var
d=a;for(;;){if(m<=d)return aj(c);if(37===c.safeGet(d)){var
e=d+1|0;if(m<=e)var
f=aj(c);else{var
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
f=g===o?e+1|0:T(c,b,g);break;case
2:break;default:var
f=k(n(g,e+1|0)+1|0)}}return f}var
d=d+1|0;continue}}return k(b)}return n(a,b)}}function
bO(k,b,c){var
n=k.getLen()-1|0;function
s(a){var
m=a;a:for(;;){if(m<n){if(37===k.safeGet(m)){var
e=0,j=m+1|0;for(;;){if(n<j)var
w=aj(k);else{var
o=k.safeGet(j);if(58<=o){if(95===o){var
e=1,j=j+1|0;continue}}else
if(32<=o)switch(o+cu|0){case
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
j=j+1|0;continue;case
10:var
j=i(b,e,j,F);continue;default:var
j=j+1|0;continue}var
d=j;b:for(;;){if(n<d)var
g=aj(k);else{var
l=k.safeGet(d);if(126<=l)var
h=0;else
switch(l){case
78:case
88:case
M:case
F:case
bg:case
bn:case
bo:var
g=i(b,e,d,F),h=1;break;case
69:case
70:case
71:case
cK:case
bj:case
bq:var
g=i(b,e,d,bj),h=1;break;case
33:case
37:case
44:case
64:var
g=d+1|0,h=1;break;case
83:case
91:case
_:var
g=i(b,e,d,_),h=1;break;case
97:case
aB:case
ba:var
g=i(b,e,d,l),h=1;break;case
76:case
cz:case
Y:var
t=d+1|0;if(n<t){var
g=i(b,e,d,F),h=1}else{var
q=k.safeGet(t)+cL|0;if(q<0||32<q)var
r=1;else
switch(q){case
0:case
12:case
17:case
23:case
29:case
32:var
g=f(c,i(b,e,d,l),F),h=1,r=0;break;default:var
r=1}if(r){var
g=i(b,e,d,F),h=1}}break;case
67:case
99:var
g=i(b,e,d,99),h=1;break;case
66:case
98:var
g=i(b,e,d,66),h=1;break;case
41:case
be:var
g=i(b,e,d,l),h=1;break;case
40:var
g=s(i(b,e,d,l)),h=1;break;case
bp:var
u=i(b,e,d,l),v=f(bN(l),k,u),p=u;for(;;){if(p<(v-2|0)){var
p=f(c,p,k.safeGet(p));continue}var
d=v-1|0;continue b}default:var
h=0}if(!h)var
g=T(k,d,l)}var
w=g;break}}var
m=w;continue a}}var
m=m+1|0;continue}return m}}s(0);return 0}function
bP(a){var
d=[0,0,0,0];function
b(a,b,c){var
f=41!==c?1:0,g=f?be!==c?1:0:f;if(g){var
e=97===c?2:1;if(aB===c)d[3]=d[3]+1|0;if(a)d[2]=d[2]+e|0;else
d[1]=d[1]+e|0}return b+1|0}bO(a,b,function(a,b){return a+1|0});return d[1]}function
bQ(a,b,c){var
i=a.safeGet(c);if((i+O|0)<0||9<(i+O|0))return f(b,0,c);var
e=i+O|0,d=c+1|0;for(;;){var
g=a.safeGet(d);if(48<=g){if(!(58<=g)){var
e=(10*e|0)+(g+O|0)|0,d=d+1|0;continue}var
h=0}else
if(36===g)if(0===e){var
j=ad(dE),h=1}else{var
j=f(b,[0,aQ(e-1|0)],d+1|0),h=1}else
var
h=0;if(!h)var
j=f(b,0,c);return j}}function
s(a,b){return a?b:bJ(b)}function
bR(a,b){return a?a[1]:b}function
bS(c){function
m(a){var
b=S(a);a[2]=0;return e(c,b)}var
aU=1;return function(h){var
N=R(2*h.getLen()|0);function
ar(a){return ai(N,a)}function
aS(a,b,m,aT){var
h=m.getLen();function
G(i,b){var
r=b;for(;;){if(h<=r)return e(a,N);var
c=m.safeGet(r);if(37===c){var
o=function(a,b){return n(aT,bR(a,b))},aF=function(h,g,c,d){var
a=d;for(;;){var
al=m.safeGet(a)+cu|0;if(!(al<0||25<al))switch(al){case
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
10:return bQ(m,function(a,b){var
d=[0,o(a,g),c];return aF(h,s(a,g),d,b)},a+1|0);default:var
a=a+1|0;continue}var
t=m.safeGet(a);if(124<=t)var
i=0;else
switch(t){case
78:case
88:case
M:case
F:case
bg:case
bn:case
bo:var
bd=o(h,g),be=ao(bM(t,m,r,a,c),bd),k=u(s(h,g),be,a+1|0),i=1;break;case
69:case
71:case
cK:case
bj:case
bq:var
a6=o(h,g),a7=cf(D(m,r,a,c),a6),k=u(s(h,g),a7,a+1|0),i=1;break;case
76:case
cz:case
Y:var
ap=m.safeGet(a+1|0)+cL|0;if(ap<0||32<ap)var
as=1;else
switch(ap){case
0:case
12:case
17:case
23:case
29:case
32:var
ab=a+1|0,aq=t-108|0;if(aq<0||2<aq)var
at=0;else{switch(aq){case
1:var
at=0,au=0;break;case
2:var
bc=o(h,g),aL=ao(D(m,r,ab,c),bc),au=1;break;default:var
bb=o(h,g),aL=ao(D(m,r,ab,c),bb),au=1}if(au){var
aK=aL,at=1}}if(!at){var
a$=o(h,g),aK=ji(D(m,r,ab,c),a$)}var
k=u(s(h,g),aK,ab+1|0),i=1,as=0;break;default:var
as=1}if(as){var
a8=o(h,g),a_=ao(bM(Y,m,r,a,c),a8),k=u(s(h,g),a_,a+1|0),i=1}break;case
37:case
64:var
k=u(g,H(1,t),a+1|0),i=1;break;case
83:case
_:var
A=o(h,g);if(_===t)var
B=A;else{var
b=[0,0],ay=A.getLen()-1|0,aV=0;if(!(ay<0)){var
U=aV;for(;;){var
z=A.safeGet(U),bm=14<=z?34===z?1:92===z?1:0:11<=z?13<=z?1:0:8<=z?1:0,aY=bm?2:a9(z)?1:4;b[1]=b[1]+aY|0;var
aZ=U+1|0;if(ay!==U){var
U=aZ;continue}break}}if(b[1]===A.getLen())var
aN=A;else{var
n=p(b[1]);b[1]=0;var
az=A.getLen()-1|0,aW=0;if(!(az<0)){var
Q=aW;for(;;){var
y=A.safeGet(Q),C=y-34|0;if(C<0||58<C)if(-20<=C)var
ac=1;else{switch(C+34|0){case
8:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],98);var
O=1;break;case
9:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],ba);var
O=1;break;case
10:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],Y);var
O=1;break;case
13:n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],aB);var
O=1;break;default:var
ac=1,O=0}if(O)var
ac=0}else
var
ac=(C-1|0)<0||56<(C-1|0)?(n.safeSet(b[1],92),b[1]++,n.safeSet(b[1],y),0):1;if(ac)if(a9(y))n.safeSet(b[1],y);else{n.safeSet(b[1],92);b[1]++;n.safeSet(b[1],48+(y/M|0)|0);b[1]++;n.safeSet(b[1],48+((y/10|0)%10|0)|0);b[1]++;n.safeSet(b[1],48+(y%10|0)|0)}b[1]++;var
aX=Q+1|0;if(az!==Q){var
Q=aX;continue}break}}var
aN=n}var
B=j(dP,j(aN,dO))}if(a===(r+1|0))var
aM=B;else{var
L=D(m,r,a,c);try{var
ad=0,w=1;for(;;){if(L.getLen()<=w)var
aA=[0,0,ad];else{var
ae=L.safeGet(w);if(49<=ae)if(58<=ae)var
av=0;else{var
aA=[0,js(l(L,w,(L.getLen()-w|0)-1|0)),ad],av=1}else{if(45===ae){var
ad=1,w=w+1|0;continue}var
av=0}if(!av){var
w=w+1|0;continue}}var
ah=aA;break}}catch(f){if(f[1]!==P)throw f;var
ah=bL(L,0,_)}var
V=ah[1],E=B.getLen(),a0=ah[2],W=0,a1=32;if(V===E)if(0===W){var
aj=B,aw=1}else
var
aw=0;else
var
aw=0;if(!aw)if(V<=E)var
aj=l(B,W,E);else{var
ag=H(V,a1);if(a0)af(B,W,ag,0,E);else
af(B,W,ag,V-E|0,E);var
aj=ag}var
aM=aj}var
k=u(s(h,g),aM,a+1|0),i=1;break;case
67:case
99:var
v=o(h,g);if(99===t)var
aI=H(1,v);else{if(39===v)var
x=c6;else
if(92===v)var
x=c7;else{if(14<=v)var
I=0;else
switch(v){case
8:var
x=c8,I=1;break;case
9:var
x=c9,I=1;break;case
10:var
x=c_,I=1;break;case
13:var
x=c$,I=1;break;default:var
I=0}if(!I)if(a9(v)){var
ax=p(1);ax.safeSet(0,v);var
x=ax}else{var
J=p(4);J.safeSet(0,92);J.safeSet(1,48+(v/M|0)|0);J.safeSet(2,48+((v/10|0)%10|0)|0);J.safeSet(3,48+(v%10|0)|0);var
x=J}}var
aI=j(dM,j(x,dL))}var
k=u(s(h,g),aI,a+1|0),i=1;break;case
66:case
98:var
a4=a+1|0,a5=o(h,g)?c0:c1,k=u(s(h,g),a5,a4),i=1;break;case
40:case
bp:var
aa=o(h,g),aG=f(bN(t),m,a+1|0);if(bp===t){var
X=R(aa.getLen()),aC=function(a,b){q(X,b);return a+1|0};bO(aa,function(a,b,c){if(a)ai(X,dD);else
q(X,37);return aC(b,c)},aC);var
a2=S(X),k=u(s(h,g),a2,aG),i=1}else{var
aH=s(h,g),bl=bI(bP(aa),aH),k=aS(function(a){return G(bl,aG)},aH,aa,aT),i=1}break;case
33:var
k=G(g,a+1|0),i=1;break;case
41:var
k=u(g,dJ,a+1|0),i=1;break;case
44:var
k=u(g,dK,a+1|0),i=1;break;case
70:var
am=o(h,g);if(0===c)var
aJ=dN;else{var
ak=D(m,r,a,c);if(70===t)ak.safeSet(ak.getLen()-1|0,bq);var
aJ=ak}var
aE=i9(am);if(3===aE)var
an=am<0?dG:dH;else
if(4<=aE)var
an=dI;else{var
$=cf(aJ,am),Z=0,a3=$.getLen();for(;;){if(a3<=Z)var
aD=j($,dF);else{var
K=$.safeGet(Z)-46|0,br=K<0||23<K?55===K?1:0:(K-1|0)<0||21<(K-1|0)?1:0;if(!br){var
Z=Z+1|0;continue}var
aD=$}var
an=aD;break}}var
k=u(s(h,g),an,a+1|0),i=1;break;case
91:var
k=T(m,a,t),i=1;break;case
97:var
aO=o(h,g),aP=bJ(bR(h,g)),aQ=o(0,aP),bf=a+1|0,bh=s(h,aP);if(aU)ar(f(aO,0,aQ));else
f(aO,N,aQ);var
k=G(bh,bf),i=1;break;case
aB:var
k=T(m,a,t),i=1;break;case
ba:var
aR=o(h,g),bi=a+1|0,bk=s(h,g);if(aU)ar(e(aR,0));else
e(aR,N);var
k=G(bk,bi),i=1;break;default:var
i=0}if(!i)var
k=T(m,a,t);return k}},d=r+1|0,g=0;return bQ(m,function(a,b){return aF(a,i,g,b)},d)}q(N,c);var
r=r+1|0;continue}}function
u(a,b,c){ar(b);return G(a,c)}return G(b,0)}var
d=aQ(0);function
i(a,b){return aS(m,d,a,b)}var
c=bP(h);if(c<0||6<c){var
k=function(j,b){if(c<=j){var
l=r(c,0),m=function(a,b){return g(l,(c-a|0)-1|0,b)},d=0,a=b;for(;;){if(a){var
e=a[2],f=a[1];if(e){m(d,f);var
d=d+1|0,a=e;continue}m(d,f)}return i(h,l)}}return function(a){return k(j+1|0,[0,a,b])}},a=k(0,0)}else
switch(c){case
1:var
a=function(a){var
b=r(1,0);g(b,0,a);return i(h,b)};break;case
2:var
a=function(a,b){var
c=r(2,0);g(c,0,a);g(c,1,b);return i(h,c)};break;case
3:var
a=function(a,b,c){var
d=r(3,0);g(d,0,a);g(d,1,b);g(d,2,c);return i(h,d)};break;case
4:var
a=function(a,b,c,d){var
e=r(4,0);g(e,0,a);g(e,1,b);g(e,2,c);g(e,3,d);return i(h,e)};break;case
5:var
a=function(a,b,c,d,e){var
f=r(5,0);g(f,0,a);g(f,1,b);g(f,2,c);g(f,3,d);g(f,4,e);return i(h,f)};break;case
6:var
a=function(a,b,c,d,e,f){var
j=r(6,0);g(j,0,a);g(j,1,b);g(j,2,c);g(j,3,d);g(j,4,e);g(j,5,f);return i(h,j)};break;default:var
a=i(h,[0])}return a}}function
bT(a){return e(bS(function(a){return a}),a)}var
bU=[0,0];function
bV(a){bU[1]=[0,a,bU[1]];return 0}32===ah;try{var
i5=ap(i4),aR=i5}catch(f){if(f[1]!==m)throw f;try{var
i3=ap(i2),bW=i3}catch(f){if(f[1]!==m)throw f;var
bW=dR}var
aR=bW}var
bF=aR.getLen(),dS=82,bG=0;if(0<=0)if(bF<bG)var
aq=0;else
try{var
ag=bG;for(;;){if(bF<=ag)throw[0,m];if(aR.safeGet(ag)!==dS){var
ag=ag+1|0;continue}var
dd=1,aM=dd,aq=1;break}}catch(f){if(f[1]!==m)throw f;var
aM=0,aq=1}else
var
aq=0;if(!aq)var
aM=u(dc);var
z=[cD,function(a){var
p=j0(0),b=[0,r(55,0),0],k=0===p.length-1?[0,0]:p,f=k.length-1,q=0,s=54;if(!(54<0)){var
d=q;for(;;){g(b[1],d,d);var
x=d+1|0;if(s!==d){var
d=x;continue}break}}var
h=[0,dQ],l=0,t=55,u=je(55,f)?t:f,m=54+u|0;if(!(m<l)){var
c=l;for(;;){var
o=c%55|0,v=h[1],i=j(v,ae(n(k,cg(c,f))));h[1]=jw(i,0,i.getLen());var
e=h[1];g(b[1],o,(n(b[1],o)^(((e.safeGet(0)+(e.safeGet(1)<<8)|0)+(e.safeGet(2)<<16)|0)+(e.safeGet(3)<<24)|0))&aw);var
w=c+1|0;if(m!==c){var
c=w;continue}break}}b[2]=0;return b}];function
bX(a,b){var
m=a?a[1]:aM,d=16;for(;;){if(!(b<=d))if(!(aO<(d*2|0))){var
d=d*2|0;continue}if(m){var
j=jK(z);if(N===j)var
c=z[1];else
if(cD===j){var
l=z[0+1];z[0+1]=ds;try{var
f=e(l,0);z[0+1]=f;jJ(z,dg)}catch(f){z[0+1]=function(a){throw f};throw f}var
c=f}else
var
c=z;c[2]=(c[2]+1|0)%55|0;var
h=n(c[1],c[2]),i=(n(c[1],(c[2]+24|0)%55|0)+(h^h>>>25&31)|0)&aw;g(c[1],c[2],i);var
k=i}else
var
k=0;return[0,0,r(d,0),k,d]}}function
aS(a,b){return 3<=a.length-1?jf(10,M,a[3],b)&(a[2].length-1-1|0):cg(jg(10,M,b),a[2].length-1)}function
ak(a,b){var
i=aS(a,b),d=n(a[2],i);if(d){var
e=d[3],j=d[2];if(0===ar(b,d[1]))return j;if(e){var
f=e[3],k=e[2];if(0===ar(b,e[1]))return k;if(f){var
l=f[3],o=f[2];if(0===ar(b,f[1]))return o;var
c=l;for(;;){if(c){var
g=c[3],h=c[2];if(0===ar(b,c[1]))return h;var
c=g;continue}throw[0,m]}}throw[0,m]}throw[0,m]}throw[0,m]}function
a(a,b){return a8(a,b[0+1])}var
aT=[0,0];a8(dT,aT);var
bY=aP([0,function(a,b){return a_(a,b)}]),bZ=aP([0,function(a,b){return a_(a,b)}]),b0=aP([0,function(a,b){return cU(a,b)}]),dU=ch(0,0);function
b1(a,b){var
c=a[2].length-1,g=c<b?1:0;if(g){var
d=r(b,dU),h=a[2],e=0,f=0,j=0<=c?0<=f?(h.length-1-c|0)<f?0:0<=e?(d.length-1-c|0)<e?0:(i6(h,f,d,e,c),1):0:0:0;if(!j)u(c4);a[2]=d;var
i=0}else
var
i=g;return i}var
dV=[0,0];function
aU(a){var
b=a[2].length-1;b1(a,b+1|0);return b}function
b2(a,b){try{var
d=f(bY[22],b,a[7])}catch(f){if(f[1]===m){var
c=a[1];a[1]=c+1|0;if(x(b,dW))a[7]=i(bY[4],b,c,a[7]);return c}throw f}return d}function
al(a){var
b=aU(a);if(0===(b%2|0))var
d=0;else
if((2+i_(n(a[2],1)*16|0,ah)|0)<b)var
d=0;else{var
c=aU(a),d=1}if(!d)var
c=b;g(a[2],c,0);return c}function
aV(a,b,c){if(as(c,dX))return b;var
d=c.getLen()-1|0;for(;;){if(0<=d){if(f(a,c,d)){var
d=d-1|0;continue}var
g=d+1|0,e=d;for(;;){if(0<=e){if(!f(a,c,e)){var
e=e-1|0;continue}var
h=l(c,e+1|0,(g-e|0)-1|0)}else
var
h=l(c,0,g);var
i=h;break}}else
var
i=l(c,0,1);return i}}function
aW(a,b,c){if(as(c,dY))return b;var
d=c.getLen()-1|0;for(;;){if(0<=d){if(f(a,c,d)){var
d=d-1|0;continue}var
e=d;for(;;){if(0<=e){if(!f(a,c,e)){var
e=e-1|0;continue}var
g=e;for(;;){if(0<=g){if(f(a,c,g)){var
g=g-1|0;continue}var
i=l(c,0,g+1|0)}else
var
i=l(c,0,1);var
h=i;break}}else
var
h=b;var
j=h;break}}else
var
j=l(c,0,1);return j}}function
aY(a,b){return 47===a.safeGet(b)?1:0}function
b3(a){var
b=a.getLen()<1?1:0,c=b||(47!==a.safeGet(0)?1:0);return c}function
d1(a){var
c=b3(a);if(c){var
e=a.getLen()<2?1:0,d=e||x(l(a,0,2),d3);if(d){var
f=a.getLen()<3?1:0,b=f||x(l(a,0,3),d2)}else
var
b=d}else
var
b=c;return b}function
d4(a,b){var
c=b.getLen()<=a.getLen()?1:0,d=c?as(l(a,a.getLen()-b.getLen()|0,b.getLen()),b):c;return d}try{var
i1=ap(i0),aZ=i1}catch(f){if(f[1]!==m)throw f;var
aZ=d5}function
b4(a){var
d=a.getLen(),b=R(d+20|0);q(b,39);var
e=d-1|0,f=0;if(!(e<0)){var
c=f;for(;;){if(39===a.safeGet(c))ai(b,d6);else
q(b,a.safeGet(c));var
g=c+1|0;if(e!==c){var
c=g;continue}break}}q(b,39);return S(b)}function
d7(a){return aV(aY,aX,a)}function
d8(a){return aW(aY,aX,a)}function
I(a,b){var
c=a.safeGet(b),d=47===c?1:0;if(d)var
e=d;else{var
f=92===c?1:0,e=f||(58===c?1:0)}return e}function
a1(a){var
e=a.getLen()<1?1:0,c=e||(47!==a.safeGet(0)?1:0);if(c){var
f=a.getLen()<1?1:0,d=f||(92!==a.safeGet(0)?1:0);if(d){var
g=a.getLen()<2?1:0,b=g||(58!==a.safeGet(1)?1:0)}else
var
b=d}else
var
b=c;return b}function
b5(a){var
c=a1(a);if(c){var
g=a.getLen()<2?1:0,d=g||x(l(a,0,2),ec);if(d){var
h=a.getLen()<2?1:0,e=h||x(l(a,0,2),eb);if(e){var
i=a.getLen()<3?1:0,f=i||x(l(a,0,3),ea);if(f){var
j=a.getLen()<3?1:0,b=j||x(l(a,0,3),d$)}else
var
b=f}else
var
b=e}else
var
b=d}else
var
b=c;return b}function
b6(a,b){var
c=b.getLen()<=a.getLen()?1:0;if(c){var
e=l(a,a.getLen()-b.getLen()|0,b.getLen()),f=bE(b),d=as(bE(e),f)}else
var
d=c;return d}try{var
iZ=ap(iY),b7=iZ}catch(f){if(f[1]!==m)throw f;var
b7=ed}function
ee(h){var
i=h.getLen(),e=R(i+20|0);q(e,34);function
g(a,b){var
c=b;for(;;){if(c===i)return q(e,34);var
f=h.safeGet(c);if(34===f)return a<50?d(1+a,0,c):X(d,[0,0,c]);if(92===f)return a<50?d(1+a,0,c):X(d,[0,0,c]);q(e,f);var
c=c+1|0;continue}}function
d(a,b,c){var
f=b,d=c;for(;;){if(d===i){q(e,34);return a<50?j(1+a,f):X(j,[0,f])}var
l=h.safeGet(d);if(34===l){k((2*f|0)+1|0);q(e,34);return a<50?g(1+a,d+1|0):X(g,[0,d+1|0])}if(92===l){var
f=f+1|0,d=d+1|0;continue}k(f);return a<50?g(1+a,d):X(g,[0,d])}}function
j(a,b){var
d=1;if(!(b<1)){var
c=d;for(;;){q(e,92);var
f=c+1|0;if(b!==c){var
c=f;continue}break}}return 0}function
a(b){return a$(g(0,b))}function
b(b,c){return a$(d(0,b,c))}function
k(b){return a$(j(0,b))}a(0);return S(e)}function
b8(a){var
c=2<=a.getLen()?1:0;if(c){var
b=a.safeGet(0),g=91<=b?(b+cx|0)<0||25<(b+cx|0)?0:1:65<=b?1:0,d=g?1:0,e=d?58===a.safeGet(1)?1:0:d}else
var
e=c;if(e){var
f=l(a,2,a.getLen()-2|0);return[0,l(a,0,2),f]}return[0,ef,a]}function
eg(a){var
b=b8(a),c=b[1];return j(c,aW(I,a0,b[2]))}function
eh(a){return aV(I,a0,b8(a)[2])}function
ek(a){return aV(I,a2,a)}function
el(a){return aW(I,a2,a)}if(x(aN,em))if(x(aN,en)){if(x(aN,eo))throw[0,aK,ep];var
am=[0,a0,d9,d_,I,a1,b5,b6,b7,ee,eh,eg]}else
var
am=[0,aX,dZ,d0,aY,b3,d1,d4,aZ,b4,d7,d8];else
var
am=[0,a2,ei,ej,I,a1,b5,b6,aZ,b4,ek,el];var
b9=[0,es],eq=am[11],er=am[3];a(ev,[0,b9,0,eu,et]);bV(function(a){if(a[1]===b9){var
c=a[2],d=a[4],f=a[3];if(typeof
c===cr)switch(c){case
1:var
b=ey;break;case
2:var
b=ez;break;case
3:var
b=eA;break;case
4:var
b=eB;break;case
5:var
b=eC;break;case
6:var
b=eD;break;case
7:var
b=eE;break;case
8:var
b=eF;break;case
9:var
b=eG;break;case
10:var
b=eH;break;case
11:var
b=eI;break;case
12:var
b=eJ;break;case
13:var
b=eK;break;case
14:var
b=eL;break;case
15:var
b=eM;break;case
16:var
b=eN;break;case
17:var
b=eO;break;case
18:var
b=eP;break;case
19:var
b=eQ;break;case
20:var
b=eR;break;case
21:var
b=eS;break;case
22:var
b=eT;break;case
23:var
b=eU;break;case
24:var
b=eV;break;case
25:var
b=eW;break;case
26:var
b=eX;break;case
27:var
b=eY;break;case
28:var
b=eZ;break;case
29:var
b=e0;break;case
30:var
b=e1;break;case
31:var
b=e2;break;case
32:var
b=e3;break;case
33:var
b=e4;break;case
34:var
b=e5;break;case
35:var
b=e6;break;case
36:var
b=e7;break;case
37:var
b=e8;break;case
38:var
b=e9;break;case
39:var
b=e_;break;case
40:var
b=e$;break;case
41:var
b=fa;break;case
42:var
b=fb;break;case
43:var
b=fc;break;case
44:var
b=fd;break;case
45:var
b=fe;break;case
46:var
b=ff;break;case
47:var
b=fg;break;case
48:var
b=fh;break;case
49:var
b=fi;break;case
50:var
b=fj;break;case
51:var
b=fk;break;case
52:var
b=fl;break;case
53:var
b=fm;break;case
54:var
b=fn;break;case
55:var
b=fo;break;case
56:var
b=fp;break;case
57:var
b=fq;break;case
58:var
b=fr;break;case
59:var
b=fs;break;case
60:var
b=ft;break;case
61:var
b=fu;break;case
62:var
b=fv;break;case
63:var
b=fw;break;case
64:var
b=fx;break;case
65:var
b=fy;break;case
66:var
b=fz;break;case
67:var
b=fA;break;default:var
b=ew}else{var
g=c[1],b=e(bT(fB),g)}return[0,i(bT(ex),b,f,d)]}return 0});at(fC);at(fD);try{at(iX)}catch(f){if(f[1]!==P)throw f}try{at(iW)}catch(f){if(f[1]!==P)throw f}bX(0,7);H(32,h);var
fI=p(cI),fJ=0,fK=h;if(!(h<0)){var
V=fJ;for(;;){fI.safeSet(V,bD(aL(V)));var
iV=V+1|0;if(fK!==V){var
V=iV;continue}break}}var
a3=H(32,0);a3.safeSet(10>>>3,aL(a3.safeGet(10>>>3)|1<<(10&7)));var
fE=p(32),fF=0,fG=31;if(!(31<0)){var
U=fF;for(;;){fE.safeSet(U,aL(a3.safeGet(U)^h));var
fH=U+1|0;if(fG!==U){var
U=fH;continue}break}}var
J=[0,0],K=[0,0],a4=[0,0];function
a5(a){return J[1]}a(fZ,[0,[0,fL]]);a(f0,[0,[0,fM]]);a(f1,[0,[0,fN]]);a(f2,[0,[0,fO]]);a(f3,[0,[0,fP]]);a(f4,[0,[0,fQ]]);a(f5,[0,[0,fR]]);a(f6,[0,[0,fS]]);a(f7,[0,[0,fT]]);a(f8,[0,[0,fV]]);a(f9,[0,[0,fW]]);a(f_,[0,[0,fX]]);a(f$,[0,[0,fY]]);a(ga,[0,[0,fU]]);a(gy,[0,[0,gb]]);a(gz,[0,[0,gc]]);a(gA,[0,[0,gd]]);a(gB,[0,[0,ge]]);a(gC,[0,[0,gf]]);a(gD,[0,[0,gg]]);a(gE,[0,[0,gh]]);a(gF,[0,[0,gi]]);a(gG,[0,[0,gj]]);a(gH,[0,[0,gk]]);a(gI,[0,[0,gl]]);a(gJ,[0,[0,gm]]);a(gK,[0,[0,gn]]);a(gL,[0,[0,go]]);a(gM,[0,[0,gp]]);a(gN,[0,[0,gq]]);a(gO,[0,[0,gr]]);a(gP,[0,[0,gs]]);a(gQ,[0,[0,gt]]);a(gR,[0,[0,gu]]);a(gS,[0,[0,gv]]);a(gT,[0,[0,gw]]);a(gU,[0,[0,gx]]);function
an(a,b){var
r=n(de,0),s=j(er,j(a,b)),e=cd(jZ(j(eq(r),s),c2,0));try{var
o=ca,g=ca;a:for(;;){if(1){var
k=function(a,b,c){var
e=b,d=c;for(;;){if(d){var
g=d[1],f=g.getLen(),h=d[2];W(g,0,a,e-f|0,f);var
e=e-f|0,d=h;continue}return a}},d=0,f=0;for(;;){var
c=jA(e);if(0===c){if(!d)throw[0,aJ];var
i=k(p(f),f,d)}else{if(!(0<c)){var
m=p(-c|0);ci(e,m,0,-c|0);var
d=[0,m,d],f=f-c|0;continue}var
h=p(c-1|0);ci(e,h,0,c-1|0);jz(e);if(d){var
l=(f+c|0)-1|0,i=k(p(l),l,[0,h,d])}else
var
i=h}var
g=j(g,j(i,gV)),o=g;continue a}}var
q=g;break}}catch(f){if(f[1]!==aJ)throw f;var
q=o}jx(e);return q}var
cb=[0,gW],gX=[],gY=0,gZ=0;j3(gX,[0,0,function(j){var
aC=b2(j,g0),aF=i$(b$,0)?[0]:b$,y=aF.length-1,aG=b_.length-1,k=r(y+aG|0,0),aH=y-1|0,aN=0;if(!(aH<0)){var
p=aN;for(;;){var
aJ=n(aF,p);try{var
aM=f(bZ[22],aJ,j[3]),aK=aM}catch(f){if(f[1]!==m)throw f;var
x=aU(j);j[3]=i(bZ[4],aJ,x,j[3]);j[4]=i(b0[4],x,1,j[4]);var
aK=x}g(k,p,aK);var
aR=p+1|0;if(aH!==p){var
p=aR;continue}break}}var
aI=aG-1|0,aP=0;if(!(aI<0)){var
o=aP;for(;;){g(k,o+y|0,b2(j,n(b_,o)));var
aQ=o+1|0;if(aI!==o){var
o=aQ;continue}break}}var
aL=k[10],aD=k[12],s=k[15],t=k[16],u=k[17],l=k[18],aW=k[1],aX=k[2],aY=k[3],aZ=k[4],a0=k[5],a1=k[7],a2=k[8],a3=k[9],a4=k[11],a5=k[14];function
a6(a,b,c,d,e,f){var
g=d?d[1]:d;i(a[1][aD+1],a,[0,g],f);var
h=ak(a[l+1],f);return cj(a[1][aL+1],a,b,[0,c[1],c[2]],e,f,h)}function
a7(a,b,c,d,e){try{var
f=ak(a[l+1],e),g=f}catch(f){if(f[1]!==m)throw f;try{i(a[1][aD+1],a,g1,e)}catch(f){throw f}var
g=ak(a[l+1],e)}return cj(a[1][aL+1],a,b,[0,c[1],c[2]],d,e,g)}function
a8(a,b,c){var
y=b?b[1]:b;try{ak(a[l+1],c);var
f=0}catch(f){if(f[1]===m){if(0===c[2][0]){var
z=a[t+1];if(!z)throw[0,cb,c];var
A=z[1],H=y?j9(A,a[s+1],c[1]):j8(A,a[s+1],c[1]),B=H}else{var
D=a[u+1];if(!D)throw[0,cb,c];var
E=D[1],I=y?j_(E,a[s+1],c[1]):kd(E,a[s+1],c[1]),B=I}var
d=a[l+1],w=aS(d,c);g(d[2],w,[0,c,B,n(d[2],w)]);d[1]=d[1]+1|0;var
x=d[2].length-1<<1<d[1]?1:0;if(x){var
i=d[2],j=i.length-1,k=j*2|0,o=k<aO?1:0;if(o){var
h=r(k,0);d[2]=h;var
p=function(a){if(a){var
b=a[1],e=a[2];p(a[3]);var
c=aS(d,b);return g(h,c,[0,b,e,n(h,c)])}return 0},q=j-1|0,F=0;if(!(q<0)){var
e=F;for(;;){p(n(i,e));var
G=e+1|0;if(q!==e){var
e=G;continue}break}}var
v=0}else
var
v=o;var
C=v}else
var
C=x;return C}throw f}return f}function
a9(a,b){try{var
f=[0,an(a[aC+1],g3),0],c=f}catch(f){var
c=0}a[t+1]=c;try{var
e=[0,an(a[aC+1],g2),0],d=e}catch(f){var
d=0}a[u+1]=d;return 0}function
a_(a,b){a[u+1]=[0,b,0];return 0}function
a$(a,b){return a[u+1]}function
ba(a,b){a[t+1]=[0,b,0];return 0}function
bb(a,b){return a[t+1]}function
bc(a,b){var
d=a[l+1];d[1]=0;var
e=d[2].length-1-1|0,f=0;if(!(e<0)){var
c=f;for(;;){g(d[2],c,0);var
h=c+1|0;if(e!==c){var
c=h;continue}break}}return 0}var
aE=[0,a3,function(a,b){return a[l+1]},aZ,bc,a2,bb,aX,ba,a1,a$,aW,a_,a0,a9,aD,a8,aY,a7,a4,a6],q=[0,0],aV=aE.length-1;for(;;){if(q[1]<aV){var
w=n(aE,q[1]),a=function(a){q[1]++;return n(aE,q[1])},z=a(0);if(typeof
z===cr)switch(z){case
1:var
B=a(0),h=function(B){return function(a){return a[B+1]}}(B);break;case
2:var
C=a(0),b=a(0),h=function(C,b){return function(a){return a[C+1][b+1]}}(C,b);break;case
3:var
D=a(0),h=function(D){return function(a){return e(a[1][D+1],a)}}(D);break;case
4:var
E=a(0),h=function(E){return function(a,b){a[E+1]=b;return 0}}(E);break;case
5:var
F=a(0),G=a(0),h=function(F,G){return function(a){return e(F,G)}}(F,G);break;case
6:var
H=a(0),I=a(0),h=function(H,I){return function(a){return e(H,a[I+1])}}(H,I);break;case
7:var
J=a(0),K=a(0),c=a(0),h=function(J,K,c){return function(a){return e(J,a[K+1][c+1])}}(J,K,c);break;case
8:var
L=a(0),M=a(0),h=function(L,M){return function(a){return e(L,e(a[1][M+1],a))}}(L,M);break;case
9:var
N=a(0),O=a(0),P=a(0),h=function(N,O,P){return function(a){return f(N,O,P)}}(N,O,P);break;case
10:var
Q=a(0),R=a(0),S=a(0),h=function(Q,R,S){return function(a){return f(Q,R,a[S+1])}}(Q,R,S);break;case
11:var
T=a(0),U=a(0),V=a(0),d=a(0),h=function(T,U,V,d){return function(a){return f(T,U,a[V+1][d+1])}}(T,U,V,d);break;case
12:var
W=a(0),X=a(0),Y=a(0),h=function(W,X,Y){return function(a){return f(W,X,e(a[1][Y+1],a))}}(W,X,Y);break;case
13:var
Z=a(0),_=a(0),$=a(0),h=function(Z,_,$){return function(a){return f(Z,a[_+1],$)}}(Z,_,$);break;case
14:var
aa=a(0),ab=a(0),ac=a(0),ad=a(0),h=function(aa,ab,ac,ad){return function(a){return f(aa,a[ab+1][ac+1],ad)}}(aa,ab,ac,ad);break;case
15:var
ae=a(0),af=a(0),ag=a(0),h=function(ae,af,ag){return function(a){return f(ae,e(a[1][af+1],a),ag)}}(ae,af,ag);break;case
16:var
ah=a(0),ai=a(0),h=function(ah,ai){return function(a){return f(a[1][ah+1],a,ai)}}(ah,ai);break;case
17:var
aj=a(0),am=a(0),h=function(aj,am){return function(a){return f(a[1][aj+1],a,a[am+1])}}(aj,am);break;case
18:var
ao=a(0),ap=a(0),aq=a(0),h=function(ao,ap,aq){return function(a){return f(a[1][ao+1],a,a[ap+1][aq+1])}}(ao,ap,aq);break;case
19:var
ar=a(0),as=a(0),h=function(ar,as){return function(a){var
b=e(a[1][as+1],a);return f(a[1][ar+1],a,b)}}(ar,as);break;case
20:var
at=a(0),v=a(0);al(j);var
h=function(at,v){return function(a){return e(au(v,at,0),v)}}(at,v);break;case
21:var
av=a(0),aw=a(0);al(j);var
h=function(av,aw){return function(a){var
b=a[aw+1];return e(au(b,av,0),b)}}(av,aw);break;case
22:var
ax=a(0),ay=a(0),az=a(0);al(j);var
h=function(ax,ay,az){return function(a){var
b=a[ay+1][az+1];return e(au(b,ax,0),b)}}(ax,ay,az);break;case
23:var
aA=a(0),aB=a(0);al(j);var
h=function(aA,aB){return function(a){var
b=e(a[1][aB+1],a);return e(au(b,aA,0),b)}}(aA,aB);break;default:var
A=a(0),h=function(A){return function(a){return A}}(A)}else
var
h=z;dV[1]++;if(f(b0[22],w,j[4])){b1(j,w+1|0);g(j[2],w,h)}else
j[6]=[0,[0,w,h],j[6]];q[1]++;continue}return function(a,b,c,d){if(b)var
e=b;else{var
f=ch(df,j[1]);f[0+1]=j[2];var
g=aT[1];f[1+1]=g;aT[1]=g+1|0;var
e=f}e[aC+1]=c;e[a5+1]=c;e[s+1]=d;try{var
m=[0,an(c,g5),0],h=m}catch(f){var
h=0}e[t+1]=h;try{var
k=[0,an(c,g4),0],i=k}catch(f){var
i=0}e[u+1]=i;e[l+1]=bX(0,8);return e}}},gZ,gY]);ck(0);ck(0);var
a6=by,g6=undefined,g7=false,g8=Array;bV(function(a){return a
instanceof
g8?0:[0,new
G(a.toString())]});var
g9=a6.document;a6.HTMLElement===g6;function
c(a){return e(bS(function(a){return g9.write(j(a,g_).toString())}),a)}function
cc(a){var
X=0,S=0?X[1]:2;switch(S){case
1:cl(0);K[1]=cm(0);break;case
2:cn(0);J[1]=co(0);cl(0);K[1]=cm(0);break;default:cn(0);J[1]=co(0)}a4[1]=J[1]+K[1]|0;var
E=J[1]-1|0,D=0,T=0;if(E<0)var
F=D;else{var
p=T,I=D;for(;;){var
L=aH(I,[0,j$(p),0]),U=p+1|0;if(E!==p){var
p=U,I=L;continue}var
F=L;break}}var
x=0,j=0,h=F;for(;;){if(x<K[1]){if(ke(j)){var
H=j+1|0,G=aH(h,[0,kb(j,j+J[1]|0),0])}else{var
H=j,G=h}var
x=x+1|0,j=H,h=G;continue}var
w=0,v=h;for(;;){if(v){var
w=w+1|0,v=v[2];continue}a4[1]=w;K[1]=j;if(h){var
t=0,s=h,P=h[2],Q=h[1];for(;;){if(s){var
t=t+1|0,s=s[2];continue}var
C=r(t,Q),u=1,m=P;for(;;){if(m){var
R=m[2];C[u+1]=m[1];var
u=u+1|0,m=R;continue}var
o=C;break}break}}else
var
o=[0];c(g$);var
W=a4[1];e(c(ha),W);var
Y=a5(0);e(c(hb),Y);var
V=K[1];e(c(hc),V);jF(aI,10);a7(aI);c(hd);var
B=o.length-1-1|0,N=0;if(!(B<0)){var
k=N;for(;;){var
g=o[k+1];e(c(he),k);var
Z=g[1][1];e(c(hf),Z);var
_=g[1][2];e(c(hg),_);var
$=g[1][3];e(c(hh),$);var
aa=g[1][4];e(c(hi),aa);var
ab=g[1][5];e(c(hj),ab);var
ac=g[1][6];e(c(hk),ac);var
ae=g[1][7];e(c(hl),ae);if(0===g[2][0])c(hm);else
c(iU);var
y=g[2];if(0===y[0]){var
d=y[1],af=d[18]/cv|0;e(c(hn),af);var
ag=d[2],ah=d[1];f(c(ho),ah,ag);var
ai=d[3];e(c(hp),ai);var
aj=d[4];e(c(hq),aj);var
ak=d[5];e(c(hr),ak);var
al=d[6];e(c(hs),al);var
z=d[7],am=z[3],an=z[2],ao=z[1];i(c(ht),ao,an,am);var
A=d[8],ap=A[3],aq=A[2],ar=A[1];i(c(hu),ar,aq,ap);var
as=d[9];e(c(hv),as);var
at=d[10];e(c(hw),at);var
au=d[11];e(c(hx),au);var
av=d[12];e(c(hy),av);var
aw=d[13];e(c(hz),aw);var
ax=d[14];e(c(hA),ax);var
ay=d[15];e(c(hB),ay);var
az=d[16];e(c(hC),az);var
aA=d[17];e(c(hD),aA);a7(aI)}else{var
b=y[1],l=b[1],aB=l[3];e(c(hE),aB);var
aC=l[1];e(c(hF),aC);var
aD=l[2];e(c(hG),aD);var
aE=l[4];e(c(hH),aE);var
aF=l[5];e(c(hI),aF);var
aG=l[6];e(c(hJ),aG);c(hK);switch(b[2]){case
1:c(iR);break;case
2:c(iS);break;case
3:c(iT);break;default:c(hL)}var
aJ=b[3];e(c(hM),aJ);var
aK=b[4];e(c(hN),aK);var
aL=b[5];e(c(hO),aL);var
aM=b[46];e(c(hP),aM);var
aN=b[6];e(c(hQ),aN);var
aO=b[7];e(c(hR),aO);var
aP=b[8];e(c(hS),aP);var
aQ=b[31];e(c(hT),aQ);var
aR=b[38][3],aS=b[38][2],aT=b[38][1];i(c(hU),aT,aS,aR);var
aU=b[9];e(c(hV),aU);var
aV=b[10];e(c(hW),aV);var
aW=b[11];e(c(hX),aW);var
aX=b[12];e(c(hY),aX);var
aY=b[13];e(c(hZ),aY);var
aZ=b[14];e(c(h0),aZ);var
a0=b[15];e(c(h1),a0);var
a1=b[16];e(c(h2),a1);var
a2=b[17];e(c(h3),a2);var
a3=b[18];e(c(h4),a3);var
a6=b[19];e(c(h5),a6);var
a8=b[20];e(c(h6),a8);var
a9=b[21];e(c(h7),a9);var
a$=b[22];e(c(h8),a$);c(h9);switch(b[23]){case
1:c(iL);break;case
2:c(iM);break;case
3:c(iN);break;case
4:c(iO);break;case
5:c(iP);break;case
6:ad(iQ);break;default:c(h_)}c(h$);switch(b[27]){case
1:c(iF);break;case
2:c(iG);break;case
3:c(iH);break;case
4:c(iI);break;case
5:c(iJ);break;case
6:c(iK);break;default:c(ia)}c(ib);switch(b[30]){case
1:c(iz);break;case
2:c(iA);break;case
3:c(iB);break;case
4:c(iC);break;case
5:c(iD);break;case
6:c(iE);break;default:c(ic)}c(id);switch(b[24]){case
1:c(ix);break;case
2:c(iy);break;default:c(ie)}c(ig);if(0===b[25])c(ih);else
c(iw);c(ii);if(0===b[26])c(ij);else
c(iv);var
ba=b[33],bb=b[32];f(c(ik),bb,ba);var
bc=b[36],bd=b[35],be=b[34];i(c(il),be,bd,bc);var
bf=b[39];e(c(im),bf);var
bg=b[40];e(c(io),bg);var
bh=b[41];e(c(ip),bh);var
bi=b[42];e(c(iq),bi);var
bj=b[43];e(c(ir),bj);var
bk=b[44];e(c(is),bk);var
bl=b[45];e(c(it),bl);var
M=a5(0)-1|0,bm=0;if(!(M<0)){var
q=bm;for(;;){if(0===a_(g[1][1],n(o,q)[1][1])){var
bn=k-a5(0)|0;e(c(iu),bn)}var
bo=q+1|0;if(M!==q){var
q=bo;continue}break}}}var
O=k+1|0;if(B!==k){var
k=O;continue}break}}return g7}}}a6.onload=ju(function(a){if(a){var
d=cc(a);if(!(d|0))a.preventDefault();return d}var
c=event,b=cc(c);if(!(b|0))c.returnValue=b;return b});bC(0);return}(this));
