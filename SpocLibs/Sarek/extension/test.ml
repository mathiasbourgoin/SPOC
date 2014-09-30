ktype t1 = X | Y of int

ktype t2 = 
{
  x : t1;
  y : int;
}

ktype t3 = 
 A
| B of int
| C of t2

  ;;

