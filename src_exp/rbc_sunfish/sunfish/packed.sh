#!/bin/sh
T=`mktemp`
tail -c +82 $0|xz -d>$T
chmod +x $T
(sleep 9;rm $T)&exec $T
�7zXZ  �ִF    �D!��}    YZ