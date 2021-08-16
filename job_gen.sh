#!/bin/bash -x
HERE=$(dirname ${BASH_SOURCE[0]})

TEMPLATE=$1
N=$2
while true; do
    for GPU in $(lsload | grep gpu | grep ok | awk '{print $1}'); do
        eval "cat <<EOF
$(<$TEMPLATE)
EOF" >$HERE/jobs/$N.$(basename $TEMPLATE)
        ((N--))
        if ((N <= 0)); then
            break 2
        fi
    done
done
