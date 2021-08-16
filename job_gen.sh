#!/bin/bash -x
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBSDIR="$HERE/jobs"
rm -rf "$JOBSDIR"
mkdir -p "$JOBSDIR"
TEMPLATE=$1
N=$2
while true; do
    for GPU in $(bhosts | grep gpu | grep ok | awk '{print $1}'); do
        FILE="$JOBSDIR/$N.$(basename $TEMPLATE)"
        eval "cat <<EOF
$(<$TEMPLATE)
EOF" >$FILE
        ((N--))
        if ((N <= 0)); then
            break 2
        fi
    done
done
