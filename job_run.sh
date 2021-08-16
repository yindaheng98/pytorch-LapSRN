#!/bin/bash -x
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBSDIR="$HERE/jobs"
for JOB in $JOBSDIR/*; do
    bsub < "$JOB"
done
