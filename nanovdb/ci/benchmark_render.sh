#!/usr/bin/env bash
function extract_benchmark {
    awk '/Average duration :/{sub(/.*Average duration : /, ""); print $1}'
}

EXEC=$1; shift
NAME=$1; shift

GRIDS=(
    "internal://ls_sphere_100" 
    "internal://ls_torus_100" 
    "internal://ls_bbox_100" 
    "internal://ls_box_100" 
)

# find available platforms...

#RENDER_PLATFORMS=(cuda)
RENDER_PLATFORMS=(`eval $EXEC -l`)

NUM_RENDER_PLATFORMS=${#RENDER_PLATFORMS[@]}

echo "Found $NUM_RENDER_PLATFORMS platforms:"
for i in ${RENDER_PLATFORMS[@]}; do
    printf "$i, "
done
printf "\n"

outname=benchmark-render-$NAME
outfile="$outname.tsv"
outputname="$outname.png"

# write file...

printf "Platform" > $outfile
for i in ${RENDER_PLATFORMS[@]}; do
    printf "\t\"$i\"" >> $outfile
done
printf "\n" >> $outfile

for i in ${GRIDS[@]}; do

    shortname=`basename $i`

    echo $shortname
    printf "$shortname" >> $outfile
    for j in ${RENDER_PLATFORMS[@]}; do
        echo $j
        v=`eval "$EXEC" -b -p $j -n 10 --turntable $i | extract_benchmark`
        if [ -z $v ]; then v=-1; fi
        printf "\t%f" "$v" >> $outfile
    done
    printf "\n" >> $outfile
done

cat $outfile

gnuplot -e "NUM=$NUM_RENDER_PLATFORMS" -e "IN='$outfile'" -e "OUT='$outputname'" ./ci/benchmark-render.gnuplot
