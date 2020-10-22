#!/bin/sh

# usage mk_turntable __out_images test cuda internal://ls_torus_100:[0-10]

set -e

if [ $# -lt 4 ]; then
    echo "$0 <outdir> <name> <platform> <asset>"
    exit 0
fi

OUTDIR=$1; shift
NAME=$1; shift
PLATFORM=$1; shift

echo "OUTDIR = $OUTDIR"
echo "NAME = $NAME"
echo "PLATFORM = $PLATFORM"
echo "URL = $URL"

mkdir -p ${OUTDIR}

nanovdb_viewer -b -p ${PLATFORM} --render-tonemapping no --render-camera-turntable yes --render-output "${OUTDIR}/${NAME}-${PLATFORM}.%04d.png" $@

ffmpeg -y -loglevel quiet -stats -framerate 30 -i ${OUTDIR}/${NAME}-${PLATFORM}.%04d.png ${OUTDIR}/${NAME}-${PLATFORM}.mp4
rm ${OUTDIR}/${NAME}-${PLATFORM}.*.png

mplayer ${OUTDIR}/${NAME}-${PLATFORM}.mp4 -loop 0