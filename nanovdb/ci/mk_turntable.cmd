echo off

set OUTDIR=%1
shift
set NAME=%1
shift
set PLATFORM=%1
shift

set args=%1
shift
:start
if [%1] == [] goto done
set args=%args% %1
shift
goto start
:done

IF "%NAME%"=="" GOTO Usage
IF "%PLATFORM%"=="" GOTO Usage

echo "OUTDIR = %OUTDIR%"
echo "NAME = %NAME%"
echo "PLATFORM = %PLATFORM%"

mkdir %OUTDIR%

set OUTFILEPATHBASE=%OUTDIR%\%NAME%-%PLATFORM%
IF "%OUTFILEPATHBASE%"=="" GOTO Usage

nanovdb_viewer -b -p %PLATFORM% --render-tonemapping no --render-camera-turntable yes --render-output "%OUTFILEPATHBASE%.%%04d.png" %args%

ffmpeg -y -loglevel quiet -stats -framerate 30 -i %OUTFILEPATHBASE%.%%04d.png %OUTFILEPATHBASE%.mp4
del %OUTFILEPATHBASE%.*.png

GOTO Exit

:Usage
echo "usage mk_turntable.cmd __out_images test cuda <nanovdb_viewer args...>"
GOTO Exit

:Exit
