set terminal png size 1024,768
set output OUT
set style data histogram
set style histogram clustered
set style fill solid border
set xtics noenhanced out
set grid ytics
set key tmargin
set ylabel "ms"
set yrange [0:30]
plot for [ROW=2:NUM+1] IN using ROW:xticlabels(1) title columnheader
