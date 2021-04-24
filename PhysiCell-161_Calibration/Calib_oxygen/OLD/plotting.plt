# Set the output to a png file
#set terminal png size 1280, 480 font "Helvetica,20"
# The file we'll write to
set output 'OxyCalib.png'

set style fill transparent solid 0.2
set colorsequence classic

set key left

set xlabel 'Distance from tumor core (mm)'
set ylabel 'PO_2 (mmHg)'

set yrange [0:]
set xrange [-0.05:2.05]

plot "OxyoutputMAP" using 1:($2) with lines lc rgb '#0000FF' lw 2 t "MAP",\
	 "OxyoutputMIN" using 1:($2) with lines lc 1 lw 2 t "Min",\
"DataOxy.dat" using ($1):($2):($3) with yerrorbars lc rgb '#000000' lt 7 t "Data"
	 
set output 'Data.png'
plot "DataOxy.dat" using ($1):($2):($3) with yerrorbars lc rgb '#0000FF' lt 7 notitle,\
	"DataOxy.dat" using ($1):($2):($3) with line lc rgb '#0000FF' lt 7 notitle