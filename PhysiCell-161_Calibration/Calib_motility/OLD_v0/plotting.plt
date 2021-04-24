# Set the output to a png file
set terminal png size 1280, 480 font "Helvetica,20"
# The file we'll write to
set output 'RedCalib.png'

set style fill transparent solid 0.2
set colorsequence classic

set xlabel 'Time (minutes)'
set ylabel 'Displacement ({/Symbol m}m)'

set yrange[0:]

set key left

plot "RedoutputMAP" using 1:($2 + $3):($2 - $3) with filledcu fill lc rgb '#e56b5d' notitle,\
"RedoutputMAP" using 1:($2) with lines lc rgb '#e56b5d' lw 2 t "MAP",\
"RedoutputMIN" using 1:($2) with lines lc rgb '#808080' lw 2 t "Min",\
"../DataMot.dat" using ($1*15):($2):($3) with yerrorbars lc rgb '#000000' lt 7 t "Data"

set output 'GreenCalib.png'
plot "GreenoutputMAP" using 1:($2 + $3):($2 - $3) with filledcu fill lc rgb '#27ad81' notitle,\
	 "GreenoutputMAP" using 1:($2) with lines lc rgb '#27ad81' lw 2 t "MAP",\
	 "GreenoutputMIN" using 1:($2) with lines lc rgb '#808080' lw 2 t "Min",\
	 "../DataMot.dat" using ($1*15):($4):($5) with yerrorbars lc rgb '#000000' lt 7 t "Data"
	 
set output 'Data.png'
plot "../DataMot.dat" using ($1*15):($2):($3) with yerrorbars lc rgb '#e56b5d' lt 7 t "DsRed",\
"../DataMot.dat" using ($1*15):($4):($5) with yerrorbars lc rgb '#27ad81' lt 7 t "GFP"