#!/bin/bash
cd output
Files=`ls snapshot*.svg | wc -l`
for i in $(seq 1 ${Files})
do
    ext=`ls snapshot*.svg |tr -s ' ' '\n' | awk NR==${i} | cut -d. -f1` 
    convert ${ext}.svg -resize 1366x768 ${ext}.png
done

#convert *.svg -set filename:basename "%[basename]" -resize 1366x768 "%[filename:basename].png" #CONVERT JUST SOME IMAGES

# CREATE GIF OPTION 1
#convert *.png -deconstruct -delay 1.6 out-convert.gif

# CREATE GIF OPTION 2
ffmpeg -i snapshot00000000.png -vf palettegen=256 palette.png 
#ffmpeg -framerate 60 -pattern_type glob -i 'snap*.png' -i palette.png -filter_complex "fps=20,scale=720:-1:flags=lanczos[x];[x] [1:v]paletteuse" out.gif

# CREATE MOVIE
#mencoder mf://snapshot*.png -mf w=800:h=600:fps=20:type=png -ovc copy -oac copy -o Simulacao.avi
ffmpeg -framerate 60 -pattern_type glob -i 'snap*.png' -i palette.png -filter_complex "fps=20,scale=720:-1:flags=lanczos[x];[x] [1:v]paletteuse" out.mp4