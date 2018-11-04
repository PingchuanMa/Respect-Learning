mkdir -p video
ffmpeg -framerate 25 -i $1/Frame%04d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p video/$1.mp4
