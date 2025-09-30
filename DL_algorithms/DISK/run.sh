start_time=$(date +%s)

python detect.py /home/buybluepants/Documents/Object-detection-through-ML-and-DL-methods/DL_algorithms/DISK/h5_artifacts_destination /home/buybluepants/Documents/Object-detection-through-ML-and-DL-methods/data --image-extension png --width 176 --height 128

python match.py /home/buybluepants/Documents/Object-detection-through-ML-and-DL-methods/DL_algorithms/DISK/h5_artifacts_destination

python view_h5.py /home/buybluepants/Documents/Object-detection-through-ML-and-DL-methods/DL_algorithms/DISK/h5_artifacts_destination /home/buybluepants/Documents/Object-detection-through-ML-and-DL-methods/data matches --image-extension png --save .

end_time=$(date +%s)

elapsed_time=$((end_time - start_time))

echo "The script took $elapsed_time seconds to run"

#ffmpeg -i loftr-matches.mp4 -vf "select=not(mod(n\,1))" -vsync vfr frame_%04d.png
