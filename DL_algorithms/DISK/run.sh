start_time=$(date +%s.%N)
echo "Start time: $start_time"

python detect.py /home/buybluepants/Documents/Object-detection-through-ML-and-DL-methods/DL_algorithms/DISK/h5_artifacts_destination /home/buybluepants/Documents/Object-detection-through-ML-and-DL-methods/data --image-extension png --width 176 --height 128

python match.py /home/buybluepants/Documents/Object-detection-through-ML-and-DL-methods/DL_algorithms/DISK/h5_artifacts_destination

python view_h5.py /home/buybluepants/Documents/Object-detection-through-ML-and-DL-methods/DL_algorithms/DISK/h5_artifacts_destination /home/buybluepants/Documents/Object-detection-through-ML-and-DL-methods/data matches --image-extension png --save .

end_time=$(date +%s.%N)
echo "End time: $end_time"


elapsed=$(echo "$end_time - $start_time" | bc)

elapsed_formatted=$(printf "%.3f" "$elapsed")

echo "The script took $elapsed_formatted seconds to run"

#ffmpeg -i loftr-matches.mp4 -vf "select=not(mod(n\,1))" -vsync vfr frame_%04d.png
