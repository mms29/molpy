for i in $(seq -f "%05g" 1 100)
do
  xmipp_image_convert -i ${1}/${i}_reconstructed.vol --oext mrc -o ${1}/${i}_reconstructed.mrc
done

