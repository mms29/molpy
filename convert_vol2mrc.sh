for i in $(seq -f "%05g" 1 1000)
do
  xmipp_image_convert -i ${1}/${i}_reconstructed.spi --oext mrc -o ${1}/${i}_reconstructed.mrc
done

