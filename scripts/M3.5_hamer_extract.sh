python M3_mean_shape_smplerx.py --input_path ${ROOT_PATH} --output_path ${OUTPUT_PATH}
echo "neutral" > ${OUTPUT_PATH}/gender.txt
cd hamer
CUDA_VISIBLE_DEVICES=0 python demo.py --img_folder ${ROOT_PATH}/images --out_folder ${OUTPUT_PATH}/hamer --batch_size=48 --side_view --save_mesh --full_frame