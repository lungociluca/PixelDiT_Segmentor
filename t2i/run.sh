rm -r vis;
rm trash/*;
python inference.py \
  --config configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml \
  --model_path pixeldit_t2i_v1.pth \
  --txt_file prompts.txt \
  --custom_height 512 --custom_width 512 \
  --cfg_scale 2.75 --seed 2025 \
  --work_dir "."