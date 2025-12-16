#Example command to run adult dataset
cd ..
python -m data_pre_process_pipeline --input_csv adult.csv --output_dir adult --label-col income --discrete_to_label 
python -m main --al_method DA+ALFA --al_function clue --classifier MLP --dataset adult --budget 5 --generator RTF 