python 4_check_weights_distant.py \
    --bert_model bert-base-cased \
    --data_path ../data/PARAREL/data_all.json \
    --tmp_data_path ../data/PARAREL/data_all_allbags.json \
    --distant_data_dir ../data/BingDistantTREx/final/ \
    --query_pairs_path ../data/BingDistantTREx/query_pairs.json \
    --kn_dir ../results/kn \
    --output_dir ../results/distant \
    --gpus 0 \
    --max_seq_length 512 \
    --debug 100000 \