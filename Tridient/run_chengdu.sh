mkdir -p logs 

# chengdu
sed -i "s/^dataset:.*/dataset: chengdu/" config.yaml
sed -i "s/^train_set_size:.*/train_set_size: 20000/" config.yaml
sed -i "s/^vali_set_size:.*/vali_set_size: 15000/" config.yaml

{
    date
    python preprocess.py
    python spatial_similarity.py
    python temporal_similarity.py
    python data_utils.py
    date
} > logs/chengdu_preprocess.log 2>&1

{
    echo "chengdu main log"
    date
    python main.py
    date
    echo "finished chengdu"
} > logs/chengdu.log 2>&1