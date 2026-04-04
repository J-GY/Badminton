mkdir -p logs

# xian
sed -i "s/^dataset:.*/dataset: xian/" config.yaml
sed -i "s/^train_set_size:.*/train_set_size: 60000/" config.yaml
sed -i "s/^vali_set_size:.*/vali_set_size: 20000/" config.yaml

{
    date
    python preprocess.py
    python spatial_similarity.py
    python temporal_similarity.py
    python data_utils.py
    date
} > logs/xian_preprocess.log 2>&1

{
    echo "xian main log"
    date
    python main.py
    date
    echo "finished xian"
} > logs/xian.log 2>&1