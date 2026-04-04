mkdir -p logs

# tdrive
sed -i "s/^dataset:.*/dataset: tdrive/" config.yaml
sed -i "s/^train_set_size:.*/train_set_size: 10000/" config.yaml
sed -i "s/^vali_set_size:.*/vali_set_size: 4000/" config.yaml

{
    date
    python preprocess.py
    python spatial_similarity.py
    python temporal_similarity.py
    python data_utils.py
    date
} > logs/tdrive_preprocess.log 2>&1

{
    echo "tdrive main log"
    date
    python main.py
    date
    echo "finished tdrive"
} > logs/tdrive.log 2>&1