echo "training neural net on dataset: $1"
python3 raw_to_processed.py $1
python3 train.py
