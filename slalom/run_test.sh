cd SGXDNN
make
cd ../
make clean
make
# python -m python.slalom.scripts.eval_transformer bert --batch_size=1 --max_index=30522
python -m python.slalom.scripts.eval_transformer llama --batch_size=8 --max_index=32000 --seq_len=8