
# step 1
run_subgraphs:
	python gen_subgraphs.py --dataset 'patients_graphs/' --output 'input/' --test_time_slice 1.0 --test_size 0.25

# step 2
run_sub2vec:
	python src/main.py --input input --output output --walkLength 5000 --iter 10 --property s

# step 3
run_model:
	python create_model_input.py --graph_emb_path 'output' --graph_label_path 'labels.csv'
	python XGBoost.py

# Run all
run:
	run_subgraphs
	run_sub2vec
	run_model

clean_bak:
	rm -r ./bak/*

bak_all:
	mv ./input/ ./labels.csv ./bak/
	mkdir ./input

	mv ./input.walk ./output ./bak/

	mv ./model_input.csv ./result/ ./save_model/xgboost_model.json ./bak/
	mkdir ./result

# bak step 2, 3
bak_sub2vec:
	mv ./input.walk ./output ./bak/

	mv ./model_input.csv ./result/ ./save_model/xgboost_model.json ./bak/
	mkdir ./result

# bak step 3
bak_model:
	mv ./model_input.csv ./result/ ./save_model/xgboost_model.json ./bak/
	mkdir ./result

# clean and backup all
clean:
	clean_bak
	bak_all
