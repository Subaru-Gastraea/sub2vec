# run_subgraphs
TRAIN_TIME_SLICE_PCT = 1.0
TEST_TIME_SLICE_PCT = 1.0

# run_sub2vec
WALKLENGTH = 50

# run_model
MODEL_NAME = 'non_weighted_xgboost_model.json'

##################################################################

# step 1
run_subgraphs:
	python gen_subgraphs.py --train_time_slice_pct $(TRAIN_TIME_SLICE_PCT) --test_time_slice_pct $(TEST_TIME_SLICE_PCT)

# step 2
run_sub2vec:
	python src/main.py --input train_input --output train_output --walkLength $(WALKLENGTH) --iter 10 --property s
	python src/main.py --input test_input --output test_output --walkLength $(WALKLENGTH) --iter 10 --property s

# step 3
run_model:
	python create_model_input.py --graph_emb_path 'train_output' --graph_label_path 'train_labels.csv' --model_input_filename 'train.csv'
	python create_model_input.py --graph_emb_path 'test_output' --graph_label_path 'test_labels.csv' --model_input_filename 'test.csv'
	python XGBoost.py --model_name $(MODEL_NAME) --train_file 'train.csv' --test_file 'test.csv'

# Run all
run:
	make run_subgraphs
	make run_sub2vec
	make run_model

clean_bak:
	rm -r ./bak/*

bak_all:
	mv ./*input/ ./*labels.csv ./bak/

	mv ./*input.walk ./*output ./bak/

	mv ./train.csv ./test.csv ./result/ ./save_model/*.json ./bak/

# bak step 2, 3
bak_sub2vec:
	mv ./*input.walk ./*output ./bak/

	mv ./train.csv ./test.csv ./result/ ./save_model/*.json ./bak/

# bak step 3
bak_model:
	mv ./train.csv ./test.csv ./result/ ./save_model/*.json ./bak/

# clean and backup all
clean:
	make clean_bak
	make bak_all
