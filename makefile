# run_subgraphs
TRAIN_TIME_SLICE_PCT = 1.0
TEST_TIME_SLICE_PCT = 0.5

# run_sub2vec
WALKLENGTH = 50

# run_model
MODEL_NAME = 'non_weighted_xgboost_model.json'
AUX_PROB = --aux_prob	# '--aux_prob'
PROB_RATIO = 1.0	# \alpha
AUX_PROB_RATIO = 1.0	# \beta

##################################################################

# step 1
run_subgraphs:
	python gen_subgraphs.py --train_time_slice_pct $(TRAIN_TIME_SLICE_PCT) --test_time_slice_pct $(TEST_TIME_SLICE_PCT)

# step 2
run_sub2vec:
	python src/main.py --input train_input --output train_output --walkLength $(WALKLENGTH) --iter 10 --property s
	python src/main.py --input test_input --output test_output --walkLength $(WALKLENGTH) --iter 10 --property s

# step 3
run_model_input:
	python create_model_input.py --graph_emb_path 'train_output' --graph_label_path 'train_labels.csv' --model_input_filename 'train.csv'
	python create_model_input.py --graph_emb_path 'test_output' --graph_label_path 'test_labels.csv' --model_input_filename 'test.csv'

# step 4
run_model:
	python XGBoost.py --model_name $(MODEL_NAME) --train_file 'train.csv' --test_file 'test.csv' $(AUX_PROB) --prob_ratio $(PROB_RATIO) --aux_prob_ratio $(AUX_PROB_RATIO)

# Run all
run:
	make run_subgraphs
	make run_sub2vec
	make run_model_input
	make run_model

clean_bak:
	rm -r ./bak/*

bak_all:
	mv ./*input/ ./*labels.csv ./bak/

	mv ./*input.walk ./*output ./bak/

	mv ./train.csv ./test.csv ./bak/

	mv ./result/ ./bak/

# bak step 2, 3, 4
bak_sub2vec:
	mv ./*input.walk ./*output ./bak/

	mv ./train.csv ./test.csv ./bak/

	mv ./result/ ./bak/

# bak step 3, 4
bak_model_input:
	mv ./train.csv ./test.csv ./bak/

	mv ./result/ ./bak/

# bak step 4
bak_model:
	mv ./result/ ./bak/

# clean and backup all
clean:
	make clean_bak
	make bak_all
