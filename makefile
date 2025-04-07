
run:
	python gen_subgraphs.py --dataset 'patients_graphs/' --output 'input/' --test_time_slice 1.0 --test_size 0.25
	python src/main.py --input input --output output --walkLength 50 --iter 10 --property s
	python create_model_input.py --graph_emb_path 'output' --graph_label_path 'labels.csv'
	python XGBoost.py

clean:
	rm -r ./bak/*
	mv ./input/ ./labels.csv ./input.walk ./output ./model_input.csv ./result/ ./save_model/xgboost_model.json ./bak/
	mkdir ./input ./result
