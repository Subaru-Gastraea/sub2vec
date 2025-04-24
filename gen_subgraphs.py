import os
import pickle
import networkx as nx
from tqdm import tqdm
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
import argparse

# Add the parent directory to sys.path
# Executing path: ddx-on-ehr/models/sub2vec/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))     # ddx-on-ehr/models/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))  # ddx-on-ehr/
from utils.graph_proc import GraphDataset

def gen_split_subgraphs(dataset, time_slice_pct, output_path, split_set):
    dataset.set_split(split_set)
    dataset.time_slice_graphs(time_slice_pct)

    label_df = pd.DataFrame(columns=['subGraphID','label'])
    for i, graph_info in enumerate(tqdm(dataset, total=len(dataset), desc=f"Generating {split_set} subgraphs")):
        graph, label = graph_info

        subGraphID = f'subGraph{i}'
        label_df = pd.concat([label_df, pd.DataFrame([{'subGraphID': subGraphID, 'label': label}])], ignore_index=True)

        mapping = {node: j for j, node in enumerate(graph.nodes)}
        graph = nx.relabel_nodes(graph, mapping)

        with open(output_path / subGraphID, 'wb') as f:
            nx.write_edgelist(graph, f, data=False)

    label_df.to_csv(f'{split_set}_labels.csv', index=False)

def gen_subgraphs(args):
    train_output_path = pathlib.Path('./train_input/')
    train_output_path.mkdir(parents=True, exist_ok=True)
    test_output_path = pathlib.Path('./test_input/')
    test_output_path.mkdir(parents=True, exist_ok=True)
    
    train_time_slice_pct = args.train_time_slice_pct
    test_time_slice_pct = args.test_time_slice_pct
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sample_dataset_path = pathlib.Path(project_root) / 'dataset/preprocessed_data/sample_dataset.pkl'

    if os.path.exists(sample_dataset_path):
         with open(sample_dataset_path, 'rb') as f:
            sample_dataset = pickle.load(f)
   
    dataset = GraphDataset(sample_dataset, dev=False, project_root=project_root)
    del sample_dataset

    gen_split_subgraphs(dataset, train_time_slice_pct, train_output_path, split_set='train')
    gen_split_subgraphs(dataset, test_time_slice_pct, test_output_path, split_set='test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_time_slice_pct', type=float, default=1.0, required=True, help='Percentage of time for slicing training graphs')
    parser.add_argument('--test_time_slice_pct', type=float, default=1.0, required=True, help='Percentage of time for slicing testing graphs')
    args = parser.parse_args()

    # Create subgraph edge files for train and test sets
    # Subgraph files are stored in 'train_output_path'<var> and 'test_output_path'<var> folders
    # Corresponding label of subgraphs are stored in 'train_labels.csv' and 'test_labels.csv'
    gen_subgraphs(args)