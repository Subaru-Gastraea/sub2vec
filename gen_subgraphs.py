import os
import pickle
import networkx as nx
from tqdm import tqdm
import argparse
import pathlib
import pandas as pd

def gen_subgraphs(args):

    graph_dataset_path = args.dataset
    output_path = pathlib.Path(args.output)

    graphs_path = os.listdir(graph_dataset_path)
    # graphs_path = graphs_path[:10] # for testing

    def get_graph_info():
        for graph_path in graphs_path:
            path = os.path.join(graph_dataset_path, graph_path)
            with open(path, 'rb') as f:
                graph_sample = pickle.load(f)
                yield graph_sample['graph'], graph_sample['labels']

    label_df = pd.DataFrame(columns=['subGraphID','label'])

    for i, graph_info in enumerate(tqdm(get_graph_info(), total=len(graphs_path))):
        graph, label = graph_info

        mapping = {node: j for j, node in enumerate(graph.nodes)}
        graph = nx.relabel_nodes(graph, mapping)

        subGraphID = f'subGraph{i}'
        label_N = label.index(1)
        label_df = pd.concat([label_df, pd.DataFrame([{'subGraphID': subGraphID, 'label': label_N}])], ignore_index=True)

        with open(output_path / subGraphID, 'wb') as f:
            nx.write_edgelist(graph, f, data=False)

    label_df.to_csv('labels.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate subgraphs for sub2vec.")
    parser.add_argument('--dataset', type=str, default='patients_graphs/', required=True, help='Path to the dataset directory')
    parser.add_argument('--output', type=str, default='input/', required=True, help='Path to the output directory')
    args = parser.parse_args()

    gen_subgraphs(args)