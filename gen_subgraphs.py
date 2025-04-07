import os
import pickle
import networkx as nx
from tqdm import tqdm
import argparse
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

def time_slice_graph(graph, time_slice_pct):
    """
    Function to slice the graph based on time.
    Args:
        graph (networkx.Graph): The input graph.
        time_slice_pct (float): The percentage of the time duration for testing graphs.
    Returns:
        networkx.Graph: The sliced graph.
    """

    if not (time_slice_pct > 0 and time_slice_pct < 1):
        return graph

    time_nodes = [node for node in graph.nodes if graph.nodes[node]['type'] == 'Time']

    # Get the time range of the graph
    min_time = min(pd.to_datetime(graph.nodes[node]['timestamp']) for node in time_nodes)
    max_time = max(pd.to_datetime(graph.nodes[node]['timestamp']) for node in time_nodes)

    # Calculate the time slice
    time_slice = min_time + (max_time - min_time) * time_slice_pct

    sliced_nodes = [node for node in graph.nodes if pd.to_datetime(graph.nodes[node]['timestamp']) <= time_slice]
    sliced_graph = graph.subgraph(sliced_nodes).copy()

    return sliced_graph

def split_train_test(label_df, test_size=0.25):
    # Separate the rows where label is 5
    label_5_data = label_df[label_df['label'] == 5]
    # Randomly sample 10000 rows from label 5 data
    label_5_sampled = label_5_data.sample(n=10000, random_state=42)
    # Get the rows where label is not 5
    other_labels_data = label_df[label_df['label'] != 5]
    # Combine the sampled label 5 data with the other labels data
    label_df_filt = pd.concat([label_5_sampled, other_labels_data], ignore_index=False)

    test_df = label_df_filt.sample(frac=test_size, random_state=42)
    train_df = label_df_filt.drop(test_df.index)
    train_df['train_or_test'] = 'train'
    test_df['train_or_test'] = 'test'
    merge_df = pd.concat([train_df, test_df], ignore_index=False)

    df = label_df.merge(merge_df, on=['subGraphID','label'], how='left')

    return df

def gen_subgraphs(args):

    graph_dataset_path = args.dataset
    output_path = pathlib.Path(args.output)
    test_time_slice = args.test_time_slice
    test_size = args.test_size

    graphs_path = os.listdir(graph_dataset_path)
    # graphs_path = graphs_path[:100] # test 

    def get_graph_info():
        for graph_path in graphs_path:
            path = os.path.join(graph_dataset_path, graph_path)
            with open(path, 'rb') as f:
                graph_sample = pickle.load(f)
                yield graph_sample['graph'], graph_sample['labels']

    label_df = pd.DataFrame(columns=['subGraphID','label'])
    for i, graph_info in enumerate(tqdm(get_graph_info(), total=len(graphs_path), desc="Generating label_df")):
        graph, label = graph_info

        subGraphID = f'subGraph{i}'
        label_N = label.index(1)
        label_df = pd.concat([label_df, pd.DataFrame([{'subGraphID': subGraphID, 'label': label_N}])], ignore_index=True)

    label_df = split_train_test(label_df, test_size)

    for i, graph_info in enumerate(tqdm(get_graph_info(), total=len(graphs_path), desc="Generating subgraphs")):
        graph, label = graph_info

        subGraphID = f'subGraph{i}'
        train_test_flag = label_df[label_df['subGraphID'] == subGraphID]['train_or_test'].values[0]
        if train_test_flag == 'test':
            # Apply time slicing to the graph
            graph = time_slice_graph(graph, test_time_slice)
        elif pd.isna(train_test_flag):
            # If the graph is not in the train/test split, skip it
            continue

        mapping = {node: j for j, node in enumerate(graph.nodes)}
        graph = nx.relabel_nodes(graph, mapping)

        with open(output_path / subGraphID, 'wb') as f:
            nx.write_edgelist(graph, f, data=False)

    label_df.to_csv('labels.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate subgraphs for sub2vec.")
    parser.add_argument('--dataset', type=str, default='patients_graphs/', required=True, help='Path to the dataset directory')
    parser.add_argument('--output', type=str, default='input/', required=True, help='Path to the output directory')
    parser.add_argument('--test_time_slice', type=float, default=1.0, help='Proportion of the time duration for testing graphs (default: 1.0)')
    parser.add_argument('--test_size', type=float, default=0.25, help='Proportion of the dataset to include in the test split (default: 0.25)')
    args = parser.parse_args()

    # Create subgraph edge files (in '--output' path) and labels.csv
    gen_subgraphs(args)