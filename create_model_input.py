import argparse
import pandas as pd

def create_model_input(args):

    graph_emb_path = args.graph_emb_path
    graph_label_path = args.graph_label_path

    # Load the graph embeddings
    with open(graph_emb_path, "r") as file:
        lines = file.readlines()

    # Remove the first line (which is just the number of subgraphs)
    embedding_lines = lines[1:]

    # Parse embeddings into a DataFrame
    data = []
    for line in embedding_lines:
        parts = line.strip().split("\t")
        subgraph_id = parts[0]
        features = list(map(float, parts[1:]))
        data.append([subgraph_id] + features)

    # Create DataFrame
    embedding_df = pd.DataFrame(data)
    embedding_df.columns = ["subGraphID"] + [f"dim_{i}" for i in range(1, len(data[0]))]

    # Load the labels
    labels_df = pd.read_csv(graph_label_path)

    # Merge embeddings with labels
    merged_df = pd.merge(embedding_df, labels_df, on="subGraphID", how="left")

    # Save the merged DataFrame to a CSV file
    merged_df.to_csv("model_input.csv", index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create model input.")
    parser.add_argument('--graph_emb_path', type=str, default='output', required=True, help='Path to the graph embeddings file')
    parser.add_argument('--graph_label_path', type=str, default='labels.csv', required=True, help='Path to the graph labels file')
    args = parser.parse_args()

    # Create model_input.csv (graph embeddings + label + train/test flag)
    create_model_input(args)