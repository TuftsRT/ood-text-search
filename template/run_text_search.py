import os
import argparse
import torch
from text_search import Dataloader, SearchEngine

parser = argparse.ArgumentParser(description="This script performs text search on a dataset.")
parser.add_argument("--data_path", type=str, help="Path to the directory of text files.")
parser.add_argument("--queries", type=str, help="Path to the file containing search queries.")
parser.add_argument("--beta", type=float, default=0.5, help="Beta parameter for search.")
parser.add_argument("--use_lemmas", action='store_true', help="Whether to use lemmas for search.")
parser.add_argument("--use_tfidf", action='store_true', help="Whether to use TF-IDF for search.")
parser.add_argument("--use_semantic", action='store_true', help="Whether to use semantic search.")
parser.add_argument("--result_file_path", type=str, help="Path to the output result file.")
args = parser.parse_args()

def parse_queries(queries):
    if os.path.isfile(queries):
        with open(queries, 'r') as f:
            return [line.strip() for line in f if line.strip()] # file must be \n delimited
    else: 
        return [query for query in queries.split(',') if query.strip()] # comma delimited string

if __name__ == "__main__":
    data_path = args.data_path
    queries = parse_queries(args.queries)
    beta = args.beta
    use_lemmas = args.use_lemmas
    use_tfidf = args.use_tfidf
    use_semantic = args.use_semantic
    print("Use semantic: ", use_semantic)
    result_file_path = args.result_file_path

    if not os.path.exists(data_path):
        raise ValueError(f"Data path {data_path} does not exist.")
    
    if not queries:
        raise ValueError("No queries provided.")
    
    if not result_file_path:
        raise ValueError("Result file path must be specified.")

    dl = Dataloader(
        data_path=data_path, 
        # max_workers and batch size - COME BACK
    )
    
    search_engine = SearchEngine(
        dataloader=dl,
        use_lemmas=use_lemmas,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        tfidf=use_tfidf,
        semantic=use_semantic,
    )

    merged_results = search_engine.search(
        query=queries,
        top_k=None,
        beta=beta,
        result_file_path=result_file_path
    )