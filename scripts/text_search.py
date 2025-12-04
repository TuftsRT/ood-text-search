import os
import pypdf
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from torch import Tensor
import torch
import numpy as np
from typing import List, Tuple, Dict, Union
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import json
import pickle
from multiprocessing import Pool, cpu_count
import re

import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("pypdf").setLevel(logging.CRITICAL)
logging.getLogger("pypdf._reader").setLevel(logging.CRITICAL)

DATA_PATH = "path" # need to pull this in from argparse in the real app

class Dataloader:
    def __init__(self, data_path=None, all_files=None, max_workers=None, batch_size=None, from_csv=None) -> None:
        """
        Dataloader for reading PDF files and extracting text.
        It reads all PDF files in the specified DATA_PATH, extracts text from each file,
        and stores the results in a dictionary. It also identifies files that need OCR processing.
        """

        if from_csv:
            self._load_from_csv(from_csv)
            return

        if not data_path.endswith('/'):
            data_path += '/'

        if not data_path:
            data_path = DATA_PATH
        self.data_path = data_path
        if not all_files:
            all_files = [self.data_path + f for f in os.listdir(self.data_path)]
        
        output_path = f"{data_path}text-search-output"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_path = output_path
        print(f"Reading files from {data_path}")
        
        self.all_files = all_files

        self.max_workers = max_workers or min(cpu_count(), 8)
        self.batch_size = batch_size or 10

        self.texts = {}
        self.usable = {}
        self.files_needing_ocr = []

        self._process_files()

    def _process_files(self):
        total_files = len(self.all_files)

        for i in range(0, total_files, self.batch_size):
            batch = self.all_files[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_files + self.batch_size - 1) // self.batch_size

            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)")

            with Pool(processes=self.max_workers) as pool:
                results = list(tqdm(
                    pool.imap(self.process_single_file, batch),
                    total=len(batch),
                    desc=f"Batch {batch_num}",
                    unit="file"
                ))

            for file_path, text, needs_ocr in results:
                self.texts[file_path] = text
                if needs_ocr:
                    self.files_needing_ocr.append(file_path)
                else:
                    self.usable[file_path] = text

    def process_single_file(self, file_path: str) -> Tuple[str, str, bool]:
        try:
            if file_path.endswith('.pdf'):
                text = self._read_pdf_text(file_path)
            elif file_path.endswith('.docx'):
                text = self._read_word_text(file_path)
            elif file_path.endswith('.txt'):
                text = self._read_word_text(file_path)
            else:
                text = ''
            
            needs_ocr = len(text.strip()) == 0
            return file_path, text, needs_ocr
            
        except Exception as e:
            logger.warning(f"Error processing {file_path}: {str(e)}")
            return file_path, '', True
    
    def _read_pdf_text(self, pdf_path: str) -> str:
        read_pdf = pypdf.PdfReader(pdf_path)
        text = ''
        for page in read_pdf.pages:
            text += page.extract_text()
        return text

    def _read_word_text(self, word_path: str) -> str:
        doc = Document(word_path)
        text = ''
        for para in doc.paragraphs:
            text += para.text
        return text

    def _read_word_text(self, txt_path: str) -> str:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        return text

    def to_csv(self, csv_path: str, include_text: bool=True) -> None:
        """
        Saves the extracted text to a CSV file.
        Args:
            csv_path (str): The path to the CSV file.
        Returns:
            None
        """
        data_rows = []
        for file_path in self.texts.keys():
            text = self.texts[file_path]
            needs_ocr = file_path in self.files_needing_ocr
            has_text = file_path in self.usable
            
            try:
                file_stats = os.stat(file_path)
                file_size = file_stats.st_size
                modified_time = datetime.fromtimestamp(file_stats.st_mtime)
            except Exception as e:
                logger.warning(f"Error getting file stats for {file_path}: {str(e)}")
                file_size = None
                modified_time = None
            
            row = {
                'file_path': file_path,
                'filename': os.path.basename(file_path),
                'file_extension': os.path.splitext(file_path)[1],
                'file_size_bytes': file_size,
                'modified_date': modified_time,
                'needs_ocr': needs_ocr,
                'has_extractable_text': has_text,
                'text_length': len(text) if text else 0,
                'text_content': text if include_text else None
            }
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        df = df.sort_values('filename')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"Exported {len(df)} files to {csv_path}")
        print(f"CSV file size: {os.path.getsize(csv_path) / (1024*1024):.1f} MB")
        
        return df

    def _load_from_csv(self, csv_path):
        """
        Load dataloader results from CSV
        
        Args:
            csv_path: Path to the CSV file to load
        """
        print(f"Loading from CSV: {csv_path}")
        
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        self.texts = {}
        self.usable = {}
        self.files_needing_ocr = []
        self.all_files = df['file_path'].tolist()
        
        for _, row in df.iterrows():
            file_path = row['file_path']
            text_content = row['text_content'] if pd.notna(row['text_content']) else ''
            needs_ocr = row['needs_ocr']
            has_text = row['has_extractable_text']
            
            self.texts[file_path] = text_content
            
            if needs_ocr:
                self.files_needing_ocr.append(file_path)
            
            if has_text:
                self.usable[file_path] = text_content
        
        print(f"Loaded {len(self.texts)} files from CSV")
        print(f"Files with extractable text: {len(self.usable)}")
        print(f"Files needing OCR: {len(self.files_needing_ocr)}")

    @classmethod
    def from_csv(cls, csv_path):
        """
        Create a Dataloader instance from a CSV file
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            OptimizedDataloader instance
        """
        return cls(from_csv=csv_path)

class TFIDFSearch:
    def __init__(self, text_dict: Dict, output_path: str, use_lemmas: bool=False, lemma_save_filename="lemmatized_texts.json") -> None:
        """
        Initializes the TF-IDF search engine with a dictionary of texts.
        Args:
            text_dict (Dict): A dictionary where keys are file paths and values are the corresponding text content.
            use_lemmas (bool): If True, uses lemmatization to preprocess the text; otherwise, uses raw text.
        Returns:
            None
        """
        self.text_dict = text_dict
        self.output_path = output_path
        self.use_lemmas = use_lemmas

        self.vectorizer = TfidfVectorizer()
        lemma_save_filename = os.path.join(self.output_path, lemma_save_filename) if lemma_save_filename else None
        if use_lemmas:
            if os.path.exists(lemma_save_filename):
                print("Loading lemmatized texts from file...")
                with open(lemma_save_filename, 'r') as f:
                    self.text_dict = json.load(f)
            else:
                self.text_dict = {k:v['lemmas'] for k, v in self.text_dict.items()}
                with open(lemma_save_filename, 'w') as f:
                    json.dump(self.text_dict, f, indent=4)
        else:
            self.text_dict = {k:v['original_text'] for k, v in self.text_dict.items()}

        self.tfidf_matrix = self.vectorizer.fit_transform(list(self.text_dict.values()))
        self.feature_names = self.vectorizer.get_feature_names_out()

        self.count_vectorizer = CountVectorizer(vocabulary=self.vectorizer.vocabulary_)
        self.count_matrix = self.count_vectorizer.fit_transform(list(self.text_dict.values()))
    
    def _count_entity_in_document(self, entity: str, file_path: str) -> int:
        text = self.text_dict.get(file_path, "")
        pattern = re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE)
        matches = pattern.findall(text.lower())
        return len(matches)
    
    def search(self, query: str, top_k: Union[int, None] = 5) -> List[Tuple[str, float, List[str]]]:
        """
        Searches for the top_k most relevant texts based on the TF-IDF scores for the given query.
        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.
        Returns:
            List[Tuple[str, float]]: A list of tuples where each tuple contains the file path and its corresponding TF-IDF score.        
        """
        if isinstance(query, str):
            query_terms = query.lower().split()
        else:
            query_terms = []
            for item in query:
                query_terms.extend(str(item).lower().split())

        query_vector = self.vectorizer.transform([' '.join(query_terms)])

        cosine_similarities = (query_vector @ self.tfidf_matrix.T).toarray()[0]

        results = [[list(self.text_dict.keys())[i], cosine_similarities[i]] for i in range(len(cosine_similarities))]

        final_results = []
        for i, (key, score) in enumerate(results):
            original_query_terms = [q.lower() for q in query] if isinstance(query, list) else [query.lower()]
            found_terms = []
            for org_term in original_query_terms:
                count = self._count_entity_in_document(org_term, key)
                found_terms.append((org_term, count))
            final_results.append((key, score, found_terms))
        
        final_results.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            final_results = final_results[:top_k]
        return final_results

class FlagEmbeddingSearch:
    def __init__(self, text_dict: Dict, output_path: str, model_path: str='/cluster/tufts/tuftsai/models/BAAI_bge-m3', device='cpu', top_n: int = 3,) -> None:
        """
        Initializes the FlagEmbedding search engine with a dictionary of texts and a model path.
        Args:
            text_dict (Dict): A dictionary where keys are file paths and values are the corresponding text content.
            model_path (str): The path to the pre-trained FlagEmbedding model.
            device (str): The device to run the model on (e.g., 'cpu' or 'cuda').
        Returns:
            None
        """
        self.text_dict = text_dict
        self.output_path = output_path

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)

        self.top_n = top_n

    def _encode(self, texts: Union[str, List[str]], batch_size: int=32, normalize_embeddings: bool=True, instruction=None) -> Dict[str, Tensor]:
        """
        Encodes a list of texts into embeddings using the embedding model.
        Args:
            texts (Union[str, List[str]]): A single text string or a list of text strings to encode.
            batch_size (int): The batch size for processing texts.
            normalize_embeddings (bool): If True, normalizes the embeddings to unit length.
        Returns:
            Dict[str, Tensor]: A dictionary containing the embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if instruction:
            texts = [f"{instruction}\n{text}" for text in texts]

        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
                model_output = self.model(**encoded_input)
                attention_mask = encoded_input['attention_mask']
                token_embeddings = model_output.last_hidden_state

                # mean pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                if normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)

    def _hierarchical_sentence_chunking(self, sentences: List[str], max_chunk_size: int=512) -> List[str]:
        """
        Performs hierarchical chunking of sentences to create larger text chunks.
        Args:
            sentences (List[str]): A list of sentences to chunk.
            max_chunk_size (int): The maximum size of each text chunk.
        Returns:
            List[str]: A list of text chunks.
        """
        if not sentences:
            return []

        chunks = []
        current_chunk = sentences[0]

        for sentence in sentences[1:]:
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                current_chunk += ' ' + sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

        chunks.append(current_chunk)
        return chunks

    
    def _get_dense_score(self, query: str, top_k: Union[int, None] = 5, top_n: int = 10, batch_size: int = 32, agg_fnc=torch.mean, save_as_numpy: str="dense_vectors.npz") -> List[Tuple[str, float]]:
        """
        Computes the dense scores for the given query using the FlagEmbedding model.
        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.
            top_n (int): The number of top query terms to consider for semantic scoring.
            batch_size (int): The batch size for processing texts.
            save_as_numpy (str): The file path to save/load dense vectors as a NumPy file.
        Returns:    
            List[Tuple[str, float]]: A list of tuples where each tuple contains the file path and its corresponding dense score.
        """
        if isinstance(query, str):
            query = [query]
        query = [str(q).lower() for q in query]

        if top_n > len(query):
            top_n = len(query)
        query_embeddings = self._encode(query, batch_size=batch_size, normalize_embeddings=True)

        save_as_numpy = os.path.join(self.output_path, save_as_numpy) if save_as_numpy else None
        if save_as_numpy and os.path.exists(save_as_numpy):
            print(f"Loading dense vectors from {save_as_numpy}...")
            with open(save_as_numpy, 'rb') as f:
                all_sentence_embeddings = pickle.load(f)
            print("Loaded dense vectors from file.")

            results = []
            for path in self.text_dict.keys():
                sentence_embeddings = all_sentence_embeddings.get(path, None)
                if sentence_embeddings is not None and len(sentence_embeddings) > 0:
                    sims = query_embeddings @ sentence_embeddings.T # (num_queries, num_sentences)
                    max_sims, _ = torch.topk(sims, k=top_n, dim=1) # (num_queries, top_n)
                    agg_max_sims = agg_fnc(max_sims, dim=1) # (num_queries,)
                    scores = [(agg_max_sims[i].item(), query[i]) for i in range(len(query))]
                else:
                    scores = [(0.0, query[i]) for i in range(len(query))]
                results.append((path, scores))
        else:
            all_sentence_embeddings = {}
            results = []
            for path in self.text_dict.keys():
                sentences = self.text_dict[path]['sentences']
                if sentences:
                    chunks = self._hierarchical_sentence_chunking(sentences)                    
                    sentence_embeddings = self._encode(chunks, batch_size=batch_size, normalize_embeddings=True) # (num_sentences, embedding_dim)
                    all_sentence_embeddings[path] = sentence_embeddings
                    sims = query_embeddings @ sentence_embeddings.T # (num_queries, num_sentences)
                    max_sims, _ = torch.topk(sims, k=top_n, dim=1) # (num_queries, top_n)
                    agg_max_sims = agg_fnc(max_sims, dim=1) # (num_queries,)
                    scores = [(agg_max_sims[i].item(), query[i]) for i in range(len(query))]
                else:
                    scores = [(0.0, query[i]) for i in range(len(query))]
                results.append((path, scores))
            if save_as_numpy:
                with open(save_as_numpy, 'wb') as f:
                    pickle.dump(all_sentence_embeddings, f)
                print(f"Saved dense vectors to {save_as_numpy}.")

        return results[:top_k] if top_k else results

    def search(self, query: str, top_k: Union[int, None] = 5, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Searches for the top_k most relevant texts based on the specified scoring method.
        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.
        Returns:
            List[Tuple[str, float]]: A list of tuples where each tuple contains the file path and its corresponding score.
        """
        return self._get_dense_score(query, top_k)

class SearchEngine:
    def __init__(self, model_path: str='BAAI/bge-m3', tfidf=True, semantic=True, use_lemmas=False, dataloader=None, device='cpu', lemma_save_file="lemmatized_texts.txt") -> None:
        """
        Initializes the SearchEngine with a model path and search configurations.
        Args:
            model_path (str): The path to the pre-trained FlagEmbedding model.
            use_lemmas (bool): If True, uses lemmatization for TF-IDF search; otherwise, uses raw text.
            dataloader (Dataloader): An instance of the Dataloader class. If None, a new instance will be created.
            device (str): The device to run the model on (e.g., 'cpu' or 'cuda').
            lemma_save_file (str): The file path to save/load lemmatized texts.
        Returns:
            None
        """

        self.tfidf = tfidf
        self.semantic = semantic
        if self.tfidf:
            print("TF-IDF search enabled.")
        if self.semantic:
            print("Semantic search enabled.")
        if not self.tfidf and not self.semantic:
            raise ValueError("At least one of tfidf or semantic must be True.")

        self.use_lemmas = use_lemmas
        if self.use_lemmas:
            print("Using lemmatization for TF-IDF search.")
        else:
            print("Using raw text for TF-IDF search.")

        if not dataloader:
            self.dl = Dataloader()
        else:
            self.dl = dataloader
        self.output_path = self.dl.output_path
        
        print(f"Read documents: {self.dl.usable.keys()}")
        self.lemma_save_file = lemma_save_file
        text_dict = self.dl.usable
        self.nlp = spacy.load("/cluster/tufts/tuftsai/models/en_core_web_sm-3.7.1", disable=["ner", "parser"])
        self.nlp.add_pipe('sentencizer')
        
        text_dict = self._spacy_lemmas_and_chunking(text_dict)

        if self.tfidf:
            self.tfidf_search = TFIDFSearch(text_dict, output_path=self.output_path, use_lemmas=self.use_lemmas, lemma_save_filename=self.lemma_save_file)
        if self.semantic:
            self.flag_embedding_search = FlagEmbeddingSearch(text_dict, output_path=self.output_path, model_path=model_path, device=device)

    def _spacy_lemmas_and_chunking(self, text_dict) -> str:
        processed_docs = {}
        docs = self.nlp.pipe(text_dict.values(), batch_size=100, n_process=-1)
        for path, doc in zip(text_dict.keys(), docs):
            if self.use_lemmas:
                lemma_text = ' '.join([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])
            else:
                lemma_text = ""

            if self.semantic:
                sentences = [sent.text.strip() for sent in doc.sents]
                sentence_boundaries = [(sent.start_char, sent.end_char) for sent in doc.sents]
            else:
                sentences = []
                sentence_boundaries = []
        
            processed_docs[path] = {
                'lemmas': lemma_text,
                'sentences': sentences,
                'sentence_boundaries': sentence_boundaries,
                'original_text': doc.text.lower()
            }
        return processed_docs

    def _combine_semantic_scores(self, scores):
        softmax_weights = F.softmax(torch.tensor(scores), dim=0).numpy()
        return np.sum(np.array(scores) * softmax_weights)

    def search(self, query: str, top_k: Union[int, None] = 5, top_n: int = 3, beta: float = 0.5, result_file_path="test_results.csv") -> Dict[str, List[Tuple[str, float]]]:
        """
        Searches for the top_k most relevant texts using both TF-IDF and FlagEmbedding methods.
        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.
            top_n (int): The number of top query terms to consider for semantic scoring.
            beta (float): Weighting factor for combining keyword and semantic scores.
        Returns:
            Dict[str, List[Tuple[str, float]]]: A dictionary containing the results from keyword search, semantic search, and composite scores.
        """
        if self.tfidf:
            print("Running keyword")
            keyword = self.tfidf_search.search(query, top_k)

        if self.semantic:
            print("Running semantic")
            semantic = self.flag_embedding_search.search(query=query, top_k=top_k, top_n=top_n)

        if self.tfidf and not self.semantic:
            results = {
                'keyword': keyword
            }
            self.results = results

            path_score = pd.DataFrame((lo[:-1] for lo in self.results['keyword']), columns=["Path", "keyword score"])
            term_counts = pd.DataFrame(([t[-1] for t in lo[-1]] for lo in self.results['keyword']), columns=[f"{q.lower()}_keyword" for q in query])
            k = pd.concat([path_score, term_counts], axis=1)
            merged_results = k.sort_values(by='keyword score', ascending=False).reset_index(drop=True)
            result_file_path = os.path.join(self.output_path, result_file_path) if result_file_path else None
            if result_file_path and result_file_path.endswith('.csv'):
                merged_results.to_csv(result_file_path, index=False)
                print(f"Results saved to {result_file_path}")
                return merged_results
            elif result_file_path and result_file_path.endswith('.xlsx'):
                merged_results.to_excel(result_file_path, index=False)
                print(f"Results saved to {result_file_path}")
                return merged_results
            else:
                return merged_results
        
        if not self.tfidf and self.semantic:
            results = {
                'semantic': semantic
            }
            self.results = results

            term_scores = pd.DataFrame([[t[0] for t in lo[-1]] for lo in self.results['semantic']], columns=[f"{q.lower()}_semantic" for q in query])
            path_df = pd.DataFrame((lo[:-1] for lo in self.results['semantic']), columns=["Path"])
            s = pd.concat([path_df, term_scores], axis=1)
            merged_results = s.sort_values(by=f'{query[0].lower()}_semantic', ascending=False).reset_index(drop=True)
            result_file_path = os.path.join(self.output_path, result_file_path) if result_file_path else None
            if result_file_path and result_file_path.endswith('.csv'):
                merged_results.to_csv(result_file_path, index=False)
                print(f"Results saved to {result_file_path}")
                return merged_results
            elif result_file_path and result_file_path.endswith('.xlsx'):
                merged_results.to_excel(result_file_path, index=False)
                print(f"Results saved to {result_file_path}")
                return merged_results
            else:
                return merged_results

        if self.tfidf and self.semantic:
            print("Creating composite scores")
            composite = []
            for k, s in zip(keyword, semantic):
                keyword_score = k[1]
                semantic_score_pairs = [score for score, _ in s[1]]
                semantic_score = self._combine_semantic_scores(semantic_score_pairs)
                composite.append((k[0], keyword_score * beta + semantic_score * (1 - beta)))

            results = {
                'keyword': keyword,
                'semantic': semantic,
                'composite': composite
            }
            self.results = results

            for key, value in results.items():
                if key == 'keyword':
                    path_score = pd.DataFrame((lo[:-1] for lo in self.results[key]), columns=["Path", "keyword score"])
                    term_counts = pd.DataFrame(([t[-1] for t in lo[-1]] for lo in self.results[key]), columns=[f"{q.lower()}_keyword" for q in query])
                    k = pd.concat([path_score, term_counts], axis=1)
                elif key == 'semantic':
                    term_scores = pd.DataFrame([[t[0] for t in lo[-1]] for lo in self.results['semantic']], columns=[f"{q.lower()}_semantic" for q in query])
                    path_df = pd.DataFrame((lo[:-1] for lo in self.results[key]), columns=["Path"])
                    s = pd.concat([path_df, term_scores], axis=1)
                elif key == 'composite':
                    c = pd.DataFrame(value, columns=["Path", "composite score"])
                    c['composite score'] = c['composite score'].apply(lambda x: x)
            
            merged_results = k.merge(s, on="Path").merge(c, on="Path").sort_values(by='composite score', ascending=False).reset_index(drop=True)
            result_file_path = os.path.join(self.output_path, result_file_path) if result_file_path else None
            if result_file_path and result_file_path.endswith('.csv'):
                merged_results.to_csv(result_file_path, index=False)
                print(f"Results saved to {result_file_path}")
                return merged_results
            elif result_file_path and result_file_path.endswith('.xlsx'):
                merged_results.to_excel(result_file_path, index=False)
                print(f"Results saved to {result_file_path}")
                return merged_results
            else:
                return merged_results

# def main():
#     dl = Dataloader(
#         data_path="test_docs",
#         max_workers=4,
#         batch_size=5
#     )
    
#     se = SearchEngine(
#         model_path='/cluster/tufts/tuftsai/models/BAAI_bge-m3',
#         use_lemmas=True,
#         dataloader=dl,
#         device='cuda',
#         lemma_save_file="lemmatized_texts.json"
#     )

#     query = ["aids", "hepatitis", "cancer"]
#     merged_results = se.search(
#         query=query,
#         top_k=5,
#         beta=0.3,
#         result_file_path="test_results.xlsx"
#     )
#     print(merged_results)

# if __name__ == "__main__":
#     main()