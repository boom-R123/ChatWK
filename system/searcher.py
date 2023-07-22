import json
from pyserini.search.lucene import LuceneSearcher
import faiss
import torch
import numpy as np
from model_response import generate_response
import os
from config import *
import sys
sys.path.append("..")
from dense_model import *
from transformers import AutoConfig, AutoTokenizer, AutoModel
from adaptertransformers.src.transformers import PretrainedConfig
import requests
from enum import Enum
import time
from bs4 import BeautifulSoup
import trafilatura
from trafilatura.settings import use_config
newconfig = use_config()
newconfig.set("DEFAULT", "EXTRACTION_TIMEOUT", "0") # 防止出现signal only works in main thread. 


# This is the searcher module, we provide simple bm25-based sparse searcher and disentangled_retriever based denser searcher
class Common_Searcher:
    def __init__(self) :
        #initialize your customized searcher
        pass
    def search(self, query:str) -> list(dict()) :
        #insert your customized searcher's search function.
        #The result should be a list of dictionary.
        #The dictionary should at least contain these keys: title, contents, url.
        pass

class Dense_Searcher:
    def __init__(self):
        self.topk = topk
        # prepare encoder

        self.model, self.tokenizer = self.get_model()
        self.device = torch.device("cuda")
        self.model.to(self.device)

        # prepare reference docs
        self.all_doc = self.get_doc_from_folder()
        
        # prepare faiss index
        self.index = faiss.read_index(DENSE_INDEX_PATH)

    
    def get_model(self):
        config = PretrainedConfig.from_pretrained(DAM_NAME)
        config.similarity_metric, config.pooling = "ip", "average"
        tokenizer = AutoTokenizer.from_pretrained(DAM_NAME, config=config)
        # model = AutoModel.from_pretrained(DAM_NAME, config=config) # 使用其他模型
        model = BertDense.from_pretrained(DAM_NAME, config=config) # 使用原始的嵌入模型
        adapter_name = model.load_adapter(REM_URL)
        model.set_active_adapters(adapter_name)
        model.eval()
        return model,tokenizer

    def get_doc_from_folder(self):
        """
        get all docs & properties from a folder containing json file
        """
        files = os.listdir(DOC_PATH)
        files.sort()
        all_data = []
        for file in files:
            # print("Loading: ",file)
            file_path = os.path.join(DOC_PATH, file)
            with open(file_path, "r", encoding = "utf-8") as f:
                doc = json.load(f)
            all_data.append(doc)
        return all_data

    def search(self, query):
        tokenized_text = self.tokenizer(query, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].to(self.device)
        attention_mask = tokenized_text["attention_mask"].to(self.device)
        query_embed = self.model(input_ids = input_ids, attention_mask = attention_mask)

        # query_embed = query_embed.pooler_output # 使用chinese bert

        query_embed = query_embed.detach().cpu().numpy()
        _,doc_id = self.index.search(query_embed, self.topk)
        doc_id = doc_id[0]  

        cnt = 0
        raw_reference_list = []
        for i,id in enumerate(doc_id):
            raw_reference = self.all_doc[id]        
            raw_reference_list.append(raw_reference)
    
            cnt += 1
            if (cnt == self.topk) :
                break

        return raw_reference_list

class Sparse_Searcher:
    def __init__(self):
        self.topk = topk
        self.all_doc = self.get_doc_from_folder()
        self.all_id_list = [item['id'] for item in self.all_doc]
        self.searcher = LuceneSearcher(SPARSE_INDEX_PATH)
        self.searcher.set_language(language)
    
    def get_doc_from_folder(self):
        """
        get all docs & properties
        """
        files = os.listdir(DOC_PATH)
        files.sort()
        all_data = []
        for file in files:
            print("Loading: ",file)
            file_path = os.path.join(DOC_PATH, file)
            with open(file_path, "r", encoding = "utf-8") as f:
                doc = json.load(f)
            all_data.append(doc)
        return all_data

    def search(self, query):
        hits = self.searcher.search(query)
        raw_reference_list = []

        cnt = 0       
        for i in range(min(self.topk, len(hits))):
            doc_id = hits[i].docid
            doc_index = self.all_id_list.index(doc_id)
            
            raw_reference = self.all_doc[doc_index]
            raw_reference_list.append(raw_reference)

            cnt += 1
            if (cnt == self.topk) :
                break

        return raw_reference_list



class CONTENT_TYPE(Enum):
    SEARCH_RESULT = 0
    RESULT_TARGET_PAGE = 1


class ContentItem:
    def __init__(self, type: CONTENT_TYPE, data):
        self.type = type
        self.data = data


class Bing_Searcher:
    def __init__(self):
        self.subscription_key = bing_search_api
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
        self.mkt = 'en-US'
        self.headers = { 'Ocp-Apim-Subscription-Key': self.subscription_key}
        self.content = []
    
    def search(self, key_words, filter=None) -> list:
        start_time = time.time()
        try:
            result = requests.get(self.endpoint, headers=self.headers, params={'q': key_words, 'mkt': self.mkt }, timeout=10)
        except Exception:
            result = requests.get(self.endpoint, headers=self.headers, params={'q': key_words, 'mkt': self.mkt }, timeout=10)
        if result.status_code == 200:
            result = result.json()

            self.content = []
            self.content.append(ContentItem(CONTENT_TYPE.SEARCH_RESULT, result))
        else:
            result = requests.get(self.endpoint, headers=self.headers, params={'q': key_words, 'mkt': self.mkt }, timeout=10)
            if result.status_code == 200:
                result = result.json()

                self.content = []
                self.content.append(ContentItem(CONTENT_TYPE.SEARCH_RESULT, result))
            else:
                raise Exception('Platform search error. Do you register your Bing API key?')
        print(f'search time:{time.time() - start_time}s')
        top_num_web = topk 
        web_snippet_reference_list = self.content[-1].data["webPages"]["value"][:top_num_web]
        for r in web_snippet_reference_list:
            r["contents"] = r["snippet"]
            r["title"] = r["name"]
        if bing_search_using_snippet:
            return web_snippet_reference_list

        web_reference_list = []
        for i in range(top_num_web):
            url, contents = self.load_page(i)
            if url is not None:
                web_reference_list.append({"url": url, "contents": contents, "title": web_snippet_reference_list[i]["title"]})
        return web_reference_list
    
    def get_page_num(self) -> int:
        return len(self.content[-1].data)

    def load_page(self, idx:int) -> str:
        try:
            top = self.content[-1].data["webPages"]["value"]
            res = requests.get(top[idx]['url'], timeout=10)
            res.raise_for_status()
            res.encoding = res.apparent_encoding
            content = res.text
            soup = BeautifulSoup(content, 'html.parser')
            text = trafilatura.extract(soup.prettify(), config=newconfig)
            return top[idx]['url'], text
        except:
            return None, None