from vectorize_gadget import *
from helper import *
#from glove_vectorize import *
import json 
import pandas 
import numpy as np
import os.path
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix


CODE_STMT = "code_stmt"
TARGET = "target"
VECTOR = "vector"
FUNC_NAME = "func_name"
CODE_LINK = "code_link"
FLAW_LINE = "flaw_line"



def get_vector_a_sample(vectorizer, sample):
  pre_surrounding_context = get_context(sample["surrounding_ctx_code_pred"], sample["surrounding_ctx_code_succ"])
  bw_cdg_context = get_context(sample["cdg_bw_slicing"], sample["cdg_fw_slicing"])
  bw_ddg_context = get_context(sample["ddg_bw_slicing"], sample["ddg_fw_slicing"])
  operation_ctx_abstract = get_operation_context(sample["operation_ctx"])
  vul_type = sample["vul_type"]
  vector = vectorizer.vectorize(pre_surrounding_context, bw_cdg_context, bw_ddg_context, operation_ctx_abstract, vul_type)
  row = {VECTOR: vector,
        TARGET: sample["target"], CODE_STMT: sample[CODE_STMT]}
  return row

def get_vector_data(vectorizer, data):
    vectors = []
    count = 0
    for df_idx, row in data.iterrows():
        count += 1
        print("Collecting gadgets...", count, end="\r")
        try:
            print("Processing gadgets...", count, end="\r")
            
            pre_surrounding_context = get_context(data.at[df_idx,"surrounding_ctx_code_pred"], data.at[df_idx,"surrounding_ctx_code_succ"])
            bw_cdg_context = get_context(data.at[df_idx,"cdg_bw_slicing"], data.at[df_idx,"cdg_fw_slicing"])
            bw_ddg_context = get_context(data.at[df_idx,"ddg_bw_slicing"], data.at[df_idx,"ddg_fw_slicing"])
            operation_ctx_abstract = get_operation_context(data.at[df_idx,"operation_ctx"])
            #operation_ctx_abstract = get_operation_context(data.at[df_idx,"code_stmt"])
            vul_type = data.at[df_idx,"vul_type"]
            vector = vectorizer.vectorize(pre_surrounding_context, bw_cdg_context, bw_ddg_context, operation_ctx_abstract, vul_type)
            row = {VECTOR: vector,
                  TARGET: data.at[df_idx,"target"], CODE_STMT: data.at[df_idx, CODE_STMT]}
            
            vectors.append(row)
        except Exception as e:
            raise e
            print(e)
            print(idx)
            print("--------------------------------")
    df = pandas.DataFrame(vectors)
    del vectors
    return df


