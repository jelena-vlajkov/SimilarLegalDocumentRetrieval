from unittest import result
import pandas as pd
import ast
import json
from os import listdir
from os.path import isfile, join

RESULTS_DIR = "results/"
TEST_DIR = "data/test/"

GLOVE_SUM_IDF = "glove_sum_idf.csv"
GLOVE_SUM_NER_IDF = "glove_sum_ner_idf.csv"
GLOVE_SUM_NER = "glove_sum_ner.csv"
GLOVE_SUM_POS_NER = "glove_sum_pos_ner.csv"
GLOVE_SUM_POS = "glove_sum_pos.csv"
GLOVE_SUM = "glove_sum.csv"
TEXT_RANK = "text_rank.csv"
W2V_SUM_IDF = "w2v_sum_idf.csv"
W2V_SUM_NER_IDF = "w2v_sum_ner_idf.csv"
W2V_SUM_NER = "w2v_sum_ner.csv"
W2V_SUM_POS_NER = "w2v_sum_pos_ner.csv"
W2V_SUM_POS = "w2v_sum_pos.csv"
W2V_SUM = "w2v_sum.csv"


glove_sum_idf_df = pd.read_csv(RESULTS_DIR + GLOVE_SUM_IDF, sep = "\t")
glove_sum_ner_idf_df = pd.read_csv(RESULTS_DIR + GLOVE_SUM_NER_IDF, sep = "\t")
glove_sum_ner_df = pd.read_csv(RESULTS_DIR + GLOVE_SUM_NER, sep = "\t")
glove_sum_pos_ner_df = pd.read_csv(RESULTS_DIR + GLOVE_SUM_POS_NER, sep = "\t")
glove_sum_pos_df = pd.read_csv(RESULTS_DIR + GLOVE_SUM_POS, sep = "\t")
glove_sum_df = pd.read_csv(RESULTS_DIR + GLOVE_SUM, sep = "\t")
text_rank_df = pd.read_csv(RESULTS_DIR + TEXT_RANK, sep = "\t")
w2v_sum_idf_df = pd.read_csv(RESULTS_DIR + W2V_SUM_IDF, sep = "\t")
w2v_sum_ner_idf_df = pd.read_csv(RESULTS_DIR + W2V_SUM_NER_IDF, sep = "\t")
w2v_sum_ner_df = pd.read_csv(RESULTS_DIR + W2V_SUM_NER, sep = "\t")
w2v_sum_pos_ner_df = pd.read_csv(RESULTS_DIR + W2V_SUM_POS_NER, sep = "\t")
w2v_sum_pos_df = pd.read_csv(RESULTS_DIR + W2V_SUM_POS, sep = "\t")
w2v_sum_df = pd.read_csv(RESULTS_DIR + W2V_SUM, sep = "\t")

intersection_df = pd.DataFrame(columns=["verdict", "intersection"])

test_ids = [f for f in listdir(TEST_DIR) if isfile(join(TEST_DIR, f))] 

# In train set there is verdicts with indexes from 0 to 1395. That's why the for loop goes from 0 to 1395. 
# We check if the verdict with current index is found in more than 7 model results

intersection_df = pd.DataFrame(columns=["verdict", "intersection"])
def intersection_pretrained():
    intersection_df = pd.DataFrame(columns=["verdict", "intersection"])
    all_data_frames = [glove_sum_idf_df, glove_sum_ner_idf_df, glove_sum_ner_df, glove_sum_pos_ner_df, glove_sum_pos_df, 
                    glove_sum_df, w2v_sum_idf_df, w2v_sum_ner_df, w2v_sum_ner_idf_df, w2v_sum_pos_ner_df, w2v_sum_pos_df, w2v_sum_df]


    for verdict in test_ids:
        verdict_final_res = []
        for i in range(0, 1396):
            count = 0
            for current_df in all_data_frames:
                try:
                    current_list = (current_df.loc[current_df['verdict'] == verdict]).iloc[0]['indexes']
                    result_list = current_list[1:len(current_list) - 1].split()
                    if str(i) in result_list:
                        count += 1
                    print(i)
                except Exception as e:
                    print(e)
                    continue
            if count > 5:
                verdict_final_res.append(i)
        
        intersection_df = intersection_df.append(
                { "verdict" : verdict, 
                "intersection" : verdict_final_res}, ignore_index=True)
        
    intersection_df.to_csv("intersection2.csv", sep="\t")
    
def intersection_w_text_rank():    
    for i in range(0, len(intersection)):
        verdict = intersection.iloc[i]["verdict"]
        final_res = []
        try:
            idx = text_rank_df.index[text_rank_df["0"] == verdict]
            result_tr = text_rank_df.iloc[idx + 1]["0"]
            result_list = result_tr.values[0][1:len(result_tr.values[0]) - 1].split()
            
            intersection_str = intersection.iloc[i]["intersection"]
            intersection_list = intersection_str[1:len(intersection_str)].split(", ")
            
            
            for value in intersection_list:
                if value in result_list:
                    final_res.append(value)
                    
            intersection_text_rank = intersection_text_rank.append(
                    { "verdict" : verdict, 
                    "intersection" : final_res}, ignore_index=True)
        except Exception as e:
            print(e)
            continue
        
    intersection_text_rank.to_csv("intersection_w_textrank2.csv", sep = "\t")
        
    
intersection_pretrained()

intersection_text_rank = pd.DataFrame(columns=["verdict", "intersection"])
intersection = pd.read_csv("intersection2.csv", sep = "\t")

intersection_w_text_rank()
    




