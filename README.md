# SimilarLegalDocumentRetrieval
Similar Legal Document Retrieval 

## Authors 
| Student          | Record        |
| -----------------|:-------------:|
| Ilija Brdar      | E2 5/2021     |
| Jelena Vlajkov   | E2 61/2021    |


## Experiment 

There are 2 experiments: 
1. TextRank + TF-IDF
2. Pretrained Word Vectors

The goal is to find similar legal documents using different document representations. 

### TextRank + TF-IDF
Using TextRank algorithm, that is based on PageRank algorithm, we can sort N most important sentences in documents.
Then, we can represent a smaller amount of sentences with TF-IDF and GTF (*Global Term Frequency*). We use cosine similarity for finding N most similar legal documents.

### Pretrained Word Vectors
Using Google News Word2Vector and GloVe, we represent documents as vectors. With different techniques (POS, NER, IDF), we can emphasize more different words. We also use cosine similarity for finding N most similar legal documents.

## Evaluation
More accurate evaluation demands help from legal experts, giving the fact that the dataset is unlabeled. 
As simple evaluation, we use intersection between different models. If 5 or more models based on pretrained vectors + TextRank algorithm says a documents are similar, we mark them as similar.

## Instructions 
To run our experiment, you will first need to clone the repository to your computer. 

### TextRank + TF-IDF 
To run TextRank experiment, just run the notebook called DocumentVectorEmbeddings2.ipynb. To be less time consuming, data frames containing train and test data are already saved in csv files.
If you want to read the datasets and apply text rank from beginning, just change the flag RUN_TRAIN to True.

### Pretrained Word Vecotrs
To run experiment with Pretrained Vectors, just run the notebook called PretrainedWordVectors2.ipynb. To be less time consuming, IDF, POS, NER values of words have been previously saved in csv files. 


