### jt_ds_recruitement_task

To produce predictions go through all steps:
1. 01_text_preprocessing: this code produce text columns ready to tokenize and fill into language model.
2. 02_embedding_transformer: this code create embeddings from text features (right now using pretrained Fasttext model or train own one).
3. 03_train: classifiers attemps on embedded features.
4. Multilayer logistic regression run. (optional)
