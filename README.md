# Sentence Compression for Portuguese

[![tag](https://i.imgur.com/ZhZ9Mw7.png)](http://nilc.icmc.usp.br/nilc/index.php)

Hello, here you will find two datasets for the **Sentence Compression** (SC) task focused on the Portuguese language based on the deletion approach.

For instance: 

- ``input``: Olá, aqui você encontrará datasets para projetos de SC em Português. 
- ``output``: aqui você encontrará datasets para SC em Português.

This repository is organized as follow:

-  ``data``: the folder with the two datasets

	-  ``PCSC-Pares``: has 874 pairs of original sentences and the respective compressed version that were extracted from the **Priberam Compressive Summarization Corpus**  [Almeida et. al (2014)](https://www.aclweb.org/anthology/L14-1193/).  
		1. ``alignment_sentences.csv``: it is a csv file with the alignment sentences (a pair for row) with the follwing columns (original_sent_id, original_sent, compressend_sent, compressend_sent_id)
		2. ``data.csv``: It is tsv file with the data above organized into tokens. Each row has a single token with respective label (must be removed or not from the input sentence?) and some linguistic features that were extracted previously.
	-  ``G1-Pares``: has 7,024 pairs of long and reduced sentences. These sentences were alignment based on the approach of [Filippova (2015)](https://www.aclweb.org/anthology/D15-1042/) from 1.008.356 news that were automatically crawled from [G1 news portal](\url{http://www.g1.com.br)

   