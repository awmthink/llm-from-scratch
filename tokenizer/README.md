# Tokenizer

* [bpe.py](./bpe.py): BPE算法的示例实现
* [covert_dataset_to_text.py](./covert_dataset_to_text.py): 该脚本读取huggingface中的一个文本数据集，然后将其保存为csv格式，每行是一个段落，后续用于训练tokenizer。
* [build_domain_tokenzier.py](./build_domain_tokenizer.py): 该脚本是对于SentencePiece中`SentencePieceTrainer.train`的包装，用于训练一个新的Tokenizer。
* [merge_tokenizers.py](./merge_tokenizers.py)：该脚本用于Tokenizer的合并，它在一个base tokenizer的基础上扩充新训练的特定领域的tokenizer，进一步扩充了百川的词汇表和结巴分词中的高频词汇。
* [nlp course tokenizer](./nlp_course_tokenizer.ipynb)：HuggingFace官方的NLP Course中tokenizer库的文档，介绍了如何训练一个用于代码的tokenizer。