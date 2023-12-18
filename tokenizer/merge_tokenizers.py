import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from transformers import AutoTokenizer, LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse


def is_chinese(uchar):
    return "\u4e00" <= uchar <= "\u9fa5"


def is_chinese_string(string):
    return all(is_chinese(c) for c in string)


def load_baichuan_vocab(vocab_file):
    words = set()
    with open(vocab_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                words.add(line.strip().split()[0])
    return words


def load_jieba_vocab(jieba_vocab_file):
    with open(jieba_vocab_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        word_freqs = [line.strip().split() for line in lines]
        word_freqs.sort(key=lambda x: int(x[1]), reverse=True)
    return word_freqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_tokenizer_dir", default=None, type=str, required=True)
    parser.add_argument("--domain_sp_model", default="./medical_sp.model", type=str)
    parser.add_argument(
        "--baichuan_vocab_file", default="../data/baichuan_vocab.txt", type=str
    )
    parser.add_argument(
        "--add_jieba", action="store_true", help="Whether to add jieba vocab."
    )
    parser.add_argument(
        "--jieba_word_freq_file", default="../data/word_freq.txt", type=str
    )
    parser.add_argument("--jieba_word_size", default=20000, type=int)

    args = parser.parse_args()
    print(args)

    llama_tokenizer = AutoTokenizer.from_pretrained(
        args.base_tokenizer_dir, trust_remote_code=True
    )
    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(args.domain_sp_model)

    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

    print("Llama Tokenizer: ")
    print(len(llama_tokenizer), len(chinese_sp_model))
    print(llama_tokenizer.all_special_tokens)
    print(llama_tokenizer.all_special_ids)
    print(llama_tokenizer.special_tokens_map)

    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
    print(len(llama_spm_tokens_set))

    added_set = set()
    for p in chinese_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)
            added_set.add(piece)

    print(f"add {len(added_set)} tokens to base tokenizer vocab")

    # Add BaiChuan vocab
    vocab = load_baichuan_vocab(args.baichuan_vocab_file)
    print(f"Baichuan vocab size: {len(vocab)}")
    baichuan_vocab_set = set([word for word in vocab if is_chinese_string(word)])
    print(f"BaiChuan Chinese vocab size: {len(baichuan_vocab_set)}")
    print("BaiChuan vocab head(10):", list(baichuan_vocab_set)[:10])

    for p in baichuan_vocab_set:
        piece = p
        if piece not in llama_spm_tokens_set and piece not in added_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)
            added_set.add(piece)

    print(f"add {len(added_set)} tokens to base tokenizer vocab")

    if args.add_jieba:
        jieba_vocab = load_jieba_vocab(args.jieba_word_freq_file)
        jieba_vocab = jieba_vocab[: args.jieba_word_size]
        jieba_vocab_set = set([item[0] for item in jieba_vocab if item])
        print("jieba_vocab_set size:", len(jieba_vocab_set))
        print("jieba_vocab head:", list(jieba_vocab_set)[:3])
        for p in jieba_vocab_set:
            piece = p
            if piece not in llama_spm_tokens_set and piece not in added_set:
                new_p = sp_pb2_model.ModelProto().SentencePiece()
                new_p.piece = piece
                new_p.score = 0
                llama_spm.pieces.append(new_p)
                added_set.add(piece)

    print(f"add {len(added_set)} tokens to base tokenizer vocab")

    # Save
    output_sp_dir = "merge_tokenize_sp"
    output_hf_dir = "merge_tokenize_hf"
    os.makedirs(output_sp_dir, exist_ok=True)
    with open(output_sp_dir + "/chinese_llama.model", "wb") as f:
        f.write(llama_spm.SerializeToString())

    tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + "/chinese_llama.model")
    tokenizer.save_pretrained(output_hf_dir)

    chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)

    text = """this is a test, hello world. thisisatesthelloworld, 
慕容复来到河边，姑苏慕容氏在外面丢了人。
1号店一周岁了，我们一古脑儿买了10斤零食。
巴塞罗那足球俱乐部简称巴萨（Barça），是一家位于西班牙加泰罗尼亚巴塞罗那的足球俱乐部，于1899年由瑞士企业家胡安·甘伯所创立，世界球坛顶级足球俱乐部之一。俱乐部主场可容纳接近十万名观众，是全欧洲最大及世界第二大的足球场。
白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"""
    print("Test text:\n", text)
    print(
        f"Tokenized by Chinese-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}"
    )


if __name__ == "__main__":
    main()
