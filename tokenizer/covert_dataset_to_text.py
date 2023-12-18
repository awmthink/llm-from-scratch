from datasets import load_dataset


pretrain_data = load_dataset("shibing624/medical", "pretrain", split="train")

pretrain_data.to_csv("../data/pretrain.csv")
