- Hugging Face libraries we will use to train a pre-trained model:
	- Transformers
	- Dataset
	- Tokenizers
	- Accelerate - run distributed training on any setup
	- Evaluate
	- Trainer - to fine-tune a model with modern best practices
- Goal - fine-tuned a BERT model for text classification
### Datasets
- Hugging Face hub datasets - https://huggingface.co/datasets
- https://huggingface.co/docs/datasets/index
- https://huggingface.co/docs/transformers/main/en/notebooks
```python
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

def tokenize_function(example):
	return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

raw_datasets = load_dataset("glue", "mrpc")
raw_train_dataset = raw_datasets["train"]

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_sentences_1 = tokenizer(raw_datasets["train"]["sentence1"])
tokenized_sentences_2 = tokenizer(raw_datasets["train"]["sentence2"])

inputs = tokenizer("This is the first sentence.", "This is the second one.")
tokenizer.convert_ids_to_tokens(inputs["input_ids"])

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
[len(x) for x in samples["input_ids"]]
batch = data_collator(samples)

{k: v.shape for k, v in batch.items()}

```
## Fine-Tuning Model with Trainer API
- 