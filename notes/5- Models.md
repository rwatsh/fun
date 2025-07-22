```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
```
- the `from_pretrained()` method will download and cache the model data from the Hugging Face Hub. As mentioned previously, the checkpoint name corresponds to a specific model architecture and weights, in this case a BERT model with a basic architecture (12 layers, 768 hidden size, 12 attention heads) and cased inputs (meaning that the uppercase/lowercase distinction is important).
- The `AutoModel` class and its associates are actually simple wrappers designed to fetch the appropriate model architecture for a given checkpoint. It’s an “auto” class meaning it will guess the appropriate model architecture for you and instantiate the correct model class. However, if you know the type of model you want to use, you can use the class that defines its architecture directly:
```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
model.save_pretrained("directory_on_my_computer")
```
- `save_pretrained()` method, which saves the model’s weights and architecture configuration
- It will save 2 files:
	- config.json - attributes of model architecture + some metadata
	- pytorch_model.bin - model weights (state dictionary)
- To reuse a saved model, use the `from_pretrained()`
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("directory_on_my_computer")
```
- sharing the model
```python
model.push_to_hub("my-awesome-model")
from transformers import AutoModel

model = AutoModel.from_pretrained("your-username/my-awesome-model")
```
- Encoding Text
	- Models can only process numbers, so tokenizers need to convert our text inputs to numerical data.
	- Translating text to numbers is known as _encoding_. Encoding is done in a two-step process: the tokenization, followed by the conversion to input IDs.
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)

#{'input_ids': [101, 8667, 117, 1000, 1045, 1005, 1049, 2235, 17662, 12172, 1012, 102], 
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
tokenizer.decode(encoded_input["input_ids"])
# "[CLS] Hello, I ' m a single sentence! [SEP]"

encoded_input = tokenizer("How are you?", "I'm fine, thank you!",padding=True, return_tensors="pt")

print(encoded_input)
#{'input_ids': tensor([[ 101, 1731, 1132, 1128, 136, 102, 0, 0, 0, 0], [ 101, 146, 112, 182, 2503, 117, 6243, 1128, 106, 102]]), 
#'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
#'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}

encoded_input = tokenizer(
    "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence.",
    truncation=True,
)
print(encoded_input["input_ids"])
# [101, 1188, 1110, 170, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1505, 1179, 5650, 119, 102]

encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,
    truncation=True,
    max_length=5,
    return_tensors="pt",
)
print(encoded_input)
# {'input_ids': tensor([[ 101, 1731, 1132, 1128, 102], [ 101, 146, 112, 182, 102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])}

import torch

  

model_inputs = torch.tensor(encoded_input["input_ids"])
output = model(model_inputs)
```
- We get a dictionary with the following fields:

	- input_ids: numerical representations of your tokens
	- token_type_ids: these tell the model which part of the input is sentence A and which is sentence B 
	- attention_mask: this indicates which tokens should be attended to and which should not
- We can decode the input IDs to get back the original text
- tokenizer has added special tokens — `[CLS]` and `[SEP]` — required by the model. Not all models need special tokens; they’re utilized when a model was pretrained with them, in which case the tokenizer needs to add them as that model expects these tokens.
- Note that when passing multiple sentences, the tokenizer returns a list for each sentence for each dictionary value. We can also ask the tokenizer to return tensors directly from PyTorch
- But there’s a problem: the two lists don’t have the same length! Arrays and tensors need to be rectangular, so we can’t simply convert these lists to a PyTorch tensor (or NumPy array). The tokenizer provides an option for that: padding.
- Note that the padding tokens have been encoded into input IDs with ID 0, and they have an attention mask value of 0 as well. This is because those padding tokens shouldn’t be analyzed by the model: they’re not part of the actual sentence.
- If you have sequences longer than the model can handle, you’ll need to truncate them with the `truncation` parameter
- By combining the padding and truncation arguments, you can make sure your tensors have the exact size you need
### Tokenizers
- Models can only process numbers, so tokenizers need to convert our text inputs to numerical data.
- 3 types of tokenizers:
	- Word-based 
		- Each word gets assigned an ID, starting from 0 and going up to the size of the vocabulary. The model uses these IDs to identify each word.
		- If we want to completely cover a language with a word-based tokenizer, we’ll need to have an identifier for each word in the language, which will generate a huge amount of tokens. For example, there are over 500,000 words in the English language, so to build a map from each word to an input ID we’d need to keep track of that many IDs.
		- words like “dog” are represented differently from words like “dogs”, and the model will initially have no way of knowing that “dog” and “dogs” are similar: it will identify the two words as unrelated.
		- Finally, we need a custom token to represent words that are not in our vocabulary. This is known as the “unknown” token, often represented as `”[UNK]” or ”<unk>”.`
	- Character-based
		- Character-based tokenizers split the text into characters, rather than words. This has two primary benefits:

			- The vocabulary is much smaller.
			- There are much fewer out-of-vocabulary (unknown) tokens, since every word can be built from characters.
		- But it has limitations:
			- each character doesn’t mean a lot on its own
			- whereas a word would only be a single token with a word-based tokenizer, it can easily turn into 10 or more tokens when converted into characters
	- Subword-based
		- Subword tokenization algorithms rely on the principle that
			- frequently used words should not be split into smaller subwords
			- rare words should be decomposed into meaningful subwords.
		- For instance, “annoyingly” might be considered a rare word and could be decomposed into “annoying” and “ly”. These are both likely to appear more frequently as standalone subwords, while at the same time the meaning of “annoyingly” is kept by the composite meaning of “annoying” and “ly”.
		- Techniques implementing subword-based tokenization:
			- Byte-level BPE, as used in GPT-2
			- WordPiece, as used in BERT
			- SentencePiece or Unigram, as used in several multilingual models
- Loading and saving tokenizers is as simple as it is with models.
	- `from_pretrained()` and `save_pretrained()`. These methods will load or save the algorithm used by the tokenizer (a bit like the _architecture_ of the model) as well as its vocabulary (a bit like the _weights_ of the model).
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
encoded_input = tokenizer("Using a Transformer network is simple")

print(encoded_input)

tokenizer.save_pretrained("directory_on_my_computer")
```
- As we’ve seen, the first step is to split the text into words (or parts of words, punctuation symbols, etc.), usually called _tokens_.
- The second step is to convert those tokens into numbers, so we can build a tensor out of them and feed them to the model. To do this, the tokenizer has a _vocabulary_, which is the part we download when we instantiate it with the `from_pretrained()` method. Again, we need to use the same vocabulary used when the model was pretrained.

- Tokenization
	- The tokenization process is done by the `tokenize()` method of the tokenizer
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
# ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
# [7993, 170, 13809, 23763, 2443, 1110, 3014]

decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
# Using a transformer network is simple
```
	
	This tokenizer is a subword tokenizer: it splits the words until it obtains tokens that can be represented by its vocabulary.

- From tokens to input IDs - `convert_tokens_to_ids()` tokenizer method
- These outputs, once converted to the appropriate framework tensor, can then be used as inputs to a model.
- Decoding
	- is going the other way around: from vocabulary indices, we want to get a string. This can be done with the `decode()` method
	- `decode` method not only converts the indices back to tokens, but also groups together the tokens that were part of the same words to produce a readable sentence.
- In summary, atomic operations a tokenizer can handle: tokenization, conversion to IDs, and converting IDs back to a string
- Handling multiple sequences
```python
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)

# Input IDs: [[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607, 2026,  2878,  2166,  1012]]
# Logits: [[-2.7276,  2.8789]]
```
- _Batching_ is the act of sending multiple sentences through the model, all at once.
- When you’re trying to batch together two (or more) sentences, they might be of different lengths. If you’ve ever worked with tensors before, you know that they need to be of rectangular shape, so you won’t be able to convert the list of input IDs into a tensor directly. To work around this problem, we usually _pad_ the inputs.
- Padding makes sure all our sentences have the same length by adding a special word called the _padding token_ to the sentences with fewer values. For example, if you have 10 sentences with 10 words and 1 sentence with 20 words, padding will ensure all the sentences have 20 words.
- The padding token ID can be found in `tokenizer.pad_token_id`.
- This is because the key feature of Transformer models is attention layers that _contextualize_ each token. These will take into account the padding tokens since they attend to all of the tokens of a sequence. To get the same result when passing individual sentences of different lengths through the model or when passing a batch with the same sentences and padding applied, we need to tell those attention layers to ignore the padding tokens. This is done by using an attention mask.
- With Transformer models, there is a limit to the lengths of the sequences we can pass the models. Most models handle sequences of up to 512 or 1024 tokens, and will crash when asked to process longer sequences. There are two solutions to this problem:

	- Use a model with a longer supported sequence length.
	- Truncate your sequences.
- Transformers API can handle all of this (tokenization, conversion to input IDs, padding, truncation, and attention masks) for us with a high-level function. When you call your `tokenizer` directly on the sentence, you get back inputs that are ready to pass through your model.
```python
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```
- Here, the `model_inputs` variable contains everything that’s necessary for a model to operate well. For DistilBERT, that includes the input IDs as well as the attention mask. Other models that accept additional inputs will also have those output by the `tokenizer` object.
- 