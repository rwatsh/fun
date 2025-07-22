- Features of Transformers library are:
	- All models are simple PyTorch `nn.Module` classes.
- `tokenizer()` API, which is the other main component of the `pipeline()` function. Tokenizers take care of the first and last processing steps, handling the conversion from text to numerical inputs for the neural network, and the conversion back to text when it is needed.
![[Pasted image 20250705114729.png]]
- Preprocessing with Tokenizer
	- Transformer models can’t process raw text directly, so the first step of our pipeline is to convert the text inputs into numbers that the model can make sense of. To do this we use a _tokenizer_, which will be responsible for:

		- Splitting the input into words, subwords, or symbols (like punctuation) that are called _tokens_
		- Mapping each token to an integer
		- Adding additional inputs that may be useful to the model
	- All this preprocessing needs to be done in exactly the same way as when the model was pretrained, so we first need to download that information from the [Model Hub](https://huggingface.co/models). To do this, we use the `AutoTokenizer` class and its `from_pretrained()` method. Using the checkpoint name of our model, it will automatically fetch the data associated with the model’s tokenizer and cache it.
	- Once we have the tokenizer, we can directly pass our sentences to it and we’ll get back a dictionary that’s ready to feed to our model! The only thing left to do is to convert the list of input IDs to tensors.
	- Transformer models only accept _tensors_ as input. If this is your first time hearing about tensors, you can think of them as NumPy arrays instead. A NumPy array can be a scalar (0D), a vector (1D), a matrix (2D), or have more dimensions.
	- To specify the type of tensors we want to get back (PyTorch or plain NumPy), we use the `return_tensors` argument

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

output:
```
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```
- `input_ids` contains two rows of integers (one for each sentence) that are the unique identifiers of the tokens in each sentence.
- We can download our pretrained model the same way we did with our tokenizer using AutoModel class.
```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

# torch.Size([2, 16, 768])

```
- given some inputs, it outputs what we’ll call _hidden states_, also known as _features_. For each model input, we’ll retrieve a high-dimensional vector representing the **contextual understanding of that input by the Transformer model**.
- While these hidden states can be useful on their own, they’re usually inputs to another part of the model, known as the _head_.
- Adaptation heads, also known simply as heads, come up in different forms: language modeling heads, question answering heads, sequence classification heads -  usually made up of one or a few layers, to convert the transformer predictions to a task-specific output.
- The vector output by the Transformer module is usually large. It generally has three dimensions:

	- **Batch size**: The number of sequences processed at a time (2 in our example).
	- **Sequence length**: The length of the numerical representation of the sequence (16 in our example).
	- **Hidden size**: The vector dimension of each model input.
- It is said to be “high dimensional” because of the last value. The hidden size can be very large (768 is common for smaller models, and in larger models this can reach 3072 or more).
![[Pasted image 20250705121336.png]]
- The output of the Transformer model is sent directly to the model head to be processed.
-  the model is represented by its embeddings layer and the subsequent layers. The embeddings layer converts each input ID in the tokenized input into a vector that represents the associated token. The subsequent layers manipulate those vectors using the attention mechanism to produce the final representation of the sentences.
- There are many different architectures available in 🤗 Transformers, with each one designed around tackling a specific task.
	- `*Model` (retrieve the hidden states)
	- `*ForCausalLM`
	- `*ForMaskedLM`
	- `*ForMultipleChoice`
	- `*ForQuestionAnswering`
	- `*ForSequenceClassification`
	- `*ForTokenClassification`
	- etc
```python
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)
# torch.Size([2, 2])
print(outputs.logits)
# tensor([[-1.5607,  1.6123],
#       [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
```
- the model head takes as input the high-dimensional vectors we saw before, and outputs vectors containing two values (one per label)
- Our model predicted `[-1.5607, 1.6123]` for the first sentence and `[ 4.1692, -3.3464]` for the second one. Those are not probabilities but _logits_, the raw, unnormalized scores outputted by the last layer of the model. To be converted to probabilities, they need to go through a [SoftMax](https://en.wikipedia.org/wiki/Softmax_function) layer (all 🤗 Transformers models output the logits, as the loss function for training will generally fuse the last activation function, such as SoftMax, with the actual loss function, such as cross entropy)
```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

# tensor([[4.0195e-02, 9.5980e-01],
#        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
model.config.id2label
# {0: 'NEGATIVE', 1: 'POSITIVE'}
```
- Now we can see that the model predicted `[0.0402, 0.9598]` for the first sentence and `[0.9995, 0.0005]` for the second one. These are recognizable probability scores.
- To get the labels corresponding to each position, we can inspect the `id2label` attribute of the model config
- Now we can conclude that the model predicted the following:

	- First sentence: NEGATIVE: 0.0402, POSITIVE: 0.9598
	- Second sentence: NEGATIVE: 0.9995, POSITIVE: 0.0005

- We have successfully reproduced the three steps of the pipeline: preprocessing with tokenizers, passing the inputs through the model, and postprocessing!