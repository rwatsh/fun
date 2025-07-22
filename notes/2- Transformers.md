- `pipeiine()` function - to solve NLP tasks for text generation and classification
- NLP tasks are speech recognition, text generation and sentiment analysis.
- Aims to understand the context or intent and not just individual words.
	- classifying sentences - sentiment analysis, spam detection
	- classifying words - part of speech tagging, NER
	- generating text - auto completion, translation, summarization
	- extracting answers - Q/A based on context
- Transformer models are used to solve all kinds of tasks across different modalities, including natural language processing (NLP), computer vision, audio processing, and more.
- https://github.com/huggingface/transformers - transformer library
- https://huggingface.co/models - model and datasets hub
- Working with pipelines
	- `pipeline()` - connects model with pre and post processing steps, allowing us to directly input any text and get an answer.
	- Tasks: - https://huggingface.co/docs/hub/en/models-tasks
		- Text classification
		- Zero-shot classification
		- Text generation
		- Text completion (mask filling)
		- Token classification (NER)
		- Question answering
		- Summarization
		- Translation
	- The `pipeline()` function supports multiple modalities, allowing you to work with text, images, audio, and even multimodal tasks. I
Hugging Face token - https://huggingface.co/settings/tokens/new?tokenType=write
https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter1/section3.ipynb
```
HF_TOKEN=""
```
Setup:
```
pip install datasets evaluate 'transformers[sentencepiece]'
pip install 'transformers[torch]'
```
```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```
output:
```
[{'label': 'POSITIVE', 'score': 0.9598047137260437}]
```
Other uses of pipeline function:
```python
classifier = pipeline("zero-shot-classification")

classifier(
"This is a course about the Transformers library",
candidate_labels=["education", "politics", "business"],
)

generator = pipeline("text-generation", model="distilgpt2")

generator(
"In this course, we will teach you how to",
max_length=30,
num_return_sequences=2,)

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")


question_answerer = pipeline("question-answering")

question_answerer( question="Where do I work?",
context="My name is Sylvain and I work at Hugging Face in Brooklyn",)

summarizer = pipeline("summarization")

summarizer("""
America has changed dramatically during recent years. Not only has the number of

graduates in traditional engineering disciplines such as mechanical, civil,

electrical, chemical, and aeronautical engineering declined, but in most of

the premier American universities engineering curricula now concentrate on

and encourage largely the study of engineering science. As a result, there

are declining offerings in engineering subjects dealing with infrastructure,

the environment, and related issues, and greater concentration on high

technology subjects, largely supporting increasingly complex scientific

developments. While the latter is important, it should not be at the expense

of more traditional engineering.
"""
)

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")
```
## Architecture
- https://arxiv.org/abs/1706.03762 - Attention is all you need paper
- GPT3 - zero-shot learning - no fine-tuning required for performing tasks in a specific domain (as the LLM was trained for a very large data corpus spanning multiple domains)
- 3 types of transformer models:
	- GPT-like - auto-regressive models - used for text generation
	- BERT-like - auto-encoding models
	- T5-like - sequence-to-sequence models
- Transformer models are trained using `self-supervised `learning`. Self-supervised learning is a type of training in which the objective is automatically computed from the inputs of the model. That means that humans are not needed to label the data!
- This type of model develops a statistical understanding of the language it has been trained on, but it’s less useful for specific practical tasks. Because of this, the general pretrained model then goes through a process called _transfer learning_ or _fine-tuning_. During this process, the model is fine-tuned in a supervised way — that is, using human-annotated labels — on a given task.
- An example of a task is predicting the next word in a sentence having read the _n_ previous words. This is called _causal language modeling_ because the output depends on the past and present inputs, but not the future ones.
- Another example is _masked language modeling_, in which the model predicts a masked word in the sentence.
- general strategy to achieve better performance is by increasing the models’ sizes as well as the amount of data they are pretrained on.
- sharing the trained weights and building on top of already trained weights reduces the overall compute cost and carbon footprint of the community.
- _Pretraining_ is the act of training a model from scratch: the weights are randomly initialized, and the training starts without any prior knowledge.
- This pretraining is usually done on very large amounts of data. Therefore, it requires a very large corpus of data, and training can take up to several weeks.
- _Fine-tuning_, on the other hand, is the training done **after** a model has been pretrained. To perform fine-tuning, you first acquire a pretrained language model, then perform additional training with a dataset specific to your task.
- For example, one could leverage a pretrained model trained on the English language and then fine-tune it on an arXiv corpus, resulting in a science/research-based model. The fine-tuning will only require a limited amount of data: the knowledge the pretrained model has acquired is “transferred,” hence the term _transfer learning_.
- Fine-tuning a model therefore has lower time, data, financial, and environmental costs. It is also quicker and easier to iterate over different fine-tuning schemes, as the training is less constraining than a full pretraining.
- you should always try to leverage a pretrained model — one as close as possible to the task you have at hand — and fine-tune it.
![[Pasted image 20250703153856.png]]
- Encoder - The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
- Decoder - The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs. Auto-regressive.
Each of these parts can be used independently, depending on the task:

- **Encoder-only models**: These models use a bidirectional approach to understand context from both directions. They’re best suited for tasks that require deep understanding of text, such as classification, named entity recognition, and question answering. t each stage, the attention layers can access all the words in the initial sentence. These models are often characterized as having “bi-directional” attention, and are often called _auto-encoding models_. The pretraining of these models usually revolves around somehow corrupting a given sentence (for instance, by masking random words in it) and tasking the model with finding or reconstructing the initial sentence.For e.g. BERT, DistilBERT, ModernBERT
- **Decoder-only models**: These models process text from left to right and are particularly good at text generation tasks. They can complete sentences, write essays, or even generate code based on a prompt. At each stage, for a given word the attention layers can only access the words positioned before it in the sentence. These models are often called _auto-regressive models_. The pretraining of decoder models usually revolves around predicting the next word in the sentence. For e.g GPT-3, Llama
- **Encoder-decoder models** or **sequence-to-sequence models**: These models combine both approaches, using an encoder to understand the input and a decoder to generate output. They excel at sequence-to-sequence tasks like translation, summarization, and question answering.For e.g. T5, BART, mBART, Marian
### Attention Layer
- A key feature of Transformer models is that they are built with special layers called _attention layers_.
- this layer will tell the model to pay specific attention to certain words in the sentence you passed it (and more or less ignore the others) when dealing with the representation of each word.
	- Given the input “You like this course”, a translation model will need to also attend to the adjacent word “You” to get the proper translation for the word “like”, because in French the verb “like” is conjugated differently depending on the subject. The rest of the sentence, however, is not useful for the translation of that word. In the same vein, when translating “this” the model will also need to pay attention to the word “course”, because “this” translates differently depending on whether the associated noun is masculine or feminine. Again, the other words in the sentence will not matter for the translation of “course”. With more complex sentences (and more complex grammar rules), the model would need to pay special attention to words that might appear farther away in the sentence to properly translate each word.
- The same concept applies to any task associated with natural language: a word by itself has a meaning, but that meaning is deeply affected by the context, which can be any other word (or words) before or after the word being studied.
	- The Transformer architecture was originally designed for translation. During training, the encoder receives inputs (sentences) in a certain language, while the decoder receives the same sentences in the desired target language. In the encoder, the attention layers can use all the words in a sentence (since, as we just saw, the translation of a given word can be dependent on what is after as well as before it in the sentence). The decoder, however, works sequentially and can only pay attention to the words in the sentence that it has already translated (so, only the words before the word currently being generated). For example, when we have predicted the first three words of the translated target, we give them to the decoder which then uses all the inputs of the encoder to try to predict the fourth word.
![[Pasted image 20250704001528.png]]
- **Architecture**: This is the skeleton of the model — the definition of each layer and each operation that happens within the model.
- **Checkpoints**: These are the weights that will be loaded in a given architecture.
- **Model** - is an umbrella term
- For example, BERT is an architecture while `bert-base-cased`, a set of weights trained by the Google team for the first release of BERT, is a checkpoint. However, one can say “the BERT model” and “the `bert-base-cased` model.”
- Models and tasks:
	- [Wav2Vec2](https://huggingface.co/docs/transformers/model_doc/wav2vec2) for audio classification and automatic speech recognition (ASR)
	- [Vision Transformer (ViT)](https://huggingface.co/docs/transformers/model_doc/vit) and [ConvNeXT](https://huggingface.co/docs/transformers/model_doc/convnext) for image classification
	- [DETR](https://huggingface.co/docs/transformers/model_doc/detr) for object detection
	- [Mask2Former](https://huggingface.co/docs/transformers/model_doc/mask2former) for image segmentation
	- [GLPN](https://huggingface.co/docs/transformers/model_doc/glpn) for depth estimation
	- [BERT](https://huggingface.co/docs/transformers/model_doc/bert) for NLP tasks like text classification, token classification and question answering that use an encoder
	- [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2) for NLP tasks like text generation that use a decoder
	- [BART](https://huggingface.co/docs/transformers/model_doc/bart) for NLP tasks like summarization and translation that use an encoder-decoder
- Language models work by being trained to predict the probability of a word given the context of surrounding words. This gives them a foundational understanding of language that can generalize to other tasks.
- There are two main approaches for training a transformer model:
	- **Masked language modeling (MLM)**: Used by **encoder** models like BERT, this approach randomly masks some tokens in the input and trains the model to predict the original tokens based on the surrounding context. This allows the model to learn bidirectional context (looking at words both before and after the masked word). 
	- **Causal language modeling (CLM)**: Used by **decoder** models like GPT, this approach predicts the next token based on all previous tokens in the sequence. The model can only use context from the left (previous tokens) to predict the next token.
- Language models are typically pretrained on large amounts of text data in a self-supervised manner (without human annotations), then fine-tuned on specific tasks. This approach, known as transfer learning, allows these models to adapt to many different NLP tasks with relatively small amounts of task-specific data.
- Tasks requiring bidirectional context use encoders, tasks generating text use decoders, and tasks converting one sequence to another use encoder-decoders.
- Tasks:
	- Text generation involves creating coherent and contextually relevant text based on a prompt or input.
	- Text classification involves assigning predefined categories to text documents, such as sentiment analysis, topic classification, or spam detection.
		- [BERT](https://huggingface.co/docs/transformers/model_doc/bert) is an encoder-only model and is the first model to effectively implement deep bidirectionality to learn richer representations of the text by attending to words on both sides.
	- Token classification involves assigning a label to each token in a sequence, such as in named entity recognition or part-of-speech tagging. Use BERT
	- Question answering involves finding the answer to a question within a given context or passage. Use BERT
	- Summarization involves condensing a longer text into a shorter version while preserving its key information and meaning. Encoder-decoder models like [BART](https://huggingface.co/docs/transformers/model_doc/bart) and [T5](https://huggingface.co/learn/llm-course/chapter1/model_doc/t5) are designed for the sequence-to-sequence pattern of a summarization task.
	- Translation involves converting text from one language to another while preserving its meaning. Translation is another example of a sequence-to-sequence task, which means you can use an encoder-decoder model like [BART](https://huggingface.co/docs/transformers/model_doc/bart) or [T5](https://huggingface.co/learn/llm-course/chapter1/model_doc/t5) to do it.
- Multi-modal models
	- Whisper - encoder-decoder transformer model for speech and audio data.
	- Computer Vision Task - 2 approaches:
		- Split an image into a sequence of patches and process them in parallel with a Transformer.
		- Use a modern CNN, like [ConvNeXT](https://huggingface.co/docs/transformers/model_doc/convnext), which relies on convolutional layers but adopts modern network designs
		- ViT and ConvNeXT are commonly used for image classification, but for other vision tasks like object detection, segmentation, and depth estimation, we’ll look at DETR, Mask2Former and GLPN, respectively; these models are better suited for those tasks.
		- ViT uses an attention mechanism while ConvNeXT uses convolutions.
- LLMs are trained in 2 phases:
	- Pretraining - The model learns to predict the next token on vast amounts of text data
	- Instruction tuning - 1. The model is fine-tuned to follow instructions and generate helpful responses
![[Pasted image 20250704102117.png]]
Figure shows speech translation system using:
- OpenAI's Whisper Base to translate speech in any language to English text
- Microsoft's SpeechT5 TTS model for text to speech generation
![[Pasted image 20250704102457.png]]
- When in doubt about which model to use, consider:
	
	1. What kind of understanding does your task need? (Bidirectional or unidirectional)
	2. Are you generating new text or analyzing existing text?
	3. Do you need to transform one sequence into another?
