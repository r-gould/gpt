# GPT

An implementation of the GPT model in PyTorch, as described originally in the paper '[Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)'.

A demonstration for training the model is shown in main.py, and continuations can be generated as shown in the example below. 

Config files for GPT-1, GPT-2, and GPT-3 models can be found in configs, and can be chosen from using model_name in main.py

# Example

Due to limited resources, examples of a model with GPT-1 parameters only trained for 8 epochs on WikiText2 (very little training compared to what is typical for language models) are shown below:

```
(see main.py for definitions)
>>> from src.generator import Generator
>>> generator = Generator(model, tokenizer, vocab, device)
>>> generator.generate("the song was", steps=30)
['the', 'song', 'was', 'released', 'on', 'may', '1', ',', '2010', ',', 'in', 'the', 'united', 'states', 'on', 'the', 'billboard', 'hot', '100', ',', 'and', 'the', 'song', 'peaked', 'at', 'number', 'one', 'on', 'the', 'billboard', 'hot', '100', '.']

>>> generator.generate("england is most known for", steps=30)
['england', 'is', 'most', 'known', 'for', 'the', 'english', 'team', ',', 'as', 'the', 'most', 'popular', 'team', 'in', 'the', 'world', '.', 'it', 'was', 'the', 'first', 'time', 'since', 'the', 'world', 'war', 'ii', 'when', 'the', 'team', 'won', 'the', 'world', 'cup']

>>> generator.generate("the united states is most known for", steps=30)
['the', 'united', 'states', 'is', 'most', 'known', 'for', 'the', 'american', 'civil', 'war', ',', 'but', 'the', 'united', 'states', 'department', 'of', 'war', 'in', 'the', 'united', 'states', 'is', 'the', 'most', 'important', 'part', 'of', 'the', 'world', 'war', 'and', 'the', 'united', 'states', 'army']
```

An example of a very bad continuation,

```
>>> generator.generate("the meaning of life is", steps=30)
['the', 'meaning', 'of', 'life', 'is', 'the', 'most', 'common', 'in', 'the', 'world', '.', 'it', 'is', 'the', 'most', 'common', 'name', 'for', 'the', 'world', 'of', 'ireland', 'and', 'the', 'most', 'common', 'starling', 'in', 'the', 'united', 'states', '.', 'it', 'is']
```

With sufficient resources, WikiText103 would be much more effective for training.

# References

**Improving Language Understanding by Generative Pre-Training (GPT-1):** https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

**Language Models are Unsupervised Multitask Learners (GPT-2):** https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

**Language Models are Few-Shot Learners (GPT-3):** https://arxiv.org/abs/2005.14165
