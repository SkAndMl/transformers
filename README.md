# Building Transformers
Run the following command once the repository has been cloned
```python
pip install -r requirements.txt
```
## BERTMaskedLM
You can train the BERTMaskedLM model built by running the following command
```bash
python train.py
```
Or you can load the pre-trained weigths using
```python
bert = BERTMaskedLM(config=config, vocab_size=vocab_size)
bert.load_state_dict(torch.load("../weights/bert_masked_lm.pt",
                                map_location=torch.device(device=device)))
```
Pre-trained weights results
| Epoch | Loss | Accuracy |
| ----- | ---- | -------- |
|   1   | 6.21 |  13.92%  |
|   2   | 4.93 |  21.74%  |
|   3   | 4.23 |  25.67%  |
|   4   | 3.59 |  29.35%  |
|   5   | 2.99 |  35.15%  |
|   6   | 2.46 |  43.26%  |
|   7   | 2.03 |  50.69%  |
|   8   | 1.70 |  56.99%  |
|   9   | 1.45 |  62.2%   |
|   10  | 1.24 |  67.02%  |

## PoemGPT
PoemGPT is a decoder-only transformer model that generates poems similar to that of Shakespeare's
You can generate poems by running the following command in the terminal
```bash
python gen_poem.py --num_tokens <num_chars in poem>
```
```
PRINCE PEY:
Very well! I will be naked to take when.

MARIANA:
Now sword? and who have friend out of the place
Artend I of your vental, and am not pratise
He disture friends in Playets to the comprove.

LUCIO:
By you women, I do put on cominition.
And whils woratil orp we moforey any
Moris: y maingive, os wacking, waras, t, ouns
Olyonesig? fagonad! cin, t, s,
I; fo, foce lelinsts!-t tate nope's; war!
Trimply titaved ps, ge
Fingeivedy, bequbupe, po.
Trhokeringe at bous ot: bys; wined iooualy; on
```


## Reference links for this repository
* [Attention is all you need](https://arxiv.org/abs/1706.03762)
* [Understanding Transformers](https://medium.com/mlearning-ai/understanding-transformers-388a0ff97799)
* [BERT](https://arxiv.org/abs/1810.04805)
* [Datapipes](https://sebastianraschka.com/blog/2022/datapipes.html)
* [Datapipes-repo](https://github.com/rasbt/datapipes-blog/blob/main/0_download-and-prep-data.ipynb)
* [Bert Data preparation](https://coaxsoft.com/blog/building-bert-with-pytorch-from-scratch)
* [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
* [Karpathy - minGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=699s)
* [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
