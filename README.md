# pytorch-skipthoughts

Simple, plug & play pytorch implementation of a skipthoughts encoder.
Pretrained models and design are taken from the paper [Skip-Thought Vectors](http://arxiv.org/abs/1506.06726) and the corresponding [theano based implementation](https://github.com/ryankiros/skip-thoughts).

## To do

* add support for vocabulary extension
* implement decoders
* add training script for custom model creation


## Example

```python
from skipthoughts import Encoder

dirStr = 'models'
encoder = Encoder(dirStr)

sentences = ["Hey, how are you?", "This sentence is a lie"]

encodedSentences = encoder.encode(sentences)
print(encodedSentences)
```

## Reference

Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. **"Skip-Thought Vectors."** *arXiv preprint arXiv:1506.06726 (2015).*

    @article{kiros2015skip,
      title={Skip-Thought Vectors},
      author={Kiros, Ryan and Zhu, Yukun and Salakhutdinov, Ruslan and Zemel, Richard S and Torralba, Antonio and Urtasun, Raquel and Fidler, Sanja},
      journal={arXiv preprint arXiv:1506.06726},
      year={2015}
    }

