# Word Embedding Annealing Using Sequence-to-sequence Neural Networks 

We propose a new technique to improve text classification accuracy by annealing the word embedding layer using Seq2seq Neural Networks. 
Our model Sequence-to-convolution Neural Networks(Seq2CNN) consists of two blocks: Sequential Block that summarizes input texts and Convolution Block that classifies the original text to a label. 

We also present Gradual Weight Shift(GWS) method that stabilize training. GWS is applied to our model's loss function. We compared our model with word-based TextCNN trained with different data preprocessing methods. 

We obtained some improvement in classification accuracy over word-based TextCNN without any ensemble or data augmentation.

Here's the overview of Seq2CNN model.

![alt text](https://github.com/tgisaturday/Seq2CNN/blob/master/seq2CNN.png)
