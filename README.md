# Sequence-to-convolution Neural Networks

arxiv link : https://arxiv.org/abs/1805.07745

We propose a new deep neural network model and its training scheme for text
classification. Our model Sequence-to-convolution Neural Networks(Seq2CNN)
consists of two blocks: Sequential Block that summarizes input texts and Convolution
Block that receives summary of input and classifies it to certain label.

Seq2CNN is trained end-to-end to classify various-length texts without preprocessing
inputs into fixed length. We also present Gradual Weight Shift(GWS)
method that stabilize training. GWS is applied to our model’s loss function.

We compared our model with word-based TextCNN trained with different data preprocessing
methods. We obtained significant improvement of in classification accuracy
over word-based TextCNN without any ensemble or data augmentation.

Here's the overview of Seq2CNN model.

![alt text](https://github.com/tgisaturday/Seq2CNN/blob/master/seq2CNN.png)
