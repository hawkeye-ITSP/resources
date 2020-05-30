# Resources

* [x] [OpenCV python tutorial - GeekforGeeks](https://www.geeksforgeeks.org/opencv-python-tutorial/)

* [x] [CRNN (CNN+RNN)](https://github.com/qjadud1994/CRNN-Keras)

* [x] [A gentle introduction to OCR - Towards Data Science](https://towardsdatascience.com/a-gentle-introduction-to-ocr-ee1469a201aa)

* [x] [OCR wit Python, OpenCV and pyTesseract - Medium](https://medium.com/@jaafarbenabderrazak.info/ocr-with-tesseract-opencv-and-python-d2c4ec097866)

* [x] [Neural Network Playground](https://playground.tensorflow.org/) - Great website for visualising Neural Networks

* [x] [Loss functions in PyTorch](https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7)

* [ ] [Image to Latex - Luopeixiang - Github](https://github.com/luopeixiang/im2latex)

* [ ] [im2markup - Harvard NLP - Github](https://github.com/harvardnlp/im2markup)

* [ ] [Permute vs Transpose vs View in PyTorch](https://discuss.pytorch.org/t/different-between-permute-transpose-view-which-should-i-use/32916)

### Dropout

<details><summary>Dropout is a . . .</summary>

Dropout is a regularization technique that “drops out” or “deactivates” few neurons in the neural network randomly in order to avoid the problem of over-fitting.

Dropout deactivates the neurons randomly at each training step instead of training the data on the original network, we train the data on the network with dropped out nodes. In the next iteration of the training step, the hidden neurons which are deactivated by dropout changes because of its probabilistic behavior. In this way, by applying dropout i.e deactivating certain individual nodes at random during training we can simulate an ensemble of neural network with different architectures.

*Dropout roughly doubles the number of iterations required to converge. However, training time for each epoch is less.*

```python
#create a neural network with out dropout
N_h = 100 #hidden nodes

model = torch.nn.Sequential(
    nn.Linear(1, N_h),
    nn.ReLU(),
    nn.Linear(N_h, N_h),
    nn.ReLU(),
    nn.Linear(N_h, 1)
)

#create a network with dropout
model_dropout = nn.Sequential(
    # Input Layer
    nn.Linear(1, N_h),
    nn.Dropout(0.5), # 50 % probability of dropping nodes in the 1st layer 
    nn.ReLU(),

    # Hidden Layer
    torch.nn.Linear(N_h, N_h),
    torch.nn.Dropout(0.2), # 20% probability of dropping nodes in the 2nd layer
    torch.nn.ReLU(),
    
    torch.nn.Linear(N_h, 1),
)
```
</details>


* [x] [Dropout in Deep Learning - Medium](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5) : Check [this](https://github.com/budhiraja/DeepLearningExperiments/blob/master/Dropout%20Analysis%20for%20Deep%20Nets/Dropout%2BAnalysis.ipynb) for implementation.

* [x] [Batch Normalization and Dropout in Neural Networks with Pytorch](https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd) Dropout section of the article.

## PyTorch 

* [x] [Object detection/segmentation using PyTorch](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/) - Detectron2 library (Facebook AI)

* [x] [PyTorch Tutorials](https://pytorch.org/tutorials/) - Official PyTorch Tutorial

* [x] [PyTorch for Deep Learning and Computer Vision - Udemy](https://www.udemy.com/course/pytorch-for-deep-learning-and-computer-vision/?LSNPUBID=QhjctqYUCD0&ranEAID=QhjctqYUCD0&ranMID=39197&ranSiteID=QhjctqYUCD0-1hYZOGjDH3dISFFHX6uK7g)

* [x] [Linear Regression with PyTorch - Medium](https://medium.com/learn-the-part/linear-regression-with-pytorch-ac8f163a14f) - Code: [Linear Regression](https://github.com/hawkeye-ITSP/resources/blob/master/Implementation_PyTorch/Linear_Regression.ipynb)

## References

* [x] [Image to LaTeX - report, CS229 Stanford](http://cs229.stanford.edu/proj2017/final-reports/5243453.pdf)

* [x] [Image to LaTeX - poster, CS229 Stanford](http://cs229.stanford.edu/proj2017/final-posters/5140564.pdf)
