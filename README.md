# Comparison of the RNN-based Generative Models:

In this project, three recurrent network architectures, namely Clockwork RNN, Simple RNN, and the Long-Short Term Memory networks 
are compared for the sequence generation task. All the codes are written from scratch on MATLAB.

In the comparison:
- All the networks include approximately 8000 parameters.
- Initial values for all the weights were drawn from a Gaussian distribution with zero mean and standard deviation of 0.1.
- All networks were trained using Stochastic Gradient Descent (SGD) with Nesterov-style momentum
- For the target value, a piece of music, taken from [here](https://www.youtube.com/watch?v=KLL3DKZAzig), is sampled at 44.1 Hz for 7 ms.

# Results
For the demo, click [here](https://youtu.be/R44aZCndydg).

# References
- J. Koutnnik, K. Greff, F. J. Gomez, and J. Schmidhuber, “A clockwork RNN,” CoRR, vol. abs/1402.3511, 2014.
- Sepp Hochreiter; Jürgen Schmidhuber (1997). "Long short-term memory". Neural Computation. 9 (8): 1735–1780. 
- I. Sutskever, J. Martens, G. Dahl, and G. Hinton, “On the importance of initialization and momentum in deep learning,” in Proceedings of the 30th International Conference on International Conference on Machine Learning - Volume 28, ser. ICML’13. JMLR.org, 2013, pp. III–1139–III–1147. 
