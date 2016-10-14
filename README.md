##Documentation

Simple code to demonstrate Direct Feedback Alignment training of neural networks. Network with 2 hidden layers is trained to perform the xor computation. Same weight initializations trained using both DFA and Backpropagation for comparison.

DFA algorithm is implemented as described in this [paper](https://arxiv.org/abs/1609.01596).

dfa.m will run and plot all results

##Figures

**Sample training**

![Alt text](/figs/DirectFeedbackAlignement_xor.png?raw=true "Sample Training")

**Average Convergence**

![Alt text](/figs/DirectFeedbackAlignement_xor_1000.png?raw=true "Average Convergence")

##References

Nøkland, Arild. "Direct Feedback Alignment Provides Learning in Deep Neural Networks." arXiv preprint arXiv:1609.01596 (2016).
