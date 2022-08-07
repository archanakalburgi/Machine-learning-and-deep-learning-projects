### Answer 4 

**Some of the potential problem for translating very long sentence are** 
- The gradient get really small as the back-propogation algorithm moves through the network, which causes the earlier layer to learn slower than the later ones. This causes the effectiveness of RNNs as they are not able to take into account long sequences. 
- As the  gap between information that is needed becomes larger, an RNN gets less effective this is the common problem associated with translating longer sentences with RNN 
- A very common solution to this problem is to use _activation functions_ that don't cause vanishing gradient problem. For example Relu. 
- Using attention based RNN models we can focus on different parts of the input for each output item, so in general we can say that attention based RNN models are better suited for translating long sentences compare to the no-attention based RNNs.
- Using _LSTM_** and _GRU_ can solve the problem of vanishing gradients.

**Attention and NoAttention:**
- Attention over comes the limitation of encoder-decoder model encoding where the input sequence is modeled to one fixed length vector from which to decode each output time step, which is more of a problem when decoding long sequence
- In this case the validation loss of NoAttention model is : Val loss: 1.818 and validation loss of Attention model is Val loss: 0.313 
- Instead of encoding the input into a single fixed context vector, the attention model developes a context vector that is filtered for each output time step.
- The above can be verified in the following two log files:
    1. with_atten.log 
    2. without_atten.log 

