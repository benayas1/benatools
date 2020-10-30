import torch

class CategoricalCrossEntropy(torch.nn.Module):
    """
    Categorical Cross Entropy in Keras style.
    Equivalent to tf.keras.losses.CategoricalCrossEntropy.

    Adapted to receive one-hot tensors as the true label.

    In order to get the cross entropy loss, values must follow the pipeline:
    logits --> probs --> log --> loss

    Values can come in two versions from the model: logits (raw values) or
    probs (all together adding 1).

    This class includes label_smoothing feature

    Parameters
    ----------
    Args:
        nn ([type]): [description]
    """
    def __init__(self, from_logits=False, label_smoothing=0, reduction='sum'):
        super(CategoricalCrossEntropy, self).__init__()
        self.from_logits=from_logits
        self.label_smoothing=label_smoothing
        self.reduction=reduction

    def forward(self, y_pred, y_true):
        """Forward

        Args:
            y_pred ([type]): [description]
            y_true ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Label smoothing
        if self.label_smoothing > 0:
            y_true = y_true.clone()
            y_true *= (1-self.label_smoothing)
            y_true += (self.label_smoothing / y_true.shape[1])

        # Check if it comes from logits (no softmax applied) or the softmax needs to be applied
        if self.from_logits:
            y_pred = torch.nn.LogSoftmax(dim=1)(y_pred)
        else:
            y_pred = torch.log(y_pred)
            
        # Reduction method
        if self.reduction=='mean':
            return torch.mean(torch.sum(-y_true * y_pred, dim=1))
        else:
            return torch.sum(torch.sum(-y_true * y_pred, dim=1))
