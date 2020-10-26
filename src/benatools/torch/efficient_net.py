
# from https://github.com/lukemelas/EfficientNet-PyTorch
# from https://github.com/rwightman/gen-efficientnet-pytorch

from efficientnet_pytorch import EfficientNet
from torch import nn
import geffnet.gen_efficientnet as efn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def create_efn2(b, weights='imagenet', include_top=False, n_classes=1000):
    """ Create EfficientNet model using https://github.com/lukemelas/EfficientNet-PyTorch

    Parameters
    ----------
    b : int
        EfficientNet version
    weights : str or None
        path indicating weights from local or 'imagenet' if weigths from the internet. None for no transfer learning
    include_top : bool
        Whether to include the last layer or not
    n_classes : int
        Number of classes in the last layer

    Returns
    -------
    torch.nn.Module
        An EfficientNet model

    """
    if weights == 'imagenet' and include_top and n_classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                        ' as true, `classes` should be 1000')

    version = 'efficientnet-b'+str(b)
    if weights is None:
        model = EfficientNet.from_name(version, num_classes=n_classes)
    else:
        model = EfficientNet.from_pretrained(version, num_classes=n_classes)

    if include_top == False:  # If not including top, it invalidates the n_classes parameter
        model._fc = Identity()

    return model

_EFNS = [efn.tf_efficientnet_b0, efn.tf_efficientnet_b1, efn.tf_efficientnet_b2, efn.tf_efficientnet_b3,
        efn.tf_efficientnet_b4, efn.tf_efficientnet_b5, efn.tf_efficientnet_b6, efn.tf_efficientnet_b7]

def create_efn(b, pretrained=True, include_top=False, n_classes=1000):
    """ Create EfficientNet model using https://github.com/rwightman/gen-efficientnet-pytorch

    Parameters
    ----------
    b : int
        EfficientNet version
    pretrained : bool
        whether to use transfer learning or not
    include_top : bool
        Whether to include the last layer or not
    n_classes : int
        Number of classes in the last layer

    Returns
    -------
    torch.nn.Module
        An EfficientNet model

    """
    model = _EFNS[b](pretrained, **{'n_classes':n_classes})

    if include_top == False:  # If not including top, it invalidates the n_classes parameter
        model.classifier = Identity()

    return model
