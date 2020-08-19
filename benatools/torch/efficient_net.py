
# from https://github.com/lukemelas/EfficientNet-PyTorch

from efficientnet_pytorch import EfficientNet
from torch import nn


class Identity(nn.Module):
        def __init__(self):
                super(Identity, self).__init__()

        def forward(self, x):
                return x


def create_efn(b, weights='imagenet', include_top=False, n_classes=1000):
        if weights == 'imagenet' and include_top and n_classes != 1000:
                raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                                 ' as true, `classes` should be 1000')

        version = 'efficientnet-b'+str(b)
        if weights is None:
                model = EfficientNet.from_name(version, num_classes=n_classes)
        else:
                model = EfficientNet.from_pretrained(version, num_classes=n_classes)

        if include_top == 'False':  # If not including top, it invalidates the n_classes parameter
                model._fc = Identity()

        return model
