from efficientnet_pytorch import EfficientNet

def create_efn(b, dim=128, weights='imagenet', include_top=False):
        version = 'efficientnet-b0'+str(b)
        if weights is None:
                model = EfficientNet.from_name(version)
        else:
                model = EfficientNet.from_pretrained(version)

        return model