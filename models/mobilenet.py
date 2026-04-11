import torch.nn as nn
import torchvision.models as models


def get_model(num_classes):
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    except AttributeError:
        model = models.mobilenet_v2(pretrained=True)

    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
