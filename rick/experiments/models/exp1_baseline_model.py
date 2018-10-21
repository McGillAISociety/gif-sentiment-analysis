import torchvision
import torch.nn as nn

from torchsummary import summary


def get_model(num_classes=1):
    model = torchvision.models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, num_classes)
    )

    return model


if __name__ == '__main__':
    _model = get_model()
    summary(_model, input_size=(3, 224, 224), device='cpu')


