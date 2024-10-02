import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, backbone):
        super(Net, self).__init__()

        self.backbone = backbone

    def forward(self, image):
        output = self.backbone(image)

        return output

model = torch.load('best_model_fold_0.pt')
torch.save(model.state_dict(), 'best_model_fold_0.pt')