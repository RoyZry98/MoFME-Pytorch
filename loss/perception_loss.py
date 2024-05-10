# --- Imports --- #
import torch
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms.functional as TF

# --- Perceptual Loss Network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred, gt):
        loss = []
        pred_features = self.output_features(pred)
        gt_features = self.output_features(gt)
        for pred_feature, gt_feature in zip(pred_features, gt_features):
            loss.append(F.mse_loss(pred_feature, gt_feature))

        return sum(loss)/len(loss)
    
class Perception_Loss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # --- Define the perceptual loss network --- #
        vgg_model = vgg16(weights=VGG16_Weights.DEFAULT).features[:16]
        vgg_model = vgg_model.to('cuda')
        for param in vgg_model.parameters():
            param.requires_grad = False
        
        self.loss_network = LossNetwork(vgg_model)

        self.loss_network.eval()

    def forward(self, pred, gt):

        return self.loss_network(pred, gt)