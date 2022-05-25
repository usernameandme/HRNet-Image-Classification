import argparse
import _init_paths
from config import config
from config import update_config
import models
import numpy as np
import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from utils.modelsummary import get_model_summary
import matplotlib.pyplot as plt
from torch.utils.mobile_optimizer import optimize_for_mobile


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')

    parser.add_argument('--testModel',
                        help='testModel',
                        type=str,
                        default='')

    args = parser.parse_args()
    update_config(config, args)

    return args

args = parse_args()
model = eval('models.'+config.MODEL.NAME+'.get_cls_net')(
        config)
dump_input = torch.rand(
        (1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0])
    )
print(get_model_summary(model, dump_input))
print('=> loading model from {}'.format(config.TEST.MODEL_FILE))
model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
model.eval()

valdir = os.path.join(config.DATA_DIR)
print(valdir)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            #transforms.Resize(int(config.MODEL.IMAGE_SIZE[0] / 0.875)),
            #transforms.Resize((224,224)),
            #transforms.CenterCrop(config.MODEL.IMAGE_SIZE[0]),
            transforms.ToTensor(),
            normalize,
        ])), batch_size=1)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()

print(int(config.MODEL.IMAGE_SIZE[0] / 0.875))

for i, (input, cls) in enumerate(valid_loader):
    print(i, input.shape, cls)
    #out = torchvision.utils.make_grid(inputs)
    #imshow(out, title=[x.item() for x in classes])
    scores = model(input).detach().numpy()
    pred_score = np.max(scores, axis=1)[0]
    pred_label = np.argmax(scores, axis=1)[0]
    print(pred_score, pred_label)
    from imagenet import CLASSES
    print(CLASSES[pred_label])
    if i == 0:
        #script_module = torch.jit.script(model)
        traced_model = torch.jit.trace(model, input, strict=False)
        print(input.shape)
        #scores = script_module(input)
        scores = traced_model(input).detach().numpy()
        pred_score = np.max(scores, axis=1)[0]
        pred_label = np.argmax(scores, axis=1)[0]
        print('traced model:', pred_score, pred_label)
        from imagenet import CLASSES
        print('traced model:', CLASSES[pred_label])
        optimized_model = optimize_for_mobile(traced_model)
        optimized_model._save_for_lite_interpreter("model.ptl")

