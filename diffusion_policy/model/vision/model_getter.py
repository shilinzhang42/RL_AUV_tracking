import torch
import torchvision

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)
    print(f"DEBUG: get_resnet is called for {name}!") # 加上这一行
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    # 1. 关键修改：固定输出为 4x4 的网格
    # 这样图像特征维度永远是 512 * 4 * 4 = 8192
    resnet.avgpool = torch.nn.AdaptiveAvgPool2d((2, 2)) 

    # 2. 移除原有的全连接层
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model
