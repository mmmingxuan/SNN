from .sew_resnet import multi_step_resnet18 as sew_resnet18
from .sew_resnet import multi_step_resnet19 as sew_resnet19
from .resnet import multi_step_resnet18 as resnet18
from .resnet import multi_step_resnet19 as resnet19
model_dict={
    'sew_resnet18':sew_resnet18,
    'sew_resnet19':sew_resnet19,
    'resnet18':resnet18,
    'resnet19':resnet19,
}

def init_model(model):
    return model_dict[model]