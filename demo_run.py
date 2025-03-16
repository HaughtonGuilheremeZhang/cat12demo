from plamform.ui import ui_na
from resnet_model.resnet_train import resnet18
# checkpoint and model structure
ui_inst = ui_na('./resnet_model/resnet_checkpoint.pth', resnet18)
ui_inst.ui_run()

