from rocknet_1sta_model import fusion_unet, wf_unet
from rocknet_associate_model import associate_net

class RockNet:
    def __init__(self):
        pass      
    def fusion_unet(self, weights=None):
        return fusion_unet(
            pretrained_weights=weights
        )
    def associate_net(self, weights=None):
        return associate_net(
            single_sta_model_h5=None, 
            station_num=4, 
            pretrained_weights=weights 
        )