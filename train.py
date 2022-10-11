from prior.net import FeatureInverter
from prior.trainer_simple import FeatureInverterTrainer
from prior.encoders import FrozenCLIPTextEmbedder, FrozenCLIPVisionEmbedder

prior = FeatureInverter()
trainer = FeatureInverterTrainer(
    prior,
    txt_enc=FrozenCLIPTextEmbedder(device='cpu'),
    img_enc=FrozenCLIPVisionEmbedder(device='cpu'),
    folder='../dataset',
    null_cond='',
    device='cpu',
    batch_size=1,
    grad_accum_every=1,
)

