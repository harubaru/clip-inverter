from prior.net import FeatureInverter
from prior.trainer_simple import FeatureInverterTrainer
from prior.encoders import FrozenCLIPTextEmbedder, FrozenCLIPVisionEmbedder

prior = FeatureInverter()
trainer = FeatureInverterTrainer(
    prior,
    txt_enc=FrozenCLIPTextEmbedder(),
    img_enc=FrozenCLIPVisionEmbedder(),
    folder='../dataset',
    null_cond='',
    device='cpu',
    batch_size=1,
    grad_accum_every=1,
)

