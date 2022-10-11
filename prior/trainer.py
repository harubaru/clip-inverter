import torch
import pytorch_warmup
from contextlib import nullcontext
from accelerate import Accelerator
from pathlib import Path
from prior.net import FeatureInverter
from prior.version import __version__
from prior.helpers import *

from packaging import version

def get_optimizer(
    params,
    lr = 1e-4,
    wd = 1e-2,
    betas = (0.9, 0.99),
    eps = 1e-8,
    filter_by_requires_grad = False,
    group_wd_params = True,
    **kwargs
):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    if wd == 0:
        return torch.optim.Adam(params, lr = lr, betas = betas, eps = eps)

    if group_wd_params:
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0},
        ]

    return torch.optim.AdamW(params, lr = lr, weight_decay = wd, betas = betas, eps = eps)

class FeatureInverterTrainer(torch.nn.Module):
    def __init__(
        self,
        prior,
        accelerator = None,
        lr = 3e-4,
        wd = 1e-2,
        eps = 1e-6,
        max_grad_norm = None,
        group_wd_params = True,
        warmup_steps = None,
        cosine_decay_max_steps = None,
        **kwargs
    ) -> None:
        super().__init__()
        assert isinstance(prior, FeatureInverter)

        accelerator_kwargs = groupby_prefix_and_trim('accelerator_', kwargs)
        if accelerator is not None:
            accelerator = Accelerator(**accelerator_kwargs)
        self.accelerator = accelerator

        self.device = self.accelerator.device
        self.prior = prior.to(self.device)

        self.optimizer = get_optimizer(
            self.prior.parameters(),
            **dict(lr=lr, wd=wd, eps=eps, group_wd_params=group_wd_params),
            **kwargs
        )

        if exists(cosine_decay_max_steps):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = cosine_decay_max_steps)
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda = lambda _: 1.0)
        self.warmup_scheduler = pytorch_warmup.LinearWarmup(self.optimizer, warmup_period = warmup_steps) if exists(warmup_steps) else None

        self.max_grad_norm = max_grad_norm

        self.register_buffer('step', torch.tensor([0], device=self.device))
    
    def save(self, path, overwrite = True, **kwargs):
        print(f'Saving checkpoint at step: {self.step.item()}')
        path = Path(path)
        assert not (path.exists() and not overwrite)
        path.parent.mkdir(parents = True, exist_ok = True)

        # FIXME: LambdaLR can't be saved due to pickling issues
        save_obj = dict(
            optimizer = self.optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
            warmup_scheduler = self.warmup_scheduler,
            model = self.accelerator.unwrap_model(self.prior).state_dict(),
            version = version.parse(__version__),
            step = self.step,
            **kwargs
        )

        torch.save(save_obj, str(path))
    
    def load(self, path_or_state, overwrite_lr = True, strict = True):
        # all processes need to load checkpoint. no restriction here
        if isinstance(path_or_state, str):
            path = Path(path_or_state)
            assert path.exists()
            loaded_obj = torch.load(str(path), map_location=self.device)

        elif isinstance(path_or_state, dict):
            loaded_obj = path_or_state

        if version.parse(__version__) != loaded_obj['version']:
            print(f'loading saved diffusion prior at version {loaded_obj["version"]} but current package version is at {__version__}')

        # unwrap the model when loading from checkpoint
        self.accelerator.unwrap_model(self.prior).load_state_dict(loaded_obj['model'], strict = strict)
        self.step.copy_(torch.ones_like(self.step, device=self.device) * loaded_obj['step'].to(self.device))

        self.optimizer.load_state_dict(loaded_obj['optimizer'])
        self.scheduler.load_state_dict(loaded_obj['scheduler'])

        # set warmupstep
        if exists(self.warmup_scheduler):
            self.warmup_scheduler.last_step = self.step.item()

        # ensure new lr is used if different from old one
        if overwrite_lr:
            new_lr = self.optim_kwargs["lr"]

            for group in self.optimizer.param_groups:
                group["lr"] = new_lr if group["lr"] > 0.0 else 0.0

        return loaded_obj

    def update(self):

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.prior.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        self.optimizer.zero_grad()

        # accelerator will ocassionally skip optimizer steps in a "dynamic loss scaling strategy"
        if not self.accelerator.optimizer_step_was_skipped:
            sched_context = self.warmup_scheduler.dampening if exists(self.warmup_scheduler) else nullcontext
            with sched_context():
                self.scheduler.step()

        if self.use_ema:
            self.ema_prior.update()

        self.step += 1

    @cast_torch_tensor
    def forward(
        self,
        *args,
        max_batch_size = None,
        **kwargs
    ):
        total_loss = 0.

        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs):
            with self.accelerator.autocast():
                loss = self.prior(*chunked_args, **chunked_kwargs)
                loss = loss * chunk_size_frac

            total_loss += loss.item()

            if self.training:
                self.accelerator.backward(loss)

        return total_loss