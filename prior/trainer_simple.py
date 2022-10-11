import glob
import torch
import torchvision
import pytorch_warmup
import PIL
import io
import random
import re
import json
from contextlib import nullcontext
from pathlib import Path
from shutil import rmtree
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


class CaptionProcessor(object):
    def __init__(self, copyright_rate, character_rate, general_rate, artist_rate, normalize, caption_shuffle, transforms, max_size, resize, random_order):
        self.copyright_rate = copyright_rate
        self.character_rate = character_rate
        self.general_rate = general_rate
        self.artist_rate = artist_rate
        self.normalize = normalize
        self.caption_shuffle = caption_shuffle
        self.transforms = transforms
        self.max_size = max_size
        self.resize = resize
        self.random_order = random_order

    def clean(self, text: str):
        text = ' '.join(set([i.lstrip('_').rstrip('_') for i in re.sub(r'\([^)]*\)', '', text).split(' ')])).lstrip().rstrip()
        if self.caption_shuffle:
            text = text.split(' ')
            random.shuffle(text)
            text = ' '.join(text)
        if self.normalize:
            text = ', '.join([i.replace('_', ' ') for i in text.split(' ')]).lstrip(', ').rstrip(', ')
        return text

    def get_key(self, val_dict, key, clean_val = True, cond_drop = 0.0, prepend_space = False, append_comma = False):
        space = ' ' if prepend_space else ''
        comma = ',' if append_comma else ''
        if random.random() < cond_drop:
            if (key in val_dict) and val_dict[key]:
                if clean_val:
                    return space + self.clean(val_dict[key]) + comma
                else:
                    return space + val_dict[key] + comma
        return ''

    def __call__(self, sample):
        # preprocess caption
        caption_data = json.loads(sample['caption'])
        if not self.random_order:
            character = self.get_key(caption_data, 'tag_string_character', True, self.character_rate, False, True)
            copyright = self.get_key(caption_data, 'tag_string_copyright', True, self.copyright_rate, True, True)
            artist = self.get_key(caption_data, 'tag_string_artist', True, self.artist_rate, True, True)
            general = self.get_key(caption_data, 'tag_string_general', True, self.general_rate, True, False)
            tag_str = f'{character}{copyright}{artist}{general}'.lstrip().rstrip(',')
        else:
            character = self.get_key(caption_data, 'tag_string_character', False, self.character_rate, False)
            copyright = self.get_key(caption_data, 'tag_string_copyright', False, self.copyright_rate, True, False)
            artist = self.get_key(caption_data, 'tag_string_artist', False, self.artist_rate, True, False)
            general = self.get_key(caption_data, 'tag_string_general', False, self.general_rate, True, False)
            tag_str = self.clean(f'{character}{copyright}{artist}{general}').lstrip().rstrip(' ')
        sample['caption'] = tag_str

        # preprocess image
        image = sample['image']
        image = PIL.Image.open(io.BytesIO(image)).convert('RGB')
        sample['image'] = self.transforms(image)
        return sample

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        txt_enc,
        img_enc,
        ext = ['image', 'caption']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.txt_enc = txt_enc
        self.img_enc = img_enc

        print('Fetching data.')

        self.image_files = []
        [self.image_files.extend(glob.glob(f'{folder}' + '/*.' + e)) for e in ext]

        print(f'Constructing image-caption map. Found {len(self.image_files)} images')

        self.examples = {}
        self.hashes = []
        for i in self.image_files:
            hash = i[len(f'{folder}/'):].split('.')[0]
            self.examples[hash] = {
                'image': i,
                'text': f'{folder}/{hash}.caption'
            }
            self.hashes.append(hash)

        print(f'image-caption map has {len(self.examples.keys())} examples')
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.CenterCrop(image_size),
            torchvision.transforms.Lambda(lambda img: img_enc.encode(img))
        ])

        self.captionprocessor = CaptionProcessor(1.0, 1.0, 1.0, 1.0, True, True, self.transform, 768, False, True)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.get_image(index)

    def get_image(self, i):
        image = {}
        try:
            image_file = self.examples[self.hashes[i]]['image']
            with open(image_file.replace('.caption', '.image'), 'rb') as f:
                image['image'] = f.read()
            text_file = self.examples[self.hashes[i]]['text']
            with open(text_file, 'rb') as f:
                image['caption'] = f.read()
            image = self.captionprocessor(image)
        except Exception as e:
            print(f'Error with {self.examples[self.hashes[i]]["image"]} -- {e} -- skipping {i}')
            return self.skip_sample(i)

        return image

class FeatureInverterTrainer(torch.nn.Module):
    def __init__(
        self,
        prior,
        txt_enc,
        img_enc,
        folder,
        null_cond='',
        device = None,
        batch_size = 4,
        grad_accum_every = 1,
        num_train_steps = 100000,
        lr = 3e-4,
        wd = 1e-2,
        eps = 1e-6,
        group_wd_params = True,
        warmup_steps = None,
        cosine_decay_max_steps = None,
        valid_frac=0.05,
        random_split_seed = 42,
        save_model_every = 1000,
        track_every = 1,
        results_folder = 'results',
        amp = False,
        **kwargs
    ) -> None:
        super().__init__()
        assert isinstance(prior, FeatureInverter)

        self.device = device
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

        self.amp = amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)

        self.step = torch.tensor([0], device=self.device)

        self.ds = ImageDataset(folder, image_size=224, txt_enc=txt_enc, img_enc=img_enc)

        if valid_frac > 0:
            train_size = int((1 - valid_frac) * len(self.ds))
            valid_size = len(self.ds) - train_size
            self.ds, self.valid_ds = torch.utils.data.random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
            print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
        else:
            self.valid_ds = self.ds
            print(f'training with shared training and valid dataset of {len(self.ds)} samples')

        # dataloader

        self.dl = cycle(torch.utils.data.DataLoader(
            self.ds,
            batch_size = batch_size,
            shuffle = True
        ))

        self.valid_dl = cycle(torch.utils.data.DataLoader(
            self.valid_ds,
            batch_size = batch_size,
            shuffle = True
        ))

        self.save_model_every = save_model_every
        self.track_every = track_every
        self.grad_accum_every = grad_accum_every
        self.results_folder = Path(results_folder)
        self.txt_enc = txt_enc.to(self.device)
        self.img_enc = img_enc.to(self.device)
        self.num_train_steps = num_train_steps

        self.null_cond = self.txt_enc.encode([null_cond]).to(self.device)

        if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
            rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents = True, exist_ok = True)

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
            model = self.prior.state_dict(),
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
        self.prior.load_state_dict(loaded_obj['model'], strict = strict)
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

    def train_step(self):
        logs = {'step': self.step.item()}

        for _ in range(self.grad_accum_every):
            batch = next(self.dl)
            img = batch['image'].to(self.device)
            emb = self.txt_enc.encode(batch['caption']).to(self.device)
            with torch.cuda.amp.autocast(enabled = self.amp):
                loss = self.prior(
                    img[0], emb[0], self.null_cond, True
                )

                self.scaler.scale(loss / self.grad_accum_every).backward()

            accum_log(logs, {'loss': loss.item() / self.grad_accum_every})

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        sched_context = self.warmup_scheduler.dampening if exists(self.warmup_scheduler) else nullcontext
        with sched_context():
            self.scheduler.step()

        if not (self.step % self.save_model_every):
            self.save(str(self.results_folder) + '/model.ckpt')

        self.step += 1
        return logs

    def train(self, log_fn = None):
        while self.step.item() < self.num_train_steps:
            logs = self.train_step()
            if log_fn == None:
                print(f'{logs["step"]}: loss: {logs["loss"]}')

        print('training complete')