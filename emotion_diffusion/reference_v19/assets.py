"""Obtain explicit pretrained assets; record remote segmentation revision."""
import argparse
import json
from pathlib import Path
import torch


def main():
    from torchvision.models import ResNet18_Weights, ResNet34_Weights
    from huggingface_hub import snapshot_download, model_info
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()
    path = Path(args.config)
    config = json.loads(path.read_text())
    for key, weights in (('appearance_a_weights', ResNet18_Weights.IMAGENET1K_V1),
                         ('appearance_b_weights', ResNet34_Weights.IMAGENET1K_V1)):
        destination = Path(config[key])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            state = torch.hub.load_state_dict_from_url(weights.url, map_location='cpu', check_hash=True)
            torch.save(state, destination)
        print('READY', key, flush=True)
    if config.get('segmentation_model') and Path(config['segmentation_model'], 'model.safetensors').exists():
        print('READY local pinned segmentation', config['segmentation_revision'], flush=True)
        return
    repo = config.get('segmentation_repo', 'ZhengPeng7/BiRefNet_lite')
    info = model_info(repo)
    directory = snapshot_download(repo, revision=info.sha,
        allow_patterns=['*.json', '*.py', '*.safetensors', '*.bin'])
    config['segmentation_model'] = directory
    config['segmentation_revision'] = info.sha
    path.write_text(json.dumps(config, indent=2))
    print('READY segmentation', repo, info.sha, flush=True)


if __name__ == '__main__':
    main()
