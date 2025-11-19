import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
from pathlib import Path

# MLP projection network
# CLIP feature를 GPT2가 이해할 수 있는 embedding space로 변환
class MLP(nn.Module):
    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[i], sizes[i+1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ClipCap model - image에서 caption 생성
class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length=10, prefix_size=512, num_layers=8):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.prefix_size = prefix_size

        # GPT2 pretrained model
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        # projection layer
        self.clip_project = MLP((
            self.prefix_size,
            (self.gpt_embedding_size * self.prefix_length) // 2,
            self.gpt_embedding_size * self.prefix_length
        ))

    def forward(self, tokens, prefix, mask=None, labels=None):
        pass

    def generate_with_log_probs(self, prefix, tokenizer, max_length=40, temperature=1.0, device='cuda'):
        # caption 생성 + log probability 계산 (REINFORCE에 필요)
        batch_size = prefix.shape[0]
        # projection
        prefix_projections = self.clip_project(prefix).view(batch_size, self.prefix_length, self.gpt_embedding_size)
        all_generated_tokens = []
        all_log_probs = []
        all_captions = []
        stop_token_id = tokenizer.encode('.')[0]

        for i in range(batch_size):
            generated_tokens = []
            log_probs = []
            past_key_values = None
            inputs_embeds = prefix_projections[i:i+1]

            for step in range(max_length):
                outputs = self.gpt(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                logits = outputs.logits[:, -1, :] / temperature
                past_key_values = outputs.past_key_values
                # greedy decoding
                probs = F.softmax(logits, dim=-1)
                token_id = torch.argmax(probs, dim=-1)
                log_prob = torch.log(probs[0, token_id] + 1e-10)
                generated_tokens.append(token_id.item())
                log_probs.append(log_prob)
                if token_id.item() == stop_token_id or token_id.item() == tokenizer.eos_token_id:
                    break
                inputs_embeds = self.gpt.transformer.wte(token_id.unsqueeze(0))
            caption = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            all_generated_tokens.append(generated_tokens)
            all_log_probs.append(torch.stack(log_probs).sum())
            all_captions.append(caption)
        return all_captions, torch.stack(all_log_probs)


# discriminative reward 계산 함수
def compute_discriminative_reward_batch(captions, images, clip_model, device='cuda'):
    batch_size = len(captions)

    with torch.no_grad():
        # text feature 추출
        text_tokens = clip.tokenize(captions, truncate=True).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # image feature 추출
        image_features = clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # similarity matrix (cosine similarity)
        similarity = text_features @ image_features.T

        # 대각선 원소가 정답 (i번째 caption <-> i번째 image)
        targets = similarity.diag().unsqueeze(1)

        similarity_masked = similarity.clone()
        similarity_masked.fill_diagonal_(float('-inf'))

        # 정답을 첫 번째 열로
        scores = torch.cat([targets, similarity_masked], dim=1)

        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        # NLL loss as reward (paper Appendix)
        log_probs = F.log_softmax(scores, dim=-1)
        rewards = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

    return rewards

# baseline for REINFORCE -> like moving average filter 
class RunningMeanBaseline:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.mean = 0.0
        self.initialized = False

    def get(self):
        return self.mean

    def update(self, reward):
        if not self.initialized:
            self.mean = reward
            self.initialized = True
        else:
            # exponential moving average
            self.mean = self.mean * self.momentum + reward * (1-self.momentum)

# Dataset - caption 제외 이미지만 로드 (caption은 모델이 생성)
class DiscriminativeDataset(Dataset):
    def __init__(self, image_dir, image_list_file, clip_preprocess, max_images=None):
        self.image_dir = Path(image_dir)
        self.clip_preprocess = clip_preprocess

        with open(image_list_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        print(f"Loaded {len(self.image_paths)} images from {image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_dir / self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        image = self.clip_preprocess(image)
        return{
            'image': image,
            'image_path': str(image_path)
        }

class InBatchDistractorDataLoader:
    def __init__(self, dataset, batch_size=100, shuffle=True, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

    def __iter__(self):
        for batch in self.dataloader:
            yield {
                'images': batch['image']
            }

    def __len__(self):
        return len(self.dataloader)

# DiscriTune Trainer
class DiscriTuneTrainer:
    def __init__(self, captioner, clip_model, clip_preprocess, tokenizer, optimizer, device='cuda', baseline_momentum=0.9):
        self.captioner = captioner
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.baseline = RunningMeanBaseline(momentum = baseline_momentum)

        # CLIP 학습 안 함
        for param in self.clip_model.parameters():
            param.requires_grad = False

        print("DiscriTune Trainer initialized")
        print(f"Device: {device}")
        print(f"Baseline momentum: {baseline_momentum}")

    def train_epoch(self, train_loader, epoch):
        self.captioner.train()

        rewards_list = []
        loss_list = []
        acc_list = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            batch_size = images.shape[0]

            # CLIP으로 image feature 추출
            with torch.no_grad():
                clip_features = self.clip_model.encode_image(images)
                clip_features = clip_features.float()

            # caption 생성 + log prob
            captions, log_probs = self.captioner.generate_with_log_probs(
                clip_features,
                self.tokenizer,
                max_length=40,
                device=self.device
            )

            # reward 계산
            rewards = compute_discriminative_reward_batch(
                captions,
                images,
                self.clip_model,
                device=self.device
            )

            # 이동평균 baseline
            baseline_value = self.baseline.get()
            advantages = rewards - baseline_value
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            

            # policy gradient loss
            loss = -(advantages.detach() * log_probs).mean()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.captioner.parameters(), max_norm=1.0)
            self.optimizer.step()

            # baseline update
            self.baseline.update(rewards.mean().item())

            rewards_list.extend(rewards.detach().cpu().tolist())
            loss_list.append(loss.item())


            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'reward': f'{rewards.mean().item():.4f}',
                    'baseline': f'{baseline_value:.4f}',
                    'loss': f'{loss.item():.4f}'
                })

        avg_reward = np.mean(rewards_list)
        avg_loss = np.mean(loss_list)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Baseline: {self.baseline.get():.4f}")

        return {
            'avg_reward': avg_reward,
            'avg_loss': avg_loss,
            'baseline': self.baseline.get()
        }

    def save_checkpoint(self, save_path, epoch, stats):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.captioner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'baseline_mean': self.baseline.mean,
            'stats': stats
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

def train_discritune(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # CLIP 모델 로드
    print('Loading CLIP model')
    clip_model, clip_preprocess = clip.load('ViT-B/32', device=device, jit=False)
    clip_model.eval()

    # GPT2 tokenizer
    print('Load GPT2 tokenizer')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # ClipCap 모델 로드
    print(f"loading clipcap model from {args.clipcap_checkpoint}")
    captioner = ClipCaptionModel(prefix_length=args.prefix_length)
    checkpoint = torch.load(args.clipcap_checkpoint, map_location=device)
    captioner.load_state_dict(checkpoint, strict=False)
    captioner = captioner.to(device)

    print('clipcap model loaded successfully')

    # map이랑 gpt2 파라미터 학습
    optimizer = torch.optim.Adam(
        [
            {'params': captioner.clip_project.parameters()},
            {'params': captioner.gpt.parameters()}
        ],
        lr=args.learning_rate
    )

    # dataset 준비
    print(f"Loading dataset from {args.data_dir}")
    dataset = DiscriminativeDataset(
        image_dir = args.data_dir,
        image_list_file=args.image_list_file,
        clip_preprocess=clip_preprocess,
        max_images=args.max_images
    )

    train_loader = InBatchDistractorDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    print(f"Training with {len(dataset)} images")
    print(f"Batches per epoch: {len(train_loader)}")

    # trainer 초기화
    trainer = DiscriTuneTrainer(
        captioner=captioner,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        tokenizer=tokenizer,
        optimizer=optimizer,
        device=device,
        baseline_momentum=args.baseline_momentum
    )

    print(f"\nStarting training for {args.num_epochs} epochs...")

    # training loop
    for epoch in range(1, args.num_epochs+1):
        stats = trainer.train_epoch(train_loader,epoch)

        if epoch % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.output_dir,
                f'discritune_{args.dataset_name}_epoch_{epoch}.pt'
            )
            trainer.save_checkpoint(checkpoint_path, epoch, stats)

    # 최종 checkpoint
    final_checkpoint_path = os.path.join(
        args.output_dir,
        f'discritune_{args.dataset_name}_final.pt'
    )
    trainer.save_checkpoint(final_checkpoint_path, args.num_epochs, stats)

    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--clipcap_checkpoint', type=str, required=True)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--image_list_file', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-7)
    parser.add_argument('--baseline_momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_every', type=int, default=5)

    args = parser.parse_args()

    print("Start DiscriTune Training")

    train_discritune(args)
