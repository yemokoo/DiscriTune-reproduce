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

import sys
sys.path.append('..')

from discritune_train import (
    MLP,
    ClipCaptionModel,
    RunningMeanBaseline,
    DiscriminativeDataset,
    InBatchDistractorDataLoader
)

# beam search로 caption 생성
def generate_captions(captioner, prefix, tokenizer, beam_size=5, max_length=40, device='cuda'):
    batch_size = prefix.shape[0]
    with torch.no_grad():
        prefix_projections = captioner.clip_project(prefix).view(
            batch_size, captioner.prefix_length, captioner.gpt_embedding_size
        )

    all_captions = []

    for i in range(batch_size):
        prefix_embed = prefix_projections[i:i+1]
        attention_mask = torch.ones(
            1, captioner.prefix_length,
            dtype=torch.long,
            device=device
        )

        with torch.no_grad():
            outputs = captioner.gpt.generate(
                inputs_embeds=prefix_embed,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=beam_size,
                num_return_sequences=1,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        generated_tokens = outputs[0]

        # EOS 토큰 찾기
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id in generated_tokens:
            eos_idx = (generated_tokens == eos_token_id).nonzero(as_tuple=True)[0][0]
            generated_tokens = generated_tokens[:eos_idx]

        caption = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if not caption:
            caption = "a photo."

        if not caption.endswith('.'):
            if '.' in caption:
                caption = caption.split('.')[0] + '.'
            else:
                caption = caption + '.'

        all_captions.append(caption)

    return all_captions

# Evaluator class
class Evaluator:
    def __init__(self, captioner, clip_model, tokenizer,device='cuda', beam_size=5):
        self.captioner = captioner
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.device = device
        self.beam_size = beam_size

    def evaluate(self, eval_loader):
        self.captioner.eval()
        self.clip_model.eval()

        total_accuracy = []

        with torch.no_grad():
            pbar = tqdm(eval_loader, desc="Evaluating")

            for batch_idx, batch in enumerate(pbar):
                images = batch['images'].to(self.device)
                batch_size = images.shape[0]

                image_features = self.clip_model.encode_image(images).float()

                # caption 생성
                captions = generate_captions(
                    self.captioner,
                    image_features,
                    self.tokenizer,
                    beam_size=self.beam_size,
                    max_length=40,
                    device=self.device
                )

                # text feature
                text_tokens = clip.tokenize(captions, truncate=True).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens).float()
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                image_features = image_features / image_features.norm(dim=1, keepdim=True)

                # similarity 계산
                logit_scale = self.clip_model.logit_scale.exp().float()
                similarity = logit_scale * (text_features @ image_features.T)

                predictions = similarity.argmax(dim=1)
                correct_indices = torch.arange(batch_size, device=self.device)
                correct = (predictions == correct_indices).float()
                accuracy = correct.mean().item()

                total_accuracy.append(accuracy)

                avg_accuracy = np.mean(total_accuracy)
                pbar.set_postfix({
                    'batch_acc': f'{accuracy:.4f}',
                    'avg_acc': f'{avg_accuracy:.4f}'
                })

        final_accuracy = np.mean(total_accuracy)
        print(f"\nTotal batches: {len(eval_loader)}")
        print(f"Total images evaluated: {len(eval_loader.dataset)}")
        print(f"Average Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

        return {
            'accuracy': final_accuracy,
        }




def evaluation(args):
    device = torch.device('cuda')
    os.makedirs(args.output_dir, exist_ok=True)

    print("Start Eval")
    clip_model, clip_preprocess = clip.load('ViT-B/32', device=device, jit=False)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    print('load GPT2 and model')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    captioner = ClipCaptionModel(prefix_length=args.prefix_length)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Loading model from {args.checkpoint}...")

    # checkpoint 로드
    if 'model_state_dict' in checkpoint:
        captioner.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        captioner.load_state_dict(checkpoint, strict=False)

    captioner.to(device)
    captioner.eval()

    print("load datset")
    dataset = DiscriminativeDataset(
        image_dir=args.data_dir,
        image_list_file=args.image_list_file,
        clip_preprocess=clip_preprocess,
        max_images=args.max_images
    )

    eval_loader = InBatchDistractorDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print(f"Dataset loaded: {len(dataset)} images")

    evaluator = Evaluator(captioner, clip_model, tokenizer, device=device, beam_size=args.beam_size)
    results = evaluator.evaluate(eval_loader)
    output_file = os.path.join(args.output_dir, args.output_file_name)

    results_to_save = {
        'accuracy': float(results['accuracy']),
        'total_images': len(eval_loader.dataset),
        'checkpoint': args.checkpoint,
        'dataset': args.data_dir,
        'batch_size': args.batch_size
    }

    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\nResults saved to {output_file}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--image_list_file', type=str, required=True)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--save_detailed', action='store_true')
    parser.add_argument('--output_file_name', type=str, default='eval_results.json')

    args = parser.parse_args()

    evaluation(args)
