#!/usr/bin/env python3
from transformers import PaliGemmaConfig, PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
base_model_id = "google/paligemma2-3b-pt-224"
checkpoint = "mateoguaman/paligemma2_racer_tartandrive"
processor = PaliGemmaProcessor.from_pretrained(base_model_id)
model = PaliGemmaForConditionalGeneration.from_pretrained(
    checkpoint,
    ignore_mismatched_sizes=True
).to(device)

print("MODEL LOADED")

dummy_image = torch.random([])