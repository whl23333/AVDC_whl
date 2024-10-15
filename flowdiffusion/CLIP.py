from transformers import CLIPTextModel, CLIPTokenizer
import torch
pretrained_model = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model)
text_encoder = CLIPTextModel.from_pretrained(pretrained_model)
torch.save(tokenizer, "../openai_clip/tokenizer.pth")
torch.save(text_encoder, "../openai_clip/text_encoder.pth")