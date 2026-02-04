import torch 
from rm_utils import RewardModel
from transformers import AutoTokenizer, AutoModelForCausalLM

YOUR_HF_TOKEN = "12345678HA!ha"
model = RewardModel.from_pretrained("Gen-Verse/ReasonFlux-PRM-1.5B", auth_token=YOUR_HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained("Gen-Verse/ReasonFlux-PRM-1.5B", trust_remote_code=True)