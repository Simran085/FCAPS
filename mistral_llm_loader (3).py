# /RAG/mistral_llm_loader_fallback.py
import os
import gc
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.llms import HuggingFacePipeline
from huggingface_hub import login
import torch

# Force CUDA to release memory
torch.cuda.empty_cache()
gc.collect()

#  Load token from token.env
env_path = os.path.join(os.path.dirname(__file__), '..', 'token.env') 
load_dotenv(dotenv_path=env_path)
hf_token = os.getenv("HF_TOKEN_mistral")
assert hf_token, "HF_TOKEN_mistral not loaded — check token.env and env path."

#  Login
login(token=hf_token)

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Model & Tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

# Load model with extreme memory optimization
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        offload_folder="offload_folder",
        max_memory={0: "20GiB", "cpu": "32GiB"}
    )
except Exception as e:
    print(f"Error with GPU loading: {e}")
    print("Falling back to CPU-only mode...")
    # Fallback to CPU-only if GPU fails
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

# Define pipeline with minimal parameters
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample = True,
    max_new_tokens=450,  # Further reduced
    temperature=0.3,
    top_p=0.9,
    repetition_penalty=1.1,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id
)

# Wrap with LangChain
llm = HuggingFacePipeline(pipeline=text_pipeline)

# Optional: function to structure prompt injection better (if needed later)
def structured_prompt(prompt_body: str) -> str:
    system_prompt = (
        "You are an expert semantic log analyst specialized in FCAPS (Fault, Configuration, Accounting, Performance, Security).\n"
        "When answering, prefer structure: Summary → Root Cause → Recommended Actions → Risk Assessment (if relevant).\n"
        "If evidence from logs is partial, infer sensibly and suggest additional data needed.\n"
        "Provide actionable insights wherever possible.\n"
        "---\n"
    )
    return system_prompt + prompt_body



# /RAG/mistral_llm_loader.py
# solution 1
# import os
# import gc
# from dotenv import load_dotenv
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain_community.llms import HuggingFacePipeline
# from huggingface_hub import login
# import torch

# # Force CUDA to release memory and clear cache
# torch.cuda.empty_cache()
# gc.collect()

# # Check available GPU memory first
# print(f"GPU Memory before model load: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reserved")
# print(f"Max memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# #  Load token from specific token.env
# env_path = os.path.join(os.path.dirname(__file__), '..', 'token.env') 
# load_dotenv(dotenv_path=env_path)
# hf_token = os.getenv("HF_TOKEN_mistral")
# assert hf_token, "HF_TOKEN_mistral not loaded — check token.env and env path."

# #  Login
# login(token=hf_token)

# # Model & Tokenizer
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

# # Modified loading strategy
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     token=hf_token,
#     device_map="auto",
#     torch_dtype=torch.bfloat16,  # Be explicit about dtype
#     load_in_8bit=True,           # Try 8-bit quantization to save memory
#     use_cache=True,
#     offload_folder="offload_folder",  # Add disk offloading
#     max_memory={
#         0: "25GiB",     # Reduce memory usage further
#         "cpu": "32GiB"
#     }
# )

# # Print model memory usage
# print(f"GPU Memory after model load: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated")

# # Define pipeline with reduced parameters
# text_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,  # Reduced from 768
#     temperature=0.3,
#     top_p=0.9,
#     do_sample=True,
#     repetition_penalty=1.1,
#     return_full_text=False,
#     pad_token_id=tokenizer.eos_token_id
# )

# # Wrap with LangChain
# llm = HuggingFacePipeline(pipeline=text_pipeline)

# # Optional: function to structure prompt injection better (if needed later)
# def structured_prompt(prompt_body: str) -> str:
#     system_prompt = (
#         "You are an expert semantic log analyst specialized in FCAPS (Fault, Configuration, Accounting, Performance, Security).\n"
#         "When answering, prefer structure: Summary → Root Cause → Recommended Actions → Risk Assessment (if relevant).\n"
#         "If evidence from logs is partial, infer sensibly and suggest additional data needed.\n"
#         "Provide actionable insights wherever possible.\n"
#         "---\n"
#     )
#     return system_prompt + prompt_body
    
#==============================================================================================================

# # /RAG/mistral_llm_loader.py

# import os
# from dotenv import load_dotenv
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain_community.llms import HuggingFacePipeline
# from huggingface_hub import login

# import torch
# torch.cuda.empty_cache()

# #  Load token from specific token.env
# env_path = os.path.join(os.path.dirname(__file__), '..', 'token.env') 
# load_dotenv(dotenv_path=env_path)

# hf_token = os.getenv("HF_TOKEN_mistral")
# assert hf_token, "HF_TOKEN_mistral not loaded — check token.env and env path."

# #  Login
# login(token=hf_token)

# # Model & Tokenizer
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)


# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     token=hf_token,
#     device_map="auto",  # ✅ auto split across GPU and CPU
#     torch_dtype="auto",
#     use_cache=True,
#     offload_buffers=True,
#     max_memory={
#         0: "28GiB",     # ✅ prevent allocator from using all 40GB
#         "cpu": "32GiB"
#     }
# )


# # model = AutoModelForCausalLM.from_pretrained(
# #     model_name,
# #     device_map="cuda",
# #     torch_dtype="auto",
# #     token=hf_token,
# #     use_cache=True,
# #     offload_buffers=True
# # )

# # Define pipeline
# text_pipeline = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=768,
#     temperature=0.3,
#     top_p=0.9,
#     do_sample=True,
#     repetition_penalty=1.1,
#     return_full_text=False,
#     pad_token_id=tokenizer.eos_token_id
# )

# # Wrap with LangChain
# llm = HuggingFacePipeline(pipeline=text_pipeline)

# # Optional: function to structure prompt injection better (if needed later)
# def structured_prompt(prompt_body: str) -> str:
#     system_prompt = (
#         "You are an expert semantic log analyst specialized in FCAPS (Fault, Configuration, Accounting, Performance, Security).\n"
#         "When answering, prefer structure: Summary → Root Cause → Recommended Actions → Risk Assessment (if relevant).\n"
#         "If evidence from logs is partial, infer sensibly and suggest additional data needed.\n"
#         "Provide actionable insights wherever possible.\n"
#         "---\n"
#     )
#     return system_prompt + prompt_body


