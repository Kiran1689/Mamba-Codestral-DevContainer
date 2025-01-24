from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path

# Define the model path
MODEL_PATH = str(Path.home().joinpath("mistral_models", "Mamba-Codestral-7B-v0.1"))

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found. Using CPU, which may be slower.")

# Load the tokenizer and model
print("Loading the tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    bnb_4bit_compute_dtype=torch.float16
)

print("Model loaded successfully!")

# Function to detect and suggest fixes for bugs in the code
def bug_finder_in_code(code_snippet: str, max_length=300):
    prompt = f"Identify any bugs in the following code and suggest possible fixes:\n\n{code_snippet}\n\nBug fixes and explanations:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Ensure attention mask is set explicitly if the pad_token is same as eos_token
    attention_mask = inputs.get('attention_mask', None)
    if attention_mask is None:
        attention_mask = torch.ones_like(inputs['input_ids'], device=device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    bug_fix_suggestions = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return bug_fix_suggestions

code_snippet = '''
def calculate_sum(a, b):
    return a + b

result = calculate_sum(5)
print(result)
'''

bug_fixes = bug_finder_in_code(code_snippet)
print("Bug Fix Suggestions: ", bug_fixes)
