from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Define the model path
MODEL_PATH = "~/mistral_models/Mamba-Codestral-7B-v0.1"

# Load the tokenizer and model
print("Loading the tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",  # Automatically map layers to available devices
    load_in_4bit=True,  # Enable 4-bit quantization for memory optimization
    torch_dtype=torch.float16  # Use float16 for further optimization
)

print("Model and tokenizer loaded successfully!")

# Function to generate a response
def chat_with_model(prompt, history=None):
    """
    Generate a response from the model based on the given prompt and conversation history.
    
    Args:
        prompt (str): User input text.
        history (list): List of previous conversation turns [(user1, Mamba1), (user2, Mamba2), ...].
        
    Returns:
        str: Model's response.
    """
    if history is None:
        history = []
    
    # Combine history and new prompt
    conversation = ""
    for user_input, bot_response in history:
        conversation += f"User: {user_input}\nMamba: {bot_response}\n"
    conversation += f"User: {prompt}\nMamba:"

    # Tokenize input
    inputs = tokenizer(conversation, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    # Generate output
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,  # Controls randomness (lower is less random)
        top_p=0.9,       # Nucleus sampling for diversity
        do_sample=True   # Enable sampling instead of greedy decoding
    )

    # Decode response
    output = tokenizer.decode(output_tokens[:, inputs["input_ids"].shape[1]:][0], skip_special_tokens=True)
    return output.strip()

# Main loop for chatting
def main():
    print("\nWelcome to the Mamba chatbot! Type 'exit' to quit.\n")
    history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Generate response
        response = chat_with_model(user_input, history)

        # Display response
        print(f"Mamba: {response}")

        # Update history
        history.append((user_input, response))

if __name__ == "__main__":
    main()
