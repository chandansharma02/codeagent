from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16)

def generate_answer(input_text):
    # Tokenize the input and send it to the model
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    # Generate the response with settings to make the output concise
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.4, top_k=50, top_p=0.9, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    # Decode and return the generated response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def read_python_file(file_path):
    """Reads the Python file and returns its content as a string."""
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' does not exist."
    with open(file_path, 'r') as file:
        return file.read()

# QA Assistant loop
print("Welcome to the QA Assistant. You can upload a Python file for code review or improvement suggestions.")
while True:
    # Get input from the user (file path)
    file_path = input("Enter the path to your Python file (or 'exit' to quit): ")
    
    if file_path.lower() in ['exit', 'quit', 'exit qa']:
        print("Exiting the QA Assistant. Goodbye!")
        break
    
    # Read the content of the Python file
    code_content = read_python_file(file_path)
    
    if code_content.startswith("Error"):
        print(code_content)
        continue
    
    # Prepare the prompt for the model to handle the code review
    prompt = f"""
    You are a highly skilled assistant. You help programmers by suggesting improvements to their code in a clear, actionable, and concise manner.

    The following is the code that needs suggestions or improvements:
    {code_content}

    Suggest improvements or provide feedback on the code.
    """
    
    # Generate an answer or improvement suggestion
    answer = generate_answer(prompt)
    
    print("QA Assistant:", answer)