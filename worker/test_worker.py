import requests
import json
import sys

# --- CONFIGURATION ---
# Replace this with the link from your Colab (e.g., https://xyz.loca.lt)
# Ensure there is NO trailing slash at the end
COLAB_URL = "http://zzijm-35-243-230-50.run.pinggy-free.link/" 
MODEL_NAME = "gemma4:e2b"
# ---------------------

def ask_gemma():
    print(f"\n--- Connected to {COLAB_URL} ---")
    print("Type 'exit' or 'quit' to stop.\n")

    endpoint = f"{COLAB_URL}/api/generate"

    while True:
        user_prompt = input("Prompt: ")
        
        if user_prompt.lower() in ['exit', 'quit']:
            break

        payload = {
            "model": MODEL_NAME,
            "prompt": user_prompt,
            "stream": True  # This makes it feel like ChatGPT
        }

        try:
            # We use stream=True in requests to handle the chunked response
            with requests.post(endpoint, json=payload, stream=True, timeout=60) as response:
                if response.status_code == 200:
                    print("\nGemma: ", end="", flush=True)
                    
                    for line in response.iter_lines():
                        if line:
                            # Parse the JSON chunk from Ollama
                            chunk = json.loads(line.decode('utf-8'))
                            token = chunk.get("response", "")
                            print(token, end="", flush=True)
                            
                            if chunk.get("done"):
                                print("\n" + "-"*30 + "\n")
                else:
                    print(f"\n❌ Error: Server returned status {response.status_code}")
                    print(f"Details: {response.text}")
                    
        except requests.exceptions.ConnectionError:
            print("\n❌ Connection Error: Could not reach the Colab server.")
            print("Check if: \n1. The Colab cell is still running.\n2. You 'unlocked' the Localtunnel link in your browser once.")
        except Exception as e:
            print(f"\n⚠️ Unexpected error: {e}")

if __name__ == "__main__":
    ask_gemma()