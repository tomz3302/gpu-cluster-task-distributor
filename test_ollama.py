from ollama import chat
import time

MODEL = "smollm:135m"

start = time.perf_counter()

response = chat(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": "Explain load balancing in one short paragraph."
        }
    ],
    options={
        "num_predict": 64,
        "temperature": 0.2
    }
)

end = time.perf_counter()

print("Response:")
print(response.message.content)
print()
print(f"Latency: {end - start:.2f} seconds")