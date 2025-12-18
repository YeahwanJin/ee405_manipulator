import google.generativeai as genai

genai.configure(api_key="AIzaSyB-v0fEyoKMCk1_6pRbzOK46-8XWtGR94M")

print("Available Models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f" - {m.name}")