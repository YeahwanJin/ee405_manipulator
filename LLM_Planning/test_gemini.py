from google import genai
from google.genai import types
import os 

API_KEY = os.environ['API_KEY']

with open('./capture/capture.jpg', 'rb') as f:
    image_bytes = f.read()

client = genai.Client(api_key=API_KEY)

image = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')   # png -> mime_type='images/png'

prompt= "What do you see in the image?"

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[image,prompt]
)

print(response.text)
