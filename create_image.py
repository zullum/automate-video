import requests
import base64

# Replace with your Cloudflare account ID and API token
account_id = 'your_account_id'
api_token = 'your_api_token'

# Set the model name
model_name = '@cf/black-forest-labs/flux-1-schnell'

# Define the API endpoint
url = f'https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}'

# Set the headers, including the authorization token
headers = {
    'Authorization': f'Bearer {api_token}',
    'Content-Type': 'application/json',
}

# Define the payload with your prompt and optional parameters
payload = {
    'prompt': 'a cyberpunk lizard',
    'steps': 4,  # Optional: Number of diffusion steps; higher values can improve quality but take longer
}

# Make the POST request to the Cloudflare API
response = requests.post(url, json=payload, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Extract the base64-encoded image from the response
    image_base64 = response.json().get('result', {}).get('image')
    if image_base64:
        # Decode the base64 string to binary data
        binary_data = base64.b64decode(image_base64)
        # Convert binary data to a byte array
        img = bytearray(binary_data)
        # Return the image as a response
        print('Image generated successfully.')
    else:
        print('No image data found in the response.')
else:
    print(f'Error: {response.status_code} - {response.text}')
