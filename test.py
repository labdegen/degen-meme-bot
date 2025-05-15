import requests
from requests_oauthlib import OAuth1
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# For OAuth 1.0a, we need API Key and API Secret
# (not the OAuth 2.0 Client ID and Client Secret)
api_key = os.getenv("X_API_KEY")  # Consumer Key
api_secret = os.getenv("X_API_SECRET")  # Consumer Secret

# Print first few characters to check if they're loaded
print(f"API Key starts with: {api_key[:5]}..." if api_key else "API Key not found")
print(f"API Secret starts with: {api_secret[:5]}..." if api_secret else "API Secret not found")

# Create OAuth1 session
auth = OAuth1(api_key, api_secret)

# Test with a simple API call to verify credentials
# Using the account/verify_credentials endpoint
response = requests.get(
    "https://api.twitter.com/1.1/account/verify_credentials.json",
    auth=auth,
    timeout=10
)

print(f"Status code: {response.status_code}")
if response.status_code == 200:
    print("Success! Your OAuth 1.0a credentials are valid.")
    user_data = response.json()
    print(f"Authenticated as: @{user_data.get('screen_name')}")
else:
    print("Error:")
    print(response.text)

# Let's also test creating an application-only Auth Token
# This is similar to the OAuth 2.0 approach but with OAuth 1.0a credentials
def get_bearer_token(consumer_key, consumer_secret):
    # Encode consumer key and secret
    bearer_token_credentials = f"{consumer_key}:{consumer_secret}"
    import base64
    encoded_credentials = base64.b64encode(bearer_token_credentials.encode()).decode()
    
    # Set headers
    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
    }
    
    # Make request
    token_response = requests.post(
        "https://api.twitter.com/oauth2/token",
        headers=headers,
        data="grant_type=client_credentials",
        timeout=10
    )
    
    return token_response

# Try to get a bearer token
print("\nAttempting to get application-only Bearer Token:")
bearer_response = get_bearer_token(api_key, api_secret)
print(f"Status code: {bearer_response.status_code}")
if bearer_response.status_code == 200:
    print("Success! Bearer token obtained:")
    print(bearer_response.json())
else:
    print("Error getting Bearer Token:")
    print(bearer_response.text)