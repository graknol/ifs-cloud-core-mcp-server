"""
GitHub Copilot API integration module
Based on the reference code provided for authentication and API calls
"""

import requests
import json
import time
import threading


class CopilotAPI:
    def __init__(self):
        self.token = None
        self.access_token = None
        # Start token refresh thread
        self.token_thread = threading.Thread(
            target=self._token_refresh_thread, daemon=True
        )
        self.token_thread.start()

    def setup(self):
        """Setup GitHub Copilot authentication"""
        resp = requests.post(
            "https://github.com/login/device/code",
            headers={
                "accept": "application/json",
                "editor-version": "Neovim/0.6.1",
                "editor-plugin-version": "copilot.vim/1.16.0",
                "content-type": "application/json",
                "user-agent": "GithubCopilot/1.155.0",
                "accept-encoding": "gzip,deflate,br",
            },
            data='{"client_id":"Iv1.b507a08c87ecfe98","scope":"read:user"}',
        )

        # Parse the response json, isolating the device_code, user_code, and verification_uri
        resp_json = resp.json()
        device_code = resp_json.get("device_code")
        user_code = resp_json.get("user_code")
        verification_uri = resp_json.get("verification_uri")

        # Print the user code and verification uri
        print(
            f"Please visit {verification_uri} and enter code {user_code} to authenticate."
        )

        while True:
            time.sleep(5)
            resp = requests.post(
                "https://github.com/login/oauth/access_token",
                headers={
                    "accept": "application/json",
                    "editor-version": "Neovim/0.6.1",
                    "editor-plugin-version": "copilot.vim/1.16.0",
                    "content-type": "application/json",
                    "user-agent": "GithubCopilot/1.155.0",
                    "accept-encoding": "gzip,deflate,br",
                },
                data=f'{{"client_id":"Iv1.b507a08c87ecfe98","device_code":"{device_code}","grant_type":"urn:ietf:params:oauth:grant-type:device_code"}}',
            )

            # Parse the response json, isolating the access_token
            resp_json = resp.json()
            access_token = resp_json.get("access_token")

            if access_token:
                break

        # Save the access token to a file
        with open(".copilot_token", "w") as f:
            f.write(access_token)

        print("Authentication success!")
        self.access_token = access_token

    def get_token(self):
        """Get session token for API calls"""
        # Check if the .copilot_token file exists
        while True:
            try:
                with open(".copilot_token", "r") as f:
                    self.access_token = f.read().strip()
                    break
            except FileNotFoundError:
                self.setup()

        # Get a session with the access token
        resp = requests.get(
            "https://api.github.com/copilot_internal/v2/token",
            headers={
                "authorization": f"token {self.access_token}",
                "editor-version": "Neovim/0.6.1",
                "editor-plugin-version": "copilot.vim/1.16.0",
                "user-agent": "GithubCopilot/1.155.0",
            },
        )

        # Parse the response json, isolating the token
        resp_json = resp.json()
        self.token = resp_json.get("token")

    def _token_refresh_thread(self):
        """Background thread to refresh tokens"""
        while True:
            self.get_token()
            time.sleep(25 * 60)  # Refresh every 25 minutes

    def is_token_invalid(self, token):
        """Check if token is expired"""
        if token is None or "exp" not in token:
            return True
        exp_time = self.extract_exp_value(token)
        if exp_time is None or exp_time <= time.time():
            return True
        return False

    def extract_exp_value(self, token):
        """Extract expiration time from token"""
        try:
            pairs = token.split(";")
            for pair in pairs:
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    if key.strip() == "exp":
                        return int(value.strip())
        except:
            pass
        return None

    def copilot_completion(self, prompt, max_tokens=4000, temperature=0.3):
        """
        Call Copilot API for text completion
        Note: This uses the codex completion endpoint, but we'll adapt it for our needs
        """
        # If the token is None, get a new one
        if self.token is None or self.is_token_invalid(self.token):
            self.get_token()

        try:
            resp = requests.post(
                "https://copilot-proxy.githubusercontent.com/v1/engines/copilot-codex/completions",
                headers={"authorization": f"Bearer {self.token}"},
                json={
                    "prompt": prompt,
                    "suffix": "",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 1,
                    "n": 1,
                    "stop": ["---END---"],  # Custom stop sequence
                    "nwo": "github/copilot.vim",
                    "stream": True,
                    "extra": {
                        "language": "markdown"  # We want markdown formatted output
                    },
                },
            )
        except requests.exceptions.ConnectionError:
            return ""

        result = ""
        # Parse the response text, splitting it by newlines
        resp_text = resp.text.split("\n")
        for line in resp_text:
            # If the line contains a completion, parse it
            if line.startswith("data: {"):
                try:
                    json_completion = json.loads(line[6:])
                    choices = json_completion.get("choices", [])
                    if choices:
                        completion = choices[0].get("text", "")
                        if completion:
                            result += completion
                except json.JSONDecodeError:
                    continue

        return result

    def chat_completion(self, messages, max_tokens=4000, temperature=0.3):
        """
        Attempt to use chat completion if available, fallback to completion
        This is experimental - the GitHub Copilot API might not support chat format
        """
        # Try to construct a prompt from messages for the completion API
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        prompt += "Assistant: "

        return self.copilot_completion(prompt, max_tokens, temperature)
