"""
GitHub Copilot API integration module
Based on the reference code provided for authentication and API calls
"""

import requests
import json
import time
import threading
import uuid


class CopilotAPI:
    def __init__(self):
        self.token = None
        self.chat_token = None
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

    def get_chat_token(self):
        """
        Get chat token for GitHub Copilot Chat API
        First tries to load from saved file, then tries API endpoint
        """
        # Try to load existing chat token first
        try:
            with open(".copilot_chat_token", "r") as f:
                self.chat_token = f.read().strip()
                if self.chat_token:
                    print("✅ Loaded existing GitHub Chat token")
                    return True
        except FileNotFoundError:
            pass

        # Check if the .copilot_token file exists
        while True:
            try:
                with open(".copilot_token", "r") as f:
                    self.access_token = f.read().strip()
                    break
            except FileNotFoundError:
                self.setup()

        # The chat token endpoint requires the GitHub session to be authenticated
        # We'll use the access token in a cookie format as shown in the curl example
        resp = requests.post(
            "https://github.com/github-copilot/chat/token",
            headers={
                "authority": "github.com",
                "accept": "application/json",
                "accept-language": "en-US,en;q=0.9",
                "content-length": "0",
                "content-type": "application/json",
                "authorization": f"token {self.access_token}",  # Try both approaches
                "github-verified-fetch": "true",
                "origin": "https://github.com",
                "referer": "https://github.com/copilot/",
                "sec-ch-ua": '"Chromium";v="105", "Not)A;Brand";v="8"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "sec-fetch-dest": "empty",
                "sec-fetch-mode": "cors",
                "sec-fetch-site": "same-origin",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36",
            },
        )

        if resp.status_code == 200:
            try:
                resp_json = resp.json()
                self.chat_token = resp_json.get("token")
                if self.chat_token:
                    print("✅ Successfully obtained GitHub Chat token")
                    # Save token for future use
                    with open(".copilot_chat_token", "w") as f:
                        f.write(self.chat_token)
                    return True
                else:
                    print("❌ No token in response")
                    return False
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON response: {resp.text}")
                return False
        else:
            print(f"❌ Failed to get chat token: {resp.status_code} - {resp.text}")
            print("💡 You can manually set the chat token using setup_chat_token.py")
            # Try the legacy token method as fallback
            self.get_token()
            return False

    def _token_refresh_thread(self):
        """Background thread to refresh tokens"""
        while True:
            self.get_chat_token()
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

    def copilot_completion(
        self,
        prompt,
        max_tokens=4000,
        temperature=0.3,
    ):
        """
        Call Copilot API for text completion using the reliable legacy completion API
        """
        return self.copilot_legacy_completion(prompt, max_tokens, temperature)

    def copilot_legacy_completion(self, prompt, max_tokens=4000, temperature=0.3):
        """
        Original Copilot completion method (fallback)
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

    def get_token(self):
        """Legacy method for getting session token (for fallback)"""
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

    def copilot_chat_completion(
        self,
        prompt,
        max_tokens=4000,
        temperature=0.3,
        model="claude-3-5-sonnet-20241022",
    ):
        """
        Call GitHub Copilot Chat API for text completion with model selection
        Uses the GitHub Chat API to access different models including Claude Sonnet 4
        """
        # If the chat token is None, get a new one
        if self.chat_token is None:
            if not self.get_chat_token():
                return ""  # Fallback will be handled by parent method

        # Generate a thread ID similar to the curl example format
        thread_id = f"{uuid.uuid4().hex[:8]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:12]}"

        try:
            # Try OpenAI-compatible message format first
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }

            print(f"🔄 Making request with model: {model}")
            print(
                f"📡 Endpoint: https://api.individual.githubcopilot.com/github/chat/threads/{thread_id}/messages"
            )

            # Make the request to the chat messages endpoint
            resp = requests.post(
                f"https://api.individual.githubcopilot.com/github/chat/threads/{thread_id}/messages",
                headers={
                    "authority": "api.individual.githubcopilot.com",
                    "accept": "application/json",  # Changed to JSON instead of */*
                    "accept-language": "en-US,en;q=0.9",
                    "authorization": f"GitHub-Bearer {self.chat_token}",
                    "content-type": "application/json",
                    "copilot-integration-id": "copilot-chat",
                    "origin": "https://github.com",
                    "referer": "https://github.com/",
                    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36",
                },
                json=payload,
                timeout=30,
            )

            print(f"📊 Response status: {resp.status_code}")
            print(f"📄 Response headers: {dict(resp.headers)}")

            if resp.status_code == 200:
                try:
                    resp_json = resp.json()
                    print(f"📋 Response keys: {list(resp_json.keys())}")

                    # Try different response formats
                    if "choices" in resp_json and resp_json["choices"]:
                        content = (
                            resp_json["choices"][0]
                            .get("message", {})
                            .get("content", "")
                        )
                        if content:
                            print(
                                f"✅ Chat API response received ({len(content)} chars)"
                            )
                            return content
                    elif "content" in resp_json:
                        print(
                            f"✅ Chat API response received ({len(resp_json['content'])} chars)"
                        )
                        return resp_json["content"]
                    elif "message" in resp_json:
                        print(
                            f"✅ Chat API response received ({len(resp_json['message'])} chars)"
                        )
                        return resp_json["message"]
                    elif "response" in resp_json:
                        print(
                            f"✅ Chat API response received ({len(resp_json['response'])} chars)"
                        )
                        return resp_json["response"]
                    else:
                        print(f"⚠️ Unknown response format: {resp_json}")
                        return ""

                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    print(f"Raw response: {resp.text[:500]}")
                    return ""
            else:
                error_text = resp.text[:500]
                print(f"❌ Chat API error: {resp.status_code}")
                print(f"Error response: {error_text}")

                # Try alternative endpoint or method
                if resp.status_code == 400:
                    print("🔄 Trying alternative payload format...")
                    # Try simpler payload
                    simple_payload = {"content": prompt, "model": model}

                    resp2 = requests.post(
                        f"https://api.individual.githubcopilot.com/github/chat/threads/{thread_id}/messages",
                        headers={
                            "authorization": f"GitHub-Bearer {self.chat_token}",
                            "content-type": "application/json",
                            "copilot-integration-id": "copilot-chat",
                        },
                        json=simple_payload,
                        timeout=30,
                    )

                    print(f"📊 Alternative request status: {resp2.status_code}")
                    if resp2.status_code == 200:
                        try:
                            resp2_json = resp2.json()
                            if "content" in resp2_json:
                                return resp2_json["content"]
                        except:
                            pass

                return ""

        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            return ""

    def _parse_streaming_response(self, response_text):
        """Parse streaming response from chat API"""
        result = ""
        lines = response_text.split("\n")

        for line in lines:
            if line.startswith("data: "):
                try:
                    json_data = json.loads(line[6:])
                    if "choices" in json_data and json_data["choices"]:
                        delta = json_data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            result += content
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
