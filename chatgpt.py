import openai

class ChatGPT:    
    def __init__(self, model="gpt-3.5-turbo", completion_params=None):
        self.model = model
        self.completion_params = completion_params or {}
        self.history = []
        self._messages = []

    # The messages object for the current conversation.
    def messages(self):
        messages = [{"role": "system", "content": self._system}] + self._messages
        return messages

    # Set the system message and optionally reset the conversation.
    def system(self, message, do_reset=True):
        if do_reset:
            self.reset()
        self._system = message

    # Add a user message to the conversation.
    def user(self, message):
        self._messages.append({"role": "user", "content": message})

    # Add an assistant message to the conversation.
    def assistant(self, message):
        self._messages.append({"role": "assistant", "content": message})

    # Reset the current conversation.
    def reset(self):
        self._messages = []

    # Make a completion with the current messages.
    def _make_completion(self, messages):
        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            **self.completion_params
        )

        self.history.append((messages, completion))
        return completion

    # Call ChatGPT with the current messages and return the assistant's message.
    def call(self):
        completion = self._make_completion(self.messages)
        return completion["choices"][0]["message"]["content"]

    # Add a user message and append + return the assistant's response. Optionally replace the last user message and response.
    def chat(self, message, replace_last=False):
        if replace_last:
            self._messages = self._messages[:-2]

        self.user(message)
        response = self.call()
        self._messages.append({"role": "assistant", "content": response})
        return response
