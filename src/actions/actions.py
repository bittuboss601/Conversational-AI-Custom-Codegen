# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from transformers import pipeline
import requests


class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Hello World!")

        return []
    
class ActionGenerateResponse(Action):
    def name(self) -> Text:
        return "action_generate_response"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        generator = pipeline('text-generation', model='gpt2')
        user_message = tracker.latest_message['text']
        generated_text = generator(user_message, max_length=50, num_return_sequences=1)[0]['generated_text']
        
        dispatcher.utter_message(text=generated_text)
        
        return []

class ActionGenerateCode(Action):
    def name(self):
        return "action_generate_code"

    def run(self, dispatcher, tracker, domain):
        user_msg = tracker.latest_message.get('text')
        if len(user_msg) > 1000:
            user_msg = user_msg[:1000]
        response = requests.post("http://localhost:8000/generate", json={"prompt": user_msg})
        result = response.json().get("response")
        dispatcher.utter_message(text=result)
        return []

class ActionDebugCode(Action):
    def name(self):
        return "action_debug_code"

    def run(self, dispatcher, tracker, domain):
        user_msg = tracker.latest_message.get('text')
        if len(user_msg) > 1000:
            user_msg = user_msg[:1000]
        response = requests.post("http://localhost:8000/debug", json={"prompt": user_msg})
        result = response.json().get("response")
        dispatcher.utter_message(text=result)
        return []
