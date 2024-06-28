###############################################################################
#                             CHATBOT                                         #
###############################################################################
# Setup
## for speech-to-text
import speech_recognition as sr #3.10.4

## for text-to-speech
from gtts import gTTS #2.5.1

## for language model
import ollama #0.5.0
model = "maya"
'''
/set system Your name is Maya. You are a spoiled russian girl with a nasty attitude and strong accent. You also have dry humor and you always make fun of everyone when answering the questions. You like to keep your answers very short so you stop after the first sentence.
'''

## for data
import os
from datetime import datetime
import numpy as np


# Build the AI
class ChatBot():
    def __init__(self, name):
        print("--- starting up", name, "---")
        self.name = name.lower()

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=1)
            print("listening...")
            audio = recognizer.listen(mic)
        try:
            self.text = recognizer.recognize_google(audio)
            print("me --> ", self.text)
        except:
            print("me -->  ERROR")

    @staticmethod
    def text_to_speech(text):
        print("ai --> ", text)
        speaker = gTTS(text=text, lang="en", slow=False)
        speaker.save("res.mp3")
        os.system("afplay res.mp3")  #mac->afplay | windows->start
        os.remove("res.mp3")

    ## predetermined commands
    def wake_up(self, text):
        lst = ["wake up "+self.name, self.name+"wake up ", "hey "+self.name]
        return True if any(i in ai.text.lower() for i in lst) else False

    def what(self, text):
        lst = ["what are you", "who are you"]
        return True if any(i in ai.text.lower() for i in lst) else False

    @staticmethod
    def action_time():
        return datetime.now().time().strftime('%H:%M')


# Run the AI
if __name__ == "__main__":
    
    ai = ChatBot(name="Maya")
    
    while True:
        ai.speech_to_text()

        ## wake up
        if ai.wake_up(ai.text) is True:
            res = "Hello I am Maya the AI, what can I do for you?"

        ## what
        elif ai.what(ai.text) is True:
            res = "I am an AI created by Mauro"

        ## action time
        elif "time" in ai.text:
            res = ai.action_time()
        
        ## respond politely
        elif any(i in ai.text for i in ["thank","thanks"]):
            res = np.random.choice(["you're welcome!","anytime!","no problem!","cool!","I'm here if you need me!","peace out!"])
        
        ## conversation
        else:   
            res = ollama.generate(model=model, prompt=ai.text)["response"]
            res = res.split("\n")[0]
            #for trash in ["answer:", "support=", "**Note:", "(Note:", "- Reply:", "- reply:", "===", "(", "-output:", "Response:", "Maya: "]:
            #    res = res[0:res.index(trash)].strip() if trash in res else res

        ai.text_to_speech(res)
