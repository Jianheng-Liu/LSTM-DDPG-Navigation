# -*- coding: utf-8 -*-
import speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source) 
    print('say something')

    # print("")
    audio = r.listen(source)

try:
    print("Sphinx thinks you said " + r.recognize_sphinx(audio))
    print("ok")
except sr.UnknownValueError:
    print("Sphinx could not understand audio")
except sr.RequestError as e:
    print("Sphinx error; {0}".format(e))

