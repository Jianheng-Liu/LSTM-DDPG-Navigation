# -*- coding:UTF-8 -*-
import pyttsx
import time

engine = pyttsx.init(driverName='espeak')

voices = engine.getProperty('voices')
'''
for voice in voices:
    print("Voice:")
    print(" - ID: %s" % voice.id)
    print(" - Name: %s" % voice.name)
    print(" - Languages: %s" % voice.languages)
    print(" - Gender: %s" % voice.gender)
    print(" - Age: %s" % voice.age)
'''

voice_id = 'zh'
engine.setProperty('voice', voice_id)

said = True

engine.say(u'到达桌子目标位置')
engine.say(u'到达桌子目标位置')
engine.runAndWait()
engine.stop()
del engine


engine = pyttsx.init(driverName='espeak')

voices = engine.getProperty('voices')

voice_id = 'zh'
engine.setProperty('voice', voice_id)

engine.say(u'到达桌子目标位置')
engine.runAndWait()

said = True
