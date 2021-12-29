import os
from playsound import playsound

while True:
    if not os.path.exists("./play_sound.txt"):
        continue
    playsound("./tip.mp3")
    os.remove("./play_sound.txt")

