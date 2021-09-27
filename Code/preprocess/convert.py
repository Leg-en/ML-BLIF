"""
Dieses Skript dient dem Ziel des Konvertierens von einem Ordner mit DNG bildern zu einem ordner mit PNG Bildern.
Dieses Skript benötigt eine Installation von ImageMagick, welche zusätzlich in der PATH Variable hinterlegt ist dass diese über ein CLI (Windows Terminal) aufrufbar ist.
Das Skript benötigt zusätzlich den Ausgangsordnern mit den Bildern als auch einen Ziel Ordner.
Ferner ist dieses Skript angepasst auf das gegebene Benennungsschema der Bilddateien.
"""

import os

path = r"" #Ordner zu DNG Datein
output = r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\PNG" #Ordner in dem die PNGs abgelegt werden
for i in os.listdir(path):
    c_path = os.path.join(path, i)
    x = "magick convert {} {}".format(c_path, os.path.join(output, (i[0:8] + ".png")))
    print(x)
    os.system(x)
