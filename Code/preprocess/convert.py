import os
path = r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\DNG"
output = r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\PNG"
#path = "/home/phoenix/Documents/ImageSeg-Kurs/Drohnenbilder/"
#output = "/home/phoenix/Documents/ImageSeg-Kurs/Drohnenbilder_convertet/"
for i in os.listdir(path):
    c_path = os.path.join(path,i)
    x = "magick convert {} {}".format(c_path, os.path.join(output,( i[0:8] +".png")))
    print(x)
    os.system(x)
