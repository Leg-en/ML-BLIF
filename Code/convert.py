import os
path = r"C:\Users\Emily\Documents\Bachelor\Jens_Drohnenbilder"
output = r"C:\Users\Emily\Documents\Bachelor\convertet_png"

for i in os.listdir(path):
    c_path = os.path.join(path,i)
    x = "magick convert {} {}".format(c_path, os.path.join(output,( i[0:8] +".png")))
    print(x)
    os.system(x)