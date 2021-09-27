"""
Diese Klasse dient dem Ziel, die XML Files mit den Label Informationen in ein JSON zu Formatieren, welches aus
filename, width, height, class, xmin, ymin, xmax, ymax besteht.
Als Input wird der Ordnerpfad zu den XMLs erwartet.
Als output der Zielpfad.
"""


import xml.etree.ElementTree as ET
import os
import csv

input = r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Klassifizierung" #Input
data_array = []
for item in os.listdir(input):
    data = os.path.join(input, item)
    tree = ET.parse(data)
    root = tree.getroot()
    x = root.findall("object")
    for i in x:
        #print("---------------------------")
        x_min = 0
        y_min = 0
        x_max = 0
        y_max = 0
        file = root.find("filename").text
        y = i.iter()
        for j in y:
            name = j.find("name")
            bbox = j.find("bndbox")
            if not name is None:
                type = name.text
            if not bbox is None:
                x_min = bbox.find("xmin").text
                y_min = bbox.find("ymin").text
                x_max = bbox.find("xmax").text
                y_max = bbox.find("ymax").text
            d = {"file":file, "x_min":x_min, "y_min":y_min, "x_max":x_max, "y_max":y_max, "type":type}
            data_array.append(d)
#print(data_array)


with open("../../Artefakte/image_data.csv", "w", newline='') as csvfile: #Output muss als csv typ gespeichert werden.
    writer = csv.DictWriter(csvfile, fieldnames=["file","x_min", "y_min","x_max","y_max","type"])
    writer.writeheader()
    for i in data_array:
        print(i)
        writer.writerow(i)
