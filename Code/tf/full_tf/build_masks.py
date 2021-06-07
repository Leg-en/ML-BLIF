import os
import xml.etree.ElementTree as ET
import cv2



def buildXMLPaths(XMLs):
    res = []
    for i in os.listdir(XMLs):
        res.append(os.path.join(XMLs,i))
    return res


def read(XMLs, outpath):
    XMLs_ = buildXMLPaths(XMLs)
    for XML in XMLs_:
        if XML.endswith(".xml"):
            tree = ET.parse(XML)
            root = tree.getroot()
            objects = root.findall("object")
            labels = {}
            for object in objects:
                iter = object.iter()
                name = None
                bndbox = []
                for element in iter:
                    if element.tag == "name":
                        name = element.text
                    if element.tag == "bndbox":
                        bbox = element.iter()
                        for coords in bbox:
                            if coords.tag == "xmin" or coords.tag == "ymin" or coords.tag == "xmax" or coords.tag == "ymax":
                                bndbox.append(int(coords.text)) #xmin ymin xmax ymax
                if name in labels:
                    labels[name].append(bndbox)
                else:
                    labels[name] = [bndbox]
            path = root.find("path")
            fileName = root.find("filename")
            buildMask(path.text,labels, outpath, fileName.text)



def buildMask(Image, labels,outpath, ImageName):
    color_code = {
        "Wasser": [0,0,102],
        "Strand": [255, 255, 0],
        "Himmel": [0, 204, 0]
    }
    img = cv2.imread(Image)
    img[:,:,:] = [255,0,0] #nutzt das bgr=
    for label in labels:
        for coords in labels[label]:
            img[coords[1]:coords[3],coords[0]:coords[2], :] = color_code[label]
    cv2.imwrite(os.path.join(outpath,ImageName),img)
    return

if __name__ == '__main__':
    path = r"C:\Users\Emily\Documents\Bachelor\Klassifizierung"
    out = r"C:\Users\Emily\Documents\Bachelor\masks"
    read(path,out)


