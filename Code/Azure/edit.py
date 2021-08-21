import os
import json

path = r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\Klassifikation2"
res_class = r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\res_class"
dir = os.listdir(path)

vott = r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\Bachelor Projekt.vott.json"

def main():
    for file in dir:
        if ".json" in file:
            with open(os.path.join(path, file)) as js:
                elem = json.load(js)
                name = elem["asset"]["name"][-8:-4]
                name_sheme = "Drohnen_Bilder%2FDJI_{}.png".format(name)
                path_sheme = "https://mlbachelor6677487339.blob.core.windows.net/ml-container/{}?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacuptfx&se=2021-08-16T01:07:24Z&st=2021-08-15T17:07:24Z&spr=https,http&sig=8U7qPuyDss2Q0jNmgM62ZVjT66%2FLmUmVJEWz2zJpxik%3D"
                elem["asset"]["path"] = path_sheme.format(name_sheme)
                elem["asset"]["name"] = name_sheme
                with open(os.path.join(res_class, file), "w", encoding='utf-8') as f:
                    json.dump(elem, f, ensure_ascii=False, indent=4)
                edit_vott(elem["asset"]["id"], name_sheme, path_sheme.format(name_sheme))



def edit_vott(id, name, path):
    elem = None
    with open(vott) as v:
        elem = json.load(v)
    elem["assets"][id]["name"] = name
    elem["assets"][id]["path"] = path
    with open(vott, "w", encoding='utf-8') as f:
        json.dump(elem, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
