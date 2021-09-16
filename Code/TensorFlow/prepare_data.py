import os
import uuid
from multiprocessing import Pool

import cv2
import pandas


w_dir = r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\test"
csv_path = r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Artefakte\image_data.csv"
img_dir = r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\PNG"


def create_dirs(w_dir=r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Code\TensorFlow\w_dir"):
    try:
        os.mkdir(os.path.join(w_dir, "Wasser"))
    except FileExistsError:
        print("Wasser ordner existiert bereits")
        pass
    try:
        os.mkdir(os.path.join(w_dir, "Himmel"))
    except FileExistsError:
        print("Himmel ordner existiert bereits")
        pass
    try:
        os.mkdir(os.path.join(w_dir, "Strand"))
    except FileExistsError:
        print("Strand ordner existiert bereits")
        pass


def create_imgs(csv, w_dir, img_dir):
    for idx, row in csv.iterrows():
        img = cv2.imread(os.path.join(img_dir, row[0]))
        img = img[row[5]:row[7], row[4]:row[6]]
        cv2.imwrite(os.path.join(w_dir, row[3], str(uuid.uuid1())) + ".png", img)


def create_imgs_np(csv, w_dir, img_dir):
    with Pool(8) as p:
        p.map(intern, csv)


def intern(row):
    img = cv2.imread(os.path.join(img_dir, row[0]))
    img = img[row[5]:row[7], row[4]:row[6]]
    cv2.imwrite(os.path.join(w_dir, row[3], str(uuid.uuid1())) + ".png", img)





def main(w_dir_in=r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\test",
         csv_path_in=r"C:\Users\Emily\Documents\GitHub\ML-BLIF\Artefakte\image_data.csv",
         img_dir_in=r"C:\Users\Emily\Documents\Bachelor_Drohnen_Bilder\PNG"):
    global w_dir, csv_path, img_dir
    w_dir, csv_path, img_dir = w_dir_in, csv_path_in, img_dir_in
    csv = pandas.read_csv(csv_path)
    create_dirs(w_dir)
    # create_imgs(csv,w_dir, img_dir)
    create_imgs_np(csv.values, w_dir, img_dir)

if __name__ == '__main__':
    main()
