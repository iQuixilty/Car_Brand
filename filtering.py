import os

directory = "cars"


for file_name in os.listdir(directory):
    brand = file_name.split('_')[0]
    if brand != "Ford" and brand != "Kia" and brand != "BMW":
        os.remove(directory + "/" + file_name)