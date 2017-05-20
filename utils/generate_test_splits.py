import os

from utils.generate_split import generate_split


def read_ids(file,base):
    ids = []
    with open(base+"/"+file,"r") as f:
        for line in f:
           ids.append(int(line))
        return ids

def split(dataset, base="splits"):
    if not (os.path.exists(base+"/"+"training_ids.txt")
            and os.path.exists(base+"/"+"dev_ids.txt") and os.path.exists(base+"/"+"test_ids.txt")):
        raise Exception("There is an error and the dataset reader cannot find the "
                        "{training_ids|test_ids|dev_ids}.txt file. Please make sure your python paths "
                        "are configured correctly")

    training_ids = read_ids("training_ids.txt",base)
    dev_ids = read_ids("dev_ids.txt",base)
    test_ids = read_ids("test_ids.txt",base)

    #return the stances that meet these criteria
    training_stances = []
    dev_stances = []
    test_stances = []

    for stance in dataset.stances:
        if stance['Body ID'] in training_ids:
            training_stances.append(stance)
        elif stance['Body ID'] in dev_ids:
            dev_stances.append(stance)
        elif stance['Body ID'] in test_ids:
            test_stances.append(stance)


    return {"training":training_stances, "dev":dev_stances, "test": test_stances}
