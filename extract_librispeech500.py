import tarfile
tar = tarfile.open("train-other-500.tar.gz")
tar.extractall(path="training_data")
tar.close()