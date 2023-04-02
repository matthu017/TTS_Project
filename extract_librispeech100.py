import tarfile
tar = tarfile.open("train-clean-100.tar.gz")
tar.extractall(path="training_data")
tar.close()