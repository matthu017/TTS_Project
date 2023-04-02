from zipfile import ZipFile

with ZipFile('LibriSpeech-Alignments.zip', 'r') as f:
    
    #extract in current directory
    f.extractall(path="training_data")