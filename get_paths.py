import os
import shutil
import random

def estrai_hash_e_costruisci_path(file_path, folder):
    # Estrai il nome del file dal percorso completo
    file_name = os.path.basename(file_path)
    
    # Separa il nome del file dalla sua estensione
    name, ext = os.path.splitext(file_name)
    
    # Estrai la parte hash del nome del file, dopo l'ultimo underscore
    hash_value = name.split('_')[-1]
    
    # Crea il nuovo percorso usando la cartella fornita e l'hash con l'estensione
    nuovo_path = os.path.join(folder, hash_value + ext)
    
    return nuovo_path

def seleziona_file_casuali(cartella, numero_file):
    # Ottieni tutti i file nella cartella specificata
    tutti_i_file = [os.path.join(cartella, f) for f in os.listdir(cartella) if os.path.isfile(os.path.join(cartella, f))]
    random.shuffle(tutti_i_file)
    # Seleziona casualmente il numero di file specificato
    file_selezionati = random.sample(tutti_i_file, numero_file)
    
    return file_selezionati

def copia_groundtruth(paths, folder, dest_folder=None, check_folder=None):
    for file_path in paths:
        print("Source: ", file_path)
        
        groundtruth_path = estrai_hash_e_costruisci_path(file_path, folder)
        print("groundtruth: ", groundtruth_path)
        
        # Se è specificata una cartella di destinazione, esegui il controllo e copia il file
        if dest_folder:
            # Assicura che la cartella di destinazione esista
            os.makedirs(dest_folder, exist_ok=True)
            
            # Crea il percorso completo di destinazione
            dest_path = os.path.join(dest_folder, os.path.basename(groundtruth_path))
            
            # Copia il file se non esiste nella cartella di check
            shutil.copy(groundtruth_path, dest_path)
            print(f"File copiato a: {dest_path}")

def check_folder(new_data, training_folder):
    #print(f"\n\nChecking folder {training_folder} against {new_data}")
    files = os.listdir(training_folder)
    files = [f.split('_')[-1] for f in files]
    
    #print("Files in training folder: ", files[:10])
    new_files = os.listdir(new_data)
    new_files = [f.split('_')[-1] for f in new_files]
    #print("Files in new data: ", new_files)
    for f in new_files:
        if f in files:
            print(f"File {f} già presente nella cartella di training: {os.path.join(training_folder, f)}")
        
    print("Check ok.\n")
def seleziona_e_copia_file_casuali(cartella_origine, numero_file, cartella_groundtruth, cartella_destinazione_groundtruth, cartella_destinazione_selezionati, folder_to_check=None):
    # Seleziona i file casuali dalla cartella di origine
    file_selezionati = seleziona_file_casuali(cartella_origine, numero_file)
    
    # Copia i file selezionati nella cartella di destinazione per i file selezionati
    os.makedirs(cartella_destinazione_selezionati, exist_ok=True)
    for file in file_selezionati:
        shutil.copy(file, cartella_destinazione_selezionati)
        print(f"File selezionato copiato a: {os.path.join(cartella_destinazione_selezionati, os.path.basename(file))}")
    
    # Esegui la copia dei file groundtruth
    copia_groundtruth(file_selezionati, cartella_groundtruth, cartella_destinazione_groundtruth, check_folder)
    if os.path.isdir(folder_to_check):
        check_folder(cartella_destinazione_selezionati, folder_to_check)
    
# Esempio di utilizzo
cartella_origine = "/home/prrgpp000/cpa_enhanced/datasets/reconstructions/val_set"
numero_file = 32
cartella_groundtruth = "/home/prrgpp000/cpa_enhanced/datasets/reconstructions/y"
cartella_destinazione_groundtruth = "/home/prrgpp000/image-restoration-sde/examples_data/y"
cartella_destinazione_selezionati = "/home/prrgpp000/image-restoration-sde/examples_data/x"
folder_to_check = "/home/prrgpp000/cpa_enhanced/datasets/reconstructions/train_set"

# elimina tutti file da cartella destinazione groundtruth e selezionati
shutil.rmtree(cartella_destinazione_groundtruth, ignore_errors=True)
shutil.rmtree(cartella_destinazione_selezionati, ignore_errors=True)

seleziona_e_copia_file_casuali(cartella_origine, numero_file, cartella_groundtruth, cartella_destinazione_groundtruth, cartella_destinazione_selezionati, folder_to_check)
