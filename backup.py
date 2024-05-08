import os
import shutil

def backup_tensorflow_settings(backup_dir):
    # Ensure the backup directory exists or create it if not
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # Paths to TensorFlow configuration files
    tensorflow_config_path = os.path.expanduser("~/.keras")
    if not os.path.exists(tensorflow_config_path):
        print("TensorFlow configuration directory not found.")
        return
    
    # Backup TensorFlow configuration files
    try:
        # List of configuration files to backup
        config_files = ["keras.json"]
        
        # Copy each configuration file to the backup directory
        for file in config_files:
            src_path = os.path.join(tensorflow_config_path, file)
            dest_path = os.path.join(backup_dir, file)
            shutil.copyfile(src_path, dest_path)
            print(f"Backed up {file} to {backup_dir}")
        
        print("TensorFlow settings backup completed successfully.")
    
    except Exception as e:
        print("An error occurred during backup:", str(e))

# Specify the directory where you want to store the backups
backup_directory = "/path/to/backup/directory"

# Call the function to perform the backup
backup_tensorflow_settings(backup_directory)
