import os
import shutil
from pathlib import Path
import subprocess
import time

def copy_script_to_all_folders(script_name):
    folders = [f for f in Path('.').iterdir() if f.is_dir()]
    for folder in folders:
        destination = folder / script_name
        if not destination.exists():
            shutil.copy2(script_name, destination)

def submit_jobs(start, end):
    base_folder_name = "split_"
    max_cores = 64
    running_jobs = []

    for i in range(start, end + 1):
        folder_name = base_folder_name + str(i)
        folder_path = Path(folder_name)
        
        if not folder_path.exists():
            print(f"Folder {folder_name} does not exist. Skipping...")
            continue
        
        if len(running_jobs) >= max_cores:
            # Wait for a job to finish before starting a new one
            running_jobs = [job for job in running_jobs if job.poll() is None]
            time.sleep(1)
        

        print(f"Starting job in {folder_name}...")
        job = subprocess.Popen("nohup python run_neighbor.py &> run.log &", shell=True, cwd=folder_path)
        running_jobs.append(job)
    

    for job in running_jobs:
        job.wait()

    print("All specified tasks completed.")

if __name__ == "__main__":
    copy_script_to_all_folders('run_neighbor.py')
    # Example usage: submit jobs from folder split_1 to split_64
    submit_jobs(1, 64)
