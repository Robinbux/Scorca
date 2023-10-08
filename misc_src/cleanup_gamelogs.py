import os
import shutil

def get_new_name(old_name, index):
    """Generate new directory name based on its new index."""
    return f"games_{str(index).zfill(3)}"

def main():
    root_dir = "/Users/robinbux/Desktop/RBC_New/game_logs"
    sub_dirs = sorted(os.listdir(root_dir))
    complete_dirs = []

    # Check each directory for 'settings.json'
    for dir_name in sub_dirs:
        path = os.path.join(root_dir, dir_name)
        if os.path.isdir(path) and "settings.json" in os.listdir(path):
            complete_dirs.append(path)
        else:
            # Remove the directory if 'settings.json' is not present
            shutil.rmtree(path)

    # Rename remaining directories to maintain continuous numbering
    for index, old_path in enumerate(complete_dirs):
        new_name = get_new_name(old_path, index)
        new_path = os.path.join(root_dir, new_name)
        os.rename(old_path, new_path)

if __name__ == "__main__":
    main()
