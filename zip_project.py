import os
import zipfile

def zip_project():
    zip_path = "project_colab.zip"
    exclude_dirs = {".venv", "model", "__pycache__", ".git"}

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk("."):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                if file.endswith(".zip") or file == "zip_project.py":
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, ".")
                zipf.write(file_path, arcname)

    print(f"Project zipped successfully to {zip_path}")

if __name__ == "__main__":
    zip_project()
