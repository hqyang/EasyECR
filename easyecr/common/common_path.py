import os
import tempfile

# tempfile.tempdir = "/home/nobody/tmp/"
tempfile.tempdir = "/home/nobody/code/tmp/"


project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"

model_dir = os.path.join(project_dir, "model")
os.makedirs(model_dir, exist_ok=True)

cache_dir = os.path.join(project_dir, "cache")
# cache_dir = "/shared_space/nobody/ecr-code"
os.makedirs(cache_dir, exist_ok=True)

if __name__ == "__main__":
    print(project_dir)
