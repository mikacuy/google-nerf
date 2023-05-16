import gdown
import zipfile
import subprocess
import urllib.request
import os

# nerf
print('Downloading NeRF blender files')
gdown.download("https://drive.google.com/uc?id=1RjwxZCUoPlUgEWIUiuCmMmG0AhuV8A2Q",
               "nerf_blender.zip")
os.system("unzip nerf_blender.zip")
os.system("mv blend_files/*.blend .")
os.system("rm nerf_blender.zip")
os.system("rm -r blend_files/")
os.system("rm -r __MACOSX")
