import os
import subprocess
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

def get_requirements():
   return [
      "numpy",
   ]

class CMakeExtension(Extension):

    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=True, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)

class cmake_build_ext(build_ext):
    def run(self):
        print("Running cmake build")
        build_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "build")
        os.makedirs(build_dir, exist_ok=True)
        
        current_dir = os.getcwd()
        os.chdir(build_dir)
        
        try:
            subprocess.check_call(['cmake', '..'])
            subprocess.check_call(['make'])
            subprocess.check_call(['make', 'install'])
        finally:
            os.chdir(current_dir)

ext_modules = []
ext_modules.append(CMakeExtension(name="mytorch._C"))

cmdclass = {
    "build_ext": cmake_build_ext
}


setup(
    name='mytorch',
    version='0.1',
    author='Your Name',
    author_email='your.email@example.com',
    description='A custom implementation of PyTorch',
    packages=find_packages(exclude=("docker", "csrc", "tests*")),
    python_requires=">=3.9",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
) 