from setuptools import setup

setup(
    name='FaceRipper',
    version='0.1.0',
    packages=['FaceRipper'],
    include_package_data=True,
    install_requires=[
        'Flask',
        'torch',
        'torchvision',
        'numpy',
        'imageio'
    ],
    python_requires='>=3.6',
)