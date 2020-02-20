import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ecog2txt",
    version="0.6.0",
    author="J.G. Makin",
    author_email="jgmakin@gmail.com",
    description="Code for decoding speech as text from neural data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jgmakin/ecog2txt",
    packages=setuptools.find_packages(),
    #packages=setuptools.find_packages(
    #    exclude=[
    #        'neural_models', 'robotics', 'toys',
    #        'machine_learning.undirected_graphical_models',
    #        '*exponential_families*'
    #    ],
    #),
    package_data={
        '': [
            'block_breakdowns.json',
            'example_experiment_manifest.yaml',
            'vocab.mocha-timit.1806',
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy', 'scipy', 'h5py', 'matplotlib', 'pandas', 'seaborn',
        'tikzplotlib', 'tfmpl', 'tensor2tensor==1.14.1', 'hickle', 'pickle',
        'python_speech_features', 'pyyaml', 'tensorflow-probability>=0.7',
        'protobuf>=3.7',
        # 'samplerate', 'tensorflow-gpu==1.14'
        # 'bamboo', 'RT'
    ],
)