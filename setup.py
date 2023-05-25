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
    package_data={
        'ecog2txt': [
            'auxiliary/block_breakdowns.json',
            'auxiliary/example_experiment_manifest.yaml',
            'auxiliary/vocab.mocha-timit.1806',
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
        'numpy==1.22.4',
        'scipy', 'matplotlib', 'pandas', 'seaborn', 'tikzplotlib',
        'dopamine-rl==2.0.5', 'jax==0.3.13', 'jaxlib==0.3.10', 'flax==0.4.2',
        'python_speech_features', 'pyyaml', 'hickle==3.4.6',
        # 'protobuf==3.7',
        'protobuf==3.12.2',
        'tensor2tensor==1.15.7', 'tensorflow-probability==0.7',
        # 'samplerate',
        # 'tensorflow-gpu==1.15.3'  # the cpu version will also work
    ],
)
