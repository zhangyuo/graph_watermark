from setuptools import setup, find_packages

setup(
    name="graph_watermark",
    version="0.1",
    description="Graph watermarking library",
    author="XXX",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "numpy",
        "scipy",
    ],
    include_package_data=True,
)