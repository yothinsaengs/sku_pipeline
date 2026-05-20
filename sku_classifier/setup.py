from setuptools import setup, find_packages

setup(
    name="sku_classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "timm",
        "pyyaml",
        "numpy",
        "opencv-python",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "sku-clf=sku_clf.cli:main",
            "sku-clf-infer=sku_clf.infer:main",
        ],
    },
)
