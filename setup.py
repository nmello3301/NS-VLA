from setuptools import setup, find_packages

setup(
    name="nsvla",
    version="0.1.0",
    description="NS-VLA: Towards Neuro-Symbolic Vision-Language-Action Models",
    author="Ziyue Zhu, Shangyang Wu, Shuai Zhao, et al.",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
    ],
)
