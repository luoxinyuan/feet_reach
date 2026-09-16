from setuptools import find_packages, setup

setup(
    name="active_adaptation",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "mujoco",
        "wandb",
        "moviepy",
        "imageio",
        "xxhash",
        "einops",
        "termcolor",
        "setproctitle",
        "torch==2.7.0",
        "tensordict==0.7.2",
        "torchrl==0.7.2",
        "onnxscript",
        "onnxruntime"
    ],
)
