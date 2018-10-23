from setuptools import setup, find_packages

setup(
    name="crypto-defense",
    use_scm_version=True,
    packages=find_packages(),
    setup_requires=['setuptools_scm'],

    # put your dependencies here
    install_requires=[
        # our packages
        # 3rd party packages
        'setuptools>=38.5.1,<=39.1.0',
        'matplotlib>=2.1.2',
        'pillow',
        'tensorflow_gpu>=1.10.0',
        'PyYAML'
    ],
    include_package_data=True,
    author="Autonomos GmbH",
    author_email="support@autonomos-systems.de",
    description="Utility library to implement neural networks with TensorFlow.",
    license="proprietary",
    keywords="machine learning, TensorFlow, library",
    url="https://www.autonomos-systems.de",
    platforms="linux")
