from setuptools import setup, find_namespace_packages

setup(
    name='apprentissage',  # Nom de votre projet
    version='0.1.0',  # Version initiale
    author='venantvr',  # Votre nom ou celui de votre organisation
    author_email='r.venant.valery@gmail.com',  # Votre adresse email
    description='Une librairie Python simple avec une classe Test dans le namespace caching',
    long_description=open('README.md').read(),  # Description longue, souvent le contenu de README.md
    long_description_content_type='text/markdown',  # Type de contenu pour la longue description
    url='https://github.com/venantvr/Python.Rsi.ML.git',  # URL du projet (par exemple, GitHub)
    packages=find_namespace_packages(include=['apprentissage.*']),  # Trouve tous les packages dans le répertoire courant
    classifiers=[
        'Programming Language :: Python :: 3',  # Compatible avec Python 3
        'License :: OSI Approved :: MIT License',  # Type de licence
        'Operating System :: OS Independent',  # Système d'exploitation compatible
    ],
    python_requires='>=3.6',  # Version de Python requise
)
