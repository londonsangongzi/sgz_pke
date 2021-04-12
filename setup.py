from distutils.core import setup

setup(name='sgz-pke',
      version='1.8.1',
      description='Python Keyphrase Extraction module',
      author='pke contributors',
      author_email='florian.boudin@univ-nantes.fr',
      license='gnu',
      packages=['pke', 'pke.unsupervised', 'pke.supervised',
                'pke.supervised.feature_based', 'pke.unsupervised.graph_based',
                'pke.unsupervised.statistical', 'pke.supervised.neural_based'],
      url="https://github.com/londonsangongzi/sgz_pke",
      install_requires=[
          'nltk',
          'networkx',
          'numpy',
          'scipy',
          'spacy',
          'six',
          'sklearn',
          'unidecode',
          'future',
          'joblib'
      ],
      package_data={'sgz_pke': ['models/*.pickle', 'models/*.gz']}
      )
