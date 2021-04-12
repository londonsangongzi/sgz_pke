from distutils.core import setup

setup(name='sgz-pke',
      version='1.8.1',
      description='Python Keyphrase Extraction module',
      author='pke contributors',
      author_email='florian.boudin@univ-nantes.fr',
      license='gnu',
      packages=['sgz_pke', 'sgz_pke.unsupervised', 'sgz_pke.supervised',
                'sgz_pke.supervised.feature_based', 'sgz_pke.unsupervised.graph_based',
                'sgz_pke.unsupervised.statistical', 'sgz_pke.supervised.neural_based'],
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
