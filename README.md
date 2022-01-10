<div align="right" style="text-align:right"><i>Asif Rahman<br>2022</i></div>

# Index of Jupyter Notebooks

1. [Physionet 2019 preprocessing](ehr/physionet2019-prepare-data.ipynb)
2. [Physionet 2012 preprocessing](ehr/physionet2012-prepare-data.ipynb)

# Setup

Some methods use Spark for data processing. To install Spark on MacOS, run `brew install apache-spark`. You will need to install the python modules [pyspark](https://pypi.org/project/pyspark/) and [findspark](https://pypi.org/project/findspark/). All required python modules are in [requirements.txt](requirements.txt).

Generic utility functions are in [utils.py](utils.py). During development, you can load (and reload) the utils module using:

```
# Add the parent notebook folder to the modules path
parent = '../'
if parent not in sys.path: sys.path.append(parent)

import importlib
import utils
importlib.reload(utils)
```
