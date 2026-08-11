Installation
============

Python support
--------------

neembed v0.1 requires Python 3.10 or newer and is tested on Python 3.10,
3.11, and 3.12.

Install from PyPI
-----------------

Install the published release from PyPI with:

.. code-block:: bash

   pip install neembed-geoopt

The PyPI distribution is named ``neembed-geoopt`` while the Python import
package remains ``neembed``.

Runtime dependencies
--------------------

The runtime dependency surface is intentionally small:

* ``torch``
* ``sentence-transformers``
* ``geoopt``

neembed uses Sentence Transformers for the pretrained encoder and Geoopt for
Poincare-ball geometry rather than reimplementing either subsystem.

Install from source
-------------------

For development, clone the repository and install the development extra:

.. code-block:: bash

   git clone https://github.com/t-yamsaki/neembed.git
   cd neembed
   pip install -e ".[dev]"

The development extra currently adds pytest for the test suite.
