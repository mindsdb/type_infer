.. -*- coding: utf-8 -*-
.. documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

****************************************
Type Infer
****************************************

:Release: |release|
:Date: |today|
|
Welcome to the ``type_infer`` documentation. ``type_infer`` is a Python package aimed at automatically inferring the data type for each column in a tabular dataset.

Quick Guide
=======================
- :ref:`Installation <Installation>`
- :ref:`Quick start <Quick start>`
- :ref:`Contribute <Contributions>`

Installation
============

You can install ``type_infer`` as follows:

.. code-block:: bash

   pip install type_infer

We recommend doing the above inside a newly-created python virtual environment.

Setting up a dev environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Clone the repository
- Run ``cd type_infer && pip install --editable .``
- Add it to your python path (e.g. by adding ``export PYTHONPATH='/where/you/cloned/repo':$PYTHONPATH`` as a newline at the end of your ``~/.bashrc`` file)
- Check that the unit-tests are passing by going into the directory where you cloned and running: ``python -m unittest discover tests``

.. warning:: If ``python`` default to python2.x on your environment use ``python3`` and ``pip3`` instead


Quick start
=======================

``type_infer`` works with ``pandas.DataFrames``.

.. code-block:: python

   import type_infer


Contributions
=======================

We love to receive contributions from the community and hear your opinions! We want to make contributing as easy as it can be.

Please continue reading this guide if you are interested.

How can you help us?
^^^^^^^^^^^^^^^^^^^^^^^^
* Report a bug
* Improve documentation
* Solve an issue
* Propose new features
* Discuss feature implementations
* Submit a bug fix
* Test with your own data and let us know how it went!

Code contributions
^^^^^^^^^^^^^^^^^^^^^^^^
In general, we follow the `fork-and-pull <https://docs.github.com/en/github/collaborating-with-pull-requests/getting-started/about-collaborative-development-models#fork-and-pull-model>`_ git workflow. Here are the steps:

1. Fork the repository
2. Checkout the ``staging`` branch, which is the development version that gets released into ``stable`` (there can be exceptions, but make sure to ask and confirm with us).
3. Make changes and commit them
4. Make sure that the CI tests pass. You can run the test suite locally with ``flake8 .`` to check style and ``python -m unittest discover tests`` to run the automated tests. This doesn't guarantee it will pass remotely since we run on multiple envs, but should work in most cases.
5. Push your local branch to your fork
6. Submit a pull request from your repo to the ``staging`` branch of ``mindsdb/type_infer`` so that we can review your changes. Be sure to merge the latest from staging before making a pull request!

.. note:: You will need to sign a CLI agreement for the code since the repository is under a GPL license.


Feature and Bug reports
^^^^^^^^^^^^^^^^^^^^^^^^
We use GitHub issues to track bugs and features. Report them by opening a `new issue <https://github.com/mindsdb/type_infer/issues/new/choose>`_ and fill out all of the required inputs.


Code review process
^^^^^^^^^^^^^^^^^^^^^^^^^
Pull request (PR) reviews are done on a regular basis. **If your PR does not address a previous issue, please make an issue first**.

If your change can affect performance, we will run our private benchmark suite to validate it.

Please, make sure you respond to our feedback/questions.


Community
^^^^^^^^^^^^^^^^^^^^^^^^^
If you have additional questions or you want to chat with MindsDB core team, you can join our community:

.. raw:: html

    <embed>
    <a href="https://join.slack.com/t/mindsdbcommunity/shared_invite/zt-o8mrmx3l-5ai~5H66s6wlxFfBMVI6wQ" target="_blank"><img src="https://img.shields.io/badge/slack-@mindsdbcommunity-blueviolet.svg?logo=slack " alt="MindsDB Community"></a>
    </embed>

To get updates on MindsDB’s latest announcements, releases, and events, sign up for our `Monthly Community Newsletter <https://mindsdb.com/newsletter/?utm_medium=community&utm_source=github&utm_campaign=lightwood%20repo>`_.

Join our mission of democratizing machine learning and allowing developers to become data scientists!

Contributor Code of Conduct
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Please note that this project is released with a `Contributor Code of Conduct <https://github.com/mindsdb/lightwood/blob/stable/CODE_OF_CONDUCT.md>`_. By participating in this project, you agree to abide by its terms.


License
=======================
.. raw:: html

    <embed>
    <img src="https://img.shields.io/pypi/l/lightwood" alt="PyPI - License">
    </embed>

| `License <https://github.com/mindsdb/type_infer/blob/stable/LICENSE>`_


Other Links
=======================
.. toctree::
   :maxdepth: 8

   philosophy
   tutorials
   base
   dtype
   infer
   helpers
