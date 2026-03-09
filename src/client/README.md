See `examples/usage.py` for how to consume the client

To run it locally install the `requirements.txt` + `pip install -e .` on the `setup.py`


To deploy to Pypi:

Manually change `client/setup.py` `version=` to the desired version, for instance `version=0.5.0`
```commandline
pip install wheel
pip install twine
Python setup.py sdist
Python setup.py bdist_wheel
twine upload dist/*
```
(currently pypi account is configured with Shay's private account - we need to change that)
