import os

# Keras 3 defaults to the tensorflow backend, which this fork does not lock.
os.environ.setdefault("KERAS_BACKEND", "torch")
