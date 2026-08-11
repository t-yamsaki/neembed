Inference
=========

The public inference helpers are ``encode()`` and ``distance()`` on
:class:`neembed.ManifoldSentenceTransformer`.

Encoding text
-------------

``encode()`` accepts either one string or a sequence of strings.

.. code-block:: python

   single = model.encode("Shiba Inu")
   batch = model.encode(["Shiba Inu", "dog", "mammal"])

For ``manifold="poincare"``, a single string has shape ``(embedding_dim,)`` and
a sequence has shape ``(batch_size, embedding_dim)``.

For ``manifold="lorentz"``, ``embedding_dim`` remains the intrinsic projected
dimension, while the hyperboloid representation adds one ambient time-like
coordinate. The corresponding shapes are therefore ``(embedding_dim + 1,)``
and ``(batch_size, embedding_dim + 1)``.

NumPy and Tensor output
-----------------------

By default, ``encode()`` returns a NumPy array on CPU:

.. code-block:: python

   embeddings = model.encode(["dog", "cat"])

Set ``convert_to_tensor=True`` to keep the result as a ``torch.Tensor``:

.. code-block:: python

   embeddings = model.encode(
       ["dog", "cat"],
       convert_to_tensor=True,
   )

The Tensor stays on the model device. Lorentz manifold outputs use ``float64``
for numerical stability; Poincare outputs keep the model's ordinary parameter
dtype.

Inference mode
--------------

``encode()`` switches the model to evaluation mode and executes the forward
pass inside ``torch.inference_mode()``. Returned embeddings therefore do not
track gradients and the helper is intended for inference rather than training.

Training code should call the model through its normal ``forward()`` path; the
provided loss does this automatically.

Geodesic distance
-----------------

``distance()`` computes the Geoopt manifold distance between two already
encoded manifold embeddings:

.. code-block:: python

   embeddings = model.encode(["Shiba Inu", "dog"])
   distance = model.distance(embeddings[0], embeddings[1])
   print(float(distance))

Array-like inputs are converted to tensors on the model device. Poincare
distance uses the model parameter dtype, while Lorentz distance is evaluated in
``float64`` to preserve the precision used by the Lorentz manifold path. The
distance calculation runs under ``torch.no_grad()`` and returns a Tensor.
This helper is therefore also inference-oriented; the training loss calls the
manifold distance directly so gradients remain available during optimization.
