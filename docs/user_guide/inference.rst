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

For a single string, the returned shape is ``(embedding_dim,)``. For a
sequence, the returned shape is ``(batch_size, embedding_dim)``.

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

The Tensor stays on the model device.

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

Array-like inputs are converted to tensors using the model's device and dtype.
The distance calculation runs under ``torch.no_grad()`` and returns a Tensor.
This helper is therefore also inference-oriented; the training loss calls the
manifold distance directly so gradients remain available during optimization.
