..
    The empty line below should not be removed. It is added such that the `rst_prolog`
    is added before the :mod: directive. Otherwise, the rendering will show as a
    paragraph instead of a header.

:mod:`{{module}}`.\ :spelling:word:`{{objname}}`
{{ underline }}=================================

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __call__
   {% endblock %}

.. raw:: html

    <div class="clearer"></div>
