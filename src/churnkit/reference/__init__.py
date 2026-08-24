"""The Telco reference implementation — the only dataset-specific code in src/.

This subpackage predates the platform. It proves the serving architecture end to
end (one Pipeline artifact, a threshold that travels with the model, an API and
a UI that never load a model twice), and it is the thing every platform
component is checked against. It also hardcodes Telco's columns and categories,
which the platform must never do.

Rather than pretend that violation away, it is fenced in here: I11 is enforced
by ``tests/test_layout_guards.py``, which fails if a Telco identifier appears
anywhere in ``src/churnkit/`` outside this directory. The fence shrinks as the
platform modules land — schema.py and data.py go when schema inference and the
label builder replace them, api/ goes when churnkit.serving replaces it.

Keep this package importable without the ML stack: the Streamlit container
installs churnkit without scikit-learn, xgboost, shap or mlflow precisely so it
cannot load a model by accident, and it imports labels.py and schema.py from
here.
"""
