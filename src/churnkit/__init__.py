"""churnkit — an open-source churn intelligence platform.

The package is deliberately split in two:

``churnkit.ingest``, ``churnkit.validation``, ``churnkit.features`` and the
modules that follow them are *platform* code. Nothing in them may name a column
from any particular dataset; everything flows from the schema mapping an
operator confirms at upload time (I11).

``churnkit.reference`` is the single-dataset implementation the project started
from. It hardcodes one company's schema, which is why it is quarantined in its
own subpackage and covered by a guard test rather than being scattered through
the platform modules. Each reference module is deleted as the platform module
that replaces it lands. See docs/decisions/0003-churnkit-package-layout.md.
"""

__version__ = "0.1.0.dev0"
