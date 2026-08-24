"""Reading an operator's file well enough that nothing downstream has to guess.

Platform code: no column name, category or business rule from any particular
dataset may appear here (I11).
"""

from churnkit.ingest.reader import (
    ColumnStats,
    ParseResult,
    RowFailure,
    read_table,
)

__all__ = ["ColumnStats", "ParseResult", "RowFailure", "read_table"]
