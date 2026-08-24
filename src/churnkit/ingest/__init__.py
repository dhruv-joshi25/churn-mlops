"""Reading an operator's file well enough that nothing downstream has to guess.

Platform code: no column name, category or business rule from any particular
dataset may appear here (I11).
"""

from churnkit.ingest.infer import (
    ColumnRole,
    ConfirmedMapping,
    LeakageFinding,
    LeakageOverrideRequired,
    SchemaProposal,
    SkippedCheck,
    TargetCandidate,
    UnconfirmedProposalError,
    infer_schema,
)
from churnkit.ingest.reader import (
    MAX_FILE_BYTES,
    ColumnStats,
    FileTooLargeError,
    ParseResult,
    RowFailure,
    read_table,
)

__all__ = [
    "MAX_FILE_BYTES",
    "ColumnRole",
    "ColumnStats",
    "ConfirmedMapping",
    "FileTooLargeError",
    "LeakageFinding",
    "LeakageOverrideRequired",
    "ParseResult",
    "RowFailure",
    "SchemaProposal",
    "SkippedCheck",
    "TargetCandidate",
    "UnconfirmedProposalError",
    "infer_schema",
    "read_table",
]
