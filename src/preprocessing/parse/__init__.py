"""Parse stage: PDF → structured :class:`schemas.paper.Paper` representation.

The active backend is MinerU's cloud API, wrapped by :mod:`mineru_adapter`
(used by the agent runtime) and :mod:`mineru` (the local CLI fallback used
by the execution stage's ``prepare`` node).
"""
