"""Single source of truth for schema-level constants shared across modules.

Before 2026-08-23 these lived as duplicate literals in ``model/rhgnn.py``
and (for ``FLOW_RELATIONS``) in ``cl/ewc.py``, so a Step 2 feature-set or
relation-schema change could update one copy and silently desynchronize
the others.

``FLOW_RELATIONS`` is Flow's incoming relations in the 11-relation Step 2
schema (the reverse edges added 2026-07-19) -- see ``docs/dataset-plan.md``
§3.2. ``FLOW_FEATURE_DIM``/``HOST_FEATURE_DIM`` must match
``configs/graph.yaml``'s ``features`` list and the Host aggregates built in
``graphs.py``; a test asserting that agreement lives in
``tests/test_integration.py``.
"""

FLOW_FEATURE_DIM = 37  # len(configs/graph.yaml: features)
HOST_FEATURE_DIM = 4  # total_flows, avg_bytes_as_src, avg_bytes_as_dst, unique_ports_contacted

WELL_KNOWN_PORT_MAX = 1023  # inclusive upper bound of the IANA well-known port range

FLOW_RELATIONS = ["originates", "terminated_by", "targeted_by", "protocol_of", "service_of"]
