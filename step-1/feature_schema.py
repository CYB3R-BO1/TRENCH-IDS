"""Feature roles for NetFlow-v3 attack-class similarity analysis.

The same 53-flow-measurement schema is shared by NF-UNSW-NB15-v3 and
NF-ToN-IoT-v3. This module assigns every measurement a role so magnitude metrics
(cosine, Wasserstein) are only applied where a numeric mean/ordering is
meaningful, and protocol-defined codes are kept as categories rather than
discarded.

Roles
-----
MAGNITUDE   continuous/count features: log1p + distribution/mean comparisons
IDENTIFIER  opaque identifiers only (excluded from every numeric metric)
BITMASK     TCP flag bitmasks: analysed as bits, not magnitudes
CATEGORICAL frequency distributions (ports, protocol, encoded protocol codes)
TIMESTAMP   metadata-only (never a feature)
"""

from __future__ import annotations

FLOW_STATS_37 = [
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS",
    "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
]

IAT_8 = [
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]

# Only truly opaque identifiers are excluded from all numeric metrics.
IDENTIFIER_FEATURES = ["DNS_QUERY_ID"]

# TCP flag bitmasks: analysed at the bit level, not as magnitudes.
BITMASK_FEATURES = ["TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"]

# Categorical fields (ports, protocol, service, and protocol-defined encodings).
CATEGORICAL_FEATURES = [
    "PROTOCOL", "L7_PROTO", "L4_SRC_PORT", "L4_DST_PORT", "TCP_FLAGS",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE",
]

# Encoded protocol codes that are categorical, not magnitudes.
CATEGORICAL_NUMERIC = ["ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_TYPE", "FTP_COMMAND_RET_CODE"]

# Magnitude features: continuous/count only.
MAGNITUDE_FEATURES = [
    f for f in FLOW_STATS_37
    if f not in IDENTIFIER_FEATURES
    and f not in BITMASK_FEATURES
    and f not in CATEGORICAL_NUMERIC
] + IAT_8

# Continuous features for the Wasserstein comparison (methodology list).
DIST_FEATURES = [
    "FLOW_DURATION_MILLISECONDS", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]

TIMESTAMP_FEATURES = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]


def feature_roles() -> dict[str, str]:
    roles: dict[str, str] = {}
    for f in MAGNITUDE_FEATURES:
        roles[f] = "MAGNITUDE"
    for f in IDENTIFIER_FEATURES:
        roles[f] = "IDENTIFIER"
    for f in BITMASK_FEATURES:
        roles[f] = "BITMASK"
    for f in CATEGORICAL_FEATURES:
        roles.setdefault(f, "CATEGORICAL")  # bitmask precedence preserved
    for f in TIMESTAMP_FEATURES:
        roles[f] = "TIMESTAMP"
    return roles


def check_schema() -> None:
    assert len(FLOW_STATS_37) == 37
    assert len(IAT_8) == 8
    assert len(MAGNITUDE_FEATURES) == 37, len(MAGNITUDE_FEATURES)
    assert len(set(MAGNITUDE_FEATURES)) == 37
    for f in MAGNITUDE_FEATURES:
        assert f not in IDENTIFIER_FEATURES
        assert f not in BITMASK_FEATURES
        assert f not in CATEGORICAL_NUMERIC
    assert "ICMP_TYPE" in CATEGORICAL_FEATURES
    assert "DNS_QUERY_ID" in IDENTIFIER_FEATURES
