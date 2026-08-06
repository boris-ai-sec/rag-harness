"""Controlled synthetic corpus for metadata boundary tests."""

CONTROLLED_CORPUS = [
    {
        "tenant_id": "tenant_a",
        "case_id": "case_exp_rag_001_synthetic",
        "document_id": "doc_a_001",
        "source_id": "source_permitted",
        "document_type": "policy",
        "access_level": "permitted",
        "text": "Tenant A permitted source record.",
    },
    {
        "tenant_id": "tenant_a",
        "case_id": "case_exp_rag_001_synthetic",
        "document_id": "doc_a_002",
        "source_id": "source_other",
        "document_type": "policy",
        "access_level": "restricted",
        "text": "Tenant A other source record.",
    },
    {
        "tenant_id": "tenant_b",
        "case_id": "case_exp_rag_001_synthetic",
        "document_id": "doc_b_001",
        "source_id": "source_external",
        "document_type": "report",
        "access_level": "restricted",
        "text": "Tenant B isolated source record.",
    },
    {
        "case_id": "case_exp_rag_001_synthetic",
        "document_id": "doc_invalid_001",
        "source_id": "source_invalid",
        "document_type": "report",
        "access_level": "restricted",
        "text": "Malformed record without tenant identifier.",
    },
]
