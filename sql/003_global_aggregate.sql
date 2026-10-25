CREATE TABLE IF NOT EXISTS global_qa_build (
    qa_build_id bigserial PRIMARY KEY,
    source_build_id bigint NOT NULL,
    tokenizer_version text NOT NULL,
    signature_top_k integer NOT NULL CHECK (signature_top_k > 0),
    n_docs integer NOT NULL CHECK (n_docs >= 0),
    complete boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS global_qa_build_lookup
    ON global_qa_build (source_build_id, tokenizer_version, qa_build_id DESC)
    WHERE complete;

CREATE TABLE IF NOT EXISTS global_qa_doc (
    qa_build_id bigint NOT NULL REFERENCES global_qa_build(qa_build_id) ON DELETE CASCADE,
    paper_id text NOT NULL,
    community_id integer NOT NULL,
    text_sha256 text NOT NULL,
    PRIMARY KEY (qa_build_id, paper_id)
);
CREATE INDEX IF NOT EXISTS global_qa_doc_community
    ON global_qa_doc (qa_build_id, community_id);

CREATE TABLE IF NOT EXISTS global_qa_community (
    qa_build_id bigint NOT NULL REFERENCES global_qa_build(qa_build_id) ON DELETE CASCADE,
    community_id integer NOT NULL,
    n_docs integer NOT NULL CHECK (n_docs > 0),
    PRIMARY KEY (qa_build_id, community_id)
);

CREATE TABLE IF NOT EXISTS global_qa_term_df (
    qa_build_id bigint NOT NULL REFERENCES global_qa_build(qa_build_id) ON DELETE CASCADE,
    community_id integer NOT NULL,
    term text NOT NULL,
    df integer NOT NULL CHECK (df > 0),
    PRIMARY KEY (qa_build_id, community_id, term)
);
CREATE INDEX IF NOT EXISTS global_qa_term_df_term
    ON global_qa_term_df (qa_build_id, term);

CREATE TABLE IF NOT EXISTS global_qa_signature (
    qa_build_id bigint NOT NULL REFERENCES global_qa_build(qa_build_id) ON DELETE CASCADE,
    community_id integer NOT NULL,
    term text NOT NULL,
    keyness double precision NOT NULL,
    df integer NOT NULL CHECK (df > 0),
    PRIMARY KEY (qa_build_id, community_id, term)
);
