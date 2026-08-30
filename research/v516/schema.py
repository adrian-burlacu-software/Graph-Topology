
SCHEMA = r"""
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sources (
    source_id INTEGER PRIMARY KEY,
    dataset TEXT NOT NULL,
    source_path TEXT NOT NULL,
    record_key TEXT,
    content_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(dataset,source_path,record_key,content_type)
);

CREATE TABLE IF NOT EXISTS concepts (
    concept_id INTEGER PRIMARY KEY,
    canonical TEXT NOT NULL,
    display TEXT NOT NULL,
    concept_type TEXT NOT NULL,
    UNIQUE(canonical,concept_type)
);

CREATE TABLE IF NOT EXISTS fact_evidence (
    evidence_id INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL REFERENCES concepts(concept_id),
    predicate TEXT NOT NULL,
    object_id INTEGER REFERENCES concepts(concept_id),
    object_text TEXT,
    fact_type TEXT NOT NULL,
    source_id INTEGER NOT NULL REFERENCES sources(source_id),
    confidence REAL NOT NULL DEFAULT 1.0,
    weight REAL NOT NULL DEFAULT 1.0,
    UNIQUE(subject_id,predicate,object_id,object_text,fact_type,source_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_evidence_subject
ON fact_evidence(subject_id);

CREATE INDEX IF NOT EXISTS idx_fact_evidence_predicate
ON fact_evidence(predicate);

CREATE TABLE IF NOT EXISTS facts (
    fact_id INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL REFERENCES concepts(concept_id),
    predicate TEXT NOT NULL,
    object_id INTEGER REFERENCES concepts(concept_id),
    object_text TEXT,
    fact_type TEXT NOT NULL,
    domain TEXT,
    source_id INTEGER REFERENCES sources(source_id),
    confidence REAL NOT NULL DEFAULT 1.0,
    frequency REAL NOT NULL DEFAULT 1.0,
    answerable INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(subject_id,predicate,object_id,object_text,fact_type,source_id)
);

CREATE TABLE IF NOT EXISTS utterances (
    utterance_id INTEGER PRIMARY KEY,
    dataset TEXT NOT NULL,
    text TEXT NOT NULL,
    speaker TEXT,
    dialogue_id TEXT,
    turn_index INTEGER,
    intent TEXT,
    domain TEXT,
    source_id INTEGER REFERENCES sources(source_id),
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS dialogue_links (
    link_id INTEGER PRIMARY KEY,
    from_utterance INTEGER NOT NULL REFERENCES utterances(utterance_id),
    to_utterance INTEGER NOT NULL REFERENCES utterances(utterance_id),
    relation TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verbnet_classes (
    class_id TEXT PRIMARY KEY,
    parent_class TEXT,
    source_id INTEGER REFERENCES sources(source_id),
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS verbnet_members (
    member_id INTEGER PRIMARY KEY,
    class_id TEXT NOT NULL,
    verb TEXT NOT NULL,
    wn_refs TEXT,
    fn_refs TEXT,
    source_id INTEGER REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS verbnet_roles (
    role_id INTEGER PRIMARY KEY,
    class_id TEXT NOT NULL,
    role TEXT NOT NULL,
    restrictions_json TEXT NOT NULL DEFAULT '[]',
    source_id INTEGER REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS verbnet_frames (
    frame_id INTEGER PRIMARY KEY,
    class_id TEXT NOT NULL,
    frame_index INTEGER NOT NULL,
    description TEXT,
    example TEXT,
    syntax_json TEXT NOT NULL DEFAULT '[]',
    semantics_json TEXT NOT NULL DEFAULT '[]',
    source_id INTEGER REFERENCES sources(source_id)
);

CREATE INDEX IF NOT EXISTS idx_verbnet_members_verb
ON verbnet_members(verb);

CREATE INDEX IF NOT EXISTS idx_verbnet_roles_class
ON verbnet_roles(class_id);

CREATE INDEX IF NOT EXISTS idx_verbnet_frames_class
ON verbnet_frames(class_id);


CREATE TABLE IF NOT EXISTS udgum_sentences (
    sentence_id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    sent_id TEXT,
    text TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS udgum_tokens (
    token_id INTEGER PRIMARY KEY,
    sentence_id INTEGER NOT NULL REFERENCES udgum_sentences(sentence_id),
    position TEXT NOT NULL,
    form TEXT NOT NULL,
    lemma TEXT,
    upos TEXT,
    xpos TEXT,
    feats TEXT,
    head TEXT,
    deprel TEXT,
    deps TEXT,
    misc TEXT
);

CREATE INDEX IF NOT EXISTS idx_udgum_lemma
ON udgum_tokens(lemma);

CREATE INDEX IF NOT EXISTS idx_udgum_upos
ON udgum_tokens(upos);


CREATE TABLE IF NOT EXISTS ubuntu_pairs (
    pair_id INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,
    container_index INTEGER NOT NULL,
    row_index INTEGER NOT NULL,
    label_text TEXT,
    context_length INTEGER,
    response_length INTEGER,
    user_text TEXT,
    reply_text TEXT,
    UNIQUE(source_path,container_index,row_index)
);

CREATE TABLE IF NOT EXISTS ubuntu_pair_tokens (
    pair_id INTEGER NOT NULL REFERENCES ubuntu_pairs(pair_id),
    side TEXT NOT NULL,
    position INTEGER NOT NULL,
    token_id INTEGER,
    token_text TEXT,
    PRIMARY KEY(pair_id,side,position)
);

CREATE TABLE IF NOT EXISTS ubuntu_token_vocab (
    token_id INTEGER PRIMARY KEY,
    token_text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ubuntu_matrices (
    name TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    rows INTEGER,
    cols INTEGER,
    dtype TEXT,
    storage_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);


CREATE TABLE IF NOT EXISTS live_entities (
    entity_id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    canonical TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    mention_turn INTEGER NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    UNIQUE(session_id,ordinal)
);

CREATE INDEX IF NOT EXISTS idx_live_entities_subject
ON live_entities(session_id,canonical);

CREATE TABLE IF NOT EXISTS live_turns (
    turn_id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    timestamp REAL NOT NULL,
    UNIQUE(session_id,turn_index,speaker)
);

CREATE TABLE IF NOT EXISTS live_facts (
    live_fact_id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object_text TEXT NOT NULL,
    fact_type TEXT NOT NULL,
    negated INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 1.0,
    turn_index INTEGER NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject_id);
CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate);
CREATE INDEX IF NOT EXISTS idx_facts_domain ON facts(domain);
CREATE INDEX IF NOT EXISTS idx_facts_type ON facts(fact_type);
CREATE INDEX IF NOT EXISTS idx_utterances_dataset ON utterances(dataset);
CREATE INDEX IF NOT EXISTS idx_utterances_dialogue ON utterances(dialogue_id);
CREATE INDEX IF NOT EXISTS idx_live_facts_session ON live_facts(session_id);
CREATE INDEX IF NOT EXISTS idx_live_facts_subject ON live_facts(subject);

CREATE VIRTUAL TABLE IF NOT EXISTS utterance_fts USING fts5(
    text,
    dataset UNINDEXED,
    intent UNINDEXED,
    domain UNINDEXED,
    content='utterances',
    content_rowid='utterance_id'
);

CREATE TRIGGER IF NOT EXISTS utterances_ai AFTER INSERT ON utterances BEGIN
    INSERT INTO utterance_fts(rowid,text,dataset,intent,domain)
    VALUES(new.utterance_id,new.text,new.dataset,new.intent,new.domain);
END;
"""
