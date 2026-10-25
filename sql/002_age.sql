-- AGE is installed beside vector in the custom PostgreSQL image.
-- Existing databases install the extension through age_graph._context.
LOAD 'age';
CREATE EXTENSION IF NOT EXISTS age;
