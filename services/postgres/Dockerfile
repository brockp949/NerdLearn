
FROM postgres:15

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-15 \
    flex \
    bison \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector
RUN cd /tmp && git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install

# Install Apache AGE
# Cloning specific release compatible with PG15
RUN cd /tmp && git clone --branch PG15/v1.5.0-rc0 https://github.com/apache/age.git \
    && cd age \
    && make install

# Cleanup
RUN rm -rf /tmp/pgvector /tmp/age

