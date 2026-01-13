FROM python:3.11-slim

ARG DENO_VERSION=2.1.0
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DENO_INSTALL=/usr/local

# OS deps: unzip for Deno installer; libgfortran5 for SciPy wheels (treequest)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       curl ca-certificates bash git tini unzip libgfortran5 \
  && rm -rf /var/lib/apt/lists/*

# Deno 2.x
RUN curl -fsSL https://deno.land/install.sh | sh -s -- -q \
  && chmod 0755 /usr/local/bin/deno \
  && /usr/local/bin/deno --version \
  && /usr/local/bin/deno run --node-modules-dir=auto -A - <<'JS'
console.log("deno OK with --node-modules-dir=auto");
JS

# MCP server
RUN pip install --no-cache-dir mcp-run-python==0.0.21

# Requirements
COPY requirements.txt /tmp/requirements.txt
# Prefer wheels; so it doesn't crash the build on transient mirrors
RUN pip install --no-cache-dir --prefer-binary -r /tmp/requirements.txt

# Non-root user
RUN useradd -m -u 10001 mcp
USER mcp
WORKDIR /home/mcp

# Deno cache dir for created user
ENV DENO_DIR=/home/mcp/.deno_dir
RUN mkdir -p "$DENO_DIR"

# Entrypoint script
COPY --chown=mcp:mcp entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 3333
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/usr/local/bin/entrypoint.sh","streamable-http"]
