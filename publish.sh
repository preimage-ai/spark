#!/usr/bin/env bash
set -euo pipefail

# Publish this package to CodeArtifact.
#
# Usage:
#   npm run publish:ca            # publish the current version
#   npm run publish:ca -- 1.0.2   # bump to 1.0.2, then publish
#
# NOTE ON VERSIONS: the version here is this fork's own line and has nothing to
# do with the upstream Spark version it is built from -- `preimage.sparkBase` in
# package.json records that. Do not set the version to match an upstream tag;
# that is exactly what collided before (2.4.0-2.4.2 were published off a Spark
# 2.3.1 base, leaving upstream's real 2.4.0 with nowhere to go).
#
# This script regenerates dist/ itself (wasm then TS) before publishing, so the
# published artifact always matches the current Rust and TS sources. It does NOT
# rely on `prepublishOnly` for that, because the publish runs with
# --ignore-scripts to avoid re-entering this script via the `publish` lifecycle.
# Requires wasm-pack and the wasm32-unknown-unknown target.
#
# Auth goes into a throwaway npmrc passed via npm_config_userconfig, so nothing
# touches ~/.npmrc or the repo .npmrc. The registry is passed explicitly so
# publishConfig cannot redirect the publish elsewhere.
#
# Override the target with CA_DOMAIN / CA_OWNER / CA_REGION / CA_REPO.

CA_DOMAIN="${CA_DOMAIN:-preimage-v2}"
CA_OWNER="${CA_OWNER:-847195926226}"
CA_REGION="${CA_REGION:-us-east-2}"
CA_REPO="${CA_REPO:-preimage}"

NEW_VERSION="${1:-}"

cd "$(dirname "${BASH_SOURCE[0]}")"

CA_HOST="${CA_DOMAIN}-${CA_OWNER}.d.codeartifact.${CA_REGION}.amazonaws.com"
REGISTRY="https://${CA_HOST}/npm/${CA_REPO}/"

PKG_NAME="$(node -p "require('./package.json').name")"

# @preimage/email -> namespace "preimage", package "email"
PKG_NAMESPACE="${PKG_NAME#@}"
PKG_NAMESPACE="${PKG_NAMESPACE%%/*}"
PKG_BASENAME="${PKG_NAME##*/}"

if [[ -n "$NEW_VERSION" ]]; then
  npm version "$NEW_VERSION" --no-git-tag-version >/dev/null
fi

PKG_VERSION="$(node -p "require('./package.json').version")"

echo "publishing ${PKG_NAME}@${PKG_VERSION}"
echo "        -> ${REGISTRY}"

# This package's version is OUR line, not upstream Spark's (see `preimage` in
# package.json for why they had to be split). Print the upstream base alongside
# it so the publish log records the mapping - otherwise "which Spark is 3.0.1?"
# is only answerable by digging through git history.
SPARK_BASE="$(node -p "require('./package.json').preimage?.sparkBase ?? 'UNRECORDED'")"
echo "   spark base: ${SPARK_BASE}"
if [[ "$SPARK_BASE" == "UNRECORDED" ]]; then
  echo "warning: preimage.sparkBase is missing from package.json." >&2
  echo "         Set it to the upstream tag this fork is rebased onto." >&2
fi

# Refuse to overwrite an existing version. Republishing the same version produces
# a different tarball hash, which breaks --frozen-lockfile installs downstream.
if aws codeartifact describe-package-version \
  --domain "$CA_DOMAIN" --domain-owner "$CA_OWNER" --region "$CA_REGION" \
  --repository "$CA_REPO" --format npm \
  --namespace "$PKG_NAMESPACE" \
  --package "$PKG_BASENAME" --package-version "$PKG_VERSION" >/dev/null 2>&1; then
  echo "error: ${PKG_NAME}@${PKG_VERSION} already exists in ${CA_DOMAIN}." >&2
  echo "       bump instead: npm run publish:ca -- <new-version>" >&2
  exit 1
fi

TOKEN="$(aws codeartifact get-authorization-token \
  --domain "$CA_DOMAIN" --domain-owner "$CA_OWNER" --region "$CA_REGION" \
  --query authorizationToken --output text)"

# The repo-root .npmrc is project config, which outranks userconfig - and it reads
# the token from ${CODEARTIFACT_TOKEN}. Unset, that expands to an empty token and
# wins over anything we put in the temp npmrc, giving a confusing E401. Export it.
export CODEARTIFACT_TOKEN="$TOKEN"

# Temp npmrc covers the case where no project .npmrc exists (fresh clone, CI).
NPMRC="$(mktemp)"
trap 'rm -f "$NPMRC"' EXIT
chmod 600 "$NPMRC"
cat > "$NPMRC" <<EOF
//${CA_HOST}/npm/${CA_REPO}/:_authToken=${TOKEN}
EOF

# Built explicitly rather than through prepublishOnly, because the publish below
# passes --ignore-scripts (see the note there). build:wasm needs wasm-pack and
# the wasm32-unknown-unknown target.
echo "building wasm..."
npm run build:wasm
echo "building dist..."
npm run build

# --ignore-scripts is REQUIRED, not an optimisation. `publish` is a reserved npm
# lifecycle name that runs AFTER a successful publish, and this package exposes
# a `publish` script pointing back at this file -- so without it, npm would
# re-enter this script, re-run `npm publish`, and only stop when the version
# collision check above aborts the second pass, reporting a failure immediately
# after a publish that actually succeeded.
npm_config_userconfig="$NPMRC" npm publish --registry "$REGISTRY" --ignore-scripts

echo "published ${PKG_NAME}@${PKG_VERSION}"
