#!/usr/bin/env bash
if [[ ! $(pip freeze | grep -i MultiScaleDeformableAttention) ]]; then
  echo "Installing cuda ops"
  pushd models/ops
  sh ./make.sh
  python test.py || exit 1
  popd
fi