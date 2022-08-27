#!/usr/bin/env bash

echo "Checking cuda ops"
if [[ ! $(pip freeze | grep -i MultiScaleDeformableAttention) ]]; then
  echo "Installing cuda ops"
  pushd models/ops
  sh ./make.sh
  python test.py || exit 1
  popd
fi

echo "Checking dino backbone"
if [[ ! -f models/dino_resnet50_pretrain.pth ]]; then
  echo "Downloading pretrained dino backbone"
  wget -O models/dino_resnet50_pretrain.pth https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth || exit 1
fi
