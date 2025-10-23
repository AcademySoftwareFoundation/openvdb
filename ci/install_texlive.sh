#!/usr/bin/env bash
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

set -ex

# Download the TeX Live installer
wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz

# Extract
tar xzf install-tl-unx.tar.gz
rm install-tl-unx.tar.gz
cd install-tl-*

# Run the installer with no docs or source installation
./install-tl --no-interaction --no-doc-install --no-src-install --scheme=minimal

# Export the texlive PATH to GITHUB_PATH
TEXLIVE_PATH=/usr/local/texlive/2025/bin/x86_64-linux
export PATH=${TEXLIVE_PATH}:${PATH}
echo "${TEXLIVE_PATH}" >> $GITHUB_PATH

# Install the required packages
tlmgr install latex-bin dvips collection-fontsrecommended collection-latexrecommended

# Validate that installing dvips and latex has been successful
dvips --version
latex --version
