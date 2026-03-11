#!/usr/bin/env bash
set -euo pipefail

SUBMODULE_PATH="external/acc-models-sps"

echo "Updating submodule: $SUBMODULE_PATH"

# Make sure submodule exists
if [ ! -d "$SUBMODULE_PATH" ]; then
    echo "Error: submodule path '$SUBMODULE_PATH' does not exist."
    exit 1
fi

# Update the submodule itself
git -C "$SUBMODULE_PATH" pull

echo
echo "Submodule is now at:"
git -C "$SUBMODULE_PATH" log --oneline -n 1

echo
echo "Main repo status:"
git status --short

echo
echo "Submodule status:"
git submodule status

# Optional: automatically commit the pointer update
read -r -p "Commit submodule update in main repo? [y/N] " answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
    git add "$SUBMODULE_PATH"
    git commit -m "Update acc-models-sps"
    echo "Committed submodule update."
else
    echo "No commit made."
fi