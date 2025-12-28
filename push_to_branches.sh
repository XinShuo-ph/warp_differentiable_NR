#!/bin/bash
cd "$(dirname "$0")"

# Save the instructions file content
cp instructions_wrapup.md /tmp/instructions_wrapup_nr.md

# List of branch suffixes for NR repo
branches="0a7f 0d97 1183 16a3 2b4b 2eb4 3a28 5800 7134 8b82 9052 95d7 99cb bd28 c374 c633"

for suffix in $branches; do
  branch="cursor/following-instructions-md-$suffix"
  echo "=== Processing $branch ==="
  
  # Checkout the branch
  git checkout -B $branch origin/$branch
  if [ $? -ne 0 ]; then
    echo "Failed to checkout $branch, skipping..."
    continue
  fi
  
  # Find where NR folder is and copy the instructions file
  if [ -d "NR" ]; then
    cp /tmp/instructions_wrapup_nr.md NR/instructions_wrapup.md
  else
    cp /tmp/instructions_wrapup_nr.md instructions_wrapup.md
  fi
  
  # Add and commit
  git add -A
  git commit -m "Add wrapup instructions for branch consolidation" --allow-empty
  
  # Push
  git push origin $branch
  
  echo ""
done

# Return to main
git checkout main
echo "Done! Pushed to all 16 NR branches."

