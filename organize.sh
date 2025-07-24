#!/bin/bash

# This script organizes the BhashaSetu project files into a clean directory structure.
# Run this script from the root directory where all your current files are located.

echo "--- Starting Project Organization ---"

# Create the new directory structure
echo "Creating directories: app/, frontend/, infra/"
mkdir -p app
mkdir -p frontend
mkdir -p infra

# Move the application files
echo "Moving application files into app/..."
mv main.py app/
mv requirements.txt app/

# Move the frontend file
echo "Moving frontend file into frontend/..."
mv index.html frontend/

# Move the infrastructure files
echo "Moving infrastructure files into infra/..."
mv main.tf infra/
mv startup-script.sh infra/

# Assuming a README.md exists, if not, this will do nothing.
# If you don't have one, you can create it with: touch README.md
if [ -f "README.md" ]; then
    echo "Keeping README.md at the root."
fi

echo ""
echo "✅ Project organization complete!"
echo ""
echo "IMPORTANT NEXT STEPS:"
echo "1. Your Terraform script is now in the 'infra/' directory. Run 'terraform apply' from inside that folder."
echo "2. You MUST update your 'infra/startup-script.sh' to reflect the new file paths for your application to run correctly on GCP."
echo "3. You will also need to update your git repository with this new structure."