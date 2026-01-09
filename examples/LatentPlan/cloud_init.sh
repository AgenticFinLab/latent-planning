#!/bin/bash

# Demo:
# $ chmod +x cloud_init.sh
# $ ./cloud_init.sh ACL25 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 [ENV_NAME]

# ===============================
# STEP 0: Process management
# ===============================
echo "Checking for 'Google Drive' or 'WindowServer' processes..."

kill_process() {
    local name="$1"
    echo -n "Checking for '$name' processes... "
    pids=$(pgrep -f "$name")
    
    if [ -n "$pids" ]; then
        echo "Found (PIDs: $pids). Stopping..."
        kill -9 $pids
    else
        echo "Not found"
    fi
}

kill_process "Google Drive"
kill_process "WindowServer"
echo "Done checking for background processes."
echo

# -----------------------------
# ARGUMENT CHECKS
# -----------------------------
if [ $# -lt 2 ]; then
    echo "Usage: $0 <subdirectory> <github_token> [env_name]"
    echo "Example: $0 mytest abc123... LatentPlan"
    exit 1
fi

SUBDIR="$1"
GITHUB_TOKEN="$2"
ENV_NAME="${3:-LatentPlan}"  # Default environment name if not provided
TARGET_DIR="$HOME/$SUBDIR"

# GitHub configuration - Add more repos here following the pattern
declare -A REPOS=(
    # Format: ["repo_name"]="owner/org <setup_command>"
    ["acl25-plan-code"]="CSJDeveloper 'pip install .; git pull origin main; python examples/LatentPlan/data_prepare.py -c examples/LatentPlan/configs/MATH/gpt4o-DSynt_data_to_hf.yml -b ICMLPlan'"
    ["dmmrl"]="CSJDeveloper 'pip install .'"
    ["open-dmmrl"]="CSJDeveloper 'pip install .'"
    ["project-init"]="CSJDeveloper 'pip install .'"
    
)

# -----------------------------
# STEP 1: DIRECTORY SETUP
# -----------------------------
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || { echo "Failed to change directory to $TARGET_DIR"; exit 1; }

# -----------------------------
# STEP 2: MINICONDA INSTALLATION
# -----------------------------
MINICONDA_DIR="$TARGET_DIR/miniconda"
CONDA_SCRIPT="$MINICONDA_DIR/etc/profile.d/conda.sh"

install_miniconda() {
    if [ ! -d "$MINICONDA_DIR" ]; then
        echo "Installing Miniconda..."
        conda_installer="Miniconda3-latest-Linux-x86_64.sh"
        wget "https://repo.anaconda.com/miniconda/$conda_installer"
        bash "$conda_installer" -b -p "$MINICONDA_DIR"
        rm "$conda_installer"
    else
        echo "Miniconda already installed at $MINICONDA_DIR"
    fi
}

install_miniconda || { echo "Miniconda installation failed"; exit 1; }

# -----------------------------
# STEP 3: CONDA ENVIRONMENT SETUP
# -----------------------------
setup_conda_env() {
    source "$CONDA_SCRIPT" 2>/dev/null || { echo "Conda initialization failed"; exit 1; }
    
    if ! conda env list | grep -q "$ENV_NAME"; then
        echo "Creating conda environment '$ENV_NAME'..."
        conda create -n "$ENV_NAME" python=3.11 -y
    else
        echo "Conda environment '$ENV_NAME' already exists"
    fi
    
    conda activate "$ENV_NAME" || { echo "Failed to activate environment"; exit 1; }
}

setup_conda_env

# -----------------------------
# STEP 4: REPOSITORY MANAGEMENT
# -----------------------------
clone_and_setup() {
    local repo="$1"
    local owner=$(echo "${REPOS[$repo]}" | awk '{print $1}')
    local command=$(echo "${REPOS[$repo]}" | cut -d' ' -f2-)
    local repo_dir="$TARGET_DIR/$repo"
    
    if [ ! -d "$repo_dir" ]; then
        echo "Cloning $repo..."
        git clone "https://${GITHUB_TOKEN}@github.com/${owner}/${repo}.git" || {
            echo "Failed to clone $repo";
            return 1
        }
        
        cd "$repo_dir" || { echo "Failed to enter $repo directory"; return 1; }
        echo "Setting up $repo..."
        eval "$command" || { echo "Failed to setup $repo"; cd ..; return 1; }
        cd ..
    else
        echo "Repository $repo already exists at $repo_dir"
        cd "$repo_dir" || { echo "Failed to verify $repo installation"; return 1; }
        echo "Updating $repo..."
        git pull origin main
        cd ..
    fi
}

for repo in "${!REPOS[@]}"; do
    clone_and_setup "$repo" || echo "Warning: Failed to process $repo"
done

# -----------------------------
# STEP 5: POST-INSTALL CHECKS
# -----------------------------
echo -e "\nVerifying installations:"
for repo in "${!REPOS[@]}"; do
    repo_dir="$TARGET_DIR/$repo"
    if [ -d "$repo_dir" ]; then
        echo -e "\n✓ $repo:"
        cd "$repo_dir" && git log -1 --oneline
        cd ..
    else
        echo -e "\n✗ $repo: Missing"
    fi
done

echo -e "\nSetup completed successfully in $TARGET_DIR"
echo "Conda environment: $ENV_NAME"
echo "Installed repositories:"
printf " - %s\n" "${!REPOS[@]}"