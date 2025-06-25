#!/bin/bash

# GitHub Push Script for Quantum FFT State Analyzer
# Repository: https://github.com/xavlolo/apiQuantum.git

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "Not in a git repository. Please run this script from your quantum project directory."
    exit 1
fi

# Check if we're in the correct repository
REPO_URL=$(git remote get-url origin 2>/dev/null)
if [[ "$REPO_URL" != *"apiQuantum"* ]]; then
    print_warning "Repository URL doesn't match apiQuantum. Current: $REPO_URL"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Aborted by user."
        exit 1
    fi
fi

print_status "Starting GitHub push process..."

# Check git status
print_status "Checking git status..."
git status --porcelain > /tmp/git_status.txt

if [ ! -s /tmp/git_status.txt ]; then
    print_warning "No changes detected. Nothing to commit."
    exit 0
fi

# Show what will be committed
print_status "Files to be committed:"
git status --short

# Prompt for commit message
echo
read -p "Enter commit message (or press Enter for auto-generated message): " COMMIT_MSG

# Generate auto commit message if none provided
if [ -z "$COMMIT_MSG" ]; then
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    
    # Count different types of changes
    ADDED=$(git status --porcelain | grep "^A" | wc -l | tr -d ' ')
    MODIFIED=$(git status --porcelain | grep "^.M" | wc -l | tr -d ' ')
    DELETED=$(git status --porcelain | grep "^.D" | wc -l | tr -d ' ')
    UNTRACKED=$(git status --porcelain | grep "^??" | wc -l | tr -d ' ')
    
    # Build commit message
    COMMIT_MSG="Update quantum analyzer: "
    CHANGES=()
    
    [ "$ADDED" -gt 0 ] && CHANGES+=("$ADDED added")
    [ "$MODIFIED" -gt 0 ] && CHANGES+=("$MODIFIED modified")
    [ "$DELETED" -gt 0 ] && CHANGES+=("$DELETED deleted")
    [ "$UNTRACKED" -gt 0 ] && CHANGES+=("$UNTRACKED new files")
    
    if [ ${#CHANGES[@]} -gt 0 ]; then
        IFS=', '
        COMMIT_MSG="$COMMIT_MSG${CHANGES[*]} - $TIMESTAMP"
    else
        COMMIT_MSG="$COMMIT_MSG$TIMESTAMP"
    fi
fi

print_status "Commit message: '$COMMIT_MSG'"

# Add all changes
print_status "Adding all changes..."
git add .

if [ $? -ne 0 ]; then
    print_error "Failed to add changes. Aborting."
    exit 1
fi

# Commit changes
print_status "Committing changes..."
git commit -m "$COMMIT_MSG"

if [ $? -ne 0 ]; then
    print_error "Failed to commit changes. Aborting."
    exit 1
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
print_status "Current branch: $CURRENT_BRANCH"

# Pull latest changes first (in case of conflicts)
print_status "Pulling latest changes from origin/$CURRENT_BRANCH..."
git pull origin $CURRENT_BRANCH

if [ $? -ne 0 ]; then
    print_warning "Pull failed or had conflicts. You may need to resolve conflicts manually."
    read -p "Continue with push anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_error "Push aborted. Please resolve conflicts and try again."
        exit 1
    fi
fi

# Push to GitHub
print_status "Pushing to origin/$CURRENT_BRANCH..."
git push origin $CURRENT_BRANCH

if [ $? -eq 0 ]; then
    print_success "Successfully pushed to GitHub!"
    print_success "Repository: https://github.com/xavlolo/apiQuantum.git"
    
    # Show final status
    echo
    print_status "Final repository status:"
    git log --oneline -n 3
    
else
    print_error "Failed to push to GitHub. Please check your connection and credentials."
    echo
    print_status "You can try manually with:"
    echo "  git push origin $CURRENT_BRANCH"
    exit 1
fi

# Optional: Open GitHub repository in browser
echo
read -p "Open GitHub repository in browser? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v open >/dev/null 2>&1; then
        open "https://github.com/xavlolo/apiQuantum"
    else
        print_status "Please open: https://github.com/xavlolo/apiQuantum"
    fi
fi

print_success "Push complete! 🚀"
