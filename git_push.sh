#!/bin/bash

# Enhanced GitHub Push Script for Quantum FFT State Analyzer
# Repository: https://github.com/xavlolo/apiQuantum.git

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

print_debug() {
    echo -e "${PURPLE}[DEBUG]${NC} $1"
}

# Debug function to show current state
show_debug_info() {
    print_debug "=== DEBUG INFORMATION ==="
    print_debug "Current directory: $(pwd)"
    print_debug "Contents of current directory:"
    ls -la
    echo
    print_debug "Git repository info:"
    if [ -d ".git" ]; then
        print_debug "✓ .git directory found"
        print_debug "Remote URL: $(git remote get-url origin 2>/dev/null || echo 'No remote found')"
        print_debug "Current branch: $(git branch --show-current 2>/dev/null || echo 'No branch found')"
    else
        print_debug "✗ No .git directory found"
    fi
    echo
    print_debug "Git status details:"
    git status 2>/dev/null || print_debug "Git status failed"
    echo
    print_debug "All files in directory (including hidden):"
    find . -maxdepth 2 -type f -name "*.py" -o -name "*.sh" -o -name "*.txt" -o -name "*.csv" -o -name "*.pkl" 2>/dev/null
    print_debug "=========================="
    echo
}

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    print_error "Not in a git repository!"
    echo
    show_debug_info
    
    print_status "Let's try to fix this..."
    
    # Check if we can find a git repo nearby
    if [ -d "../.git" ]; then
        print_warning "Found git repository in parent directory"
        read -p "Move to parent directory? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd ..
            print_status "Moved to $(pwd)"
        else
            exit 1
        fi
    elif [ -d "apiQuantum/.git" ]; then
        print_warning "Found git repository in apiQuantum subdirectory"
        read -p "Move to apiQuantum directory? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd apiQuantum
            print_status "Moved to $(pwd)"
        else
            exit 1
        fi
    else
        print_error "No git repository found. Please run this script from your apiQuantum directory."
        print_status "To initialize git repository, run:"
        print_status "  git init"
        print_status "  git remote add origin https://github.com/xavlolo/apiQuantum.git"
        exit 1
    fi
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

# Enhanced git status check
print_status "Checking git status..."
git status --porcelain > /tmp/git_status.txt

# Show detailed status regardless
print_debug "=== DETAILED GIT STATUS ==="
git status
print_debug "============================"

# Check for untracked files specifically
UNTRACKED_FILES=$(git ls-files --others --exclude-standard)
if [ ! -z "$UNTRACKED_FILES" ]; then
    print_warning "Found untracked files:"
    echo "$UNTRACKED_FILES"
    echo
    read -p "Add all untracked files? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        print_status "Added all files to staging area"
    fi
fi

# Re-check status after potential add
git status --porcelain > /tmp/git_status.txt

if [ ! -s /tmp/git_status.txt ]; then
    print_warning "No changes detected. Nothing to commit."
    
    # Show what files exist vs what git knows about
    print_debug "=== FILES IN DIRECTORY ==="
    find . -maxdepth 2 -type f ! -path "./.git/*" | head -20
    print_debug "=== FILES TRACKED BY GIT ==="
    git ls-tree -r --name-only HEAD 2>/dev/null | head -20
    print_debug "=== UNTRACKED FILES ==="
    git ls-files --others --exclude-standard | head -20
    print_debug "=========================="
    
    read -p "Force add all files and try again? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add -A
        git status --porcelain > /tmp/git_status.txt
        if [ ! -s /tmp/git_status.txt ]; then
            print_error "Still no changes after adding all files."
            exit 0
        fi
    else
        exit 0
    fi
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
    ADDED=$(git status --porcelain | grep -E "^A|^\?\?" | wc -l | tr -d ' ')
    MODIFIED=$(git status --porcelain | grep "^.M" | wc -l | tr -d ' ')
    DELETED=$(git status --porcelain | grep "^.D" | wc -l | tr -d ' ')
    
    # Build commit message
    COMMIT_MSG="Update quantum analyzer: "
    CHANGES=()
    
    [ "$ADDED" -gt 0 ] && CHANGES+=("$ADDED added")
    [ "$MODIFIED" -gt 0 ] && CHANGES+=("$MODIFIED modified")
    [ "$DELETED" -gt 0 ] && CHANGES+=("$DELETED deleted")
    
    if [ ${#CHANGES[@]} -gt 0 ]; then
        IFS=', '
        COMMIT_MSG="$COMMIT_MSG${CHANGES[*]} - $TIMESTAMP"
    else
        COMMIT_MSG="Add quantum FFT analyzer files - $TIMESTAMP"
    fi
fi

print_status "Commit message: '$COMMIT_MSG'"

# Add all changes (more aggressive)
print_status "Adding all changes..."
git add -A

if [ $? -ne 0 ]; then
    print_error "Failed to add changes. Aborting."
    exit 1
fi

# Show what's actually being committed
print_status "Files being committed:"
git diff --cached --name-only

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

# Check if branch exists on remote
print_status "Checking remote branch..."
git ls-remote --heads origin $CURRENT_BRANCH >/dev/null 2>&1
BRANCH_EXISTS=$?

if [ $BRANCH_EXISTS -ne 0 ]; then
    print_warning "Branch '$CURRENT_BRANCH' doesn't exist on remote"
    print_status "Creating new branch on remote..."
    git push -u origin $CURRENT_BRANCH
else
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
fi

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