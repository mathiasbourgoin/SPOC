#!/bin/bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2024-2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Check and validate SPDX license headers in source files
# - Ensures SPDX-License-Identifier: CECILL-B is present
# - Validates copyright years (creation year and last modification year)
# - Tracks contributors based on git history

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Expected license
EXPECTED_LICENSE="CECILL-B"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counters
MISSING_COUNT=0
OUTDATED_COUNT=0
CHECKED_COUNT=0

echo "Checking SPDX license headers..."
echo "Expected: SPDX-License-Identifier: $EXPECTED_LICENSE"
echo ""

# Get copyright info from git for a file
get_file_copyright_info() {
    local file="$1"
    
    # Get first commit year
    local first_year=$(git log --follow --format=%aI --reverse "$file" 2>/dev/null | head -1 | cut -d- -f1)
    
    # Get last commit year
    local last_year=$(git log --follow --format=%aI -1 "$file" 2>/dev/null | cut -d- -f1)
    
    # Get all contributors (email addresses)
    local contributors=$(git log --follow --format=%ae "$file" 2>/dev/null | sort -u | tr '\n' ' ')
    
    # If git info not available, use current year
    if [ -z "$first_year" ]; then
        first_year=$(date +%Y)
        last_year=$(date +%Y)
        contributors=""
    fi
    
    # Return: first_year|last_year|contributors
    echo "$first_year|$last_year|$contributors"
}

# Check OCaml/ML file header
check_ocaml_header() {
    local file="$1"
    local info=$(get_file_copyright_info "$file")
    local first_year=$(echo "$info" | cut -d'|' -f1)
    local last_year=$(echo "$info" | cut -d'|' -f2)
    local contributors=$(echo "$info" | cut -d'|' -f3)
    
    CHECKED_COUNT=$((CHECKED_COUNT + 1))
    
    # Read first 20 lines
    local header=$(head -20 "$file")
    
    # Check for SPDX identifier
    if ! echo "$header" | grep -q "SPDX-License-Identifier: $EXPECTED_LICENSE"; then
        echo -e "${RED}MISSING LICENSE${NC}: $file"
        echo "  Expected: (* SPDX-License-Identifier: $EXPECTED_LICENSE *)"
        MISSING_COUNT=$((MISSING_COUNT + 1))
        return 1
    fi
    
    # Check copyright notice exists
    if ! echo "$header" | grep -q "SPDX-FileCopyrightText:"; then
        echo -e "${YELLOW}MISSING COPYRIGHT${NC}: $file"
        echo "  Should have: (* SPDX-FileCopyrightText: $first_year-$last_year ... *)"
        OUTDATED_COUNT=$((OUTDATED_COUNT + 1))
        return 1
    fi
    
    # Check if year range is current
    if [ "$first_year" != "$last_year" ]; then
        # Multi-year file, check if last year is present
        if ! echo "$header" | grep -q "SPDX-FileCopyrightText:.*$last_year"; then
            echo -e "${YELLOW}OUTDATED YEAR${NC}: $file"
            echo "  Last modified: $last_year (header may need update)"
            OUTDATED_COUNT=$((OUTDATED_COUNT + 1))
            return 1
        fi
    fi
    
    return 0
}

# Check shell script header
check_shell_header() {
    local file="$1"
    local info=$(get_file_copyright_info "$file")
    local first_year=$(echo "$info" | cut -d'|' -f1)
    local last_year=$(echo "$info" | cut -d'|' -f2)
    
    CHECKED_COUNT=$((CHECKED_COUNT + 1))
    
    local header=$(head -20 "$file")
    
    if ! echo "$header" | grep -q "SPDX-License-Identifier: $EXPECTED_LICENSE"; then
        echo -e "${RED}MISSING LICENSE${NC}: $file"
        MISSING_COUNT=$((MISSING_COUNT + 1))
        return 1
    fi
    
    if ! echo "$header" | grep -q "SPDX-FileCopyrightText:"; then
        echo -e "${YELLOW}MISSING COPYRIGHT${NC}: $file"
        OUTDATED_COUNT=$((OUTDATED_COUNT + 1))
        return 1
    fi
    
    return 0
}

# Check dune file header
check_dune_header() {
    local file="$1"
    local info=$(get_file_copyright_info "$file")
    local first_year=$(echo "$info" | cut -d'|' -f1)
    local last_year=$(echo "$info" | cut -d'|' -f2)
    
    CHECKED_COUNT=$((CHECKED_COUNT + 1))
    
    local header=$(head -20 "$file")
    
    if ! echo "$header" | grep -q "SPDX-License-Identifier: $EXPECTED_LICENSE"; then
        echo -e "${RED}MISSING LICENSE${NC}: $file"
        MISSING_COUNT=$((MISSING_COUNT + 1))
        return 1
    fi
    
    return 0
}

# Process OCaml files
echo "Checking OCaml files (.ml, .mli)..."
while IFS= read -r -d '' file; do
    check_ocaml_header "$file" || true
done < <(find sarek sarek-cuda sarek-opencl sarek-vulkan sarek-metal spoc \
    -type f \( -name "*.ml" -o -name "*.mli" \) \
    ! -path "*/.*" \
    ! -path "*/_build/*" \
    ! -path "*/_opam/*" \
    ! -path "*/dependencies/*" \
    -print0 2>/dev/null)

# Process shell scripts
echo ""
echo "Checking shell scripts (.sh)..."
while IFS= read -r -d '' file; do
    check_shell_header "$file" || true
done < <(find scripts ci \
    -type f -name "*.sh" \
    ! -path "*/.*" \
    -print0 2>/dev/null)

# Process dune files (optional check)
# Uncomment if you want to enforce headers in dune files
# echo ""
# echo "Checking dune files..."
# while IFS= read -r -d '' file; do
#     check_dune_header "$file" || true
# done < <(find sarek sarek-cuda sarek-opencl sarek-vulkan sarek-metal spoc \
#     -type f -name "dune" \
#     ! -path "*/.*" \
#     ! -path "*/_build/*" \
#     ! -path "*/_opam/*" \
#     -print0 2>/dev/null)

# Print summary
echo ""
echo "========================================"
echo "License Header Check Summary"
echo "========================================"
echo "Files checked: $CHECKED_COUNT"

if [ $MISSING_COUNT -eq 0 ] && [ $OUTDATED_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓ All files have proper SPDX headers!${NC}"
    exit 0
else
    [ $MISSING_COUNT -gt 0 ] && echo -e "${RED}✗ Files missing license: $MISSING_COUNT${NC}"
    [ $OUTDATED_COUNT -gt 0 ] && echo -e "${YELLOW}⚠ Files with outdated/incomplete copyright: $OUTDATED_COUNT${NC}"
    echo ""
    echo "SPDX Header Format:"
    echo ""
    echo "OCaml files (.ml, .mli):"
    echo "  (* SPDX-License-Identifier: CECILL-B *)"
    echo "  (* SPDX-FileCopyrightText: YYYY-YYYY Contributor Name <email> *)"
    echo ""
    echo "Shell scripts (.sh):"
    echo "  # SPDX-License-Identifier: CECILL-B"
    echo "  # SPDX-FileCopyrightText: YYYY-YYYY Contributor Name <email>"
    echo ""
    echo "For files modified across multiple years, use: YYYY-YYYY format"
    echo "Add multiple contributors with separate SPDX-FileCopyrightText lines"
    echo ""
    exit 1
fi
