RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

OK=${GREEN}OK${NC}
KO=${RED}KO${NC}
EXEC=${BLUE}${1}${NC}

printf "${EXEC}: "
