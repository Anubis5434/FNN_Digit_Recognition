#!/usr/bin/env bash

# https://stackoverflow.com/a/246128/3692822
# https://stackoverflow.com/questions/59895/getting-the-source-directory-of-a-bash-script-from-within
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

python3 "${DIR}/grader_part1.py" q1 student
python3 "${DIR}/grader_part1.py" q2 student
python3 "${DIR}/grader_part1.py" q3 student
python3 "${DIR}/grader_part1.py" q4 student
