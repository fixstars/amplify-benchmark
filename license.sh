SRC_DIR=src
EXCEPT_LAST_LIST=(
    '.json'
    '.html'
    '.yml'
)
EXCEPT_LIST=(
    '\data'
)

LICENSE=$(cat docs/license_block.txt)

FIND_COMMAND="find '$SRC_DIR' -type f"
for VALUE in ${EXCEPT_LAST_LIST[@]}; do
  FIND_COMMAND="$FIND_COMMAND | grep -v \\$VALUE$"
done
for VALUE in ${EXCEPT_LIST[@]}; do
  FIND_COMMAND="$FIND_COMMAND | grep -v \\$VALUE"
done

eval $FIND_COMMAND | while read FILE; do
  if grep -qz "Copyright (c) Fixstars Corporation and Fixstars Amplify Corporation." $FILE; then
      continue
  fi
  CONTENTS=$(cat "$FILE")
  echo """$LICENSE

$CONTENTS""" > "$FILE"
done;
