#!/bin/bash
#NOTE: DO NOT EDIT THIS FILE-- MAY RESULT IN INCOMPLETE SUBMISSIONS
set -euo pipefail

# these notebooks should ideally
# be in order of questions so
# that the generated pdf is
# in order of questions
NOTEBOOKS=(
	"forward-backward-kl.ipynb"
)

FILES=( "${NOTEBOOKS[@]}" )

LOCAL_DIR=`pwd`
ASSIGNMENT_NO=1
ZIP_FILENAME="hw7.zip"

C_R="\e[31m"
C_G="\e[32m"
C_BLD="\e[1m"
C_E="\e[0m"

for FILE in "${FILES[@]}"
do
	if [ ! -f ${FILE} ]; then
		echo -e "${C_R}Required file ${FILE} not found, Exiting.${C_E}"
		exit 0
	fi
done

echo -e "### Zipping file ###"
rm -f ${ZIP_FILENAME}
zip -q "${ZIP_FILENAME}" -r ${NOTEBOOKS[@]} $(find . -name "*.py") -x "makepdf.py"

echo -e "### Creating PDFs ###"
python makepdf.py --notebooks "${NOTEBOOKS[@]}"

echo -e "### Done! Please submit the zip and the pdfs to ETL. ###"
