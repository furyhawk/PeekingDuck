#!/bin/sh

MINLINTSCORE=9

if ! (pylint --fail-under=$MINLINTSCORE --extension-pkg-whitelist=cv2 peekingduck); then
    echo "PYLINT ERROR: score below required lint score"
    exit 123
else
    echo "PYLINT SUCCESS!!"
fi

echo "CHECKCODE: CONGRATULATIONS, ALL TESTS SUCCESSFUL!!"