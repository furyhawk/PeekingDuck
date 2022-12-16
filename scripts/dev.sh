DEV_VER='__version__ = "developer"'
TMP_VER='__version__ = "0.0.0.dev1"'
FILE='peekingduck/__init__.py'
TMP_DIR="/tmp/peekingduck-$EPOCHSECONDS"

sed -i "" "s/$DEV_VER/$TMP_VER/g" "$FILE"
pip install --pre .
sed -i "" "s/$TMP_VER/$DEV_VER/g" "$FILE"

mkdir $TMP_DIR
mv build $TMP_DIR
mv *.egg-info $TMP_DIR

echo "Build artifacts stored in $TMP_DIR"