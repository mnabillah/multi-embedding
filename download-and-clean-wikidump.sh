#!/bin/sh
set -e

# set path to download indonesian wikidump file
WIKI_DUMP_DIR=./corpus/idwiki
WIKI_DUMP_NAME=idwiki-latest-pages-articles.xml.bz2
WIKI_DUMP_DOWNLOAD_URL=https://dumps.wikimedia.org/idwiki/latest/$WIKI_DUMP_NAME

# create directory to save wikidump file (if not exist yet)
if [ ! -d $WIKI_DUMP_DIR ]; then
  mkdir -p $WIKI_DUMP_DIR
fi

# download latest Wikipedia dump in chosen language
echo "Mengunduh dari $WIKI_DUMP_DOWNLOAD_URL..."
if [ ! -f $WIKI_DUMP_DIR/$WIKI_DUMP_NAME ]; then
  wget -c $WIKI_DUMP_DOWNLOAD_URL -P $WIKI_DUMP_DIR
  echo "Unduh berhasil. Hasil unduhan ada di $WIKI_DUMP_DIR/$WIKI_DUMP_NAME"
else
  echo "Berkas wikidump sudah terunduh"
fi
#bunzip2 -kvv  $WIKI_DUMP_NAME ${WIKI_DUMP_NAME%.bz2}

# set wiki dump file input and output path
WIKI_DUMP_FILE_IN=$WIKI_DUMP_DIR/$WIKI_DUMP_NAME
WIKI_DUMP_FILE_OUT=$WIKI_DUMP_DIR/${WIKI_DUMP_FILE_IN%.xml.bz2}.txt

# check if wikiextractor is already cloned or not
echo "Meng-clone repo wikiextractor"
if [ ! -d ./wikiextractor ]; then
  git clone https://github.com/attardi/wikiextractor.git
fi
echo "Repo sudah di-clone"

# extract and clean the chosen Wikipedia dump
echo "Meng-extract dan membersihkan $WIKI_DUMP_FILE_IN ke $WIKI_DUMP_FILE_OUT dalam $WIKI_DUMP_DIR..."
python wikiextractor/WikiExtractor.py $WIKI_DUMP_FILE_IN --processes 8 -q -o - \
| sed "/^\s*\$/d" \
| grep -v "^<doc id=" \
| grep -v "</doc>\$" \
> $WIKI_DUMP_FILE_OUT
echo "Sukses meng-extract dan membersihkan $WIKI_DUMP_FILE_IN ke $WIKI_DUMP_FILE_OUT dalam $WIKI_DUMP_DIR"

# remove wikiextractor repo
rmdir wikiextractor -r -fo