wget https://ndownloader.figshare.com/files/12855005
apt install unrar
mv 12855005 data.rar
unrar x data.rar ./
rm -rf Caltech101/*/*.fig
rm -rf Caltech101/*/*.bmp