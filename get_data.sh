files=(\
13cPzQoeojyve-ZWm9U_XHbZAqhyWtC_I \
#1C48qbJyQ-O94VhKOBMdMsEKga32fF_6C \
#18ffIdGVc7vJBp4z8UOrYIto_gFAwvMfz \
#1tYJ9XZ3B0wOTbZfGdAKn6Odq59HpttdW \
#194KTHzlQ_6ZMrHY07G6ueDz9Mt5o8O7X \
#19ABuU4FOloh0-w6-6z1Gb96N_9pxf5sV \
#1p5LZ7H3_O7405PIeDg-cTVv3vkm9sMhu \
#1T1gBoS4UCfbUvNvlpZKdZXGLLWrrnLrv \
#1Q5uizGscixfawMh3BqNR6ZuIC6BbnM5I \
)

export directory=sim_data
export filename=sim_data_file

cd /opt
mkdir $directory

for fileid in "${files[@]}"
do
	## WGET ##
	wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

	wget --load-cookies cookies.txt -O $fileid \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
     
	unzip -q $fileid -d $directory
done

export DATA_PATH=/opt/sim_data
     
