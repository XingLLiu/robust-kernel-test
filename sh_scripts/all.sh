date > test.txt
echo mixture >> test.txt
sh sh_scripts/mixture.sh
date >> test.txt

echo rbm >> test.txt
sh sh_scripts/rbm.sh
date >> test.txt

echo tail >> test.txt
sh sh_scripts/tail.sh
date >> test.txt

echo ms >> test.txt
sh sh_scripts/ms_dim.sh
date >> test.txt