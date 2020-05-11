echo "Executing Events test with one event - add"
./assignment13.exe "add"

echo "Executing Events test with two event - add,sub"
./assignment13.exe "add" "sub"

echo "Executing Events test with three event - add,sub,mult"
./assignment13.exe "add" "sub" "mult"

echo "Executing Events test with three event out of order queue"
./assignment13.exe "add" "sub" "mult" -outorder

echo "Executing Events test with four event - add,sub,mult,sub"
./assignment13.exe "add" "sub" "mult" "sub"