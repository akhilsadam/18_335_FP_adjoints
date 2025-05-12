su - $1 <<!
$2
user=$whoami
echo $whoami
cd $PWD/../../../
echo "j-runtime/$(ls -rt $PWD/../../../j-runtime/| tail -n 1)"
!