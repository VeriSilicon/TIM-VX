# generate macro ops define for vsi feature ops

for line in `cat $1`
do
    if [[ $line =~ "DEF_OP" ]]
    then
        x=${line#*(}
        echo "#define VSI_FEAT_OP_${x%)*}"
    fi
done