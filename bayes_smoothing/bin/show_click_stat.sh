#!/bin/sh
# 统计query数量
cd `dirname $0`
if (( $# == 1 ))
then
    inputdate=$(date -d "$1" +%Y-%m-%d)
elif (( $# == 0 ))
then
    inputdate=$(date -d "-1days" +%Y-%m-%d)
else
    Usage="sh $0 end_date[YYYY-MM-DD]"
    echo "param num error"
    echo "Usage: $Usage"
    exit 1
fi
param_file=../params.prop
if [ -f "${param_file}" ]; then
    get_name=$(sed '/inputdate/!d;s/.*=//' ${param_file})
    if echo ${get_name} | grep -Eq "[0-9]{4}-[0-9]{2}-[0-9]{2}" && date -d ${get_name} +%Y-%m-%d > /dev/null 2>&1; then
        echo "using inputdate from ${param_file}"
        inputdate=${get_name}
    fi
fi
echo "$BASH_SOURCE $inputdate"
source ../lib/header.sh

${SPARK_SUBMIT} \
    --master yarn-cluster \
    --archives ${PY27PATH}#python27 \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=${PY27BIN} \
    --conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=${PY27BIN} \
    --name  show_click_stat.${inputdate} \
    --conf spark.default.parallelism=3000 \
    --driver-memory 8G \
    --driver-cores 1 \
    --executor-memory 4G \
    --executor-cores 3 \
    --num-executors 50 \
    ../python/show_click_stat.py  --input_dir=${ndaysInput} --output_dir=${output_dir} --inputdate=${inputdate}

